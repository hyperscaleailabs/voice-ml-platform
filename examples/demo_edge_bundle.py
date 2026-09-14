"""The edge path end to end, with nothing but the standard library installed.

Creates a fake adapter directory, plans a GGUF export as a dry run, builds an
`EdgeBundle` from the "exported" files, verifies it, tampers with one file to
show verification failing, then runs one offline turn on the restored bundle
with stub backends and prints the runtime health and a trace summary.

    python examples/demo_edge_bundle.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmp.edge.bundle import build_bundle, diff_bundles, verify_bundle
from vmp.edge.export import ExportPlan, export
from vmp.edge.policy import ACTION_CLOUD_FALLBACK, Action, EdgePolicy, check
from vmp.edge.runtime import EdgeRuntime
from vmp.observability.trace import ListSink, Tracer, format_summary, summarise_turn

BASE_MODEL = "google/gemma-3-270m"
ANSWER = "The bundle is verified. Every file matched its checksum."


class StubSTT:
    """The "audio" carries UTF-8 text, so no model is needed."""

    def transcribe(self, audio: bytes | str) -> str:
        return audio.decode("utf-8") if isinstance(audio, bytes) else str(audio)


class StubLLM:
    def stream(self, messages: list[dict[str, str]]):
        for word in ANSWER.split(" "):
            time.sleep(0.01)  # simulated decode pacing, not a benchmark
            yield word + " "


class StubTTS:
    sample_rate = 24000

    def synthesize(self, text: str) -> bytes:
        time.sleep(0.02)  # simulated synthesis pacing
        return b"\x00\x00" * (len(text) * 24)


def fake_adapter(root: Path) -> Path:
    """A LoRA adapter directory, as `vmp train` would leave behind."""
    adapter = root / "adapters" / "voice-sft-v3"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"lora-weights" * 64)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    return adapter


def fake_export_output(root: Path) -> Path:
    """What a real GGUF export would write into `out_dir`."""
    out = root / "export"
    (out / "tokenizer").mkdir(parents=True)
    (out / "model-q4_k_m.gguf").write_bytes(b"quantised-weights-v1" * 128)
    (out / "config.json").write_text(json.dumps({"base_model": BASE_MODEL, "quant": "q4_k_m"}))
    (out / "tokenizer" / "tokenizer.json").write_text(json.dumps({"vocab_size": 262144}))
    return out


def main() -> int:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vmp-edge-") as tmp:
        root = Path(tmp)
        adapter = fake_adapter(root)

        # -- 1. export (dry run) ------------------------------------------
        plan = ExportPlan(
            base_model=BASE_MODEL,
            target="gguf",
            out_dir=str(root / "export"),
            adapter_path=str(adapter),
            quantization={"method": "q4_k_m"},
        )
        manifest = export(plan, dry_run=True)
        print(f"export plan: {plan.base_model} -> {plan.target} ({plan.method})")
        print(f"  problems: {manifest['problems'] or 'none'}")
        for step in manifest["steps"]:
            requires = ", ".join(step.get("requires", [])) or "stdlib"
            print(f"  [{step['status']:>7}] {step['name']:<16} needs: {requires}")
        print(f"  would produce: {', '.join(manifest['files'])}\n")

        # -- 2. build the bundle ------------------------------------------
        policy = EdgePolicy(offline_only=True, max_ttfa_ms=3000, fallback="degrade")
        src = fake_export_output(root)
        bundle = build_bundle(
            src,
            root / "bundles" / "voice-agent-edge-1.0.0",
            name="voice-agent-edge",
            version="1.0.0",
            target="gguf",
            base_model=BASE_MODEL,
            adapter_version="voice-sft-v3",
            lineage={"artifact": "voice-sft", "version": "3", "git_sha": "0000000"},
            policy=policy,
        )
        print(f"bundle {bundle.name} {bundle.version}  ->  {bundle.path.name}")
        for rel, info in sorted(bundle.files.items()):
            print(f"  {info['size']:>6} bytes  {info['sha256'][:12]}  {rel}")
        print(f"  policy: offline_only={policy.offline_only} fallback={policy.fallback!r}")

        ok, problems = verify_bundle(bundle.path)
        print(f"  verify: ok={ok} problems={problems}\n")

        # -- 3. tamper with a file ----------------------------------------
        target = bundle.path / "model-q4_k_m.gguf"
        original = target.read_bytes()
        target.write_bytes(b"trojan-weights" * 128)
        ok, problems = verify_bundle(bundle.path)
        print("after tampering with model-q4_k_m.gguf")
        print(f"  verify: ok={ok}")
        for problem in problems:
            print(f"    - {problem}")
        try:
            EdgeRuntime(bundle.path, StubSTT(), StubLLM(), StubTTS()).start()
        except RuntimeError as e:
            print(f"  runtime refused to start: {str(e)[:70]}...")
        target.write_bytes(original)
        print(f"  restored: ok={verify_bundle(bundle.path)[0]}\n")

        # -- 4. an OTA update would ship ----------------------------------
        next_src = root / "export2"
        next_src.mkdir()
        for path in sorted(src.rglob("*")):
            if path.is_file():
                destination = next_src / path.relative_to(src)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
        (next_src / "model-q4_k_m.gguf").write_bytes(b"quantised-weights-v2" * 128)
        (next_src / "projector.bin").write_bytes(b"new-in-1.1.0" * 16)
        nxt = build_bundle(
            next_src,
            root / "bundles" / "voice-agent-edge-1.1.0",
            name="voice-agent-edge",
            version="1.1.0",
            target="gguf",
            base_model=BASE_MODEL,
            policy=policy,
        )
        diff = diff_bundles(bundle, nxt)
        print(f"OTA {diff['from']['version']} -> {diff['to']['version']}")
        print(f"  added={diff['added']} changed={diff['changed']} removed={diff['removed']}")
        print(f"  unchanged={diff['unchanged']}  download={diff['download_bytes']} bytes\n")

        # -- 5. one offline turn ------------------------------------------
        sink = ListSink()
        runtime = EdgeRuntime(
            bundle.path, StubSTT(), StubLLM(), StubTTS(), tracer=Tracer(sink)
        )
        runtime.start()
        turn = runtime.turn(b"is this bundle trustworthy", session="edge-demo")

        print(f"turn {turn.turn}  user: {turn.transcript}")
        print(f"        assistant: {turn.assistant_text.strip()}")
        for i, sentence in enumerate(turn.segments, 1):
            print(f"        sentence {i}: {sentence}")
        print(f"        time to first audio: {turn.ttfa_ms:.2f} ms (budget {policy.max_ttfa_ms})")
        print(f"        stages: {format_summary(summarise_turn(sink.rows))}")

        allowed, reason = check(policy, Action(ACTION_CLOUD_FALLBACK))
        print(f"        cloud fallback allowed: {allowed} ({reason})\n")

        print("health")
        for key, value in runtime.health().items():
            if key == "uptime_s":
                value = f"{value:.3f}"
            print(f"  {key}: {value}")

    print(f"\ndone in {time.perf_counter() - t0:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
