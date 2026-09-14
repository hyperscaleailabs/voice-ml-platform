from __future__ import annotations

import json
from pathlib import Path

import pytest

from vmp.cli import main
from vmp.training.dpo import (
    load_preference_pairs,
    preference_stats,
    rule_preference_pairs,
    run_dpo,
)
from vmp.training.merge import adapter_manifest, merge_adapter
from vmp.training.plan import (
    ComputeConfig,
    LoraConfig,
    PlanError,
    TrainingPlan,
    dataset_stats,
    estimate_steps,
    read_jsonl,
    write_jsonl,
)
from vmp.training.ray_jobs import (
    RayDataPreprocessor,
    RayTrainLauncher,
    tokenize_whitespace,
    transform_for_kind,
)
from vmp.training.sft import format_chat_example, is_spoken_style, run_sft, strip_markdown
from vmp.training.whisper_lora import (
    exact_match,
    normalize_text,
    run_whisper_lora,
    word_edits,
    word_error_rate,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _plan(kind: str = "sft", tmp_path: Path | None = None, **overrides) -> TrainingPlan:
    train = str(tmp_path / "train.jsonl") if tmp_path else "data/train.jsonl"
    data = {
        "kind": kind,
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "datasets": {"train": train},
        "output_dir": "runs/x",
        "lora": {"task_type": "SEQ_2_SEQ_LM"} if kind == "whisper-lora" else {},
    }
    data.update(overrides)
    return TrainingPlan.from_dict(data)


# -- plan ---------------------------------------------------------------------


def test_from_toml_loads_all_shipped_configs():
    kinds = {}
    for name in ("train_sft", "train_dpo", "train_whisper_lora"):
        plan = TrainingPlan.from_toml(CONFIGS / f"{name}.toml")
        kinds[name] = plan.kind
        assert plan.lora.r == 16 and plan.lora.alpha == 32
        assert len(plan.config_hash()) == 64
    assert kinds == {
        "train_sft": "sft",
        "train_dpo": "dpo",
        "train_whisper_lora": "whisper-lora",
    }


def test_from_toml_missing_file_names_path(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"nope\.toml"):
        TrainingPlan.from_toml(tmp_path / "nope.toml")


def test_from_toml_error_names_file(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('kind = "sft"\nbase_model = "m"\noutput_dir = "o"\n')
    with pytest.raises(PlanError, match=r"bad\.toml.*datasets"):
        TrainingPlan.from_toml(p)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "rlhf"}, "kind must be one of"),
        ({"base_model": ""}, "base_model"),
        ({"datasets": {"eval": "x"}}, "'train'"),
        ({"lora": {"r": 0}}, "lora.r"),
        ({"lora": {"dropout": 1.5}}, "lora.dropout"),
        ({"lora": {"target_modules": []}}, "target_modules"),
        ({"lora": {"task_type": "SEQ_2_SEQ_LM"}}, "lora.task_type"),
        ({"compute": {"backend": "slurm"}}, "compute.backend"),
        ({"compute": {"backend": "local", "num_workers": 2}}, "requires compute.backend"),
        ({"compute": {"num_workers": 0}}, "num_workers"),
        ({"hyperparams": {"learning_rate": -1}}, "learning_rate"),
        ({"hyperparams": {"per_device_batch_size": 0}}, "per_device_batch_size"),
        ({"seed": -1}, "seed"),
        ({"lora": {"rank": 4}}, "unknown field"),
    ],
)
def test_plan_validation_errors(overrides, message):
    with pytest.raises(PlanError, match=message):
        _plan(**overrides)


def test_dpo_beta_must_be_positive():
    with pytest.raises(PlanError, match="beta"):
        _plan("dpo", hyperparams={"beta": 0})


def test_whisper_plan_defaults_task_type():
    plan = TrainingPlan.from_dict(
        {
            "kind": "whisper-lora",
            "base_model": "openai/whisper-small.en",
            "datasets": {"train": "a.jsonl"},
            "output_dir": "o",
        }
    )
    assert plan.lora.task_type == "SEQ_2_SEQ_LM"


def test_config_hash_is_stable_and_canonical():
    a = _plan(hyperparams={"epochs": 2, "learning_rate": 1e-4})
    b = _plan(hyperparams={"learning_rate": 1e-4, "epochs": 2})
    assert a.config_hash() == b.config_hash()
    assert (
        a.config_hash() == TrainingPlan.from_dict(json.loads(json.dumps(a.to_dict()))).config_hash()
    )
    # explicit defaults hash the same as implicit ones
    assert _plan().config_hash() == _plan(hyperparams={"epochs": 1}).config_hash()


def test_config_hash_ignores_output_dir_but_not_recipe():
    base = _plan()
    assert base.config_hash() == _plan(output_dir="elsewhere").config_hash()
    assert base.config_hash() != _plan(hyperparams={"epochs": 3}).config_hash()
    assert base.config_hash() != _plan(lora={"r": 8}).config_hash()
    assert base.config_hash() != _plan(seed=7).config_hash()


def test_lora_and_compute_roundtrip():
    lora = LoraConfig(r=8, alpha=16, target_modules=("q_proj",))
    assert LoraConfig.from_dict(lora.to_dict()) == lora
    assert lora.to_peft_kwargs()["lora_alpha"] == 16
    compute = ComputeConfig(backend="ray", num_workers=4, gpu_per_worker=1)
    assert ComputeConfig.from_dict(compute.to_dict()) == compute


def test_estimate_steps():
    plan = _plan(hyperparams={"epochs": 2, "per_device_batch_size": 4, "gradient_accumulation": 2})
    assert estimate_steps(100, plan) == 26  # ceil(100/8)=13 per epoch * 2
    ray_plan = _plan(
        hyperparams={"epochs": 2, "per_device_batch_size": 4, "gradient_accumulation": 2},
        compute={"backend": "ray", "num_workers": 4},
    )
    assert estimate_steps(100, ray_plan) == 8  # ceil(100/32)=4 * 2
    assert estimate_steps(0, plan) == 0
    assert estimate_steps(100, _plan(hyperparams={"max_steps": 5})) == 5


def test_read_jsonl_reports_bad_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\nnot json\n')
    with pytest.raises(ValueError, match=r"x\.jsonl:3"):
        read_jsonl(p)
    p.write_text('{"a": 1}\n[1]\n')
    with pytest.raises(ValueError, match="expected an object"):
        read_jsonl(p)


def test_dataset_stats(tmp_path):
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [{"prompt": "a", "response": "b"}, {"prompt": "c"}])
    st = dataset_stats(p, required=("prompt", "response"))
    assert st["rows"] == 2 and st["exists"] and len(st["data_hash"]) == 64
    assert st["missing_required"] == ["response"]
    assert dataset_stats(tmp_path / "missing.jsonl")["exists"] is False


# -- sft ------------------------------------------------------------------------


def test_strip_markdown_and_spoken_style():
    md = "## Title\n\n- **bold** item\n- `code` item\n\n```py\nx = 1\n```\nSee [docs](http://x)."
    assert strip_markdown(md) == "Title bold item code item See docs."
    assert is_spoken_style("Sure, it is on the second shelf.")
    assert not is_spoken_style("**Sure.** It is on the shelf.")
    assert not is_spoken_style(" ".join(["word"] * 80))


def test_format_chat_example_shapes():
    for row in (
        {"prompt": "hi", "response": "**Hello** there"},
        {"user": "hi", "assistant": "**Hello** there"},
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "**Hello** there"},
            ]
        },
    ):
        out = format_chat_example(row)["messages"]
        assert out[0]["role"] == "system"
        assert out[-1] == {"role": "assistant", "content": "Hello there"}
    with pytest.raises(ValueError, match="row needs"):
        format_chat_example({"text": "x"})
    with pytest.raises(ValueError, match="unknown chat role"):
        format_chat_example({"messages": [{"role": "tool", "content": "x"}]})


def test_run_sft_dry_run_manifest(tmp_path):
    hp = {"per_device_batch_size": 2, "gradient_accumulation": 1}
    plan = _plan("sft", tmp_path, hyperparams=hp)
    rows = [{"prompt": f"q{i}", "response": f"a{i}"} for i in range(5)]
    write_jsonl(plan.datasets["train"], rows)
    m = run_sft(plan, dry_run=True)
    assert m["dry_run"] is True and m["kind"] == "sft"
    assert m["config_hash"] == plan.config_hash()
    assert m["datasets"]["train"]["rows"] == 5
    assert m["data_hash"] == m["datasets"]["train"]["data_hash"]
    assert m["estimated_steps"] == 3
    assert m["trainer"] == "trl.SFTTrainer" and m["peft"]["r"] == 16
    assert m["warnings"] == []
    json.dumps(m)  # serialisable


def test_run_sft_dry_run_missing_dataset_warns():
    m = run_sft(_plan("sft"), dry_run=True)
    assert m["estimated_steps"] == 0
    assert any("not found" in w for w in m["warnings"])


def test_runners_reject_wrong_kind():
    with pytest.raises(ValueError, match="expects kind"):
        run_dpo(_plan("sft"))
    with pytest.raises(ValueError, match="expects kind"):
        run_whisper_lora(_plan("sft"))
    with pytest.raises(ValueError, match="expects kind"):
        run_sft(_plan("dpo"))


# -- dpo ------------------------------------------------------------------------


def _pairs():
    return [
        {"prompt": "Where is it?", "chosen": "On the shelf.", "rejected": "## Location\n- shelf"},
        {"prompt": "How long?", "chosen": "About an hour.", "rejected": "**About** an hour."},
    ]


def test_load_preference_pairs_validates(tmp_path):
    p = tmp_path / "pairs.jsonl"
    write_jsonl(p, _pairs())
    pairs = load_preference_pairs(p)
    assert len(pairs) == 2 and pairs[0].chosen == "On the shelf."
    write_jsonl(p, [{"prompt": "x", "chosen": "y"}])
    with pytest.raises(ValueError, match="rejected"):
        load_preference_pairs(p)
    write_jsonl(p, [{"prompt": "x", "chosen": "y", "rejected": "y"}])
    with pytest.raises(ValueError, match="identical"):
        load_preference_pairs(p)


def test_preference_stats_and_rule_pairs(tmp_path):
    p = tmp_path / "pairs.jsonl"
    write_jsonl(p, _pairs())
    st = preference_stats(load_preference_pairs(p))
    assert st["n"] == 2 and st["chosen_spoken_fraction"] == 1.0
    assert st["rejected_spoken_fraction"] == 0.0
    assert preference_stats([]) == {"n": 0}
    rows = [{"prompt": "q", "candidates": ["Short answer.", "## Long\n- markdown", "Also short."]}]
    pairs = rule_preference_pairs(rows)
    assert len(pairs) == 2 and all(p.rejected.startswith("## Long") for p in pairs)
    assert rule_preference_pairs([{"prompt": "q", "candidates": ["a", "b"]}]) == []


def test_run_dpo_dry_run_manifest(tmp_path):
    plan = _plan("dpo", tmp_path, hyperparams={"beta": 0.2})
    write_jsonl(plan.datasets["train"], _pairs())
    m = run_dpo(plan, dry_run=True)
    assert m["trainer"] == "trl.DPOTrainer" and m["beta"] == 0.2
    assert "adapter disabled" in m["reference_model"]
    assert m["pairs"]["n"] == 2 and m["estimated_steps"] == 1
    assert m["datasets"]["train"]["missing_required"] == []


# -- whisper --------------------------------------------------------------------


def test_normalize_and_exact_match():
    assert normalize_text("  Hello,   World! ") == "hello world"
    res = exact_match(["Turn on the light.", "Set a timer"], ["turn on the light", "set the timer"])
    assert res.n == 2 and res.metrics["exact_match"] == 0.5 and res.details["correct"] == 1
    assert exact_match([], []).metrics["exact_match"] == 0.0
    with pytest.raises(ValueError, match="same length"):
        exact_match(["a"], [])


def test_word_edits_and_wer():
    assert word_edits("the cat sat", "the cat sat") == (0, 0, 0)
    assert word_edits("the cat sat", "the dog sat") == (1, 0, 0)
    assert word_edits("the cat sat", "the sat") == (0, 1, 0)
    assert word_edits("the cat sat", "the big cat sat") == (0, 0, 1)
    res = word_error_rate(["the cat sat", "on the mat"], ["the dog sat", "on mat"])
    assert res.metrics["substitutions"] == 1 and res.metrics["deletions"] == 1
    assert res.metrics["reference_words"] == 6
    assert res.metrics["wer"] == pytest.approx(2 / 6)
    assert res.metrics["word_accuracy"] == pytest.approx(4 / 6)


def test_run_whisper_lora_dry_run_manifest(tmp_path):
    plan = TrainingPlan.from_toml(CONFIGS / "train_whisper_lora.toml")
    plan.datasets = {"train": str(tmp_path / "accent.jsonl")}
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    write_jsonl(
        plan.datasets["train"],
        [
            {"audio_path": str(wav), "text": "one two", "duration_s": 1.5, "speaker": "op"},
            {"audio_path": str(tmp_path / "missing.wav"), "text": "three", "duration_s": 1.0},
        ],
    )
    m = run_whisper_lora(plan, dry_run=True)
    assert m["model_class"].endswith("WhisperForConditionalGeneration")
    assert m["peft"]["target_modules"] == ["q_proj", "v_proj"]
    assert m["audio"] == {
        "n": 2,
        "total_duration_s": 2.5,
        "rows_with_duration": 2,
        "missing_audio_files": 1,
        "speakers": ["op"],
        "mean_words": 1.5,
    }
    assert any("audio files" in w for w in m["warnings"])
    assert m["estimated_steps"] == 3  # 2 rows, batch 8, 3 epochs


def test_whisper_warns_on_extra_target_modules(tmp_path):
    lora = {"task_type": "SEQ_2_SEQ_LM", "target_modules": ["q_proj", "fc1"]}
    plan = _plan("whisper-lora", lora=lora)
    m = run_whisper_lora(plan, dry_run=True)
    assert any("fc1" in w for w in m["warnings"])


# -- ray ------------------------------------------------------------------------


def test_ray_launcher_dry_run_manifest(tmp_path):
    plan = _plan("sft", tmp_path, compute={"backend": "ray", "num_workers": 4, "gpu_per_worker": 1})
    write_jsonl(plan.datasets["train"], [{"prompt": "q", "response": "a"}] * 64)
    m = RayTrainLauncher(address="ray://head:10001").submit(plan, dry_run=True)
    ray = m["ray"]
    assert ray["trainer"] == "ray.train.torch.TorchTrainer"
    assert ray["scaling_config"] == {
        "num_workers": 4,
        "use_gpu": True,
        "resources_per_worker": {"GPU": 1.0},
    }
    assert ray["run_config"]["name"].startswith("sft-")
    assert ray["train_loop_config"] == plan.to_dict()
    assert ray["address"] == "ray://head:10001"
    assert m["estimated_steps"] == 1  # 64 rows / (4*4*4)
    json.dumps(m)


def test_ray_launcher_cpu_and_local_rejection():
    launcher = RayTrainLauncher()
    plan = _plan(compute={"backend": "ray", "num_workers": 2})
    assert launcher.scaling_config(plan)["resources_per_worker"] == {"CPU": 1}
    with pytest.raises(ValueError, match="backend"):
        launcher.submit(_plan(), dry_run=True)


def test_ray_data_preprocessor_pure_python(tmp_path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    write_jsonl(src, [{"text": "a b c"}, {"text": "d"}])
    pre = RayDataPreprocessor(tokenize_whitespace)
    assert pre.map_rows([{"text": "x y"}]) == [{"text": "x y", "tokens": ["x", "y"], "n_tokens": 2}]
    dry = pre.run(src, dst, dry_run=True)
    assert dry["rows_in"] == 2 and dry["rows_out"] is None and not dst.exists()
    m = pre.run(src, dst)
    assert m["engine"] == "python" and m["rows_out"] == 2
    assert [r["n_tokens"] for r in read_jsonl(dst)] == [3, 1]
    with pytest.raises(FileNotFoundError):
        pre.run(tmp_path / "nope.jsonl", dst)


def test_transform_for_kind():
    sft = transform_for_kind("sft")({"prompt": "q", "response": "a"})
    assert sft["messages"][-1]["content"] == "a"
    dpo = transform_for_kind("dpo")({"prompt": "q", "chosen": "a b", "rejected": "c"})
    assert dpo["chosen_words"] == 2 and dpo["rejected_words"] == 1
    assert transform_for_kind("whisper-lora")({"audio_path": "x", "text": " y "}) == {
        "audio_path": "x",
        "text": "y",
    }
    with pytest.raises(ValueError, match="unknown kind"):
        transform_for_kind("x")


# -- merge ----------------------------------------------------------------------


def test_adapter_manifest_and_merge_dry_run(tmp_path):
    empty = adapter_manifest(tmp_path / "none")
    assert empty["exists"] is False and empty["has_config"] is False
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "base-a", "r": 16})
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"\0")
    info = adapter_manifest(adapter)
    assert info["peft_type"] == "LORA" and info["r"] == 16
    assert info["weight_files"] == ["adapter_model.safetensors"]
    m = merge_adapter("base-b", adapter, tmp_path / "merged", dry_run=True)
    assert m["dry_run"] and m["method"].endswith("merge_and_unload")
    assert any("trained on 'base-a'" in w for w in m["warnings"])
    assert merge_adapter("base-a", adapter, tmp_path / "m", dry_run=True)["warnings"] == []
    with pytest.raises(FileNotFoundError):
        merge_adapter("base-a", tmp_path / "none", tmp_path / "m", dry_run=False)


# -- cli ------------------------------------------------------------------------


def test_cli_train_dry_run_and_plan(capsys):
    assert main(["train", "sft", "--config", str(CONFIGS / "train_sft.toml"), "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "sft" and out["dry_run"] is True
    assert main(["train", "plan", "--config", str(CONFIGS / "train_whisper_lora.toml")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["plan"]["kind"] == "whisper-lora" and len(out["config_hash"]) == 64


def test_cli_train_kind_mismatch_and_missing_config(capsys, tmp_path):
    assert main(["train", "dpo", "--config", str(CONFIGS / "train_sft.toml"), "--dry-run"]) == 2
    assert "expects 'dpo'" in capsys.readouterr().err
    assert main(["train", "sft", "--config", str(tmp_path / "x.toml"), "--dry-run"]) == 2


def test_cli_train_without_extras_reports_missing_dependency(capsys):
    pytest.importorskip("tomllib")
    try:
        import trl  # noqa: F401
    except ModuleNotFoundError:
        rc = main(["train", "sft", "--config", str(CONFIGS / "train_sft.toml")])
        assert rc == 1 and "missing dependency" in capsys.readouterr().err
    else:
        pytest.skip("trl installed; real run not exercised here")


# -- heavy (skipped without extras) -----------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
def test_sft_real_run_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    plan = _plan(
        "sft",
        tmp_path,
        base_model="hf-internal-testing/tiny-random-Qwen2ForCausalLM",
        output_dir=str(tmp_path / "out"),
        hyperparams={"max_steps": 2, "per_device_batch_size": 1, "gradient_accumulation": 1},
        lora={"target_modules": ["q_proj", "v_proj"]},
    )
    write_jsonl(plan.datasets["train"], [{"prompt": "hi", "response": "hello"}] * 4)
    m = run_sft(plan, dry_run=False)
    assert m["result"]["global_step"] == 2
    assert adapter_manifest(plan.output_dir)["has_config"]
