# deploy/edge/ — shipping to devices

The edge target is a device that runs the same turn loop as the cloud API with
offline-only backends. What ships is an **EdgeBundle**: exported model files, a
manifest with a checksum per file, and the policy the runtime must enforce.

```
build  ->  verify  ->  OTA diff  ->  canary rollout  ->  rollback
```

| File | What it is |
|---|---|
| `systemd/vmp-edge.service` | The unit for a plain Linux device. Verifies the bundle before starting the runtime. |
| `k3s/daemonset.yaml` | A device fleet managed by k3s. One pod per node labelled `vmp.io/device=voice-agent`. |
| `ota/rollout.toml` | The canary policy: 5% -> 25% -> 100%, with gates on time-to-first-audio and error rate. |

## 1. Build

Export first, then bundle. The export merges the adapter into the base model and
writes the target format (`onnx`, `gguf`, `mlx`); the bundle adds the manifest.

```bash
vmp edge export --config configs/edge.toml --dry-run      # plan only
vmp edge export --config configs/edge.toml

vmp edge bundle build \
  --src .vmp/edge/export \
  --out .vmp/edge/bundles/voice-agent-edge-0.2.0 \
  --name voice-agent-edge \
  --version 0.2.0 \
  --target gguf \
  --base-model google/gemma-3-270m \
  --adapter-version voice-sft-3 \
  --config configs/edge.toml
```

`--config` supplies the `[policy]` table, so the policy travels inside the
manifest. A bundle whose policy contradicts itself — `offline_only` with a
non-empty `allowed_egress`, or `offline_only` with `fallback = "cloud"` — is
rejected at build time, not discovered on a device.

`--dry-run` builds and prints the manifest without writing files.

## 2. Verify

Verification recomputes the SHA-256 of every file in the bundle, compares it to
the manifest, checks the required files are present and checks the runtime
version against `min_runtime_version`.

```bash
vmp edge bundle verify .vmp/edge/bundles/voice-agent-edge-0.2.0
# {"ok": true, "problems": []}
```

It runs three times in the life of a bundle, deliberately: in CI after the
build, on the device before installation, and again at every start
(`ExecStartPre` in the systemd unit, the `verify-bundle` init container in the
DaemonSet). A device that fails verification does not start the new bundle; it
keeps serving the previous one.

## 3. OTA diff

Before shipping, diff the candidate against what the fleet is running. The diff
is what makes the update reviewable and what sizes the transfer.

```bash
vmp edge bundle diff \
  .vmp/edge/bundles/voice-agent-edge-0.1.0 \
  .vmp/edge/bundles/voice-agent-edge-0.2.0
```

It reports files added, removed and changed (by checksum), and the manifest
fields that moved: base model, adapter version, target, policy. Two things to
look at every time:

- **A policy change.** A bundle that flips `offline_only` or widens
  `allowed_egress` is a privacy change, not a model change, and needs the review
  in `runbooks/data-retention-and-privacy.md`.
- **An unexpected weight change.** If the adapter version is unchanged but the
  model files differ, the export was not reproducible; find out why before
  shipping.

## 4. Canary rollout

`ota/rollout.toml` defines the waves and the gates. Devices are assigned by a
hash of their id, so the same devices are always in the canary and a failure can
be reproduced rather than re-rolled.

| Stage | Devices | Soak |
|---|---|---|
| `canary` | 5% (at least 3) | 60 min |
| `expand` | 25% | 240 min |
| `full` | 100% | 1440 min |

A stage advances only when its gates pass over at least 200 turns from the
devices in that stage: p95 time to first audio within budget and not more than
10% worse than the previous bundle, error rate within budget and not more than
0.2 points worse, and zero verification failures. Under-trafficked stages wait
instead of passing on noise.

Installation on a device is a symlink swap plus a restart:

```bash
# on the device, as root
install -d /var/lib/vmp/edge/voice-agent-edge-0.2.0
# ... transfer the bundle ...
vmp edge bundle verify /var/lib/vmp/edge/voice-agent-edge-0.2.0
ln -sfn /var/lib/vmp/edge/voice-agent-edge-0.2.0 /var/lib/vmp/edge/current
systemctl restart vmp-edge
```

Under k3s the same swap is a DaemonSet image update with `maxUnavailable: 1`.

## 5. Rollback

The previous bundle is still on disk; rollback is the reverse symlink swap.

```bash
ln -sfn /var/lib/vmp/edge/voice-agent-edge-0.1.0 /var/lib/vmp/edge/current
systemctl restart vmp-edge
vmp edge bundle verify /var/lib/vmp/edge/current
```

With `auto_rollback = true`, a failed gate halts the rollout and reverts the
devices already updated. Reverted devices are quarantined from new rollouts for
24 hours so a device does not oscillate between bundles while an incident is
open. The full procedure, including the registry and Ray Serve sides, is in
`runbooks/rollback-model.md`.

Keep at least the current and the previous bundle on every device. A device with
only one bundle has no rollback, only a re-download over a network the policy may
not allow it to use.

## Policy on a device

`configs/edge.toml` `[policy]` is the contract the runtime enforces before any
network call, cloud fallback or audio write:

- `offline_only = true` — no egress at all. The systemd unit enforces the same
  thing with `IPAddressDeny=any`, and the DaemonSet with a NetworkPolicy that
  permits only DNS. Config and sandbox must agree; the config alone is a
  statement of intent.
- `retention.audio = "none"` — captured audio is never written to disk.
- `fallback = "degrade"` — when a backend is unavailable the device degrades its
  answer rather than reaching for the cloud.
