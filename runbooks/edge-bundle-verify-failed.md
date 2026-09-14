# Edge bundle verification failed

## Symptom

`vmp edge bundle verify` returns `{"ok": false, "problems": [...]}`. On a device
that means the runtime refuses to start: the systemd unit fails its
`ExecStartPre`, or the DaemonSet's `verify-bundle` init container does not
complete and the new pod never becomes ready.

## Severity

SEV-3 on one device — it keeps serving the previous bundle, which is the design.
SEV-2 if verification fails across a rollout wave, or if a device is left with no
working bundle at all.

## Read the problems list first

`verify_bundle` reports exactly what is wrong. Each problem maps to a different
cause and a different fix.

| Problem | Meaning | Cause |
|---|---|---|
| `missing manifest.json` | Not a bundle | Wrong path, or an interrupted transfer |
| `manifest.json is not valid JSON` | Truncated manifest | Transfer cut short, or disk full |
| checksum mismatch on a file | Bytes differ from what was built | Corrupt or partial transfer; tampering |
| a file in the manifest is missing | Incomplete transfer | Transfer interrupted |
| a file on disk is not in the manifest | Extra content | Stale files from a previous bundle in the same directory |
| runtime version below `min_runtime_version` | The device is too old for this bundle | The fleet was not upgraded before the bundle shipped |

```bash
vmp edge bundle verify /var/lib/vmp/edge/current
vmp edge bundle verify /var/lib/vmp/edge/voice-agent-edge-0.2.0
```

## Diagnose

### 1. Is the device serving?

Answer this before anything else. A device that failed verification and kept the
previous bundle is not an outage.

```bash
systemctl status vmp-edge --no-pager
journalctl -u vmp-edge -n 100 --no-pager
ls -l /var/lib/vmp/edge/          # where does `current` point?
```

```bash
kubectl get pods -n vmp-edge -o wide
kubectl describe pod -n vmp-edge <pod>          # init container status and reason
kubectl logs -n vmp-edge <pod> -c verify-bundle
```

### 2. Transfer or build?

Verify the same bundle where it was built. This splits the problem cleanly:

```bash
vmp edge bundle verify .vmp/edge/bundles/voice-agent-edge-0.2.0
```

- **Passes at the source, fails on the device** — a transfer problem. Re-transfer
  into a fresh directory (never on top of an existing one, which produces the
  "file not in the manifest" case) and verify again before swapping the symlink.
- **Fails at the source too** — the bundle was built wrong. It must not ship.
  Halt the rollout, rebuild, and diff the rebuilt bundle against the previous
  good one.

### 3. Device-local causes

```bash
df -h /var/lib/vmp                # a full disk truncates transfers
ls -l /var/lib/vmp/edge/current   # a dangling symlink after a manual cleanup
```

A partly-deleted bundle directory is a common cause and looks like corruption.

### 4. Is it one device or the wave?

```bash
kubectl get pods -n vmp-edge -o wide | grep -v Running
```

One device is a device problem. A whole wave is a bundle problem, and the
rollout must stop before the next stage.

## Mitigate

**Halt the rollout.** `verify_failure_rate` has a maximum of 0.0 in
`deploy/edge/ota/rollout.toml`, so a single verification failure fails the
stage's gate; with `auto_rollback = true` the driver reverts devices already
updated. Confirm it actually halted rather than assuming it did.

**Make sure the device is on a good bundle.**

```bash
vmp edge bundle verify /var/lib/vmp/edge/voice-agent-edge-0.1.0
ln -sfn /var/lib/vmp/edge/voice-agent-edge-0.1.0 /var/lib/vmp/edge/current
systemctl restart vmp-edge
```

```bash
kubectl rollout undo daemonset/vmp-edge -n vmp-edge
kubectl rollout status daemonset/vmp-edge -n vmp-edge --timeout=600s
```

**Re-transfer, if the bundle itself is good.** Into a fresh directory, verify
before swapping:

```bash
install -d /var/lib/vmp/edge/voice-agent-edge-0.2.0
# ... transfer ...
vmp edge bundle verify /var/lib/vmp/edge/voice-agent-edge-0.2.0
ln -sfn /var/lib/vmp/edge/voice-agent-edge-0.2.0 /var/lib/vmp/edge/current
systemctl restart vmp-edge
```

**Rebuild, if the source bundle is bad.**

```bash
vmp edge bundle build --src .vmp/edge/export --out .vmp/edge/bundles/voice-agent-edge-0.2.1 \
  --name voice-agent-edge --version 0.2.1 --target gguf \
  --base-model google/gemma-3-270m --config configs/edge.toml
vmp edge bundle verify .vmp/edge/bundles/voice-agent-edge-0.2.1
vmp edge bundle diff .vmp/edge/bundles/voice-agent-edge-0.1.0 \
                     .vmp/edge/bundles/voice-agent-edge-0.2.1
```

Never bypass verification to get a device running. A bundle that does not verify
is a bundle whose contents are unknown, and the policy travels in the same
manifest as the checksums.

## Verify

```bash
vmp edge bundle verify /var/lib/vmp/edge/current     # {"ok": true, "problems": []}
systemctl status vmp-edge --no-pager
kubectl get pods -n vmp-edge                          # all Running, init completed
```

Then confirm the device is answering, not merely running: check that new turns
appear in its trace and that `p95_ttfa_ms_24h` for that device is not degraded.

## Follow-up

- Record which failure mode it was. Checksum mismatches on transfer and
  `min_runtime_version` failures have completely different fixes: one is the
  transport, the other is fleet sequencing.
- If it was `min_runtime_version`, the action item is ordering — the runtime
  upgrade must precede the bundle that requires it, as its own rollout.
- Keep at least two bundles on every device. A device with one bundle has no
  rollback, only a re-download over a network its policy may forbid.
