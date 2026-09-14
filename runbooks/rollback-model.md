# Rolling a model back

## Symptom

A promoted model or a shipped edge bundle is worse than what it replaced: a
failing release gate, a TTFA regression, wrong answers, a rise in error rate, or
a policy change that should not have shipped.

## Severity

SEV-2 by default. SEV-1 if the model is producing unsafe output or leaking
context between sessions.

## Principle

Roll back before diagnosing. Every artefact here is versioned and the previous
version is still present: the registry keeps stages, Ray Serve keeps the previous
cluster, devices keep the previous bundle. Nothing below rebuilds anything.

There are three places a model lives, and a full rollback touches the ones that
are actually serving it. Do them in this order — registry first, so a
reconciliation loop or a redeploy cannot re-promote what you just removed.

## 1. Registry: demote the bad version, promote the good one

Find the versions and what is currently in production.

```bash
vmp registry list --name voice-agent
vmp registry show --name voice-agent --version 7
```

Demote the bad version out of `production`, then put the known-good version
back. `retired` is the terminal stage; use it rather than deleting, so the
lineage of the incident survives.

```bash
# demote the bad version
vmp registry promote --name voice-agent --version 7 --stage retired

# restore the previous known-good version
vmp registry promote --name voice-agent --version 6 --stage production
```

Confirm:

```bash
vmp registry list --name voice-agent
```

The stages are `candidate | staging | production | retired`. A version demoted
to `retired` cannot be promoted back by accident; promoting it again is an
explicit command that shows up in the shell history of whoever ran it.

## 2. Ray Serve: roll the inference graph back

Two cases, and the distinction matters because they behave differently.

**The model is selected by config** (`serveConfigV2`, the usual case). Edit the
model back in the config and apply. KubeRay applies a `serveConfigV2`-only change
in place, deployment by deployment, without a new cluster.

```bash
# see what is running now
kubectl get rayservice -n platform vmp-voice \
  -o jsonpath='{.status.activeServiceStatus.applicationStatuses}{"\n"}'

# regenerate the config from the TOML you are rolling back to, then apply
vmp serve ray --config configs/serving.toml --yaml
kubectl apply -f deploy/ray/rayservice-voice.yaml

# watch the applications go healthy again
kubectl get rayservice -n platform vmp-voice -w
```

**The image changed too.** A change to `rayClusterConfig` triggers KubeRay's
zero-downtime upgrade: a second cluster starts, its Serve applications become
healthy, traffic moves, the old cluster is torn down. Rolling back is the same
operation in reverse — apply the previous manifest and let the same mechanism
run.

```bash
git show HEAD~1:deploy/ray/rayservice-voice.yaml | kubectl apply -f -
kubectl get rayservice -n platform vmp-voice -o wide
kubectl get pods -n platform -l ray.io/cluster --show-labels
```

The rollback is complete when `status.activeServiceStatus` lists all four
applications (`STTDeployment`, `LLMDeployment`, `TTSDeployment`,
`VoiceAgentIngress`) as `RUNNING` and no pending cluster remains.

If the API Deployment rather than Ray is serving the model:

```bash
kubectl rollout history deployment/api -n platform
kubectl rollout undo deployment/api -n platform
kubectl rollout status deployment/api -n platform --timeout=180s
```

## 3. Edge: roll the bundle back on devices

The previous bundle is on the device's disk. Rollback is a symlink swap and a
restart — no download, which matters on a device whose policy forbids egress.

**Halt the rollout first**, or the driver will re-apply the bad bundle to the
next wave. With `auto_rollback = true` in `deploy/edge/ota/rollout.toml` a failed
gate does this automatically and quarantines reverted devices for 24 hours.

On a systemd device:

```bash
ls -l /var/lib/vmp/edge/                       # which bundles are present
vmp edge bundle verify /var/lib/vmp/edge/voice-agent-edge-0.1.0
ln -sfn /var/lib/vmp/edge/voice-agent-edge-0.1.0 /var/lib/vmp/edge/current
systemctl restart vmp-edge
systemctl status vmp-edge --no-pager
vmp edge bundle verify /var/lib/vmp/edge/current
```

On a k3s fleet:

```bash
kubectl rollout undo daemonset/vmp-edge -n vmp-edge
kubectl rollout status daemonset/vmp-edge -n vmp-edge --timeout=600s
kubectl get pods -n vmp-edge -o wide
```

Before declaring it done, diff what the devices now run against what you
intended, so a partial rollout does not leave the fleet on two bundles:

```bash
vmp edge bundle diff \
  /var/lib/vmp/edge/voice-agent-edge-0.1.0 \
  /var/lib/vmp/edge/voice-agent-edge-0.2.0
```

## Verify

```bash
# the registry says what you think it says
vmp registry list --name voice-agent

# the gates pass on the restored version
vmp eval golden --set data/golden.jsonl
vmp eval gate --rules configs/gates.toml --metrics /tmp/metrics.json

# latency and error rate back within budget
vmp obs slo --trace /var/lib/vmp/trace.jsonl --config configs/slo.toml
```

All three, not one. A rollback that restores latency but leaves the registry
pointing at the bad version will be undone by the next deploy.

## Follow-up

- The bad version stays `retired`, with the incident number in its metrics or
  notes. Do not reuse the version number.
- Ask why the gate did not catch it. If the failing metric is one of the
  `required = false` gates in `configs/gates.toml` (`ttfa_p95_ms`, `win_rate`,
  `error_rate`), the action item is to decide whether it should block.
- If the rollback needed a step that is not in this runbook, add it now.
