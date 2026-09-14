#!/usr/bin/env bash
# Kustomize refuses to read files outside its root, so the base keeps copies of
# shared config under deploy/k8s/base/config. This script refreshes them from
# the canonical files. `--check` exits 1 if any copy is stale (used in CI).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
dst="$here/base/config"
pairs=(
  "configs/serving.toml:serving.toml"
  "deploy/otel/collector.yaml:collector.yaml"
  "deploy/prometheus/prometheus.yml:prometheus.yml"
  "deploy/prometheus/rules/voice-slo.yml:voice-slo.yml"
  "deploy/grafana/dashboards/voice-agent.json:voice-agent.json"
  "deploy/grafana/provisioning/datasources/prometheus.yaml:grafana-datasources.yaml"
  "deploy/grafana/provisioning/dashboards/dashboards.yaml:grafana-dashboards.yaml"
  "deploy/compose/db/init/001_schema.sql:001_schema.sql"
)
check="${1:-}"
status=0
for pair in "${pairs[@]}"; do
  src="$root/${pair%%:*}"
  out="$dst/${pair##*:}"
  if [[ ! -f "$src" ]]; then
    echo "skip (missing source): ${pair%%:*}"
    continue
  fi
  if [[ "$check" == "--check" ]]; then
    if ! cmp -s "$src" "$out"; then
      echo "stale: $out differs from ${pair%%:*}"
      status=1
    fi
  else
    cp "$src" "$out"
    echo "synced: ${pair%%:*} -> deploy/k8s/base/config/${pair##*:}"
  fi
done
exit $status
