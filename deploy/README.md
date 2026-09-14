# deploy/ — productization

Stage 3 of the progression. The same container image and the same TOML configs
run in three places; only the substrate changes.

```
local docker compose  ->  k3d / kind  ->  cloud Kubernetes
deploy/compose/           deploy/k8s/overlays/local      deploy/k8s/overlays/cloud
```

Nothing in here is specific to a cloud provider. There is no Helm chart: the
manifests are plain YAML assembled by kustomize, so `kubectl kustomize` renders
the exact objects that will be applied and a reviewer can read them.

## What each subdirectory is

| Path | Contents |
|---|---|
| `docker/` | `Dockerfile.api`, `Dockerfile.train`, `Dockerfile.edge`, `.dockerignore`. Multi-stage; the API image installs only the `serve,obs` extras by default (`--build-arg EXTRAS=...`). |
| `compose/` | `docker-compose.yml` with profiles (`core`, `rag`, `obs`, `train`), `.env.example`, and Postgres init SQL. The laptop stack. |
| `k8s/base/` | The namespace, the API (Deployment, Service, HPA, PDB, NetworkPolicy, ServiceAccount), the shared services, the observability stack, the Feast materialize CronJob, and `config/` copies of canonical config files. |
| `k8s/overlays/local/` | k3d / kind: one replica, small requests, no ingress, no PDB pressure. |
| `k8s/overlays/cloud/` | Three replicas, larger requests, pod anti-affinity, an Ingress with TLS annotation placeholders. |
| `ray/` | KubeRay: `RayCluster`, `RayJob` for SFT and DPO, `RayService` for the inference graph. |
| `feast/` | Feast repository: `feature_store.yaml` and a `features.py` generated from the in-code views. |
| `edge/` | Device fleet: systemd unit, k3s DaemonSet, OTA rollout policy. |
| `otel/`, `prometheus/`, `grafana/` | Canonical collector config, scrape config and SLO recording/alerting rules, dashboard JSON. `k8s/base/config/` holds copies; `k8s/sync-config.sh` refreshes them and CI checks they are in sync. |

## The DNS contract

Every workload reaches shared infrastructure by **service name inside the
`platform` namespace**. The names and ports are identical in docker compose and
in Kubernetes, which is why the same `serving.toml` and the same environment
variables work in both.

| Service | DNS name | Port(s) | Used for |
|---|---|---|---|
| Postgres (pgvector) | `postgres` | 5432 | RAG vector store, session metadata, Langfuse |
| Redis | `redis` | 6379 | Feature store online layer, session cache |
| Neo4j | `neo4j` | 7687 (bolt), 7474 (http) | Graph RAG entities and relations |
| Qdrant | `qdrant` | 6333 | Alternative vector store |
| MinIO | `minio` | 9000 (S3), 9001 (console) | Model artifacts, offline feature Parquet, edge bundles |
| OTel Collector | `otel-collector` | 4317 (gRPC), 4318 (HTTP) | Span export from every component |
| Prometheus | `prometheus` | 9090 | Metrics and SLO rules |
| Grafana | `grafana` | 3000 | Dashboards |

Fully qualified inside the cluster: `<name>.platform.svc.cluster.local`. Within
the namespace the short name is enough, and that is what the manifests use.

Credentials are never in these names. Connection strings that carry a password
come from the `vmp-api-secrets` Secret (`deploy/k8s/secrets.example.yaml`); the
password-free endpoints are plain environment variables in the Deployment.

### How the three workloads attach

**API** (`k8s/base/api-deployment.yaml`) — `VMP_REDIS_URL=redis://redis:6379/0`,
`VMP_QDRANT_URL=http://qdrant:6333`, `VMP_NEO4J_URI=bolt://neo4j:7687`,
`VMP_S3_ENDPOINT=http://minio:9000`,
`OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`. The Postgres DSN
arrives from the Secret. `/metrics` is scraped by Prometheus via the pod
annotations. The NetworkPolicy allows egress to exactly these services plus DNS.

**Ray** (`ray/`) — head and workers get the same environment. `RayJob` pods read
training corpora from MinIO through the S3 endpoint and write checkpoints back;
`RayService` pods export spans to `otel-collector` and are scraped on the Serve
metrics port. The Ray head's dashboard/client ports stay inside the cluster.

**Feature store** (`feast/`) — the offline store is Parquet under
`VMP_FEATURE_STORE_ROOT` (a PVC, or MinIO in the cloud overlay); the online store
is `redis:6379` database **1**, kept separate from the API's session database 0.
`deploy/k8s/base/feast-materialize-cronjob.yaml` runs `vmp features materialize`
on a schedule against both.

## Why `storageClassName` is omitted everywhere

No PVC in `k8s/` names a storage class. An omitted `storageClassName` means "use
the cluster's default StorageClass", which every target here provides under a
different name: k3d/k3s gives `local-path`, kind gives `standard`, EKS gives
`gp2`/`gp3`, GKE gives `standard-rwo`, AKS gives `default`. Hard-coding any one
of them would make the manifests fail to bind on the other four. Portability is
the point: the base stays substrate-neutral, and a cluster that needs a specific
class sets it as the default or patches it in an overlay:

```yaml
# overlay patch, when a specific class is genuinely required
- op: add
  path: /spec/storageClassName
  value: gp3
```

The same reasoning applies to volume sizes being modest in the base and raised in
the cloud overlay, and to no Ingress existing in the base: ingress controllers,
certificate issuers and DNS are cluster-specific.

## Running it

```bash
# laptop
docker compose -f deploy/compose/docker-compose.yml --profile core up

# k3d or kind
kubectl apply -k deploy/k8s/overlays/local

# cloud
kubectl apply -k deploy/k8s/overlays/cloud

# what would be applied, without applying it
kubectl kustomize deploy/k8s/overlays/cloud
```

Secrets first, in every cluster:

```bash
cp deploy/k8s/secrets.example.yaml /tmp/secrets.yaml   # edit, never commit
kubectl apply -f /tmp/secrets.yaml
```

Ray and the edge fleet are applied separately because they need operators
(KubeRay) or a different cluster (k3s on devices): see `deploy/ray/README.md` and
`deploy/edge/README.md`.
