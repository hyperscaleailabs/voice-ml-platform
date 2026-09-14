# Security policy

## Supported versions

This project is pre-1.0. Security fixes are made on `main` and released in the
next version; older versions are not patched.

| Version | Supported |
|---|---|
| `main` | yes |
| 0.1.x | yes |
| older | no |

## Reporting a vulnerability

**Do not open a public issue.**

Report privately through GitHub's
[private vulnerability reporting](https://github.com/hyperscaleailabs/voice-ml-platform/security/advisories/new)
on this repository. Include:

- what the vulnerability is, and which component or file;
- how to reproduce it, ideally minimally;
- what an attacker gains — the impact, not just the defect;
- the version or commit you tested.

What to expect: an acknowledgement within 3 working days, an assessment with a
severity and a planned fix within 10 working days, and credit in the advisory
unless you prefer otherwise. Please give us 90 days before public disclosure, or
less if the issue is already being exploited — tell us if it is.

## Scope

In scope: the `vmp` package, the deployment manifests under `deploy/`, the
container images built from `deploy/docker/`, and the CI workflows.

Out of scope: vulnerabilities in third-party dependencies (report those
upstream, though do tell us if this project's use of one is unsafe); findings
that require an attacker to already have cluster-admin or root on a device; the
deliberate placeholder credentials in `deploy/k8s/secrets.example.yaml` and
`deploy/compose/.env.example`, which exist to be replaced.

## Operating this platform securely

Things that are the operator's responsibility, listed because they are the ones
most often missed:

- **Secrets.** `deploy/k8s/secrets.example.yaml` and `deploy/compose/.env.example`
  contain placeholders. Replace every one before deploying, and keep the real
  values out of the repository.
- **Network.** The API's NetworkPolicy restricts egress to the shared services
  and DNS. The edge DaemonSet's NetworkPolicy permits DNS only, matching
  `offline_only = true`. Keep them aligned with the policy files; a policy value
  without an enforcing rule is a statement of intent.
- **Containers.** The manifests run as non-root with a read-only root filesystem,
  all capabilities dropped and `RuntimeDefault` seccomp. Preserve those settings
  when adding workloads.
- **Ingress.** `deploy/k8s/overlays/cloud/api-ingress.yaml` ships with TLS
  annotation placeholders. It is not usable until the issuer, the class and the
  hostname are set.
- **Data.** Audio and transcripts are sensitive. Retention, PII scrubbing,
  consent and the edge offline-only policy are covered in
  `runbooks/data-retention-and-privacy.md`.
- **Model artefacts.** A model trained on user data may have memorised it.
  Treat artefacts with the same care as the corpus they came from.

## Model-specific risks

Worth stating because they are not conventional software vulnerabilities:

- **Prompt injection through retrieved context.** Content in the RAG corpus
  reaches the model. Treat an indexed corpus as untrusted input.
- **Data leakage between sessions.** Session state is per-session by design;
  a change that shares context across sessions is a security change.
- **Bundle integrity.** Edge bundles carry a checksum manifest and are verified
  at build, at install and at every start. Never bypass verification to get a
  device running — a bundle that does not verify has unknown contents, and the
  policy travels in the same manifest as the checksums.
