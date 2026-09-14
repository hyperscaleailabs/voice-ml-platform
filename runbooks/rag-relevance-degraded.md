# RAG relevance degraded

## Symptom

Answers are vague, off-topic, or confidently wrong about the corpus. Users
report "it made something up" or "it read me something unrelated". Retrieval is
returning the wrong chunks, too few chunks, or junk that passed the floor.

## Severity

SEV-2 if answers are wrong rather than merely unhelpful — a confidently wrong
spoken answer is worse than a refusal, because the user cannot see the source.
SEV-3 if answers are hedged or empty but not incorrect.

## Background: the relevance floor

The retriever fuses a keyword prefilter, a vector top-k and a graph expansion
with reciprocal rank fusion, then applies a **relevance floor**
(`relevance_floor = 0.15` in `configs/rag.toml`). The floor exists because
cosine similarity separates poorly at the top of a small corpus: rank 1 and rank
20 measured 0.494 and 0.442 (alpha-core, cycle 5, 2026-09-12, `README.md`
"Talking about this codebase"). With that little spread, an off-topic question
still returns something, and without a floor the agent reads the nearest junk
aloud. `RetrievalResult.floored` counts how many candidates the floor removed;
it is the first number to look at.

Returning nothing is a correct outcome. An empty retrieval that produces "I
don't have that" is a working system, not a broken one.

## Diagnose

### 1. Reproduce the query

```bash
vmp rag query "the exact thing the user asked" --k 5 --json
```

The JSON carries the fused scores, the per-path ranks and `floored`. Read it
before touching anything.

### 2. Which of the four failure modes is it

| What the JSON shows | Cause |
|---|---|
| `floored` high, few or no results | The floor is doing its job and the corpus lacks the answer — or the floor is too high for this corpus |
| Results returned, all irrelevant | The floor is too low, or the embedder changed |
| Relevant chunk exists in the corpus but never appears | Indexing problem — it was not chunked, not embedded, or not indexed |
| Right chunk retrieved, wrong answer spoken | Not a retrieval problem. Context packing or the model; check `max_context_chars` in `configs/serving.toml` |

The fourth row is the one most often misdiagnosed. Confirm the chunk was
actually in the context before blaming retrieval: the `retrieve` stage in the
trace records what was passed on.

### 3. Is the index current

```bash
vmp rag index <corpus-dir> --dry-run     # what would be indexed, and how many chunks
```

Compare the chunk count against the last known-good run. A collapse means the
corpus moved, a glob in `[rag.corpus] include` stopped matching, or an indexing
job failed. A jump means duplicates — the same document indexed twice pushes real
answers out of the top-k.

Check the store itself is reachable and populated:

```bash
kubectl exec -n platform deploy/api -- sh -c 'echo > /dev/tcp/qdrant/6333' && echo qdrant reachable
kubectl exec -n platform statefulset/postgres -- psql -U vmp -d vmp \
  -c "select count(*) from chunks;"
kubectl logs -n platform -l app.kubernetes.io/name=vmp-api --tail=200 \
  | grep -i "embed\|retriev\|vector"
```

### 4. Did the embedder change

An embedder change invalidates every stored vector: new query vectors are
compared against old document vectors and the scores are meaningless. Check
`[rag.embedder]` in `configs/rag.toml` and the deploy history. If `kind` or `dim`
changed without a reindex, that is the cause and nothing else needs
investigating.

## Mitigate

```bash
# embedder changed, or the index is stale or duplicated: rebuild it
vmp rag index <corpus-dir>

# verify against the query that failed
vmp rag query "the exact thing the user asked" --k 5 --json
```

If the floor is wrong for the corpus, change it in `configs/rag.toml` and
redeploy — but move it in small steps and check both directions:

- **Raising it** reduces junk and increases refusals.
- **Lowering it** reduces refusals and increases junk read aloud.

Test any change against a set of questions the corpus genuinely cannot answer.
A floor tuned only on questions with answers will always look better lower.

```bash
kubectl rollout restart deployment/api -n platform
```

If the cause is a bad model rather than bad retrieval, go to
`runbooks/rollback-model.md`.

## Verify

```bash
vmp rag query "<a question the corpus answers>" --k 5 --json     # returns the right chunk
vmp rag query "<a question the corpus cannot answer>" --k 5 --json  # returns little or nothing
vmp eval gate --rules configs/gates.toml --metrics /tmp/metrics.json
```

Both query directions, every time. A change that fixes recall and silently
destroys the refusal behaviour is a regression that ships looking like a fix.

## Follow-up

- Add the failing question to the golden set so the regression is catchable.
- If an embedder change caused it, the action item is a reindex step wired into
  the same change — an embedder version belongs in the index metadata so a
  mismatch is detected rather than inferred.
- If the index silently went stale, alert on chunk count and index age.
