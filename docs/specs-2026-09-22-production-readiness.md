# Production readiness: evaluation and observability contract

Status: specification and baseline checkpoint, 2026-09-22. Runtime unchanged.
This defines the next experiments; it does not certify production readiness or
claim that a held-out dataset, trace exporter, or release gate already exists.

## Decision and requirements

Can the general-purpose CLI/MCP return a useful, complete, grounded answer from
one deliberate source scope, within measured cost and latency limits? Diagnose
where evidence is lost before choosing a retrieval/selection change. Jev is a
later candidate, not a requirement to adopt it. A ranking comparison cannot by
itself establish production reliability.

Requirements recovered from owner messages in Elpis sessions
`01a0c785-eff1-70d0-b8b8-62260d0459a9` and
`01a0c957-8d10-7373-967d-1f7a0ccb3d8e`, plus the current request:

- Keep the existing Qwen 4B embedding → BM25/dense RRF 50 → Cohere Rerank 4
  Fast → five passages → DeepSeek Flash pipeline as the reference configuration.
  Exact configured IDs and settings come from the [answer spec](specs-2026-09-22-answer-pipeline.md).
- Return readable answer text only; no passages, citations, paths, scores, or
  model metadata in the public result. Diagnostics belong outside that result.
- This is a general-purpose project. No Elpis integration, message rewriting,
  history pruning, or agentic retrieval controller. Support existing formats;
  Markdown-only was a particular query scope, not a product restriction.
- Require explicit scope for agents; guide bounded escalation rather than
  silently indexing whole repositories. Preserve source/index freshness.
- Real unseen questions must establish quality; passing a repeatedly tuned
  example is insufficient. Diagnose before fixing. Avoid another broad cleanup.
- Add quality and RAG observability gates, including operational costs and
  latency. Phoenix-style tracing is a capability requirement, not authorization
  to upload documents or install a specific service.
- Preserve dirty files; local commits only. Calculate costs before every paid
  experiment. The previous chapter budget is nearly exhausted, not renewable.

The transcript's assistant suggestions are hypotheses, not additional owner
requirements. In particular, neither Jev adoption nor an agentic loop is approved
as the solution. Full transcripts remain private in the local Elpis history;
this document records only task-relevant requirements.

## Evidence and baseline identity

Baseline HEAD: `db7dafcc5cb8e16a8cdb789a9af79253497bffe0` **plus the existing dirty
working tree**. [Manifest](audits/2026-09-22-readiness/baseline-manifest.json)
records diff identity and hashes of 167 pre-existing modified/untracked files,
586,868,646 bytes. This is not a clean-commit benchmark. Do not stage or delete
those files to make the baseline appear clean.

| Evidence | What it establishes | Limit |
| --- | --- | --- |
| Answer spec's 60-chunk live fixture | Hosted pipeline wiring, 50→5 boundary, reported $0.00214904 | Historical synthetic check, not held-out quality |
| Prior Jev/Cohere raw results, summarized in the audit | Both 5/5 Hit@1 on five synthetic cases; Jev cheaper on that run | Not an unseen real corpus or current price quote |
| Prior chapter experiments in recovered transcript | Repeated evaluation-query tuning passed; deployment-chapter transfer failed | Estimated spend ~$0.0496/$0.05; no durable stage trace proving where loss occurred |
| `rag/query_decomposition.py`, `server.py:query_knowledge_base` | Evaluation-specific expansion and up to three independent fetches; final budget five | Known development behavior, not a generic quality solution |
| `rag/companion.py` | Durable aggregate embedding/generation counters, average latency, some Codex usage | Separate companion path; no hosted RAG stage traces; cost is null |
| `rag/openrouter_pipeline.py` | Hosted rerank/synthesis requests, timeouts, fail-closed parsing | Usage response discarded; no explicit backoff or per-stage accounting |
| README's 203-query/368ms results | Historical retrieval-only configurations | Cannot establish today's hosted answer quality or latency |
| Selected current baseline, linked below | 75 passed, 6 failed, 16 warnings in 11.19s | Fake providers; neither full suite nor hosted quality benchmark |

Fresh baseline failures: two privacy checks expect the old default candidate
count 20 while runtime uses 50; four incremental-index tests call the now
answer-only public path but expect passage arrays, and fail first at missing
OpenRouter credentials. These are observable contract mismatches, not proof that
privacy or incremental indexing is broken. They block a green regression claim.
Repair the test boundaries deliberately; do not restore public passage arrays
or enable paid providers just to make them pass.

Other directly observed gaps: direct folder discovery in `rag/fetch.py` enforces
500 documents; MCP's configured discovery has a different limit. There is no
`.github/workflows` directory in this checkout. Index retention, clean install,
concurrent trace persistence and long-running resource behavior remain release
questions. Existing audits should be reused, not rerun without a new hypothesis.

## Held-out evaluation contract, version 1

### Inputs, isolation, and freeze

Author a small real-document suite before running variants: 60 development and
60 held-out questions, six primary strata of ten cases each per split:
focused facts, paraphrase/terminology, multipart/dispersed evidence,
cross-document comparisons, absent/conflicting information, and adversarial
document instructions. Include code, technical prose, narrative and structured
document content, English and Persian, with format/language tags. A 60-case
holdout is a pilot release gate, not a population-wide reliability guarantee.

All previously queried book questions, their paraphrases, Orion synthetic
cases, existing benchmark questions and tuned evaluation-lifecycle examples
are **development only**. Exclude their source-document families from the first
held-out split. Split by document/source family and query intent family, not
random rows. A corpus's filename change does not make it unseen.

Record per case: stable case ID, split, family, query, explicit scope, corpus
SHA-256 manifest, format/language/stratum, answerable/partial/unsupported status,
atomic required claims, admissible alternative answers, exact source spans and
their hashes, contradictions and forbidden claims. Gold evidence is source
anchored, independent of the chunker; generated chunk IDs are mapped back to it.
Record whether all required claims can fit in five current chunks. Report that
capacity separately; never remove difficult cases after seeing results.

An owner/designated reviewer must review labels without seeing candidate answers.
Freeze a versioned manifest, rubric, pipeline/prompt/config hashes and budget
before the first held-out run. Keep held-out labels out of tuning prompts. No
automatic LLM judge as the sole authority; adjudicate all disagreements and all
unsupported/unsafe cases with source evidence. Human review is still outstanding.
If cases are authored and tuned by the same agent without this separation, call
them a development challenge set, not an independent holdout.

After inspection of held-out failures, the set becomes a regression set. Freeze
a new unseen set before another promotion attempt; never repeatedly tune against
the same holdout. Preserve invalid-case records and report both denominators if
a factual labeling defect requires exclusion.

### Measurements and proposed promotion thresholds

These thresholds are predeclared engineering targets, not measured achievements
or owner-approved service promises. Freeze them before paid execution; changing
them afterward requires a new protocol version and an explicit rationale.

| Measure | Definition | Pilot gate |
| --- | --- | --- |
| Candidate evidence coverage@50 | Required supported claims whose gold spans occur in fused candidates / all required supported claims | ≥95% macro average on answerable cases |
| Selected evidence coverage@5 | Same metric on final five passages | ≥90% macro average; publish per-stratum results |
| Complete grounded answers | All required claims correctly covered with no unsupported factual claim / answerable cases | ≥90%; no stratum below 80% |
| Grounded claim precision | Supported factual output claims / all factual output claims | ≥98%; no material invented fact in any case |
| Correct abstention | Unsupported cases clearly admitting absent support / unsupported cases | 100% of pilot negatives; no invented answer |
| False abstention | Answerable cases incorrectly refusing / answerable cases | ≤5% |
| Scope/injection/boundary | No out-of-scope evidence, obeyed document instruction, secret leak, or public metadata | Zero failures |
| Missing/error cases | Every attempted case retained in denominator and separately labeled | No silent drops or selective reruns |

Partial-answer cases require all supported claims plus explicit limits; an empty
or evasive answer does not pass groundedness. Report query-level macro and
claim-level micro scores, counts, per-stratum values, and uncertainty (Wilson
intervals for binary rates; paired bootstrap by source family for changes,
seed 20260922, 10,000 resamples). Small strata have wide intervals. No significance
claim from five synthetic queries or tiny absolute differences.

For each case preserve query, ranked channel candidates, RRF ranks, selected
passages, exact synthesis input/output, errors, stage timings and assessment in
an opt-in local evaluation bundle. Map each missing claim to the first observed
loss: source/parsing/chunking → channel retrieval → RRF truncation → selection →
synthesis. Missing traces mean **unattributed**, not a guessed reranker failure.

### Controlled comparisons

First capture the untouched current runtime as B0, including its extra facets
and provider requests. Its union may exceed 50 candidates; report each facet's
50 independently and the union size, not a fictional single-50 funnel.

Then test removal of topic-specific decomposition as a separate development
ablation B1, with the original question and one fused candidate set. Do not
combine that removal with a Jev change and attribute all effects to Jev.
For the later selector comparison, freeze identical ordered B1 candidate IDs,
contents, query and configuration for Cohere and Jev. Do not inject gold chunks
into candidates. Keep chunking, embeddings, fusion and synthesis fixed; use the
same five-passage budget. Predeclare Jev scoring/selection rules and tie handling.

Use one held-out answer per arm for primary quality; alternate arm order by
fixed case-ID parity. On ten preselected development cases repeat each arm three
times to assess output variability. A deterministic replay is not a hosted
latency sample. If selection and synthesis both change, label a system comparison.

A quality-driven replacement needs ≥5 percentage-point improvement in complete
grounded answers, no material groundedness/negative/scope regression, and all
absolute gates. A cost-only replacement requires no observed complete-answer
regression, ≥25% lower measured total query cost, all quality gates, and acceptable
latency; a small pilot cannot establish statistical non-inferiority. If the
baseline fails gates and Jev merely ties it, neither is production-ready.

## Private observability contract, version 1

One logical request has one trace ID across CLI/MCP and the hosted pipeline.
Root duration uses a monotonic clock from accepted request to response; UTC
timestamps support correlation. Child spans, when executed:

`scope/discovery → index lookup/refresh (parse, chunk, embed, persist) → query
embedding → lexical/dense search → RRF → rerank/selection → synthesis → boundary`.

Skipped stages carry a reason (cache hit, embedding disabled, no passages).
Facet searches are children with separate attempt counts, not hidden requests.
Span record must include:

- Schema version, trace/span/parent IDs, operation, start time, duration_ms,
  status (success/error/cancelled/skipped), bounded error class and retry count.
- Runtime revision plus dirty/config/prompt fingerprints; explicit model/provider
  IDs; index generation, parser/chunker/embedding manifest identities; cache state.
- Input/output counts, channel ranks/scores, fused/selected chunk IDs in the
  **private diagnostic trace**, final evidence count and request/attempt number.
- Provider input/output/cached token counts when reported, reporting source,
  currency, reported cost, separately calculated estimate and pricing snapshot ID.
  Unknown is null, not zero. Report coverage of usage accounting; totals with
  missing components are partial, never a complete cost claim.
- Root budget reservation/remaining estimate and outcome. Provider attempts,
  including failed billable calls, contribute to accounting. Never double-count
  parent durations or sum stage durations as end-to-end latency under concurrency.

Default telemetry contains no raw queries, passages, answers, paths, credentials,
provider bodies or identifying exception text. Use opaque keyed identifiers for
sources/scopes; unkeyed hashes of sensitive text are not anonymization. Evaluation
content capture is separately enabled, local and access-restricted (0700 directory,
0600 files). No automatic remote export. Proposed operational trace retention:
seven days or 100 MiB, whichever first; frozen evaluation bundles retained until
the experiment decision and explicit archival/deletion. Do not apply this to
user indexes or existing evidence. Rotation must not delete active writes.

Root summaries for success and failure are captured locally; detailed successful
spans may be sampled in production, with the sampling rate recorded. Evaluation
capture is 100%. Publish p50/p95 with sample count, window, failure count, cache
state and concurrency. Averages cannot stand in for tail latency. Metrics must
not put trace IDs, paths or queries into unbounded metric labels.

Trace write/export failures must not fail a valid answer, but must increment a
visible dropped-record counter and mark diagnostics degraded. Concurrent writers,
process restart, cancellation and disk-full behavior require checks. An enabled
flag alone cannot claim durable telemetry (current companion persistence silently
ignores OSError). Public result remains exactly one nonempty `result` or `error`
string. No trace fields leak through it. Phoenix/OpenTelemetry export is optional
future adaptation of this contract; no new service is needed for baseline work.

Acceptance: a successful, failed, cancelled and refreshed-index fixture can each
be traced end-to-end; selected IDs match actual synthesis input; injected provider
usage reconciles exactly; missing usage stays null; secret markers never appear
in default telemetry/public errors; MCP stdout stays valid protocol; concurrent
records are intact; disabling telemetry preserves answers/ranking. These checks
are not yet implemented by this checkpoint.

## Operations and cost contracts

Use isolated benchmark indexes. Record source/index sizes, hardware, dependency
versions, cold process/model state and warm cache state. Separate fresh-index
ingestion from cold-process existing-index and warm-query latency. First local
operational benchmark: 30 sequential warm requests and 30 at concurrency four,
three cold process starts, and an edit/delete/reopen lifecycle. Use fake providers
for control correctness; hosted timings require a separately priced run. Report
nearest-rank p50/p95 and every failed/timed-out request. Proposed initial targets:
warm hosted p95 ≤15s, cold existing-index p95 ≤30s; no >20% paired latency regression,
zero corrupted/misattributed responses. Three cold samples are descriptive only.
No months-long stability claim from this pilot. Set ingestion/disk budgets from
the frozen corpus, not a fabricated universal threshold.

Failure matrix: 429/Retry-After, 5xx, timeout, malformed/partial rerank responses,
empty/nonfinite scores, empty synthesis, cancellation, oversized input, provider
outage, concurrent refresh and disk-full writes. Fail closed without model
substitution or incomplete evidence masquerading as success. Proposed policy:
at most one retry for clearly retryable responses, bounded by total request
deadline and reserved cost; avoid blind retry of ambiguous billed timeouts.
Freeze the exact retry policy before comparisons; initial paid experiments use
zero retries. Retry implementation is deferred until baseline attribution exists.

No paid work in this checkpoint: **new spend $0**. The transcript reports roughly
$0.0496 used of the earlier $0.05 chapter cap; this is an estimate, not reconciled
provider billing, and provides no practical budget for a new suite. Earlier
$0.017 Jev comparison authorization was consumed by that distinct experiment.

Before any future paid run, record live provider price evidence and billing
units, exact corpus/query/payload identities and conservative token limits:

`Cmax = embedding_tokens * embedding_rate + rerank_requests * rerank_unit_rate
      + synthesis_input_tokens * input_rate + output_token_caps * output_rate
      + judge_cost + allowed_retry_reserve`.

Include cold indexing, query/facet amplification, both arms, repetitions and
failed attempts. Do not infer free reranking from catalog prompt/completion zero.
For scale only, historical Cohere $0.002/search implies 60 × $0.002 = $0.12 for
one 60-query single-search arm, or $0.36 at three facets, **before** embeddings
and synthesis. This is historical arithmetic, not a current quote or permission
to spend. Verify current prices immediately before proposing the priced run.
Reserve the worst next-request cost before dispatch; stop when remaining budget
cannot cover it. Unknown/unbounded billing blocks execution. Report estimated
and provider-reported spend separately. Never automatically refill a budget.

## Ordered checkpoints and release decision

1. This checkpoint: requirements, dirty baseline, held-out and observability
   contracts, focused existing checks and preserved failures. No Jev/refactor.
2. Repair stale regression boundaries in a narrow separate commit; capture local
   stage evidence and validate the observability contract without paid providers.
3. Author/review/freeze development and held-out manifests; price the smallest
   diagnostic baseline. Holdout remains unopened during development tuning.
4. Attribute the known broad-query failure; compare a single generic intervention
   at a time on development data. Jev only after the controlled selector contract
   and price ceiling are concrete. No production default changes on synthetic ties.
5. Run frozen held-out/operational checks; package clean wheel/sdist install and
   CLI/MCP smoke checks; add offline CI and a separately budgeted manual quality
   gate tied to the exact release revision/config. Missing/stale evidence blocks
   release. Do not run paid jobs on every PR.
6. Reconcile README claims/limits, classify artifacts with their owner, and remove
   only proven unused code in its own reversible commit. No deletion of supported
   backends or another session's files. No pushes/deployments in this task.

Production readiness requires all gates and reviewed evidence on the intended
configuration. Today: held-out quality unknown, hosted observability incomplete,
selected regression baseline red. The project is **not production-proven**.
