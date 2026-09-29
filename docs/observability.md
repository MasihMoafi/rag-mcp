# Local RAG diagnostics

Paired Studio, CLI, and MCP queries write one private JSON trace per request under
`$XDG_STATE_HOME/ragmcp/traces` (normally `~/.local/state/ragmcp/traces`).
Set `RAG_MCP_TRACE_DIR` to a dedicated directory or `off` to disable tracing.
There is no remote exporter or new service dependency.

Traces contain parent/child spans, timings, errors by class, index build/refresh/
cache state, candidate and selected ranks, request-local opaque passage IDs,
provider model IDs, reported rerank/synthesis usage, the safe configuration
revision, the loaded workspace revision when present, and the server startup hash.
The configuration revision identifies the shared settings snapshot before a
per-run answer-model override; the effective model is recorded separately.
The root answer model is a configured setting, not evidence that generation
ran; retrieval-only runs have no synthesis stage. An off reranker has no active
model in the safe configuration or trace. Studio's retrieval funnel uses the
facet selection count when a run ends before reranking or synthesis.
Dense retrieval spans also retain the backend and configured distance metric;
their ranked scores are metric-dependent and may be negative. A top-ranked
dense score is not directly comparable with BM25, RRF, or reranker scores.
The `retrieval.rrf` span records its actual weighted reciprocal-rank values
and labels them `score_kind: rrf`; it no longer displays the source document's
retained dense or lexical score as the fusion score. These RRF values rank only
that fusion stage and are not probabilities.
When reranking succeeds, `selection.rerank` records the model scores used for
its selected order with `score_kind: reranker`. A local reranker failure or a
too-small candidate list marks that stage skipped and labels retained source
scores as `retained_source`; hosted reranker failures still propagate. Query
selection and synthesis retain source-stage scores for compatibility, and Studio
labels them separately from reranker scores. Reranker scores are model-specific,
not probabilities or cross-stage confidence measures. The private result's
evidence score still comes from its source retrieval stage and may not follow
the final order after reranking or multi-part balancing.
The public answer remains unchanged. `rag_settings_get` reports whether tracing
is enabled and the process-local dropped-trace counter. Errors writing traces
warn on stderr and do not replace the answer.

Normal traces exclude queries, document text, answers, source paths, credentials
and provider error bodies. IDs are HMACs under an ephemeral key per request;
they correlate stages within a request, not documents across requests. Trace
files use mode0600 inside a mode0700 directory. A cross-process lock and atomic
publication prevent partial/interleaved files. Operational records expire after
seven days or when retaining the next record would exceed100MiB. Rotation runs
on the next completed query and touches only owned `trace-<uuid>.json` files.

For a deliberate local evaluation, set `RAG_MCP_TRACE_CONTENT=1` and use a
dedicated `RAG_MCP_TRACE_DIR`. The `evaluations/` subdirectory then captures exact
provider request bodies, selected content/source locations, query and answer.
These are sensitive artifacts: do not upload them by default. They are never
automatically rotated; once their100MiB budget is full, new captures are dropped
with a warning until the owner archives them. Metadata-only traces continue to
use the parent directory. No credentials are captured even in evaluation mode.

Known limits:

- Embedding dispatch records timing/model/counts. OpenAI-compatible embedding
  responses now contribute reported input tokens and optional reported cost;
  direct Ollama `/api/embed` responses contribute reported input tokens when
  every batch supplies `prompt_eval_count`. Missing or invalid counts and local
  model usage remain unknown. Neither path estimates embedding cost. Legacy
  per-query fallback worker threads are contained in the parent batch span, without child
  attempt traces. Rerank/synthesis attempts have no automatic retries.
- `reported_cost_subtotal_usd` is a subtotal of known reported costs, never a
  whole-request estimate. `unreported_cost_spans` identifies missing accounting;
  the estimated cost remains null. Provider-level failover billing is not inferred.
- The runtime manifest identifies files and dependencies observed on the first
  query, not the exact Python objects already loaded in memory. Its bundled
  archive hash does not prove the web source Git commit that built it, and a
  clean-machine dependency resolution remains unverified. Evaluation bundles
  still need a separate two-repository release manifest. The server startup
  hash by itself identifies only the loaded server file.
- Index metadata persistence has its own span; vector-table writes are currently
  included in index build/refresh timing. Aggregate parsing spans include worker
  completion rather than individually tracing each parser worker.
- No latency SLO, online quality judge, retry policy, external dashboard or
  held-out quality gate is claimed implemented. The broader
  [readiness contract](specs-2026-09-22-production-readiness.md) remains the target.

Verification: failure/cancellation, missing usage, privacy markers, explicit
content capture, concurrent threads and processes, restart, retention, disk error,
real LanceDB refresh, selected-evidence identity and disabled-tracing equivalence
are covered by `tests/unit/test_tracing.py`,
`tests/unit/test_api_embedding_contract.py`, and
`tests/integration/test_trace_pipeline.py`. Provider responses in these tests
are simulated; they do not establish live model quality or hosted latency.

## Inspect saved runs

`rag runs` lists recent completed metadata traces. `rag runs <run-id>` shows
measured stages and models; add `--json` for the recorded attributes and passage
ranks in the same structured record used by Studio. `--limit 1–100` bounds the
recent listing; `--offset` pages through older records. The interactive menu also
offers “Inspect recent runs”. No query is executed by
these commands.

`rag runs <run-id> --result` explicitly opens the private saved answer and source
passages on this machine; add `--json` for the full result record. Results expire
after seven days and may be unavailable even when a metadata trace remains.
The private result now includes a process-captured runtime manifest: relative
engine-file hashes, installed distribution names/versions, Python/platform,
local Git HEAD and clean/dirty/unavailable state, and the bundled Studio archive
hash when present. The normal metadata trace carries only its SHA-256
fingerprint (or `unavailable`). Capture is local and bounded on the first query
in a process; restart a long-lived engine after code or dependency changes.
When Studio was built through the staging command, the archive also carries
its build ID, measured web-source fingerprint, installed pnpm-lock hash, and
local Git state. A skipped build is labeled `unattested`. A local `.env` file
is recorded only as present; its value and process environment are not captured.

`rag search QUESTION --workspace ID` records a retrieval-only run. Its local
terminal output includes ranked passage text; `--no-content` omits text and
scores. MCP `search_workspace` records the same run type and returns only
source locations and run/revision IDs by default. `include_content: true`
explicitly sends passage text and scores to the MCP host. Both clients use
the same index scope as paired Studio when searching every workspace source.
If a legacy index lacks an unambiguous source offset, search reports the
retrieved rank with `location_verified: false` and null offsets. The clickable
Studio evidence list still contains only passages whose source span was checked.

Start `rag studio --studio-url http://localhost:3000` against the updated Studio
checkout, then open Observability. Its local-engine inspector reads these same
files through authenticated, origin-checked `GET /v1/runs` and
`GET /v1/runs/<run-id>`. The list accepts `limit` and `offset`, returns
`next_offset`, and counts unavailable records as `skipped`. Responses are
`no-store`. Normal metadata records only are accessible: evaluation captures,
symlinks, malformed records, and records larger than 4 MiB are not served.
The reader explicitly selects supported diagnostic fields; it does not serve
arbitrary JSON files. Existing seven-day/100 MiB retention still applies.

When Studio is paired, its parsed source text is atomically published into a
private engine workspace. `rag workspaces`, `rag query --workspace <id>`, and
the MCP query tool's `workspace_id` address that same snapshot. Workspace list
and detail responses expose metadata only; raw document bodies stay in the
owner-readable local store. A separately authenticated snapshot endpoint returns
extracted document text for bidirectional source reconciliation. Replacements use
revision checks and unchanged snapshots retain their index identity.

## Shared execution and configuration

Paired Playground uses `POST /v1/runs`, the same Python `execute_run()` entry point
as CLI/MCP. It includes a run ID, workspace revision, and selected source IDs.
The gateway refuses concurrent queries rather than growing a work queue. Retrying
the same completed ID returns its saved response; reusing it for different inputs
is rejected. `POST /v1/runs/<id>/cancel` requests cooperative cancellation at stage
boundaries. A provider call already in progress may finish before cancellation is
observed. `GET /v1/runs/<id>/result` returns a private result or active-run status.

Private results contain the answer, safe configuration snapshot, and resolvable
workspace passages with source IDs and UTF-16 offsets. These are separate from the
metadata-only trace endpoints: paired clients can read them, CLI/MCP public answers
remain concise. Result files use 0600 permissions in a 0700 `traces/results` directory,
an 8 MiB record limit, and seven-day/100 MiB retention. Disabling tracing also disables
result persistence. The gateway retains at most 20 recent results in memory for retries;
a disk error is reported without discarding the answer. Restart-safe retries require
a successfully saved result. Unresolvable or ambiguous source offsets are not invented.

`GET /v1/config` and `PATCH /v1/config` share the CLI/MCP settings file. The browser
can change candidate/final counts, answer model, reranker, and contextual retrieval.
Credentials and embeddings remain configured through `rag setup`. Environment-controlled
settings are read-only in Studio. Each run freezes its settings so a concurrent CLI edit
applies to the next run. `GET /v1/capabilities` describes configured stages; a key's presence
does not imply a successful model probe. `GET /v1/health` reports engine/storage readiness,
independent of the legacy Codex companion's `/health` response.

Studio answers link to their exact metadata trace. Citation buttons open recorded
source locations; sources edited since the run are refused for navigation, while the
saved passage remains readable. Chunking, comparison, retrieval experiments, and
evaluations still use the explicitly labeled browser-preview pipeline. Bundling the
browser with the Python package remains separate work.

`tests/integration/test_control_plane.py` exercises real lexical retrieval through
browser HTTP, CLI, and MCP, comparing source identity and stage structure with a
simulated provider. Browser walkthroughs also cover cancellation, provider failure,
bidirectional settings updates, exact citation navigation, and English/Persian mobile views.
