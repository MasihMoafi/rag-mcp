# Local RAG diagnostics

CLI and MCP queries now write one private JSON trace per request under
`$XDG_STATE_HOME/ragmcp/traces` (normally `~/.local/state/ragmcp/traces`).
Set `RAG_MCP_TRACE_DIR` to a dedicated directory or `off` to disable tracing.
There is no remote exporter or new service dependency.

Traces contain parent/child spans, timings, errors by class, index build/refresh/
cache state, candidate and selected ranks, request-local opaque passage IDs,
provider model IDs, reported rerank/synthesis usage, and the server startup hash.
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

- Embedding dispatch records timing/model/counts, but provider token/cost usage
  is still unavailable at that interface and stays null. Legacy per-query
  fallback worker threads are contained in the parent batch span, without child
  attempt traces. Rerank/synthesis attempts have no automatic retries.
- `reported_cost_subtotal_usd` is a subtotal of known reported costs, never a
  whole-request estimate. `unreported_cost_spans` identifies missing accounting;
  the estimated cost remains null. Provider-level failover billing is not inferred.
- Traces do not yet carry a full Git/dependency/configuration release manifest.
  Evaluation bundles must retain that separately. The server hash identifies
  the loaded server file, not every Python module or the provider implementation.
- Index metadata persistence has its own span; vector-table writes are currently
  included in index build/refresh timing. Aggregate parsing spans include worker
  completion rather than individually tracing each parser worker.
- No latency SLO, online quality judge, retry policy, external dashboard or
  held-out quality gate is claimed implemented. The broader
  [readiness contract](specs-2026-09-22-production-readiness.md) remains the target.

Verification: failure/cancellation, missing usage, privacy markers, explicit
content capture, concurrent threads and processes, restart, retention, disk error,
real LanceDB refresh, selected-evidence identity and disabled-tracing equivalence
are covered by `tests/unit/test_tracing.py` and
`tests/integration/test_trace_pipeline.py`. Provider responses in these tests
are simulated; they do not establish live model quality or hosted latency.
