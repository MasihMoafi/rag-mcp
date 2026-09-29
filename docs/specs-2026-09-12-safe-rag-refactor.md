# Safe deletion-first RAG refactor

Status: audited mechanical pass verified, 2026-09-12; final evidence and retained
behavior boundaries below. This is not a proof of globally minimal code.
Specification, plan and atomic tasks are kept together to avoid workspace clutter.

## Approved structural follow-up

Success case: module names reveal responsibilities instead of hiding chunking,
BM25, ingestion and reranking in core.py; discovery is separate from fetch's
pipeline reuse/query entry points. Parser naming/comments describe extraction
truthfully. Compatibility imports and existing callable contracts remain usable.
Recovery checkpoint: ed443ae; baseline215 checks passed in the preceding pass.

Plan: Ohm owns core and extracted chunking/lexical/ingestion/reranking/pipeline
modules; Halley owns fetch/discovery; coordinator owns docs/parsing review and
integration. Agree new module interfaces without circular imports. No inheritance
framework or dynamic module replacement to disguise imports. Retain thin legacy
entry points where needed. No algorithm, format-support, provider or schema change.

Done checks: focused responsibility modules import independently; old imports and
method overrides pass compatibility tests; original215 checks plus new boundary
checks pass; concurrency/device behavior unchanged; README matches actual structure
and makes no AST claim. Clarify names/comments only in touched RAG responsibilities,
not unrelated CLI/UI code. Record relocation and total production counts honestly.

### Structural follow-up result

Verified against checkpoint ed443ae with the combined regression command below:
**240 passed, 38 existing LanceDB deprecation warnings**. No live-provider/GPU
throughput or new wheel-build claim; no model downloads, deployment or push.

- core.py: 787→173, now pipeline configuration/composition and search.
- fetch.py: 377→241, now scoped pipeline caching and query orchestration.
- chunking.py144, bm25.py121, ingestion.py197, reranking.py172,
  discovery.py145 contain the corresponding implementation responsibilities.
- parsers/{text,pdf,json,notebook,office}.py contain standalone readers.
  parsing.py48 is dispatch/legacy adapters; notebook_chunker.py7 is compatibility
  imports, not a second implementation. The parser package imports formats lazily.
- Canonical RAGPipeline and query_documents coexist with legacy class/query names.
  Old core chunk/BM25 imports, pipeline method overrides, parser keyword arguments
  and legacy pickle references have explicit checks.
- README now maps the actual files, identifies regex code labels rather than AST
  chunking, and points to the correct fusion and embedding modules. Moved comments
  describe responsibilities and failure behavior instead of tutorial step numbers.

The structural pass increases total production Python (including parsers/) from
5,009 to5,085 lines: **+76**, not a deletion claim. The earlier initial source
checkpoint was5,104 lines. Module boundaries, docs and compatibility account for
the organizational overhead; no new retrieval strategy or feature was added.

Before-change parser/discovery/reranker contracts passed; targeted rank-order,
legacy delegation and device-argument mutations failed expected assertions.
Extracted engine bodies were compared to the checkpoint excluding docstrings;
reranker initialization also matched exactly after indentation normalization.
Fresh imports of discovery/chunking/BM25/notebook load no core, torch, LanceDB or
unrelated OCR/PDF runtimes. Final outputs and raw commands remain in the session
tool transcript; the same original regression selection is retained.

Independent review found a boundary worth recording: callable legacy imports and
pipeline overrides are retained, but arbitrary rebinding of helper globals on a
legacy module does not forward into the new implementation module (for example,
notebook_chunker.extract_exercise_number). No repository caller performs such
rebinding. Dynamic module-forwarding machinery was deliberately not introduced;
new code should target the named implementation module. No claim of universal
monkeypatch compatibility or production perfection is made.

Execution clarification from Masih: use a reviewed local Git checkpoint, not an
elaborate backup workflow. Checkpoint 7054026 preserves the current RAG source,
tests and dependency files, including previously untracked modules. Working files
were compared with that commit. This supersedes the snapshot/restore-directory
procedure below: recover named files from the checkpoint into a separate location
if needed, never blanket-reset shared work. Benchmark corpora and unrelated dirty
experiment files are excluded and must remain untouched. ES.md is an ignored
progress log, not an implementation recovery dependency.

## Goal (intent: the success case)

rag-mcp has less unnecessary production code and a clearer, easier-to-explain
structure, while its supported functionality, retrieval results, citations and
hardware efficiency are preserved. Recovery is verified, and before/after evidence
supports that preservation rather than merely asserting it.

## Constraints

Follow the first-principles skill's
delete → simplify/optimize → accelerate sequence and dev/CODING_GUIDELINES.md.
There is no deletion quota. Fewer concepts and less duplication matter more than
fewer newlines; no minification, speculative frameworks or single-use abstractions.

## Context & Architecture Invariants

The MCP server and CLI use the same retrieval engine. Existing parsing, indexing,
storage, provider and retrieval contracts below remain authoritative; this work
does not replace the architecture.

Pre-execution observation: HEAD ea8e67c; core.py 924 lines, fetch.py 407; rag/*.py plus
server.py 5,104 physical lines. Substantial tracked and untracked changes exist.
HEAD alone was not the working baseline. Checkpoint and subsequent verification
are recorded separately below; this initial observation is retained as history.

## Acceptance Criteria

- [ ] Recovery: a restored copy matches the inventoried working baseline hashes,
  including relevant untracked source; its location and recovery steps are recorded.
- [ ] Happy path: deterministic before/after fixtures return identical chunks,
  source IDs, citations and ordered retrieval results under fixed configurations.
- [ ] Failure modes: affected malformed-input, provider/storage-failure and OOM
  paths retain their baseline error/fallback contracts, demonstrated by focused tests.
- [ ] Concurrency: affected scheduling and batch-bound tests detect serialization
  or limit violations; no concurrency/device changes are accepted without evidence.
- [ ] Test sensitivity: each changed responsibility has a protecting test that
  detects an intentional relevant fault before the corresponding deletion lands.
- [ ] Regression: applicable baseline and combined integration gates pass; existing
  failures and unavailable checks remain explicitly identified, not hidden.
- [ ] Observability: raw commands/results and production/test line counts are
  recorded; genuine deletion is distinguished from relocation. Claimed hardware
  improvements meet the independently agreed performance comparison thresholds.
- [ ] Simplicity: independent review identifies removed redundancy and explains
  retained responsibilities; no cosmetic line-count reduction or speculative layers.

## Stop Conditions & Escalation Rules

1. Stop worker edits if recovery is unverified or file ownership conflicts.
2. Stop the affected slice if existing behavior contradicts this contract; present
   the discrepancy to Masih before choosing a behavior change.
3. After three unsuccessful focused repair attempts, isolate/revert only the
   candidate slice and ask for direction with the failing evidence.
4. Ask before touching live data/configuration, paid services, private-data uploads,
   new dependencies or external deployment/publication.
5. Defer optimization when hardware evidence or agreed thresholds are unavailable;
   do not label it verified. Consult before changing an experiment's scope or cost.
6. Stop deletion where a supported behavior would be lost or necessity remains
   unresolved; surface the product decision rather than deleting to meet a quota.

## Plan: recovery and baseline prerequisites

The coordinator must complete these prerequisites:

1. Inspect current diffs, untracked/ignored source and active writers. Establish
   exclusive file ownership. Coordinate overlapping sessions or use isolation;
   never race edits to shared files.
2. Preserve the actual working baseline, including relevant untracked modules,
   tests, configuration and documentation. Prefer an isolated refactor worktree
   with a local checkpoint of the current source state, not merely HEAD. Explicitly
   enumerate and inspect inputs; exclude credentials, corpora and generated indexes.
   If committing mixed work is inappropriate, use an access-restricted local
   snapshot with an inventory and hashes. Do not stash away other sessions' work.
3. Verify recovery into a separate temporary directory, including untracked files,
   by comparing hashes. Record identifier/location, inventory and recovery procedure
   in ES.md. A tracked-only patch is insufficient. Restore separately and reconcile
   named files/hunks; no destructive blanket restore in the user's checkout.
4. Run and retain the baseline checks. Identify existing failures/skips explicitly;
   do not silently waive them. Record dependencies/configuration without secrets.

Tests use isolated temporary indexes/configuration. Do not modify live data,
restart MCP, deploy, push, publish, upload, install dependencies or invoke paid
providers without the relevant permission. Private downstream applications are
not needed for this mechanical refactor. No Studio or downstream worktree changes.

## Plan: parallel team, files and agent prompts

The user authorized parallel work. Start two implementation agents only after the
safety gate, with the coordinator providing integration and independent review.
Read-only inspection/review may run before that gate; it authorizes no worker edits.
Give each agent this entire spec, the verified checkpoint, explicit file/test
ownership, relevant callers and the skill/guideline paths to read.

### Agent A: core.py

Own rag/core.py, explicitly assigned extracted modules and focused test files.
Inspect orchestration, parsing, chunking, BM25 and reranking for duplication and
unnecessary complexity. Preserve existing callable/import contracts. Separate
cohesive responsibilities when useful, but report code moves separately from
actual deletion. Do not redesign algorithms or change chunk output.

### Agent B: fetch.py and connected RAG implementation

Own rag/fetch.py initially. Inspect embeddings.py, contextual_retrieval.py,
notebook_chunker.py, storage backends and settings.py; propose an exact additional
file list for coordinator assignment before editing. Simplify demonstrated waste
in discovery, caching, configuration orchestration and adapters. Do not create a
competing shared parser, extension registry or settings abstraction alongside A.

### Coordinator

Own ES.md, this spec and assigned integration tests. Inspect server.py, CLI and
companion consumers; allow necessary wiring or separately assigned, demonstrated
duplication removal, not UI redesign. Agree shared interfaces before workers use
them. Assign tests as well as production files exclusively. Integrate one passing
slice at a time. Review actual diffs and evidence, not just worker summaries.

### Common worker prompt

> Find the smallest clear implementation of existing required behavior in your
> assigned scope. For each deletion, identify callers, behavior at risk, evidence
> of redundancy, and a protecting test. Write missing tests BEFORE production
> edits and demonstrate that they detect a plausible loss of the behavior. Delete
> or simplify one cohesive piece, run focused checks, review the diff and checkpoint
> the passing result. When it breaks, diagnose and try a smaller equivalent repair;
> otherwise undo only your last change. Never weaken tests to make deletion pass.
> Preserve concurrency, batching, device selection and meaningful errors. Be
> creative within these contracts; coordinate shared-interface changes. Return
> changes, deletions versus moves, raw before/after evidence, risks and untested
> paths. Green existing tests alone are not proof of equivalence. Never edit or
> revert another agent's files.

## Architecture invariants: protected contracts

- Discovery, exclusions, scope isolation, supported formats, extraction, chunk
  content/order/overlap, source-qualified IDs, metadata and citations.
- Keyword-only and configured dense retrieval, fusion, reranking and contextual
  retrieval, including disabled paths; provider/model selection, dimensions,
  request payloads, result limits and established fallback behavior.
- Persistence/reopen, same-named files, cache reuse and invalidation after source
  or setting changes; empty, malformed and Unicode/Persian inputs.
- MCP/CLI/HTTP contracts, configuration precedence, optional/lazy imports, light
  installation and supported storage backends. Lack of test coverage does not
  authorize deleting a supported backend or feature.
- Concurrency bounds, batching, device placement, OOM recovery and resource cleanup.
  No accidental serialization, unbounded parallelism or increased API requests.

## Non-Goals (Out of Scope)

HTML markup and notebook container metadata are not inherently useful retrieval
content. However changing filtering, code/output inclusion, parsers or chunking is
a separate behavior change, not part of this refactor. No new AST/LaTeX/HTML parser,
model change, database migration or index-schema change. The README's AST-chunking
claim currently overstates implementation; do not treat it as an existing feature.
Reproduce and report discovered bugs separately; do not silently change contracts
while claiming a behavior-preserving refactor.

## Plan: approach, trade-offs and verification

Prefer deletion of proven redundancy over extraction-only churn. Extract cohesive
responsibilities when it materially improves comprehension, accepting that useful
compatibility wrappers can temporarily retain lines. Prefer deterministic local
contract checks over paid end-to-end calls for the mechanical refactor, while
explicitly retaining the limitation that doubles cannot prove hardware performance.

Build a behavior-to-test matrix for touched code: input, observable contract,
baseline result, candidate result and limitations. Reuse existing tests first.

1. Characterization tests: exact extraction/chunks/IDs/locations using small text,
   Markdown, generated PDF, JSON/notebook and mocked optional-parser fixtures.
   Include boundaries, repeated passages, Unicode, empty and malformed input.
2. Unit/property checks where relevant: source fidelity, chunk limits, stable IDs,
   fusion/deduplication, configuration precedence, batch/OOM transitions. New
   property-testing dependencies are not required.
3. Integration: real local storage search/reopen/refresh, deterministic embedding
   and reranker doubles, MCP stdio and CLI contract smokes. Mocks do not establish
   live API, Qdrant service or GPU correctness; label unavailable paths unverified.
4. Test sensitivity: temporarily introduce a relevant fault in an isolated copy
   (drop metadata, bypass dense search, break invalidation, serialize extraction).
   The protecting test must fail for the intended reason. Restore each mutation
   immediately. Surviving relevant mutations expose gaps to fix before deletion.
   Target changed behavior; no mandatory exhaustive mutation campaign.
5. Focused regression after each slice; combined relevant suite after integration;
   import/install/entry-point checks proportional to the changed surface. Inspect
   test collection before running scripts that might download, charge or stress GPUs.

TDD here means establishing contracts and sensitive tests first, not inventing
features to obtain a red test. Coverage percentages/test counts alone are inadequate.
Preserve existing tests; replace implementation-coupled tests only with documented
equivalent or stronger behavioral checks reviewed by the coordinator. Do not
invent tolerances after observing a regression.

## Performance and hardware guard

Use experiment-workflow for repeatable performance/system evaluations. Consult
Masih before material experiment choices: question, workloads/data, resource/API
cost, repetitions and acceptable noise/regression thresholds. Ordinary focused
local unit/contract tests do not need separate experimental approval.

Proposed question: does the refactor preserve output and resource efficiency on
the same workload? Hold data, dependencies, model/provider, device and batch limits
fixed. Separate cold ingestion, warm-cache query and refresh. Retain raw timings,
throughput, peak memory where available, call counts and effective concurrency;
separate warm-up/model loading. Run baseline/candidate without competing workers.

Use event/barrier-based tests for in-flight bounds and accidental serialization,
not flaky sleep-based speed assertions. These prove scheduling contracts, not GPU
throughput. Hardware performance remains unverified until measured. Optimize only
measured waste, one mechanism per comparison. Defer unsafe/unverifiable changes;
do not infer speed-ups from line counts or unrelated test-suite runtimes.

## Atomic tasks and dependency order

- [x] T1: Inventory the current working baseline. Done when every proposed touched
  file has recorded status and an exclusive owner.
- [x] T2 (after T1): Create the recovery checkpoint. Done when the restore hash
  comparison passes.
- [x] T3 (after T2): Capture baseline verification. Done when selected commands have
  saved raw outcomes and an explicit failure/skip inventory.
- [x] T4 (after T3, per responsibility): Fill the protecting-test gap. Done when
  the relevant intentional mutation is caught for the expected reason.
- [x] T5 (after T4, per slice): Delete/simplify one cohesive piece. Done when its
  focused contract checks pass unchanged.
- [x] T6 (after each T5): Independently review the slice. Done when its diff has
  no unexplained contract change. Repeat T4–T6 for subsequent bounded slices.
- [x] T7 (after approved slices): Verify combined integration. Done when applicable
  regression gates pass against the integrated candidate.
- [ ] T8 (before accepting performance changes): Run the separately agreed fixed-
  workload comparison. Done when results satisfy its predeclared thresholds.
  Not exercised: no hardware optimization or throughput claim in this pass.
- [x] T9: Prepare handoff. Done when the acceptance checklist is evidence-linked
  and any unchecked item is explicitly reported to Masih.

Agents refine each repeated slice into a named change and binary done-check before
editing. If a task cannot be reviewed in one session, split it. Record new facts in
this spec/plan before proceeding when they change the contract. Human review is
the final integration boundary; PR creation is not authorized by this document.

## Verify, integrate and learn: handoff

Accept each slice only with protected behavior passing, demonstrated test
sensitivity, independent diff review and required performance checks within the
pre-agreed bounds. Unresolved failures or unavailable mandatory checks block that
slice's acceptance. Passing tests do not prove universal absence of regressions.

Report checkpoint/recovery details, changed files, removed versus moved logic,
before/after core.py and fetch.py counts, all production Python counts including
new modules (tests counted separately), exact commands/raw evidence locations,
performance results or explicit gaps, and remaining risks. Explain the resulting
architecture, why remaining parts are necessary, and how failures are diagnosed.
No perfection or production-readiness claim from this refactor alone.

Stop when further deletion would remove required behavior or its necessity cannot
be safely resolved; ask about product decisions instead. No deletion quota and no
removing useful comments or splitting files merely to improve counts. Keep coherent
local checkpoints, stage named reviewed files/hunks only, never git add -A. Use
existing ES.md and one scoped evidence location, not new workspace task ledgers.
Coordinator reviews final integration and hands results to Masih; no auto-deployment.
At handoff, record which assumptions the evidence changed and any follow-on work
in ES.md; update this contract if needed rather than inventing new standing rules.

## First-pass evidence — 2026-09-12

Checkpoint 7054026 restored to /tmp/rag-refactor-baseline-UrN3DO: original selection
88 passed. New search/fetch/fusion contracts passed before their corresponding
edits (4/20/10 cases). Relevant in-memory mutations produced expected assertion
failures for lost record identity, reversed filename ranks, missing invalidation,
missing PDF support and changed duplicate-ID rank semantics.

Reviewed candidate: core.py 924→912; fetch.py 407→394; production Python total
5,104→5,079. No relocation or new production module; 211 test lines added.
Removed unused search computation, consolidated duplicate fusion loops, shared
fetch extensions and identical scoped setup. Original empty-query fast path kept.

Final command from repository root, outside sandbox for local integration:

```sh
.venv/bin/python -m pytest tests/unit tests/test_cli_audit.py tests/test_cli.py tests/test_companion.py tests/test_settings.py tests/test_model_catalog.py tests/test_prompts.py tests/test_bootstrap.py tests/integration/test_source_citations.py tests/integration/test_document_refresh.py tests/integration/test_lancedb_backend.py tests/integration/test_rag_scope.py tests/integration/test_rag_mcp_host.py -q --tb=short
```

144 passed, 38 existing LanceDB warnings. This includes 34 new contract cases and
22 additional catalog/prompt/bootstrap checks beyond the baseline selection.
Raw command output is in the session tool transcript. Timings are not a performance
comparison; test populations differ. No hardware throughput, live Qdrant or paid
API validation claimed. Concurrency/device/batch code unchanged. Broader structural
refactoring remains incomplete; parser separation requires its own characterization.

## Continued pass: evidence and deletion boundary

Starting checkpoint d36d355. Two workers and coordinator reviewed core, parsing,
discovery, provider/storage adapters, settings and MCP callers. No concurrent
edits to the same file; the coordinator reviewed worker diffs and Halley separately
reviewed coordinator settings/server changes. Original experiment edits untouched.

Final verification: the same combined command above now reports **215 passed,
38 existing LanceDB deprecation warnings**. No reported test failures remain.
New contracts were established before each production slice, with targeted
in-memory mutation failures for BM25 document frequency, notebook separators,
uppercase discovery, contextual form-feed cleanup, cache-setting invalidation,
metadata precedence, parallel fallback, setting order/types and context deduplication.
Fresh-process MCP startup proves no eager core/fetch/torch/LanceDB import. Raw
outputs are in the session tool transcript. No paid provider or live user data used.

| Production scope | Original checkpoint 7054026 | Final candidate |
| --- | ---: | ---: |
| core.py | 924 | 787 |
| fetch.py | 407 | 377 |
| settings.py | 264 | 243 |
| server.py | 463 | 447 |
| lancedb_backend.py | 205 | 201 |
| qdrant_backend.py | 202 | 188 |
| New parsing.py / formats.py | 0 | 118 / 9 |
| All rag/*.py plus server.py | 5,104 | 5,009 |

Total net production reduction is **95 lines**, including new modules. Roughly
110 parser-body lines were relocated, not deleted; direct method aliases preserve
self-dispatch and overrides without inheritance. This continuation adds 582 net
test lines (545 in new files, 37 in existing files), separate from production counts.
The preceding pass added 211 test lines. Source-code reduction is modest compared
with the reduction in core.py alone; it must not be presented otherwise.

Removed work includes duplicate term tracking, unused normalization and backend
metric work, redundant sorting/payload reconstruction, repeated configuration
merging/dispatch, repeated format declarations and an unnecessary context list.
Both backends' two-worker fallback is protected by barrier-based tests, including
a deliberately serialized mutation that failed. Existing device/OOM tests pass;
actual GPU throughput and remote-provider equivalence remain unmeasured.

The audited stopping boundary is concrete:

- Chunk validation, source text/cleaned text and location tracking preserve public
  chunk entry points, citations and contextual retrieval; deleting them changes evidence.
- Ordered ingestion state coalesces duplicate paths and preserves extraction order.
  Streaming it is a redesign, not a harmless deletion.
- Persistence, refresh, exclusions and scan limits protect stale/missing-file and
  scope behavior. Removing them loses tested functionality or safety.
- Provider/device/reranker branches, retry/fallback behavior and Qwen padding/cleanup
  implement distinct supported modes. No mode was retired to meet a line count.
- Backend compatibility and different exception contracts prevent blindly sharing
  all embedding logic; changing write strategy can change partial-failure behavior.
- Notebook metadata and exercise priority are exported functionality even though
  the core parser consumes primarily text. HTML/LaTeX filtering is still out of scope.
- CLI/companion validation, authentication, process/error handling and entry-point
  compatibility remain product behavior; they were inspected, not stripped.

No further demonstrated deletion in this audit meets the preservation contract.
This is a defensible tested stopping point, not proof that future refactoring is
impossible. A fresh wheel build was not run: build/hatchling tooling is absent in
the active environment, and dependencies were not installed for this pass. Existing
bootstrap/CLI tests and import checks pass. No deployment, restart or push performed.
