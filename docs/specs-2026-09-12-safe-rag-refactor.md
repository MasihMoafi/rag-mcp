# Safe deletion-first RAG refactor

Status: specification revised to the spec skill, 2026-09-12. Read-only preflight
delegation authorized; implementation remains gated on verified recovery and tests.
Specification, plan and atomic tasks are kept together to avoid workspace clutter.

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

Observed baseline: HEAD ea8e67c; core.py 924 lines, fetch.py 407; rag/*.py plus
server.py 5,104 physical lines. Substantial tracked and untracked changes exist.
Re-measure before implementation: HEAD alone is NOT the working baseline. This
specification does not claim a recovery checkpoint, new test run or safe refactor.

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
providers without the relevant permission. private downstream remains internal-only and is
not needed for this mechanical refactor. No Studio or private-eval worktree changes.

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

- [ ] T1: Inventory the current working baseline. Done when every proposed touched
  file has recorded status and an exclusive owner.
- [ ] T2 (after T1): Create the recovery checkpoint. Done when the restore hash
  comparison passes.
- [ ] T3 (after T2): Capture baseline verification. Done when selected commands have
  saved raw outcomes and an explicit failure/skip inventory.
- [ ] T4 (after T3, per responsibility): Fill the protecting-test gap. Done when
  the relevant intentional mutation is caught for the expected reason.
- [ ] T5 (after T4, per slice): Delete/simplify one cohesive piece. Done when its
  focused contract checks pass unchanged.
- [ ] T6 (after each T5): Independently review the slice. Done when its diff has
  no unexplained contract change. Repeat T4–T6 for subsequent bounded slices.
- [ ] T7 (after approved slices): Verify combined integration. Done when applicable
  regression gates pass against the integrated candidate.
- [ ] T8 (before accepting performance changes): Run the separately agreed fixed-
  workload comparison. Done when results satisfy its predeclared thresholds.
- [ ] T9: Prepare handoff. Done when the acceptance checklist is evidence-linked
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
