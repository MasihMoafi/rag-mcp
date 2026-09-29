# Fusion score trace

## Goal

The saved `retrieval.rrf` span must show the score that produced its ranking,
not the dense or lexical score retained on the source document. This makes
paired Studio's run inspector and `rag runs` describe the same real operation.

## Contract

- Keep the returned passage objects, order, and public query result unchanged.
- Record actual weighted reciprocal-rank values for the bounded fused candidates
  under `retrieval.rrf`, labeled as RRF scores. An empty fusion records none.
- Preserve dense and lexical candidates' own scores in their respective spans.
- Keep normal traces free of source text, paths, queries, and credentials.
- A disabled trace has no effect on retrieval. A trace write failure does not
  turn a successful retrieval into a failure.
- The CLI reader and Studio inspector retain and label the RRF score without
  treating it as a probability or comparing it with source-stage scores.

## Verification

1. A controlled two-arm fusion test proves the exact RRF calculation and
   verifies source dictionary identity and no mutation.
2. A saved metadata trace proves the fusion span carries those same values
   and no private source content, while earlier stage scores remain intact.
3. Focused engine tests, web parser/typecheck/lint, and one real-book
   retrieval trace pass with bounded CPU and no provider calls.

## Boundary

Reranker score provenance and a unified score schema for public answer evidence
remain separate work. No change to retrieval ranking, provider selection,
generation, or hosted telemetry.

## Verified 2026-09-23

- Controlled fusion identity and RRF arithmetic, public trace filtering, and
  real LanceDB trace: 21 focused tests passed.
- Adjacent fusion, trace, API retrieval, and companion tests: 41 passed.
- A real nine-chapter book retrieval emitted RRF scores 0.016393, 0.016129,
  0.015873 for the top three candidates while the lexical top score was 14.18.
  `rag runs` and the rendered Studio inspector exposed the same values; the
  Studio table visibly labeled them “RRF score.” No embedding or provider call.
- The broader 770-test suite produced 769 passes and one conversation gateway
  failure. A second `-x` run stopped at an API mock failure after one pass.
  Both failing tests passed in isolation and within the 41-test adjacent set;
  the broad-suite instability remains unresolved. No full-suite green claim.
