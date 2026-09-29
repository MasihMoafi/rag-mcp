# Reranker score provenance

## Goal

When a reranker orders passages, saved `selection.rerank` rows and paired Studio must show the actual values used for that ordering. A source-stage score retained on the passage must not masquerade as the reranker score.

## Contract

- Keep public answer text, returned passage object identity and order, original `score` fields, and provider calls unchanged.
- A successful rerank records selected passage IDs, ranks, and its actual finite model scores under `selection.rerank`, with `score_kind: reranker`.
- If reranking is skipped for too few candidates or a local reranker fails and preserves fusion order, mark the span skipped and label any retained source-stage scores as such. Hosted reranker failures continue to propagate.
- Non-reranker selection and synthesis selected rows remain labeled as retained source-stage scores; they are not claimed to be the values that determined final rank.
- Normal public traces retain opaque IDs and scores without passage text, paths, queries, or credentials. Studio and CLI read the same selected ranks and score kind. No score is called a probability or compared across models or stages.
- Tracing disabled, write failures, and no-reranker execution preserve retrieval behavior.

## Verification

1. A controlled model-score inversion proves the selected trace follows reranker scores while returned passage objects and their original scores are unchanged.
2. A simulated hosted rerank and local failure prove successful versus skipped provenance in saved and public traces, with privacy markers absent.
3. Focused engine and web checks pass; a rendered paired Studio run labels successful reranker scores and fallback scores accurately, if a bounded browser fixture is available.

## Boundary

The engine's private result evidence retains its source-stage `score` for compatibility. This change does not implement a universal cross-stage score schema, judge answer support, or test live provider quality.

## Verified 2026-09-23

- The original simulated hosted-provider test and a controlled score-inversion test failed before the change because `score_kind` was absent and selected rows reused older source scores.
- After the change, 38 focused reranker/trace checks and 24 adjacent retrieval, citation, and execution checks passed. The inversion includes NumPy `float32` model output, unchanged passage objects, and private-content exclusion. A local failure records `skipped` with `retained_source`; simulated hosted success records `reranker` in both raw and public traces.
- One real BM25-only book query reused the 884-chunk index, still found its exact expected passage at rank four, and recorded `retained_source` on its selection span. No provider call or new index build occurred.
- Paired Studio displayed the same simulated trace as CLI: reranker selection 1.0000 versus synthesis's retained retrieval score 0.2877, with distinct English labels. The Persian inspector visibly labeled the reranker score. Temporary browser and companion processes were stopped afterward.
- A bounded web typecheck, targeted lint, and ten engine-run/evidence-view tests passed. This does not verify live reranker quality, a full-suite regression run, or clean-machine installation.
- The bounded Next production build and offline wheel/sdist build passed. The wheel contains the actual reranker trace code and the staged Studio bundle byte-for-byte. Bundle build ID `QNemT-HQOw2HtWDEqJW_J`, SHA-256 `c3eceb9c73b78b971d0a5da32b51a31d34d271dd2162f50ec67e868cdf02766e`. The new English labels, earlier lexical-search warning, and corrected citation labels are present in its JavaScript assets. No package was installed or published.
