# Honest dense retrieval scores in run inspection

## Goal

The paired run inspector must show the score the engine recorded for its
top-ranked dense result, including negative scores. It must not label every
backend and distance metric as cosine similarity.

## Contract

- Each dense retrieval span records its backend and configured distance metric,
  including when dense retrieval is skipped. The metadata reader exposes only
  those allowlisted values, not queries, paths, or document text.
- Studio uses rank 1's finite score, not the numerically largest score or a
  nonnegative-only filter. When no dense candidates exist, the headline remains
  unreported.
- The UI calls the value a metric-dependent engine score and displays the
  backend/metric provenance. It never compares it directly with lexical, RRF,
  or reranker scores.

## Verification

Use a real, simulated-provider dense fixture to inspect persisted metadata
before and after reopening. Use a negative-score web fixture where rank 1 is
not the numeric maximum, plus a lexical-only fixture. Run adjacent privacy,
trace, and browser checks without contacting a model provider.
