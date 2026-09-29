# Lightweight chunk inspection

## Problem

A cold chunk-preview HTTP request imports `rag.fetch` for two default values.
That imports `rag.core` and PyTorch before returning source metadata; a
two-second loopback test times out. Published-index inspection has the same
import dependency even when no index exists.

## Contract

- Preview and metadata-only published inspection use the same index defaults
  and scope/stamp calculations as actual retrieval.
- Neither cold read imports `rag.fetch`, `rag.core`, or PyTorch, constructs a
  pipeline, opens a model, or builds an index.
- Used `rag.fetch` helper names remain available to callers and tests.

## Verification

Run both reads in a fresh subprocess and check module imports. The existing
loopback preview request must finish within its current timeout. Run focused
chunk and index-inspection regressions; no model or provider calls.
