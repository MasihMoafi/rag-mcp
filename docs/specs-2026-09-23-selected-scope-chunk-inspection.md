# Inspect published chunks for the selected query scope

## Problem

Studio, CLI, and MCP can query a selected subset of workspace sources, which
publishes an index under that exact source-set identity. Published Chunking
inspection currently accepts only a workspace and one source. It cannot name
the subset, so it can claim `not_indexed` immediately after a successful
subset query.

## Contract

- Indexed inspection accepts an optional ordered list of selected source IDs.
  Omission means every saved source; an explicit list must be nonempty,
  unique, available in the requested revision, and include the inspected
  source. Preview inspection rejects a scope option because it has no index.
- The engine derives the same selected-file scope and source stamp as a query
  for that set. The response echoes the canonical source IDs, so callers can
  reject a page from an older selection. Legacy root-scope lookup remains
  available only for the all-source selection.
- CLI `rag chunks --view indexed --scope-source ID`, MCP
  `get_workspace_chunks(scope_source_ids)`, and repeated gateway
  `scope_source_id` parameters use the same reader. They continue to omit
  private chunk text unless explicitly requested.
- Paired Studio's Published chunks view follows the currently included
  sources. If the inspected source is excluded, show that state rather than
  implying an index is missing. Selection changes invalidate an old page.
- Inspection stays read-only: no model load, index rebuild, directory creation,
  or provider call. Stale revision and invalid scope fail before opening index
  metadata.

## Verification

Build an isolated two-source index by querying only one source; prove the
all-source view is absent while the explicit subset view is current through
the engine reader, CLI, MCP, and HTTP. Confirm invalid/excluded selections
fail and default pages remain metadata-only. Cover the browser parser and
selection change, then render the paired Chunking view at least once.

## Result — 2026-09-23

Implemented and checked with a two-source BM25 fixture. Six focused Python
tests and three browser transport tests passed; CLI, MCP, and HTTP returned the
same selected page. Preview-scope requests were rejected. The cached-offline
wheel served a paired English Chunking view where the one-source selection was
current with one exact chunk, while all sources were not indexed. Excluding
the inspected source produced an explicit explanation. The Persian route was
also rendered and visually checked. Evidence and limits are in
`audits/2026-09-23-selected-scope-chunk-inspection.md`.
