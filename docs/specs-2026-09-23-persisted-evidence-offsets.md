# Persisted passage offsets in saved evidence

## Goal

When a newly indexed passage occurs more than once on a source line, a paired
Studio run should open the selected occurrence, not leave it unlinked or open
another occurrence. CLI, MCP, and Studio use the same retrieval storage.

## Contract

- LanceDB persists the Python source offsets already emitted by ingestion and
  returns them with lexical and dense hits after reopening an index.
- The saved run checks the offsets against the current workspace source text
  before using them, then converts them to UTF-16 offsets for browser navigation.
- A passage with missing or invalid offsets keeps the existing conservative
  line-search fallback; ambiguous repeated text stays unlinked.
- Legacy rows can be copied during an incremental refresh without fabricating
  a location. This change does not rebuild existing indexes or load a model.

## Verification

Use a tiny BM25-only index to check persisted search results before and after
reopening. Use a repeated Unicode fixture to check the saved run points at the
second occurrence. Check invalid and absent offsets, then run adjacent offline
retrieval tests. Do not index the user's large book to validate this contract.
