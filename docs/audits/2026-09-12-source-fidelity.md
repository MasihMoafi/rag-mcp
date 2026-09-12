# Source-fidelity repair — 2026-09-12

## Findings and changes

- Reproduced five failures before patching: non-contiguous text after whitespace normalization, headings ignored after a Markdown preamble, partial words from chunking/overlap, and missing persisted file/citation fields. A separate invalid-size test exposed a non-progressing loop (bounded test command timed out).
- Chunking now uses contiguous source slices with paragraph/sentence/word boundaries where possible. Overlap does not manufacture text or require partial short paragraphs. Unbroken tokens larger than the character budget still require hard cuts. Chunk size defaults remain unchanged.
- Markdown heading detection works after a preamble. Repeated passages advance the location cursor.
- Ingestion emits file path, readable location, line range and starting PDF page; LanceDB preserves them. Page numbers refer to the start of a chunk, not necessarily every page it spans.
- Source manifests now include chunk settings and a new schema version so old indexes and changed chunk settings trigger rebuilding through the normal fetch entry point.

## Verification

Final command (outside the sandbox because LanceDB tests timed out inside it):

```sh
.venv/bin/python -m pytest tests/unit tests/test_cli_audit.py tests/test_cli.py tests/test_companion.py tests/test_settings.py tests/integration/test_source_citations.py tests/integration/test_document_refresh.py tests/integration/test_lancedb_backend.py tests/integration/test_rag_scope.py tests/integration/test_rag_mcp_host.py -q --tb=short
```

Result: **87 passed**, 33 existing LanceDB deprecation warnings. Compile and diff-whitespace checks passed.

Book smoke check: `/home/masih/Desktop/books/rag/2. The Base RAG Stack _ Hands-On RAG for Production.md`, isolated temporary BM25-only index, contextualization and reranking disabled. All 143 chunks were exact source slices; five returned results had real source line ranges. The first result began `Recursive chunking` at L484, rather than a sliced word. This is source-fidelity evidence, not a relevance benchmark. Temporary indexes were cleaned up; the user's configured model and active index were not modified.

## Scope and remaining limits

private downstream's downstream record loader supplies canonical record IDs, citations, article metadata and filters; its article-oriented evidence boundary is different from generic file chunking. No changes were made to either private downstream checkout or Studio. Existing user edits were retained.

No paid/model-based quality evaluation was run, and earlier benchmark scores have not been re-established with the changed chunks. No table reconstruction, adaptive model selection, or general core refactor was attempted. The running MCP process must restart to load the repair; its next selected-scope query rebuilds a stale index (potential embedding cost with an API model). A separately installed package will need the updated code too.

Commit verification: exported the staged snapshot to a separate temporary directory and ran the source-fidelity, chunking, citation integration and LanceDB tests: **21 passed**. The commit includes the necessary short-document and source-cache behavior; unrelated contextual retrieval, CLI/provider, and evaluation edits remain outside it. The 87-test result above describes the full working tree, not the isolated commit.
