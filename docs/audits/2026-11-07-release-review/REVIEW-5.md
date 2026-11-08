## Verdict

**No — the CLI and MCP server are not ready to publish at `399db5a`.** The release blockers are ignored files entering scope without Git, credential-bearing provider errors reaching MCP, unrelated query rewriting, missing source references, and a failing test in the release gate.

Engineering is mixed against the sampled DeepSeek harness. rag-mcp now gates publication on quality checks and an installed-artifact smoke; the reference also gates release validation on installed SDK and MCP scenarios across Python versions. rag-mcp’s query rewriting and answer provenance still have user-visible failures. [rag-mcp release workflow](../../../.github/workflows/release.yml:55), [reference release workflow](deepseek-harness/.github/workflows/python-release.yml:33).

## Release-blocking findings

1. **RELEASE-BLOCKING — Ignored files enter scope when Git is absent.** The no-Git fallback treats every name as unignored, and folder scans then index non-hidden files covered by `.gitignore`. A configured hosted embedding provider can receive their contents. Preserve ignore rules without Git or fail closed when they cannot be checked. [Ignore fallback](../../../rag/server.py:824), [scan](../../../rag/server.py:1016).

2. **RELEASE-BLOCKING — Provider errors can expose credentials to MCP.** `set_rag_settings` returns the embedding probe’s exception text verbatim. An in-memory provider response that echoed its authorization header caused the MCP response to contain the fake bearer token. Sanitize provider errors before returning them. [Probe error](../../../rag/settings.py:425), [MCP settings path](../../../rag/server.py:677).

3. **RELEASE-BLOCKING — Ordinary questions become unrelated searches.** A two-part question asking how to assess solar-panel efficiency and recommend an installation plan is rewritten into three RAG-evaluation queries; two omit solar panels entirely. Both CLI and MCP use those facets for retrieval. Keep the user’s subjects and requested actions in every facet. [Rewrite](../../../rag/query_decomposition.py:39), [retrieval](../../../rag/server.py:1112).

4. **RELEASE-BLOCKING — Requested passages lose their references.** For CLI and MCP, passage rendering drops the available filename and line range and instead reports an unavailable PDF page, even for Markdown. Workspace answer evidence also uses the stored hash filename rather than the imported source name. Carry source identity and location through synthesis and rendering. [Renderer](../../../rag/passage_answers.py:109), [client citation setting](../../../rag/execution.py:241), [workspace filename](../../../rag/workspaces.py:41).

5. **RELEASE-BLOCKING — The requested commit’s release gate contains a failing test.** At `399db5a`, `tests/integration/test_document_refresh.py:97` matches the original bad-file path, but ingestion calls the parser with a temporary snapshot path. The invalid bytes therefore reach the real decoder before the test’s first assertion. The release workflow requires the suite containing this test. [Snapshot](../../../rag/ingestion.py:105), [quality gate](../../../.github/workflows/python-quality.yml:40).

## LATER

1. **LATER — Loopback API keys are ignored.** An explicit key for an authenticated localhost embedding service is replaced with `local`, so setup can fail authentication. Use the supplied key when present. [Embedder](../../../rag/embeddings.py:331).

2. **LATER — A named pipe can stall a scan.** Folder discovery accepts a supported-extension FIFO, then opens it before checking whether it is a regular file. Skip nonregular files and use a nonblocking open before checking the descriptor. [Discovery](../../../rag/server.py:991), [open](../../../rag/workspace_import.py:38).

3. **LATER — Re-import does not remove old workspace sources.** Removing or newly ignoring a file and adding its folder again leaves the previously imported copy searchable. Add explicit removal or synchronization, and describe imports as snapshots. [Import merge](../../../rag/workspace_import.py:120), [CLI actions](../../../rag/cli.py:311).

4. **LATER — The release smoke does not exercise a successful answer.** It checks a keyless answer failure and mocked embedding retrieval. Add a successful mocked answer through the installed CLI and MCP entry points. [Artifact smoke](../../../tests/install/smoke.sh:34).

The branch advanced to `c5df59d` during review; its only change from `399db5a` is the test in finding 5. The other findings remain anchored to `399db5a`. I made no file changes. The test suite, CI outcomes, registry installations, live providers, real MCP hosts, and other operating systems were **not checked**.