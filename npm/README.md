# ragmcp-cli (npm launcher)

Local hybrid search for your documents and codebases: exact-word BM25 is always on, and you choose the embedding model for meaning-based search (a small local download, Ollama, or your own API key). Also an MCP server for Claude Code, Codex and other MCP clients.

This npm package is a thin launcher. The engine is the Python package of the same name, run through [`uv`](https://docs.astral.sh/uv/getting-started/installation/), so `uv` must be installed once.

Exact-word retrieval works without a key. Answers send the question and selected
evidence to OpenRouter and need `rag key set` or `OPENROUTER_API_KEY`. When a key
is available, hosted reranking sends the query and candidate passages to OpenRouter
by default; `rag config reranker off` disables it. Other hosted embedding and
contextual providers receive the text they process. See the full documentation
for offline settings and the MCP trust boundary. Studio is not included in this release.

```bash
npx ragmcp-cli setup                       # choose an embedding model (or exact words only)
npx ragmcp-cli query "What changed?" --path ./docs
npx ragmcp-cli doctor
```

`pnpm dlx ragmcp-cli ...` and a global `npm install -g ragmcp-cli` (then `rag ...`) work the same way.

Environment: `RAGMCP_EXTRAS=full,office,qdrant` adds optional extras to the Python install. Full documentation: https://github.com/MasihMoafi/rag-mcp
