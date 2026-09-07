# ragmcp (npm launcher)

Local hybrid search for your documents and codebases: exact-word BM25 is always on, and you choose the embedding model for meaning-based search (a small local download, Ollama, or your own API key). Also an MCP server for Claude Code, Codex and other MCP clients.

This npm package is a thin launcher. The engine is the Python package of the same name, run through [`uv`](https://docs.astral.sh/uv/getting-started/installation/), so `uv` must be installed once.

```bash
npx ragmcp setup                       # choose an embedding model (or exact words only)
npx ragmcp query "What changed?" --path ./docs
npx ragmcp doctor
```

`pnpm dlx ragmcp ...` and a global `npm install -g ragmcp` (then `rag ...`) work the same way.

Environment: `RAGMCP_EXTRAS=full,ocr,office,qdrant` adds optional extras to the Python install. Full documentation: https://github.com/MasihMoafi/rag-mcp
