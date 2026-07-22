# rag-mcp

Standalone MCP server exposing local hybrid RAG (BM25 + vector search + Qdrant + reranking)
as a single tool, `query_knowledge_base`. Forked out of Elpis's internal `elpis-rag` sidecar
so any MCP-compatible client can use it against any workspace, not just Elpis.

Fully local: sentence-transformers embeddings, Qdrant (embedded, on-disk), CrossEncoder
reranking. No API keys required.

## Setup

```bash
cd rag-mcp
uv sync        # or: pip install -e .
```

## Add to an MCP client

The server speaks stdio JSON-RPC (MCP). Point any client at:

- command: `/absolute/path/to/rag-mcp/.venv/bin/python`
- args: `["/absolute/path/to/rag-mcp/server.py"]`
- env: `RAG_MCP_WORKSPACE_ROOT` = the directory to search by default (optional; falls back to cwd)

### Claude Code / Claude Desktop (`.mcp.json` or `claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "rag": {
      "command": "/absolute/path/to/rag-mcp/.venv/bin/python",
      "args": ["/absolute/path/to/rag-mcp/server.py"],
      "env": { "RAG_MCP_WORKSPACE_ROOT": "/absolute/path/to/your/project" }
    }
  }
}
```

Or via CLI: `claude mcp add rag /absolute/path/to/rag-mcp/.venv/bin/python /absolute/path/to/rag-mcp/server.py`

### Codex / Elpis (`~/.codex/config.toml`)

```toml
[mcp_servers.rag]
command = "/absolute/path/to/rag-mcp/.venv/bin/python"
args = ["/absolute/path/to/rag-mcp/server.py"]

[mcp_servers.rag.env]
RAG_MCP_WORKSPACE_ROOT = "/absolute/path/to/your/project"
```

## Tool

`query_knowledge_base(query: str, doc_path: str = "")` — runs hybrid search (RRF fusion of
BM25 + vector) with CrossEncoder reranking over `doc_path`, or `RAG_MCP_WORKSPACE_ROOT` /
cwd if `doc_path` is omitted. Read-only; indexes are cached and persisted under
`rag/rag_db_v2/` per scanned path.

Scan limits (`RAG_MCP_MAX_DEPTH`, default 20; `RAG_MCP_MAX_TOKENS`, default 2,000,000) guard
against accidentally indexing huge trees. `node_modules`, `.git`, `.venv`, `dist`, `build`,
`target` are always skipped.

## Layout

```
rag-mcp/
├── server.py       # MCP stdio host (JSON-RPC, no SDK dependency)
├── rag/            # hybrid RAG engine: chunking, BM25, Qdrant vector search, reranking
└── utils/proxy.py  # strips SOCKS proxy env vars so local Ollama/Qdrant stay reachable
```
