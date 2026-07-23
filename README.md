---
name: rag-mcp
type: local semantic-search MCP server for coding agents and document workflows
---

# rag-mcp

**A coding agent should not have to choose between opening files one at a time and dumping an entire repository into context.**

`rag-mcp` is a local hybrid-search MCP server: point it at a file or folder, ask a question, and get back ranked chunks with exact source paths. Embeddings, vector search, and reranking run locally; the server exposes one read-only MCP tool to compatible clients.

## Quick start

```bash
git clone https://github.com/MasihMoafi/rag-mcp
cd rag-mcp
uv sync
```

Run the automated tests:

```bash
.venv/bin/python -m pytest tests/ -v
```

Then register the server with an MCP client.

### Claude Code

```bash
claude mcp add rag -s user -- /absolute/path/to/rag-mcp/.venv/bin/python /absolute/path/to/rag-mcp/server.py
```

### Codex / Elpis

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.rag]
command = "/absolute/path/to/rag-mcp/.venv/bin/python"
args = ["/absolute/path/to/rag-mcp/server.py"]

[mcp_servers.rag.env]
RAG_MCP_WORKSPACE_ROOT = "/absolute/path/to/your/project"
```

Expected result: the client discovers `query_knowledge_base`, and a query returns ranked passages with source paths from the requested scope.

## The problem

Coding agents commonly retrieve context by either opening files one by one or loading a large portion of the repository. The first can miss relevant files; the second consumes context with material the current task may not need.

`rag-mcp` moves retrieval into one local tool call so the agent can search by meaning without making the entire tree part of every prompt.

## How it works

```text
query + optional path
        ↓
chunking
        ↓
BM25 lexical search + local embeddings / Qdrant
        ↓
Reciprocal Rank Fusion
        ↓
CrossEncoder reranking
        ↓
ranked chunks + exact source paths
```

Repository structure:

```text
rag-mcp/
├── server.py       # stdio JSON-RPC MCP host
├── rag/            # chunking, BM25, vector search, reranking
└── utils/proxy.py  # local proxy-environment handling
```

Technical boundaries:

- one MCP tool: `query_knowledge_base(query, doc_path?)`;
- default embeddings: `all-MiniLM-L6-v2`;
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`;
- local embedded/on-disk Qdrant;
- `doc_path` can scope each call to a file or directory;
- per-path indexes are persisted under `rag/rag_db_v2/`;
- common large/build directories such as `.git`, `node_modules`, `.venv`, `dist`, `build`, and `target` are rejected;
- configurable depth/token limits fail explicitly instead of scanning an unbounded tree.

## Current state

### Implemented and verified

- MCP `initialize` → `tools/list` → `tools/call` protocol path.
- Read-only `query_knowledge_base` tool.
- Workspace-root and explicit `doc_path` scoping.
- Local hybrid retrieval and reranking.
- Guardrails for excluded directories and oversized scopes.
- End-to-end registration was exercised through a real MCP client during development.

### Implemented but not yet covered by the current tests

- The alternative Ollama embedding-provider path in `rag/core.py`.

### Planned

Nothing is formally tracked yet. Extend it when a concrete retrieval failure or client requirement appears.

### Intentionally unsupported

- Hosted/remote vector databases.
- File types outside the extension allowlist in `server.py`.
- Write/mutation tools; this server is retrieval-only.

## What sets this apart

These are design choices, not novelty claims:

- **Local retrieval:** source files, embeddings, vector search, and reranking stay on the machine.
- **Small transport layer:** the MCP host uses direct stdio JSON-RPC rather than depending on an MCP SDK.
- **Per-call scope:** one server can search different files/directories instead of requiring one fixed knowledge base per project.
- **Evidence in the response:** returned chunks include source paths rather than only synthesized prose.

## Evals and test series

Five lightweight tests live under `tests/`:

```bash
uv sync --group dev
.venv/bin/python -m pytest tests/ -v
```

They cover:

- read-only tool annotations;
- default workspace scoping;
- explicit `doc_path` scoping;
- rejection of excluded directories;
- rejection of depth-limit violations.

Protocol-level check, without another MCP client:

```bash
printf '%s\n%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"how does reciprocal rank fusion combine bm25 and vector results"}}}' \
  | RAG_MCP_WORKSPACE_ROOT="$PWD" .venv/bin/python server.py
```

A successful self-query should return evidence pointing at the RRF implementation in `rag/core.py`.

What the tests prove: MCP transport/scoping/guardrail behavior covered by those cases.

What they do **not** prove: retrieval quality across arbitrary corpora, cross-client compatibility, or superiority to grep/code-search/RAG alternatives.

## Example

```text
query_knowledge_base(
  "how does retry backoff work for failed jobs",
  doc_path="codex-rs/memories"
)
```

The response is intended for the calling agent: ranked source passages it can use as task context rather than a standalone chat answer.

## Future development

Keep the surface small. Add capability only when real usage shows a retrieval, compatibility, or performance gap worth testing.

## License

MIT — see [LICENSE](LICENSE).
