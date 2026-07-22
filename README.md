# rag-mcp

A coding agent should not have to choose between reading files one at a time and
dumping an entire repo into context just to answer "where is X handled."

**rag-mcp** is a local hybrid-search MCP server: point it at any file or folder,
ask a question in plain language, get back the most relevant chunks with exact
source paths. Everything — embeddings, vector search, reranking — runs on your
machine. No API keys, no upload, no hosted service. It speaks the Model Context
Protocol, so any MCP-capable agent (Claude Code, Claude Desktop, Codex) can call
it mid-task the way it would call `grep`.

It is, in effect, a local NotebookLM: ask your files a question, get a grounded,
cited answer back — except the answer goes to your agent instead of a chat
window, and nothing leaves your disk.

## The problem

Coding agents retrieve context two ways today: open files one by one (slow,
misses files the agent didn't think to look for) or load the whole tree into
context (expensive, drowns the actual signal in noise). Hosted RAG services fix
the search problem but require sending your code or documents to a third party.
rag-mcp runs the retrieval step locally and exposes it as one tool call, so an
agent can search semantically without either tradeoff.

## How does it work? Technical specification

```
rag-mcp/
├── server.py       # stdio JSON-RPC MCP host (~300 lines, no MCP SDK dependency)
├── rag/            # retrieval engine: chunking, BM25, Qdrant vector search, reranking
└── utils/proxy.py  # strips SOCKS proxy env vars so local Qdrant/embeddings stay reachable
```

- One tool: `query_knowledge_base(query, doc_path?)`.
- Pipeline: chunk → BM25 (lexical) + sentence-transformer embeddings in Qdrant
  (semantic) → Reciprocal Rank Fusion → CrossEncoder reranking.
- Fully local models: `all-MiniLM-L6-v2` embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2`
  reranker, Qdrant in embedded/on-disk mode.
- `doc_path` scopes a call to one file or directory; omit it and the server
  searches `RAG_MCP_WORKSPACE_ROOT` (or its own cwd). Each scanned path gets its
  own persistent index under `rag/rag_db_v2/`, so repeat queries against the
  same path skip re-embedding.
- Guardrails: refuses to scan `node_modules`, `.git`, `.venv`, `dist`, `build`,
  `target`; rejects any scope over a configurable depth/token budget
  (`RAG_MCP_MAX_DEPTH`, `RAG_MCP_MAX_TOKENS`, default 20 / 2,000,000) instead of
  silently hanging on a huge tree.

## Current state

- **Implemented and verified:** the MCP handshake (`initialize` → `tools/list` →
  `tools/call`) and `query_knowledge_base`, tested end-to-end through a real
  registered Claude Code MCP connection — see Evals below.
- **Implemented, not yet exercised by tests:** the `ollama` embedding-provider
  path in `rag/core.py` (an alternative to the default sentence-transformer
  path) is present but untested here.
- **Intentionally unsupported:** remote/hosted vector DBs (Qdrant only runs
  local, on-disk); file types outside the extension allowlist in `server.py`.
- **Planned:** nothing tracked yet — open an issue if you need something.

## What sets this apart

- No API keys, no network round-trip: retrieval and reranking both run on-device.
- No MCP SDK dependency: the transport is a few hundred lines of stdio
  JSON-RPC, small enough to read in one sitting.
- Workspace-agnostic by construction: one running server can be pointed at a
  different `doc_path` on every call, so it covers every project you have, not
  one server per repo.

## Evals and test series

Automated: `tests/`. Five tests, no heavy ML deps loaded (the RAG import is
mocked out), runs in under a tenth of a second:

```bash
uv sync --group dev
.venv/bin/python -m pytest tests/ -v
```

Covers: the tool is advertised read-only with the right annotations; a call
with no `doc_path` scopes to the workspace root; an explicit `doc_path` keeps
its own scope; `node_modules` and similar scopes are rejected with an
actionable message; depth-limit violations are rejected with an actionable
message.

Protocol-level check, reproducible against this repo, no client required:

```bash
printf '%s\n%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"how does reciprocal rank fusion combine bm25 and vector results"}}}' \
  | RAG_MCP_WORKSPACE_ROOT="$PWD" .venv/bin/python server.py
```

Every stdout line must parse as JSON, and the `tools/call` response should cite
`rag/core.py`'s `_rrf_fusion` — this repo indexing and querying itself. If a
line doesn't parse, something in the dependency chain is writing to stdout
instead of stderr again (this is what the redirect in `query_knowledge_base`
guards against).

## Quick start

```bash
git clone https://github.com/MasihMoafi/rag-mcp
cd rag-mcp
uv sync        # or: pip install -e .
```

### Add to Claude Code

```bash
claude mcp add rag -s user -- /absolute/path/to/rag-mcp/.venv/bin/python /absolute/path/to/rag-mcp/server.py
```

`-s user` makes it available in every project; leave `RAG_MCP_WORKSPACE_ROOT`
unset and it defaults to whatever directory the calling session is in. Verify
with `claude mcp list` (expect `✔ Connected`), then start a **new** session —
a server registered mid-session is not hot-loaded into the one that ran `add`.

### Add to Codex / Elpis (`~/.codex/config.toml`)

```toml
[mcp_servers.rag]
command = "/absolute/path/to/rag-mcp/.venv/bin/python"
args = ["/absolute/path/to/rag-mcp/server.py"]

[mcp_servers.rag.env]
RAG_MCP_WORKSPACE_ROOT = "/absolute/path/to/your/project"
```

### Add to Claude Desktop, or any JSON-based MCP client

```json
{
  "mcpServers": {
    "rag": {
      "command": "/absolute/path/to/rag-mcp/.venv/bin/python",
      "args": ["/absolute/path/to/rag-mcp/server.py"]
    }
  }
}
```

## Why this is useful

Think NotebookLM, but local and callable instead of chat-only: NotebookLM
grounds answers in documents you upload to a hosted product; rag-mcp grounds
answers in files that never leave your machine, and hands the result to
whatever agent asked for it instead of rendering it in a browser tab. A coding
agent mid-task can call:

```
query_knowledge_base("how does retry backoff work for failed jobs", doc_path="codex-rs/memories")
```

and get back ranked, cited passages — the same motion as `grep`, but searching
by meaning instead of literal text, with a rerank pass to push the actually
relevant chunks to the top.

## Future development

Nothing tracked yet — this was built and verified in one session; extend it as
real needs surface rather than pre-building features.

## License

MIT — see [LICENSE](LICENSE).
