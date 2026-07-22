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

No CI yet — the evidence below is manual, from the session that built this:

1. **Protocol-level smoke test** (bypasses any client): piped `initialize` /
   `tools/list` / `tools/call` directly at `server.py`'s stdin, checked every
   stdout line parses as JSON. This is what caught a real bug — the retrieval
   engine logs via plain `print()`, which was leaking onto the same stdout
   stream as the protocol frames. Fixed by redirecting that stream to stderr
   around the library call.
2. **Through the actual registered tool** in Claude Code
   (`claude mcp add rag -s user -- ...`, health-checked `✔ Connected`):
   - Queried a real subdirectory of a 90-crate Rust monorepo
     (`codex-rs/tui/src/chatwidget`, asking how a UI focus-toggle worked) →
     returned the exact files implementing it (`context_ledger.rs`,
     `slash_dispatch.rs`).
   - Queried a **different, unrelated** local repo (`~/Desktop/p/skills`,
     asking about skill-authoring conventions) → indexed it fresh on the spot
     and returned on-topic results, confirming it isn't hardwired to one project.
   - Queried with `doc_path` pointed at a **single file**
     (`codex-rs/memories/README.md`) → every result came from that file alone.
   - Queried the monorepo root with no `doc_path` → correctly **rejected** for
     exceeding the token-budget guard instead of hanging or truncating silently.

Smallest reproducible check, if you'd rather verify than trust the above:

```bash
printf '%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | .venv/bin/python server.py
```

Every output line must parse as JSON — if it doesn't, something in the
dependency chain is writing to stdout again.

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
