---
name: rag-mcp
type: local hybrid-search MCP server for AI coding agents & document workflows
---

<div align="center">

<!-- TODO: Insert Experiment 1 Unified Multi-Domain Scaling & Throughput Visualizations -->
<img src="assets/retrieval-hero.svg" alt="rag-mcp retrieval reliability" width="720">

<br>

[![MCP](https://img.shields.io/badge/protocol-MCP-blue?style=flat-square)](#quick-start)
[![Local](https://img.shields.io/badge/retrieval-100%25%20local-brightgreen?style=flat-square)](#core-features)
[![Hybrid search](https://img.shields.io/badge/search-LanceDB%20%2B%20BM25%20%2B%20Rerank-orange?style=flat-square)](#how-it-works)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

[Quick Start](#quick-start) • [Features](#core-features) • [Evals](#benchmark--retrieval-evaluations) • [How It Works](#how-it-works) • [Map](#repository-map) • [State](#current-state)

</div>

## Core Features

| Feature | Specification | Impact |
| :--- | :--- | :--- |
| **100% Local Execution** | On-device Ollama + LanceDB + Cross-Encoder | Zero cloud API dependencies, zero telemetry, zero data egress. |
| **2-Stage Hybrid Search** | Vector proximity + Tantivy BM25 FTS $\rightarrow$ RRF ($k=60$) | Combines semantic intent with exact keyword and symbol matching. |
| **Cross-Encoder Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` on CUDA | Reranks Top-50 candidates down to Top-5 with pinpoint accuracy. |
| **Per-Call Dynamic Scoping** | Target subdirectories/files via `doc_path` per tool call | Avoids full-workspace re-indexing on every search. |
| **Rich Multi-Format Support** | Native code, Markdown, PDF, IPYNB, Office & OCR | Parses `.py`, `.rs`, `.ts`, `.docx`, `.xlsx`, `.pptx`, `.png`, `.jpg`. |
| **Safety Guardrails** | Exclusion filters + token & depth limits | Blocks `.git`, `node_modules`, `.venv`, and directory traversal loops. |

---

## Quick Start: Agent Installation

Add `rag-mcp` to your coding agent of choice:

### 1. Claude Code
```bash
claude mcp add rag -s user -- /absolute/path/to/rag-mcp/.venv/bin/python /absolute/path/to/rag-mcp/server.py
```

### 2. Antigravity / Google AGY
Add to `~/.gemini/antigravity-cli/mcp/rag-mcp/config.json` (or `.mcp.json`):
```json
{
  "mcpServers": {
    "rag": {
      "command": "/absolute/path/to/rag-mcp/.venv/bin/python",
      "args": ["/absolute/path/to/rag-mcp/server.py"],
      "env": {
        "RAG_MCP_WORKSPACE_ROOT": "/path/to/your/workspace",
        "RAG_MCP_BACKEND": "lancedb"
      }
    }
  }
}
```

### 3. Codex & Elpis
Add to `~/.codex/config.toml` or `~/.elpis/config.toml`:
```toml
[mcp_servers.rag]
command = "/absolute/path/to/rag-mcp/.venv/bin/python"
args = ["/absolute/path/to/rag-mcp/server.py"]

[mcp_servers.rag.env]
RAG_MCP_WORKSPACE_ROOT = "/path/to/your/workspace"
RAG_MCP_BACKEND = "lancedb"
```

### 4. Local Setup & Verification
```bash
git clone https://github.com/MasihMoafi/rag-mcp
cd rag-mcp
uv sync
.venv/bin/python -m pytest tests/ -v
```

---

## How It Works

```text
query + optional doc_path
         ↓
syntax-aware chunking (functions, markdown breadcrumbs, cells)
         ↓
Stage 1: LanceDB vector search + Tantivy BM25 full-text search
         ↓
Reciprocal Rank Fusion (RRF, k=60) -> Top-50 candidates
         ↓
Stage 2: Cross-Encoder GPU reranking (ms-marco-MiniLM-L-6-v2)
         ↓
Top-5 ranked chunks with exact file paths & line numbers
```

---

## Repository Map

```
rag-mcp/
├── rag/                                # Core 2-Stage Hybrid RAG Engine & Storage
│   ├── core.py                         # Pipeline: Chunking -> Ollama Embed -> LanceDB/BM25 -> Cross-Encoder
│   ├── lancedb_backend.py              # LanceDB Vector Table + Tantivy FTS Full-Text Search
│   ├── qdrant_backend.py               # Alternative Qdrant Vector Store
│   ├── fetch.py                        # Document Loader & Multi-Format Extractor (PDF, Office, OCR)
│   └── notebook_chunker.py             # Jupyter Notebook parser & cell-block chunker
│
├── tests/                              # Pytest Automated Test Suite (24 Passed)
│   ├── unit/                           # Chunking, heading breadcrumbs, GPU OOM guardrail tests
│   │   ├── test_chunking.py            # Markdown breadcrumb, sliding-window & symbol extraction
│   │   ├── test_batch_size_oom.py      # OOM safety guardrails on massive batch requests
│   │   └── test_*_raises_runtime_error # GPU/CUDA availability enforcement tests
│   ├── integration/                    # LanceDB CRUD, MCP tool contracts, OCR extraction tests
│   │   ├── test_lancedb_backend.py     # LanceDB vector + full-text index integration
│   │   ├── test_rag_mcp_host.py        # FastMCP protocol & read-only contract
│   │   ├── test_rag_scope.py           # Scope traversal, depth limits & manifest caching
│   │   └── test_unstructured.py        # Office docs (.docx, .xlsx, .pptx) & image OCR
│   └── benchmarks/                     # GPU throughput sweeps & hardware benchmarks
│       ├── measure_qwen_throughput.py  # Batch size throughput sweep (1 to 2048)
│       └── measure_qwen_throughput.md  # Peak GPU throughput findings report
│
├── evals/                              # Scaling Experiments & Evaluation Suites
│   └── experiments/
│       ├── experiment-unified-scaling/ # Unified multi-domain 6.3k scaling experiment
│       │   ├── data/                   # Raw documents, candidate pools, queries & judge evals
│       │   ├── experiment.md           # Benchmark report & GPU latency profile
│       │   └── run_experiment.py       # Unified benchmark runner script
│       │
│       ├── experiment-isolated-scaling/# Single-domain isolated baseline benchmarks
│       │   ├── data/                   # Isolated raw documents, candidate pools & judge evals
│       │   ├── experiment.md           # Isolated baseline report
│       │   └── run_experiment.py       # Isolated benchmark runner script
│       │
│       └── baselines/                  # Single-corpus baseline benchmarks
│           └── data/                   # Raw corpus files (Elpis Rust, Notebooks, OCR datasets)
│
├── server.py                           # FastMCP server entry point exposing tools to coding agents
├── pyproject.toml                      # Project metadata & Python dependencies (LanceDB, PyTorch, etc.)
└── README.md                           # Main repository documentation & benchmark summary
```

<br>

## Benchmark & Retrieval Evaluations

Evaluated against the **[open-rag-eval](https://github.com/vectara/open-rag-eval)** taxonomy ($top\_k=5$, isolated local retrieval with no reranker, local `qwen3-embedding:8b` + BM25 hybrid search):

| Corpus / Domain | Total Queries | Strict Relevance (Score 3 / Exact) | Lenient Relevance (Score $\ge$ 2 / Full+Partial) | Miss Rate (Score $\le$ 1 / Miss) |
| :--- | :--- | :--- | :--- | :--- |
| **Attention Paper** (Scientific / AI) | 30 | **86.7%** (26/30) | **96.7%** (29/30) | 3.3% (1/30) |
| **Brain & Behavior** (Neuroscience) | 30 | **76.7%** (23/30) | **93.3%** (28/30) | 6.7% (2/30) |
| **Napoleon V2** (1000-char hybrid) | 30 | **66.7%** (20/30) | **90.0%** (27/30) | 10.0% (3/30) |
| **Napoleon V1** (300-char chunks) | 30 | **53.3%** (16/30) | **83.3%** (25/30) | 16.7% (5/30) |
| **Fire & Blood** (Narrative Fiction) | 30 | **46.7%** (14/30) | **80.0%** (24/30) | 20.0% (6/30) |
| **Mixed Codebase** (Py/Rust/IPYNB) | 33 | **90.9%** (30/33) | **100.0%** (33/33) | 0.0% (0/33) |
| **Elpis Memories Crate** (Rust) | 15 | **73.3%** (11/15) | **100.0%** (15/15) | 0.0% (0/15) |
| **rag-mcp Codebase** (Python Server) | 15 | **80.0%** (12/15) | **93.3%** (14/15) | 6.7% (1/15) |
| **Notebook Corpus** (JSON/Code) | 10 | **100.0%** (10/10) | **100.0%** (10/10) | 0.0% (0/10) |
### Multi-Domain Scaling Experiment (Experiment 1)

Evaluated across **5 merged heterogeneous domains (6,314 chunks in a single index)** comparing isolated baselines against unified scaling on NVIDIA RTX 3070 Laptop GPU with `qwen3-embedding:0.6b` + `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker:

| Corpus / Domain | Queries | Isolated Baseline Hit@5 | Unified Scaled Hit@5 | Isolated Baseline MRR | Unified Scaled MRR | Domain Purity in Unified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention Paper** (Scientific / AI) | 30 | **90.0%** (27/30) | **90.0%** (27/30) | 0.703 | 0.703 | 96.7% |
| **Brain & Behavior** (Neuroscience) | 30 | **93.3%** (28/30) | **93.3%** (28/30) | 0.831 | 0.831 | 94.7% |
| **Napoleon Biography** (History / Phil) | 30 | **66.7%** (20/30) | **66.7%** (20/30) | 0.558 | 0.548 | 100.0% |
| **Fire & Blood** (Narrative Fiction) | 30 | **70.0%** (21/30) | **70.0%** (21/30) | 0.526 | 0.526 | 100.0% |
| **Mixed Codebase** (Py/Rust/IPYNB) | 33 | **87.9%** (29/33) | **87.9%** (29/33) | 0.812 | 0.812 | 100.0% |
| **Overall Experiment Total** | **153** | **81.7%** (125/153) | **81.7%** (125/153) | **0.686** | **0.684** | **98.4%** |

* **Empirical Scaling Finding:** Scaling to a single 6,314-chunk multi-domain index resulted in **0.0% recall loss** (81.7% vs 81.7%) and **99.3% Top-1 candidate equivalence** (152/153 queries) with an average query latency of **368ms**.

<br>

### Configuration

Every retrieval parameter is configurable via environment variables:

| Variable | Default | What it controls |
| --- | --- | --- |
| `RAG_MCP_BACKEND` | `lancedb` | `lancedb` (default vector+FTS hybrid) or `qdrant` |
| `RAG_MCP_EMBED_PROVIDER` | `sentencetransformer` | `sentencetransformer` (local), `ollama` (local GPU), or `openai_compatible` |
| `RAG_MCP_EMBED_MODEL` | `all-MiniLM-L6-v2` | embedding model name (`qwen3-embedding:0.6b`, `all-MiniLM-L6-v2`, etc.) |
| `RAG_MCP_RERANKER_TYPE` | `cross-encoder` | `cross-encoder` or `disabled` |
| `RAG_MCP_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder reranker model |
| `RAG_MCP_TOP_K` | `50` | candidates pulled from Stage 1 before fusion/reranking |
| `RAG_MCP_RERANK_TOP_K` | `5` | results returned after reranking |
| `RAG_MCP_CHUNK_SIZE` | `700` | characters per chunk before overlap |
| `RAG_MCP_CHUNK_OVERLAP` | `100` | characters shared between adjacent chunks |
| `RAG_MCP_RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RAG_MCP_MAX_DEPTH` | `20` | directory-scan depth limit |
| `RAG_MCP_MAX_TOKENS` | `2000000` | directory-scan size limit |

## Current state

### Implemented and verified

- FastMCP stdio JSON-RPC 2.0 protocol path.
- Read-only `query_knowledge_base` tool with workspace-root and dynamic `doc_path` scoping.
- 2-Stage LanceDB vector table + Tantivy FTS full-text hybrid search with GPU Cross-Encoder reranking.
- Full pytest test suite (24 unit and integration tests passing).
- Guardrails for excluded directories (`.git`, `node_modules`, `.venv`) and oversized scopes.
- Text extraction for `.pdf`, `.ipynb`, office files (`.docx`, `.pptx`, `.xlsx`, `.csv`), and local OCR.

### Intentionally unsupported

- Hosted/cloud vector databases (designed for 100% on-device execution).
- Write or filesystem mutation tools (server is strictly read-only retrieval).

## What sets this apart

- **100% Local Execution:** Embeddings, vector search, BM25 indexing, and reranking run entirely on local hardware.
- **Evidence-First Retrieval:** Returns verbatim source text chunks with exact file paths and line numbers.
- **Per-Call Dynamic Scoping:** One server instance can search different subfolders per tool call without re-indexing the entire workspace.
- **Lightweight Transport:** Direct stdio JSON-RPC without heavyweight SDK overhead.

## Future development

- Explore int8/FP16 quantization for low-memory environments.
- Optional streaming chunk responses over HTTP/SSE transports.

## License

MIT — see [LICENSE](LICENSE).
