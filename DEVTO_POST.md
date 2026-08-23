---
title: Why Your AI Coding Agent Needs Local Hybrid Search: Building rag-mcp with LanceDB & Tantivy
published: false
description: A 100% local, high-throughput hybrid RAG MCP server combining LanceDB columnar vectors, Tantivy BM25, and Reciprocal Rank Fusion for AI coding agents. Includes in-browser WASM demo.
tags: ai, mcp, python, rust
cover_image: https://masihmoafi.com/rag-mcp/retrieval-hero.svg
canonical_url: https://masihmoafi.com/projects/rag-mcp
---

**A coding agent should not have to choose between opening files one at a time and dumping an entire repository into context.**

When building autonomous AI coding assistants (like Claude Code, Codex, or Cursor), context window management is everything. Today, agents face a painful trilemma:

1. **Greedy file probing**: Running `cat`, `grep`, or `find_by_name` across dozens of directories wastes precious tokens, incurs multi-turn tool latency, and often misses cross-file abstractions.
2. **Brute-force context stuffing**: Dumping entire directories into the prompt triggers quadratic attention costs, slows inference to a crawl, and increases reasoning hallucination.
3. **Naive vector search**: Pure dense embedding search fails spectacularly on exact symbol names, compiler error codes, and unique variable identifiers like `AUTH_JWT_SECRET_KEY` or `handle_epoch_timeout`.

To solve this, I built **[rag-mcp](https://github.com/MasihMoafi/rag-mcp)**: a 100% local, high-throughput hybrid search engine engineered specifically as a Model Context Protocol (MCP) server for coding agents.

> 🚀 **Try it live in your browser (Zero Download / Zero Backend)**:  
> [https://rag.masihmoafi.com](https://rag.masihmoafi.com) *(Mirror: [https://rag.masihmoafi.tech](https://rag.masihmoafi.tech))*

---

## The Architecture: Why Pure Vector Search Fails on Code

Merging sparse lexical search (BM25) and dense semantic vectors sounds simple in theory, but in practice, linear score combination ($S = \alpha S_{\text{vector}} + (1-\alpha) S_{\text{bm25}}$) causes severe calibration skew because BM25 scores are unbounded floats while cosine similarities are bounded in $[-1, 1]$.

`rag-mcp` resolves this with a 2-stage retrieval pipeline:

![rag-mcp Architecture](https://masihmoafi.com/rag-mcp/architecture.svg)

### 1. Tantivy Rust BM25 (Lexical Accuracy)
Tantivy provides native Rust-speed full-text indexing directly over your codebase. If your prompt asks about `class TransactionHandler` or an exact error string, Tantivy surfaces the exact line ranges with zero semantic hallucination.

### 2. LanceDB Apache Arrow Vectors (Semantic Context)
Unlike traditional vector databases that serialize records into SQLite blobs or require heavy background Docker containers, **LanceDB** stores embeddings in Apache Arrow columnar memory format. It allows zero-copy vector querying directly against your filesystem.

### 3. Reciprocal Rank Fusion (RRF, $k=60$)
Instead of normalizing uncalibrated similarity scores, `rag-mcp` merges the ranked lists using Reciprocal Rank Fusion:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $k=60$ acts as a smoothing constant preventing any single outlier from dominating the final ranking.

### 4. Cross-Encoder Reranker
The top $N$ fused candidates pass through a fast local Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) that jointly attends over `(query, passage)` pairs to prune false positives before returning tokens to the agent.

---

## Empirical Benchmark Results

We evaluated `rag-mcp` against the **[open-rag-eval](https://github.com/vectara/open-rag-eval)** taxonomy across multiple heterogeneous corpora ($top\_k=5$, local Ollama embeddings):

| Corpus / Domain | Total Queries | Strict Relevance (Score 3 / Exact) | Full + Partial (Score $\ge$ 2) | Miss Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Mixed Codebase (Python + Rust + Notebooks)** | 22 | **90.9%** (20/22) | **100.0%** (22/22) | **0.0%** (0/22) |
| **Scientific Literature (Attention Paper)** | 30 | **93.3%** (28/30) | **96.7%** (29/30) | **3.3%** (1/30) |
| **Academic Textbook (Brain & Behavior)** | 35 | **85.7%** (30/35) | **94.3%** (33/35) | **5.7%** (2/35) |
| **Historical Biography (Napoleon)** | 35 | **88.6%** (31/35) | **97.1%** (34/35) | **2.9%** (1/35) |

On real-world software codebases, combining breadcrumb AST chunking with RRF achieved **100% at-least-partial evidence capture and 0% total retrieval failure**.

---

## Try the In-Browser WASM Demo (Zero Setup)

Before configuring your local CLI, you can test the full pipeline in an air-gapped WebAssembly environment:

- 🔗 **Live Demo**: [https://rag.masihmoafi.com](https://rag.masihmoafi.com)
- 🔒 **100% Private**: Runs quantized ONNX models (`all-MiniLM-L6-v2`) via Transformers.js in a dedicated Web Worker.
- 💾 **Local IndexedDB Persistence**: All documents, chunks, and vector tables persist locally in your browser with **zero backend server requests**.

---

## Quick Start & Installation

### 1. Install via `uv`

```bash
git clone https://github.com/MasihMoafi/rag-mcp
cd rag-mcp
uv sync
```

Run the automated test suite:
```bash
.venv/bin/python -m pytest tests/ -v
```

### 2. Connect to Your Coding Agent

#### Claude Code (`~/.claude.json`):
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

#### Codex CLI (`~/.codex/config.toml`):
```toml
[mcp_servers.rag]
command = "/absolute/path/to/rag-mcp/.venv/bin/python"
args = ["/absolute/path/to/rag-mcp/server.py"]
```

#### Cursor (`Settings > Features > MCP`):
- **Name**: `rag`
- **Type**: `command`
- **Command**: `/absolute/path/to/rag-mcp/.venv/bin/python /absolute/path/to/rag-mcp/server.py`

---

## Links & Resources

- ⭐️ **GitHub Repository**: [github.com/MasihMoafi/rag-mcp](https://github.com/MasihMoafi/rag-mcp)
- 🌐 **Live WASM Demo**: [rag.masihmoafi.com](https://rag.masihmoafi.com) | [rag.masihmoafi.tech](https://rag.masihmoafi.tech)
- 📖 **Deep-Dive Technical Report**: [masihmoafi.com/projects/rag-mcp](https://masihmoafi.com/projects/rag-mcp)
- 👤 **Author**: [Masih Moafi](https://masihmoafi.com)

Feel free to star the repo, test the WASM demo with your own files, or contribute optimizations!
