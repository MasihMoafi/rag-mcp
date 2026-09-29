#!/usr/bin/env python
"""
Full LanceDB vs Qdrant benchmark across every corpus in evals/ that has a locatable source.
Uses local qwen3-embedding:8b via Ollama (GPU) -- the OpenRouter path proved unreliable.
No reranking (reranker_type disabled), matching the original "first benchmark" protocol.
Dumps full retrieved-chunk text per question to JSON for manual/automated grading.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from rag.core import RAGPipelineV2

EMBED_CONFIG = {
    "embed_provider": "ollama",
    "embed_model": "qwen3-embedding:8b",
}

MIXED_CODEBASE_DIR = repo_root / "evals" / "mixed-codebase"
MIXED_CODEBASE_FILES = sorted(
    str(p) for p in MIXED_CODEBASE_DIR.rglob("*")
    if p.is_file() and p.suffix not in (".csv",)
)

RAG_MCP_CODEBASE_FILES = [
    str(repo_root / "rag" / "core.py"),
    str(repo_root / "server.py"),
    str(repo_root / "rag" / "notebook_chunker.py"),
    str(repo_root / "utils" / "proxy.py"),
]

# name -> (document_paths, index_key) ; index_key groups corpora that share one index
CORPORA = {
    "brain-and-behavior": {
        "docs": [str(repo_root / "evals" / "brain-and-behavior" / "brain_and_behavior.pdf")],
        "queries": repo_root / "evals" / "brain-and-behavior" / "queries.csv",
        "index_key": "brain-and-behavior",
    },
    "fire-and-blood": {
        "docs": [str(repo_root / "evals" / "fire-and-blood" / "Fire And Blood_George.R.R.Martin_furfalling.ir.pdf")],
        "queries": repo_root / "evals" / "fire-and-blood" / "queries.csv",
        "index_key": "fire-and-blood",
    },
    "mixed-codebase": {
        "docs": MIXED_CODEBASE_FILES,
        "queries": repo_root / "evals" / "mixed-codebase" / "queries.csv",
        "index_key": "mixed-codebase",
    },
    "notebook": {
        "docs": MIXED_CODEBASE_FILES,
        "queries": repo_root / "evals" / "notebook" / "queries.csv",
        "index_key": "mixed-codebase",
    },
    "elpis-memories-crate": {
        "docs": MIXED_CODEBASE_FILES,
        "queries": repo_root / "evals" / "elpis-memories-crate" / "queries.csv",
        "index_key": "mixed-codebase",
    },
    "rag-mcp-codebase": {
        "docs": RAG_MCP_CODEBASE_FILES,
        "queries": repo_root / "evals" / "rag-mcp-codebase" / "queries.csv",
        "index_key": "rag-mcp-codebase",
    },
}


def load_queries(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []
    for row in rows:
        q = row.get("Question") or row.get("Q: The Question")
        a = row.get("Expected_Answer") or row.get("Answer") or row.get("Exact Answer")
        if q and a:
            out.append({"question": q, "expected": a})
    return out


def get_pipeline(index_key: str, docs: list[str], backend: str, cache: dict) -> RAGPipelineV2:
    cache_key = (index_key, backend)
    if cache_key in cache:
        return cache[cache_key]
    persist_dir = str(repo_root / "rag" / f"rag_db_full_{index_key}_{backend}")
    config = {
        "version": "eval",
        "persist_dir": persist_dir,
        "document_paths": docs,
        "backend": backend,
        "top_k": 5,
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "rrf_k": 60,
        "rerank_top_k": 5,
        "reranker_type": "disabled",
        "force_reindex": True,
        "distance_metric": "cosine",
        **EMBED_CONFIG,
    }
    t0 = time.time()
    rag = RAGPipelineV2(config)
    elapsed = time.time() - t0
    sys.stderr.write(f"[BENCH] Indexed {index_key} ({backend}) in {elapsed:.1f}s\n")
    cache[cache_key] = rag
    cache[(cache_key, "index_time")] = elapsed
    return rag


def main():
    checkpoint_path = repo_root / "scripts" / "full_benchmark_results.json"
    results = json.load(open(checkpoint_path)) if checkpoint_path.exists() else {}
    pipeline_cache = {}

    for corpus_name, meta in CORPORA.items():
        queries = load_queries(meta["queries"])
        for backend in ["lancedb", "qdrant"]:
            key = f"{corpus_name}|{backend}"
            if key in results:
                sys.stderr.write(f"[BENCH] Skipping {key} (already checkpointed)\n")
                continue
            rag = get_pipeline(meta["index_key"], meta["docs"], backend, pipeline_cache)
            index_time = pipeline_cache.get(((meta["index_key"], backend), "index_time"), None)
            results[key] = {"index_time": index_time, "questions": []}
            sys.stderr.write(f"[BENCH] Running {len(queries)} queries: {corpus_name} ({backend})\n")
            for q in queries:
                t0 = time.time()
                retrieved = rag.search(q["question"])
                lat = (time.time() - t0) * 1000
                chunks = [
                    (r.get("content") or r.get("original_content") or "")[:800]
                    for r in (retrieved if isinstance(retrieved, list) else [])[:5]
                ]
                results[key]["questions"].append({
                    "question": q["question"],
                    "expected": q["expected"],
                    "latency_ms": lat,
                    "chunks": chunks,
                })
            # checkpoint after every corpus+backend in case of interruption
            json.dump(results, open(repo_root / "scripts" / "full_benchmark_results.json", "w"))
            sys.stderr.write(f"[BENCH] Checkpoint saved after {key}\n")

    print("DONE")


if __name__ == "__main__":
    main()
