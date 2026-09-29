#!/usr/bin/env python
"""
Benchmark comparison script for rag-mcp.
Evaluates LanceDB hybrid retrieval vs Qdrant hybrid retrieval using OpenRouter SOTA embeddings.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from rag.core import RAGPipelineV2

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def evaluate_query(query: str, expected_answer: str, retrieved_chunks: list[dict]) -> tuple[str, str]:
    """Grade whether the expected answer was found in the top chunks."""
    all_retrieved_text = " ".join([c.get("content", "") + " " + c.get("original_content", "") for c in retrieved_chunks]).lower()
    expected_lower = expected_answer.lower()

    # Extract key phrases / words
    keywords = [w.strip(".,;:\"'()") for w in expected_lower.split() if len(w.strip(".,;:\"'()")) > 3]

    if not keywords:
        found_ratio = 1.0 if expected_lower in all_retrieved_text else 0.0
    else:
        matches = sum(1 for kw in keywords if kw in all_retrieved_text)
        found_ratio = matches / len(keywords)

    if expected_lower in all_retrieved_text or found_ratio >= 0.80:
        return "Yes", f"Match score: {found_ratio:.2f}"
    elif found_ratio >= 0.40:
        return "Partial", f"Partial match score: {found_ratio:.2f}"
    else:
        return "No", f"Miss score: {found_ratio:.2f}"


def run_benchmark(corpus_name: str, doc_paths: list[str], queries_path: Path, backend: str = "lancedb", chunk_size: int = 1000, chunk_overlap: int = 150):
    persist_dir = str(repo_root / "rag" / f"rag_db_eval_{corpus_name}_{backend}")

    config = {
        "version": "eval",
        "persist_dir": persist_dir,
        "document_paths": doc_paths,
        "embed_provider": "ollama",
        "embed_model": "qwen3-embedding:8b",
        "backend": backend,
        "top_k": 5,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "rrf_k": 60,
        "rerank_top_k": 5,
        "reranker_type": "disabled",
        "force_reindex": True,
        "distance_metric": "cosine",
    }

    print(f"\n=======================================================")
    print(f"Running {corpus_name} Evaluation | Backend: {backend.upper()}")
    print(f"=======================================================")

    t0 = time.time()
    rag = RAGPipelineV2(config)
    index_time = time.time() - t0
    print(f"Indexing completed in {index_time:.2f} seconds.")

    queries = []
    with open(queries_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)

    results = []
    latencies = []
    grades = {"Yes": 0, "Partial": 0, "No": 0}

    for i, row in enumerate(queries, 1):
        q = row.get("Question") or row.get("Q: The Question") or ""
        expected = row.get("Answer") or row.get("Exact Answer") or row.get("Expected_Answer") or ""
        if not q:
            continue

        t_start = time.time()
        retrieved = rag.search(q)
        lat = (time.time() - t_start) * 1000
        latencies.append(lat)

        grade, note = evaluate_query(q, expected, retrieved if isinstance(retrieved, list) else [])
        grades[grade] += 1

        results.append({
            "id": i,
            "question": q,
            "expected": expected,
            "grade": grade,
            "note": note,
            "latency_ms": lat,
        })
        print(f"Q{i:02d} [{grade:7s}] ({lat:5.1f}ms): {q[:60]}...")

    total = len(results)
    yes_pct = (grades["Yes"] / total) * 100 if total else 0
    full_or_part = ((grades["Yes"] + grades["Partial"]) / total) * 100 if total else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n--- {corpus_name} ({backend}) Summary ---")
    print(f"Total Queries: {total}")
    print(f"Exact Hits (Yes): {grades['Yes']}/{total} ({yes_pct:.1f}%)")
    print(f"Partial Hits: {grades['Partial']}")
    print(f"Misses (No): {grades['No']}")
    print(f"Total Coverage (Yes + Partial): {full_or_part:.1f}%")
    print(f"Average Search Latency: {avg_lat:.1f} ms")
    print(f"Indexing Time: {index_time:.2f} s")

    return {
        "corpus": corpus_name,
        "backend": backend,
        "total": total,
        "yes": grades["Yes"],
        "partial": grades["Partial"],
        "no": grades["No"],
        "yes_pct": yes_pct,
        "coverage_pct": full_or_part,
        "avg_latency_ms": avg_lat,
        "index_time_s": index_time,
        "results": results,
    }


if __name__ == "__main__":
    print(">>> Starting SOTA OpenRouter Head-to-Head Benchmark: LanceDB vs Qdrant <<<")

    napoleon_doc = [str(repo_root / "evals" / "napoleon" / "Napoleon.pdf")]
    napoleon_queries = repo_root / "evals" / "napoleon" / "queries.csv"

    attention_doc = [str(repo_root / "evals" / "attention" / "att.pdf")]
    attention_queries = repo_root / "evals" / "attention" / "queries.csv"

    # 1. Napoleon
    nap_lance = run_benchmark("Napoleon (Biography)", napoleon_doc, napoleon_queries, backend="lancedb")
    nap_qdrant = run_benchmark("Napoleon (Biography)", napoleon_doc, napoleon_queries, backend="qdrant")

    # 2. Attention Paper
    att_lance = run_benchmark("Attention (Research Paper)", attention_doc, attention_queries, backend="lancedb")
    att_qdrant = run_benchmark("Attention (Research Paper)", attention_doc, attention_queries, backend="qdrant")

    print("\n" + "=" * 95)
    print("FINAL HEAD-TO-HEAD BENCHMARK COMPARISON TABLE (qwen3-embedding:8b via Ollama)")
    print("=" * 95)
    print(f"{'Corpus':<30} | {'Backend':<8} | {'Exact (Yes)':<12} | {'Coverage (Yes+Part)':<20} | {'Avg Latency':<12} | {'Index Time'}")
    print("-" * 105)

    for res in [nap_lance, nap_qdrant, att_lance, att_qdrant]:
        print(
            f"{res['corpus']:<30} | {res['backend']:<8} | "
            f"{res['yes']}/{res['total']} ({res['yes_pct']:.1f}%)   | "
            f"{res['yes'] + res['partial']}/{res['total']} ({res['coverage_pct']:.1f}%)        | "
            f"{res['avg_latency_ms']:>6.1f} ms    | "
            f"{res['index_time_s']:>6.2f} s"
        )
    print("-" * 105)
