#!/usr/bin/env python3
"""
Experiment 1: Unified Multi-Domain Scaling Benchmark Runner
Executes batch indexing and 2-stage retrieval (LanceDB Vector + BM25 RRF Top-50 -> GPU Cross-Encoder Rerank Top-5).
"""
import os
import sys
import csv
import json
import time
import torch

# Clean proxy environment for local execution
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if var in os.environ:
        del os.environ[var]

# Point to local codebase
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)

from rag.core import RAGPipelineV2

def run_unified_experiment():
    print("=" * 75)
    print("EXPERIMENT 1: UNIFIED MULTI-DOMAIN SCALING BENCHMARK (2-STAGE GPU PIPELINE)")
    print("=" * 75)

    # 1. Strict GPU Verification Gate
    if not torch.cuda.is_available():
        raise RuntimeError("Dedicated NVIDIA GPU is required for this experiment. Aborting.")
    print("✓ Dedicated NVIDIA GPU Active (CUDA).")

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(exp_dir, "data")
    corpus_dir = os.path.join(data_dir, "corpus")
    db_dir = os.path.join(data_dir, "lancedb_unified_storage")
    queries_file = os.path.join(data_dir, "queries_unified.csv")
    stage1_file = os.path.join(data_dir, "stage1_candidates.json")
    results_file = os.path.join(data_dir, "results_unified_benchmark.csv")

    doc_files = []
    for root, _, files in os.walk(corpus_dir):
        for f in sorted(files):
            if f.endswith((".pdf", ".rs", ".py", ".ipynb", ".md")):
                doc_files.append(os.path.join(root, f))
    print(f"✓ Found {len(doc_files)} total target domain files in corpus.")

    config = {
        "version": "experiment1_unified_v2",
        "backend": "lancedb",
        "persist_dir": db_dir,
        "document_paths": doc_files,
        "embed_provider": "ollama",
        "embed_model": "qwen3-embedding:0.6b",
        "reranker_type": "cross-encoder",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "chunk_size": 1000,

        "chunk_overlap": 150,
        "top_k": 50,         # Retrieve Top-50 candidates in Stage 1
        "rerank_top_k": 5,   # Rerank down to Top-5 final in Stage 2
        "rrf_k": 60


    }

    # 2. Ingestion Phase
    print("\n" + "=" * 55)
    print("PHASE 1: MULTI-DOMAIN CORPUS INGESTION (BATCH SIZE 8)")
    print("=" * 55)
    start_ingest = time.time()
    pipeline = RAGPipelineV2(config=config)
    ingest_time = time.time() - start_ingest
    print(f"✓ Corpus Ingestion Complete in {ingest_time:.2f}s ({ingest_time/60:.2f} min)")

    # 3. Query Execution Phase (153 Queries)
    print("\n" + "=" * 55)
    print("PHASE 2: UNIFIED 2-STAGE RETRIEVAL & GPU RERANKING")
    print("=" * 55)

    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        queries = list(reader)

    print(f"✓ Loaded {len(queries)} shuffled ground-truth queries across all 5 domains.")
    results = []
    stage1_candidates = {}

    start_retrieval = time.time()
    for idx, q_row in enumerate(queries, 1):
        q_text = q_row["query"]
        domain = q_row["domain"]
        q_id = q_row["query_id"]

        q_start = time.time()
        # Full 2-Stage Pipeline: Vector+BM25 Top-50 -> Cross-Encoder GPU Rerank Top-5
        final_chunks = pipeline.search(q_text)
        q_elapsed = time.time() - q_start

        # Record candidate outputs
        stage1_candidates[f"{domain}_{q_id}"] = {
            "query": q_text,
            "domain": domain,
            "latency": q_elapsed,
            "chunks": final_chunks
        }

        top_chunk = final_chunks[0].get("content", "") if final_chunks else ""
        top_source = final_chunks[0].get("source", final_chunks[0].get("file_path", final_chunks[0].get("metadata", {}).get("source", ""))) if final_chunks else ""

        results.append({
            "domain": domain,
            "query_id": q_id,
            "query": q_text,
            "expected_answer": q_row["expected_answer"],
            "top_hit_source": top_source,
            "top_hit_content": top_chunk[:300].replace("\n", " "),
            "latency_s": f"{q_elapsed:.3f}"
        })

        if idx % 10 == 0 or idx == len(queries):
            print(f"  [Progress: {idx:3d}/{len(queries)}] Domain: {domain:18s} | Latency: {q_elapsed:.2f}s")

    total_retrieval_time = time.time() - start_retrieval
    print(f"✓ All {len(queries)} Queries Processed & Reranked in {total_retrieval_time:.2f}s!")

    # 4. Save Artifacts to Disk
    with open(stage1_file, "w", encoding="utf-8") as f:
        json.dump(stage1_candidates, f, indent=2)
    print(f"✓ Raw Candidate Stage 1 Pool Saved: {stage1_file}")

    with open(results_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "query_id", "query", "expected_answer", "top_hit_source", "top_hit_content", "latency_s"])
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ Evaluation Benchmark Report Saved: {results_file}")

    print("\n" + "=" * 75)
    print(f"EXPERIMENT 1 RUN SUMMARY:")
    print(f"  Ingestion Time:   {ingest_time/60:.2f} minutes")
    print(f"  Retrieval Time:   {total_retrieval_time:.2f} seconds")
    print(f"  Total Wall-Clock: {(ingest_time + total_retrieval_time)/60:.2f} minutes")
    print("=" * 75)

if __name__ == "__main__":
    run_unified_experiment()
