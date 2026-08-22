#!/usr/bin/env python3
"""
Runs isolated single-domain benchmarks for all 5 domains using the EXACT same stack:
- Embedding: qwen3-embedding:0.6b (100% GPU via Ollama)
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (100% GPU via PyTorch CUDA)
- LanceDB + Tantivy FTS + RRF (k=60) Top-50 -> Top-5 Rerank
"""
import os
import sys
import csv
import json
import time
import shutil
import torch

for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if var in os.environ: del os.environ[var]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)


from rag.core import RAGPipelineV2

DOMAINS = [
    {
        "name": "attention",
        "docs": ["evals/experiments/experiment-isolated-scaling/data/attention/att.pdf"],
        "queries": "evals/experiments/experiment-isolated-scaling/data/attention/queries.csv",
        "q_col": "Q: The Question",
        "a_col": "Exact Answer"
    },
    {
        "name": "brain-and-behavior",
        "docs": ["evals/experiments/experiment-isolated-scaling/data/brain-and-behavior/brain_and_behavior.pdf"],
        "queries": "evals/experiments/experiment-isolated-scaling/data/brain-and-behavior/queries.csv",
        "q_col": "Q: The Question",
        "a_col": "Exact Answer"
    },
    {
        "name": "napoleon",
        "docs": ["evals/experiments/experiment-isolated-scaling/data/napoleon/Napoleon.pdf"],
        "queries": "evals/experiments/experiment-isolated-scaling/data/napoleon/queries.csv",
        "q_col": "Question",
        "a_col": "Answer"
    },
    {
        "name": "fire-and-blood",
        "docs": ["evals/experiments/experiment-isolated-scaling/data/fire-and-blood/Fire And Blood_George.R.R.Martin_furfalling.ir.pdf"],
        "queries": "evals/experiments/experiment-isolated-scaling/data/fire-and-blood/queries.csv",
        "q_col": "Q: The Question",
        "a_col": "Answer"
    },
    {
        "name": "mixed-codebase",
        "docs": [
            "evals/experiments/experiment-isolated-scaling/data/mixed-codebase/python",
            "evals/experiments/experiment-isolated-scaling/data/mixed-codebase/rust",
            "evals/experiments/experiment-isolated-scaling/data/mixed-codebase/notebook"
        ],
        "queries": "evals/experiments/experiment-isolated-scaling/data/mixed-codebase/queries.csv",
        "q_col": "Question",
        "a_col": "Expected_Answer"
    }
]

def run_isolated_benchmarks():
    print("=" * 80)
    print("RUNNING ISOLATED BENCHMARKS WITH IDENTICAL QWEN 0.6B + RERANKER STACK")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    isolated_results = {}
    all_isolated_candidates = {}
    
    for dom in DOMAINS:
        dom_name = dom["name"]
        print(f"\n>>> Running Domain: {dom_name.upper()} <<<")
        
        # Discover all doc files recursively if directory
        all_doc_files = []
        for p in dom["docs"]:
            abs_p = os.path.join(BASE_DIR, p)
            if os.path.isdir(abs_p):
                for root, _, files in os.walk(abs_p):
                    for f in files:
                        if f.endswith((".pdf", ".rs", ".py", ".ipynb", ".md")):
                            all_doc_files.append(os.path.join(root, f))
            elif os.path.isfile(abs_p):
                all_doc_files.append(abs_p)
                
        print(f"  Files to index: {len(all_doc_files)}")
        
        db_dir = os.path.join(BASE_DIR, f"evals/{dom_name}/lancedb_isolated_0.6b")
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)
            
        config = {
            "version": f"isolated_{dom_name}_0.6b",
            "backend": "lancedb",
            "persist_dir": db_dir,
            "document_paths": all_doc_files,
            "embed_provider": "ollama",
            "embed_model": "qwen3-embedding:0.6b",
            "reranker_type": "cross-encoder",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 50,
            "rerank_top_k": 5,
            "rrf_k": 60
        }
        
        t0 = time.time()
        pipeline = RAGPipelineV2(config=config)
        ingest_time = time.time() - t0
        
        queries_path = os.path.join(BASE_DIR, dom["queries"])
        with open(queries_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            
        queries_data = []
        for idx, row in enumerate(reader, 1):
            q_text = row.get(dom["q_col"]) or row.get("question") or row.get("query")
            ans_text = row.get(dom["a_col"]) or row.get("expected_answer") or row.get("answer")
            if q_text:
                queries_data.append((str(idx), q_text.strip(), ans_text.strip() if ans_text else ""))
                
        print(f"  Loaded {len(queries_data)} queries. Running retrieval...")
        
        hit1 = 0
        hit5 = 0
        reciprocal_ranks = []
        latencies = []
        
        for qid, q_text, expected in queries_data:
            start_q = time.time()
            final_chunks = pipeline.search(q_text)
            lat = time.time() - start_q
            latencies.append(lat)
            
            key = f"{dom_name}_{qid}"
            all_isolated_candidates[key] = {
                "domain": dom_name,
                "query_id": qid,
                "query": q_text,
                "expected_answer": expected,
                "latency": lat,
                "chunks": final_chunks
            }
            
            first_hit_rank = 0
            exp_clean = expected.lower()
            exp_tokens = [t for t in exp_clean.replace('"', '').replace("'", "").split() if len(t) > 3]
            
            for rank, chunk in enumerate(final_chunks[:5], 1):
                c = chunk.get("content", "").lower()
                orig = chunk.get("original_content", "").lower()
                is_match = False
                if exp_clean and len(exp_clean) > 3:
                    if exp_clean in c or exp_clean in orig:
                        is_match = True
                    elif exp_tokens:
                        matched = sum(1 for t in exp_tokens if t in c or t in orig)
                        if matched / len(exp_tokens) >= 0.5:
                            is_match = True
                if is_match and first_hit_rank == 0:
                    first_hit_rank = rank
                    
            if first_hit_rank == 1:
                hit1 += 1
                hit5 += 1
                reciprocal_ranks.append(1.0)
            elif first_hit_rank > 1:
                hit5 += 1
                reciprocal_ranks.append(1.0 / first_hit_rank)
            else:
                reciprocal_ranks.append(0.0)
                
        n = len(queries_data)
        isolated_results[dom_name] = {
            "total": n,
            "hit1": hit1,
            "hit5": hit5,
            "hit1_pct": (hit1 / n) * 100,
            "hit5_pct": (hit5 / n) * 100,
            "mrr": sum(reciprocal_ranks) / n,
            "avg_lat": sum(latencies) / n,
            "ingest_time": ingest_time
        }
        
        print(f"  Domain {dom_name}: Hit@1={isolated_results[dom_name]['hit1_pct']:.1f}% | Hit@5={isolated_results[dom_name]['hit5_pct']:.1f}% | MRR={isolated_results[dom_name]['mrr']:.3f}")
        
    out_file = os.path.join(BASE_DIR, "evals/experiments/experiment1-unified-scaling/isolated_0.6b_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(isolated_results, f, indent=2)
    
    cand_file = os.path.join(BASE_DIR, "evals/experiments/experiment1-unified-scaling/isolated_candidates.json")
    with open(cand_file, "w", encoding="utf-8") as f:
        json.dump(all_isolated_candidates, f, indent=2)
    print(f"\n✓ Saved exact isolated 0.6B baseline results to: {out_file} and {cand_file}")

if __name__ == "__main__":
    run_isolated_benchmarks()
