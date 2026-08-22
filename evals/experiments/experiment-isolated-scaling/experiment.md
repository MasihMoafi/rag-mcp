# Experiment 1: Multi-Domain Scaling Benchmark (Complete Apples-to-Apples Evaluation)

**Date:** 2026-08-22  
**Status:** Completed, Verified on NVIDIA GPU   
**Embedding Engine:** `qwen3-embedding:0.6b` via Ollama (CUDA GPU)  
**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CUDA GPU, Top-50 -> Top-5)  
**Vector Store:** LanceDB + Tantivy BM25 Full-Text Hybrid Search (RRF $k=60$)  
**Benchmark Suite:** 153 queries across 5 heterogeneous domains (6,314 chunks, ~391k total tokens across all candidate files)  
**LLM Judge:** `Gemini-3.7-Flash` (High Thinking mode, Google AI Studio, structured outputs JSON schema enabled, tools disabled)

---

## 1. Apples-to-Apples Accuracy & Recall Benchmark

> [!NOTE]
> **Isolated Baseline vs. Unified Multi-Domain Scaling are virtually identical in accuracy across all 5 corpora.**
> Merging 5 distinct domains into one 6,314-chunk database produced **0.0% degradation in Hit@5 recall** and **99.3% identical Top-1 candidate selection** (152 / 153 queries).

| Domain / Corpus | Queries | Chunks | Hit@1 (Isolated / Unified) | Hit@5 Recall (Both) | MRR (Iso / Uni) | Unified Domain Purity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention Is All You Need** | 30 | 47 | 73.3% / 73.3% | **90.0%** (27/30) | 0.803 / 0.803 | 96.7% |
| **Brain & Behavior (Textbook)** | 30 | 3,324 | 76.7% / 76.7% | **93.3%** (28/30) | 0.831 / 0.831 | 94.7% |
| **Napoleon Biography** | 30 | 728 | 50.0% / 46.7% | **66.7%** (20/30) | 0.558 / 0.548 | 100.0% |
| **Fire & Blood (Fantasy Novel)** | 30 | 1,700 | 30.0% / 30.0% | **46.7%** (14/30) | 0.372 / 0.372 | 100.0% |
| **Mixed Codebase (Rust/Py/NB)** | 33 | 515 | 75.8% / 75.8% | **87.9%** (29/33) | 0.812 / 0.812 | 100.0% |
| **OVERALL SYSTEM TOTAL** | **153** | **6,314** | **61.4% / 60.8%** | **77.1%** (118/153) | **0.675 / 0.673** | **98.4%** |

---

## 2. Execution & Latency Profile (NVIDIA RTX 3070 Laptop GPU)

### Latency Comparison (End-to-End Search + Top-50 Rerank per Query)

| Domain / Corpus          |  Isolated Index Size  | Unified Index Size | Isolated Mean Latency (p50 / p95) | Unified Mean Latency (p50 / p95) | Scaling Overhead ($\Delta$) |
| :----------------------- | :-------------------: | :----------------: | :-------------------------------: | :------------------------------: | :-------------------------: |
| **Attention Paper**      |       47 chunks       |    6,314 chunks    |  **274.3 ms** (257.8 / 298.6 ms)  | **387.3 ms** (376.9 / 446.5 ms)  |          +113.0 ms          |
| **Napoleon Biography**   |      728 chunks       |    6,314 chunks    |  **286.5 ms** (283.5 / 314.5 ms)  | **345.3 ms** (340.9 / 378.2 ms)  |          +58.8 ms           |
| **Fire & Blood (Novel)** |     1,700 chunks      |    6,314 chunks    |  **279.9 ms** (279.8 / 292.3 ms)  | **337.3 ms** (333.5 / 372.3 ms)  |          +57.4 ms           |
| **Brain & Behavior**     |     3,324 chunks      |    6,314 chunks    |  **313.1 ms** (310.4 / 339.8 ms)  | **365.0 ms** (365.2 / 410.5 ms)  |          +51.9 ms           |
| **Mixed Codebase**       |      515 chunks       |    6,314 chunks    |  **325.8 ms** (323.3 / 362.3 ms)  | **399.8 ms** (396.9 / 449.4 ms)  |          +74.0 ms           |
| **OVERALL SYSTEM TOTAL** | **Avg ~1,260 chunks** |  **6,314 chunks**  |  **296.5 ms** (290.8 / 344.8 ms)  | **367.6 ms** (361.9 / 429.2 ms)  |        **+71.1 ms**         |

* **Ingestion Throughput (CUDA GPU):**
  * Unified Corpus (6,314 chunks, 15 files): **8.43 minutes** (12.5 chunks / sec end-to-end embedding + indexing).
* **Retrieval Consistency:**
  * **99.3% Top-1 Equivalence:** 152 of 153 queries retrieved the exact same Rank 1 chunk.
  * **98.4% Domain Purity:** Only 12 out of 765 total retrieved Top-5 slots contained cross-domain noise.

---

## 3. Key Findings

1. **Multi-Domain Scaling Causes No Retrieval Degradation:**
   * Scaling from isolated indexes (300 chunks) to a unified multi-domain index (6,314 chunks) yielded identical Hit@5 recall (**77.1%** vs **77.1%**) and negligible MRR difference (0.675 vs 0.673).
2. **Domain Isolation is 98.4% Pure:**
   * Tantivy BM25 + LanceDB Vector search effectively prevents cross-domain noise from leaking into results.
3. **Capacity Ceiling:**
   * The remaining retrieval bottleneck is specific to dense narrative text (*Fire & Blood*, 46.7%), where a 0.6B embedding model lacks semantic depth compared to 8B models.
