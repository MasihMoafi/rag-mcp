# Experiment: True GPU Empirical Batch Size & Throughput Benchmark

**Date:** 2026-08-22  
**Model:** `qwen3-embedding:0.6b` (100% CUDA GPU Enforced via Ollama)  
**Hardware:** NVIDIA RTX 3070 Laptop GPU (8GB VRAM)  
**Status:** Verified on Hardware  

---

## 1. Objective
Measure the true GPU hardware throughput (chunks/sec) across batch sizes to identify the exact empirical peak for LanceDB document ingestion.

---

## 2. Experimental Data (True GPU Execution)

| Batch Size ($B$) | Wall-Clock Time (s) | Throughput ($\text{chunks/sec}$) | State / Analysis |
| :--- | :--- | :--- | :--- |
| **1** | 0.290s | **3.44** | Under-utilizing GPU cores; overhead-bound |
| **4** | 0.212s | **18.88** | Tensor cores engaging |
| **8** | 0.242s | **33.12** | Steady throughput ramp |
| **16** | 0.297s | **53.86** | Fast sub-second response |
| **32** | 0.405s | **79.01** | High efficiency |
| **64** | 0.606s | **105.64** | Triple-digit throughput threshold |
| **128** | 0.981s | **130.52** | High saturation |
| **256** | 1.840s | **138.93** | Tensor cores fully saturated |
| **512** | **3.350s** | **153.04** | **PEAK EMPIRICAL HARDWARE THROUGHPUT** |
| **1024** | 6.750s | **151.61** | Plateaus (Memory overhead starts) |
| **2048** | 14.480s | **141.40** | Diminishing return |

---

## 3. Conclusions & Production Setting
1. **The Peak Hardware Point:** **Batch Size 512** delivers the absolute highest chunk processing speed (**$153.04\text{ chunks/sec}$** vs. $3.44\text{ chunks/sec}$ at $B=1$).
2. **Safe High-Speed Production Setting:** **Batch Size 256 or 512** enables ingesting all 6,314 chunks into LanceDB in **~41 seconds** on GPU.
3. **Hardware Health:** 
   - Memory allocation: `2,445 MiB / 8,192 MiB` (VRAM buffer intact).
   - Operating temperature: `62°C` (cool, well below 70°C).
