import os
import requests
import time

# Clean proxy environment
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if var in os.environ:
        del os.environ[var]

def benchmark_real_gpu_qwen():
    print("=" * 70)
    print("REAL GPU BENCHMARK: MEASURING THROUGHPUT ON QWEN3-EMBEDDING:0.6B")
    print("=" * 70)

    url = "http://localhost:11434/api/embed"
    sample_text = "This is a benchmark sample document chunk representing typical technical RAG text passages."
    
    # Test batch sizes on true GPU
    test_sizes = [1, 4, 8, 16, 32, 64, 128, 256]

    for batch_size in test_sizes:
        payload = {
            "model": "qwen3-embedding:0.6b",
            "input": [sample_text] * batch_size
        }
        
        start = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=60)
            elapsed = time.time() - start
            if resp.status_code == 200:
                chunks_per_sec = batch_size / elapsed
                print(f"Batch Size: {batch_size:3d} | Time: {elapsed:6.3f}s | Throughput: {chunks_per_sec:7.2f} chunks/sec")
            else:
                print(f"Batch Size: {batch_size:3d} | Failed with HTTP {resp.status_code}")
        except Exception as e:
            print(f"Batch Size: {batch_size:3d} | Error: {e}")

    print("=" * 70)

if __name__ == "__main__":
    benchmark_real_gpu_qwen()
