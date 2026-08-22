import os
import requests
import time

# Clean proxy environment
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if var in os.environ:
        del os.environ[var]

def benchmark_qwen_batch_sizes():
    print("=" * 65)
    print("MEASURING EMPIRICAL THROUGHPUT (CHUNKS / SEC) ON QWEN3-EMBEDDING:8B")
    print("=" * 65)

    url = "http://localhost:11434/api/embed"
    sample_text = "This is a benchmark sample document chunk representing typical technical RAG text passages."
    
    # Test realistic batch sizes to measure actual chunks/sec throughput
    test_sizes = [1, 4, 8, 16, 32, 64]

    for batch_size in test_sizes:
        payload = {
            "model": "qwen3-embedding:8b",
            "input": [sample_text] * batch_size
        }
        
        start = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=30)
            elapsed = time.time() - start
            if resp.status_code == 200:
                chunks_per_sec = batch_size / elapsed
                print(f"Batch Size: {batch_size:2d} | Time: {elapsed:5.2f}s | Throughput: {chunks_per_sec:6.2f} chunks/sec")
            else:
                print(f"Batch Size: {batch_size:2d} | Failed with HTTP {resp.status_code}")
        except Exception as e:
            print(f"Batch Size: {batch_size:2d} | Error: {e}")

    print("=" * 65)

if __name__ == "__main__":
    benchmark_qwen_batch_sizes()
