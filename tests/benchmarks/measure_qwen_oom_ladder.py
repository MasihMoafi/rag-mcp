import os
import requests
import time

# Clean proxy environment
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if var in os.environ:
        del os.environ[var]

def test_qwen_gpu_oom_ladder():
    print("=" * 70)
    print("OLLAMA QWEN3-EMBEDDING:0.6B GPU OOM LADDER BENCHMARK")
    print("=" * 70)

    url = "http://localhost:11434/api/embed"
    sample_text = "This is a benchmark sample document chunk representing typical technical RAG text passages."
    
    # Scale batch size upward until OOM / error occurs
    ladder = [256, 512, 1024, 2048, 4096, 8192, 16384]

    for batch_size in ladder:
        payload = {
            "model": "qwen3-embedding:0.6b",
            "input": [sample_text] * batch_size
        }
        
        print(f"\nTesting Batch Size: {batch_size:5d} chunks ... ", end="", flush=True)
        start = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=120)
            elapsed = time.time() - start
            if resp.status_code == 200:
                throughput = batch_size / elapsed
                print(f"✓ SUCCESS in {elapsed:6.2f}s | Throughput: {throughput:7.2f} chunks/sec")
            else:
                print(f"✗ FAILED: HTTP {resp.status_code} - {resp.text[:120]}")
                break
        except Exception as e:
            print(f"✗ ERROR: {e}")
            break

    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_qwen_gpu_oom_ladder()
