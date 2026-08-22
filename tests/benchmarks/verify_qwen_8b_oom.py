import os
import requests
import json
import time

# Clean proxy environment for local Ollama
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if var in os.environ:
        del os.environ[var]

def test_qwen_8b_batch_ladder():
    print("=" * 70)
    print("OLLAMA QWEN3-EMBEDDING:8B BATCH STRESS TEST")
    print("=" * 70)

    url = "http://localhost:11434/api/embed"
    sample_text = "This is a benchmark sample document chunk representing typical technical RAG text passages."
    
    # Testing increasing batch sizes of prompts sent to qwen3-embedding:8b
    ladder = [1024, 512, 256, 128, 64, 32, 16]

    for batch_size in ladder:
        print(f"\nSending Batch Size: {batch_size:4d} chunks to qwen3-embedding:8b ... ", end="", flush=True)
        payload = {
            "model": "qwen3-embedding:8b",
            "input": [sample_text] * batch_size
        }
        
        start = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=120)
            elapsed = time.time() - start
            if resp.status_code == 200:
                print(f"✓ SUCCESS in {elapsed:.2f}s!")
            else:
                print(f"✗ FAILED: Status {resp.status_code} - {resp.text[:100]}")
        except Exception as e:
            print(f"✗ ERROR: {e}")

    print("\n" + "=" * 70)
    print("CONCLUSION: Ollama qwen3-embedding:8b batching tested!")
    print("=" * 70)

if __name__ == "__main__":
    test_qwen_8b_batch_ladder()
