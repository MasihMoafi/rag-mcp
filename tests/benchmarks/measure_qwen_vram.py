import os
import time
import requests
import json

# Clean proxy environment for local Ollama communication
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if var in os.environ:
        del os.environ[var]

import torch

def print_gpu_vram(label: str):
    """Print current GPU memory usage from PyTorch perspective."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"\n[{label}] (PyTorch)")
        print(f"  Allocated VRAM: {allocated:.2f} MB")
        print(f"  Reserved VRAM:  {reserved:.2f} MB")

def run_test_a_qwen():
    print("=" * 65)
    print("TEST A: STATIC VRAM FOR LOCAL QWEN3-EMBEDDING:8B VIA OLLAMA")
    print("=" * 65)

    print("\n--> Sending warm-up embedding request to Ollama (qwen3-embedding:8b)...")
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "qwen3-embedding:8b",
        "prompt": "Testing static VRAM memory allocation."
    }

    try:
        start = time.time()
        resp = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start
        if resp.status_code == 200:
            print(f"✓ Ollama response received in {elapsed:.2f}s!")
            print("--> Model is now fully loaded into GPU VRAM by Ollama runtime.")
            print("\nCheck nvidia-smi now to observe the 4.7 GB allocation!")
        else:
            print(f"✗ Ollama error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"✗ Failed to connect to Ollama: {e}")

if __name__ == "__main__":
    run_test_a_qwen()
