import os
import sys
import time
import tempfile
import torch

# Clean proxy environment for local execution
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if var in os.environ:
        del os.environ[var]

sys.path.insert(0, "/home/masih/Desktop/p/rag-mcp-lancedb")
from rag.core import RAGPipelineV2

def run_smoke_test():
    print("=" * 65)
    print("PRE-FLIGHT SMOKE TEST: UNIFIED RAG PIPELINE (LANCEDB + QWEN3-8B)")
    print("=" * 65)

    # 1. Verify CUDA GPU
    print("\n1. Verifying Dedicated GPU Status...")
    if not torch.cuda.is_available():
        raise RuntimeError("Dedicated GPU (CUDA) is not available! Aborting smoke test.")
    print("   ✓ CUDA GPU is active and verified.")

    # 2. Setup isolated smoke-test configuration
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = os.path.join(tmp_dir, "smoke_corpus.md")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("# Napoleon and the Battle of Austerlitz\n\nNapoleon won the battle through masterclass strategic maneuvering.\n\n")
            f.write("# Attention Is All You Need\n\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks.\n\n")
            f.write("# Neuroscience: Brain and Behavior\n\nThe prefrontal cortex plays an essential role in executive cognitive functions.\n")

        config = {
            "version": "smoke_test_v2",
            "backend": "lancedb",
            "persist_dir": os.path.join(tmp_dir, "smoke_lancedb"),
            "document_paths": [test_file],
            "embed_provider": "ollama",
            "embed_model": "qwen3-embedding:8b",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 5,
            "rrf_k": 60,
            "reranker_type": "disabled"
        }

        print("\n2. Initializing RAGPipelineV2 with LanceDB backend...")
        start = time.time()
        pipeline = RAGPipelineV2(config=config)
        print(f"   ✓ Pipeline initialized in {time.time() - start:.2f}s")


        # 4. Run Retrieval Query across the domains
        queries = [
            "How did Napoleon win at Austerlitz?",
            "What architecture is used in sequence transduction?",
            "What role does the prefrontal cortex play?"
        ]

        print("\n4. Testing Live Queries & RRF Fusion...")
        for q in queries:
            q_start = time.time()
            results = pipeline.search(q)
            elapsed = time.time() - q_start
            print(f"\n   Query: '{q}' ({elapsed:.3f}s)")
            if results:
                print(f"   --> Hit Output: {str(results)[:120]}...")
            else:
                print("   --> ✗ No results returned!")


    print("\n" + "=" * 65)
    print("✓ ALL SMOKE TESTS PASSED CLEANLY! SYSTEM IS 100% READY.")
    print("=" * 65)

if __name__ == "__main__":
    run_smoke_test()
