import os
import unittest
import torch

# Clean proxy environment for local execution
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if var in os.environ:
        del os.environ[var]


def simulate_large_batch_embedding(batch_size: int = 4096, seq_len: int = 512, dim: int = 4096):
    """
    Stress-test GPU VRAM with an intentionally massive tensor batch.
    Enforces pure CUDA execution with zero CPU offload.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this stress test.")

    # Calculate tensor memory size
    # 4096 * 512 * 4096 elements in FP32 (4 bytes) = 34.35 GB VRAM required!
    # This mathematically guarantees an OOM on an 8GB RTX 3070.
    tensor = torch.empty((batch_size, seq_len, dim), dtype=torch.float32, device="cuda")
    return tensor


class TestLargeBatchOOM(unittest.TestCase):
    def test_massive_batch_raises_cuda_oom(self):
        """Verify that a 4096 batch size triggers OutOfMemoryError on GPU without CPU fallback."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA GPU not active.")

        with self.assertRaises(torch.cuda.OutOfMemoryError):
            simulate_large_batch_embedding(batch_size=4096, seq_len=512, dim=4096)


if __name__ == "__main__":
    unittest.main()
