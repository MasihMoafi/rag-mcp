import unittest
from unittest.mock import patch
import torch


def ensure_gpu_for_reranker():
    """Enforce strict CUDA GPU availability specifically for the Reranker stage."""
    if not torch.cuda.is_available():
        import subprocess
        subprocess.run(["pkexec", "prime-select", "nvidia"])
        if not torch.cuda.is_available():
            raise RuntimeError("Dedicated GPU (CUDA) is not available for reranker stage. Please verify NVIDIA driver status.")
    return "cuda"


class TestRerankerGPU(unittest.TestCase):
    @patch("torch.cuda.is_available", return_value=False)
    @patch("subprocess.run")
    def test_cuda_unavailable_raises_runtime_error(self, mock_subp, mock_cuda):
        with self.assertRaises(RuntimeError) as ctx:
            ensure_gpu_for_reranker()
        self.assertIn("Dedicated GPU (CUDA) is not available for reranker stage", str(ctx.exception))
        mock_subp.assert_called_once_with(["pkexec", "prime-select", "nvidia"])


if __name__ == "__main__":
    unittest.main()
