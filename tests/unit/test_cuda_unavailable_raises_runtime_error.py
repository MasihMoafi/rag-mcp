import unittest
from unittest.mock import patch, MagicMock
import torch


def ensure_gpu_for_pipeline(stage_name: str = "retrieval"):
    """Enforce strict CUDA GPU availability for embedding and reranking stages."""
    if not torch.cuda.is_available():
        import subprocess
        subprocess.run(["pkexec", "prime-select", "nvidia"])
        if not torch.cuda.is_available():
            raise RuntimeError(f"Dedicated GPU (CUDA) is not available for {stage_name} stage. Please verify NVIDIA driver status.")
    return "cuda"


class TestGPUErrors(unittest.TestCase):
    @patch("torch.cuda.is_available", return_value=False)
    @patch("subprocess.run")
    def test_cuda_unavailable_raises_runtime_error_for_retriever(self, mock_subp, mock_cuda):
        with self.assertRaises(RuntimeError) as ctx:
            ensure_gpu_for_pipeline(stage_name="retrieval")
        self.assertIn("Dedicated GPU (CUDA) is not available for retrieval stage", str(ctx.exception))
        mock_subp.assert_called_once_with(["pkexec", "prime-select", "nvidia"])

    @patch("torch.cuda.is_available", return_value=False)
    @patch("subprocess.run")
    def test_cuda_unavailable_raises_runtime_error_for_reranker(self, mock_subp, mock_cuda):
        with self.assertRaises(RuntimeError) as ctx:
            ensure_gpu_for_pipeline(stage_name="reranker")
        self.assertIn("Dedicated GPU (CUDA) is not available for reranker stage", str(ctx.exception))
        mock_subp.assert_called_once_with(["pkexec", "prime-select", "nvidia"])


if __name__ == "__main__":
    unittest.main()
