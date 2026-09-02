from unittest.mock import patch

from rag.core import select_device


def test_select_device_falls_back_without_changing_the_machine():
    with patch("rag.core.torch.cuda.is_available", return_value=False), patch(
        "rag.core.torch.backends.mps.is_available", return_value=False
    ):
        assert select_device() == "cpu"

    with patch("rag.core.torch.cuda.is_available", return_value=True):
        assert select_device() == "cuda"
