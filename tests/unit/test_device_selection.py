from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag.core import select_device


def test_select_device_falls_back_without_changing_the_machine(monkeypatch):
    from rag import core

    cuda = Mock(return_value=False)
    mps = Mock(return_value=False)
    monkeypatch.setattr(core, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=mps)),
    ))
    assert select_device() == "cpu"
    cuda.return_value = True
    assert select_device() == "cuda"
    cuda.return_value = False
    mps.return_value = True
    assert select_device() == "mps"
    monkeypatch.setattr(core, "torch", None)
    assert select_device() == "cpu"


def test_remote_only_pipeline_does_not_probe_local_accelerators(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from rag import core

    def unexpected_probe():
        raise AssertionError("Hosted retrieval must not probe CUDA or MPS")

    monkeypatch.setattr(core, "select_device", unexpected_probe)
    monkeypatch.setattr(core, "build_embedder", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "LanceDBBackend", Mock())
    monkeypatch.setattr(core.RAGPipeline, "_load_or_create_database", lambda self: None)
    pipeline = core.RAGPipeline({"_staged_index": True, "persist_dir": str(tmp_path),
                                 "embed_provider": "openai_compatible", "reranker_type": "off"})
    assert pipeline.device == "cpu"


@pytest.mark.parametrize("reranker", ["off", "none", "disabled", "openrouter"])
def test_remote_only_device_selection_skips_local_probe(monkeypatch, reranker):
    from rag import core

    monkeypatch.setattr(core, "select_device", lambda: pytest.fail("Local accelerator probe ran"))
    assert core.device_for_config({"embed_provider": "openai_compatible",
                                   "reranker_type": reranker}) == "cpu"


@pytest.mark.parametrize("embedding,reranker", [("openai_compatible", "cross-encoder"),
                                                  ("fastembed", "off")])
def test_local_model_keeps_local_device_selection(monkeypatch, embedding, reranker):
    from rag import core

    monkeypatch.setattr(core, "select_device", lambda: "cuda")
    assert core.device_for_config({"embed_provider": embedding,
                                   "reranker_type": reranker}) == "cuda"
