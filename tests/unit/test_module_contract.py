"""Behavioral seams protected while separating engine responsibilities."""

from unittest.mock import Mock
from types import SimpleNamespace
import sys

import pytest

from rag.core import RAGPipelineV2


@pytest.mark.parametrize("setting,model", [("off", ""), ("auto", ""), ("fastembed", ""), ("cross-encoder", "custom"), ("cross-encoder", "Qwen3-Reranker")])
def test_reranker_initialization_contract(monkeypatch, tmp_path, setting, model):
    from rag import core, embeddings

    cross = Mock()
    fast = Mock()
    tokenizer = Mock()
    causal = Mock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None if setting == "auto" else SimpleNamespace(CrossEncoder=cross))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer, AutoModelForCausalLM=causal))
    monkeypatch.setattr(embeddings, "FastEmbedReranker", fast)
    monkeypatch.setattr(core, "select_device", lambda: "cpu")
    monkeypatch.setattr(core, "build_embedder", lambda *a, **k: None)
    monkeypatch.setattr(core, "LanceDBBackend", Mock())
    monkeypatch.setattr(core.RAGPipeline, "_load_or_create_database", Mock())
    pipeline = core.RAGPipeline({"persist_dir": str(tmp_path), "reranker_type": setting, "reranker_model": model})
    assert pipeline.device == "cpu"
    if setting in {"off", "auto"}:
        assert pipeline.reranker is None
        cross.assert_not_called()
        fast.assert_not_called()
    elif setting == "fastembed":
        fast.assert_called_once_with("Xenova/ms-marco-MiniLM-L-6-v2")
        assert pipeline.reranker is fast.return_value
    elif model == "custom":
        cross.assert_called_once_with("custom", device="cpu")
        assert pipeline.reranker is cross.return_value
    else:
        tokenizer.from_pretrained.assert_called_once_with(model, padding_side="left", local_files_only=True)
        causal.from_pretrained.assert_called_once_with(model, torch_dtype=core.torch.float16, device_map="cuda", local_files_only=True)
        assert pipeline.reranker_type == "qwen3"
        assert pipeline.reranker_model is causal.from_pretrained.return_value.eval.return_value


def test_legacy_imports_and_bound_methods_use_canonical_implementations():
    from rag import bm25, chunking, core, ingestion, reranking

    assert core.BM25Index is bm25.BM25Index
    assert core.RAGPipelineV2 is core.RAGPipeline
    assert core.chunk_text is chunking.chunk_text
    assert core.chunk_document is chunking.chunk_document
    assert core.RAGPipelineV2._index_documents is ingestion.index_documents
    assert core.RAGPipelineV2._crossencoder_rerank is reranking.rerank_documents


def test_pipeline_class_supports_legacy_pickle_reference():
    import pickle
    from rag.core import RAGPipeline

    assert pickle.loads(b"crag.core\nRAGPipelineV2\n.") is RAGPipeline
    instance = RAGPipeline.__new__(RAGPipeline)
    instance.config = {"backend": "lancedb"}
    restored = pickle.loads(pickle.dumps(instance))
    assert type(restored) is RAGPipeline
    assert restored.config == instance.config


@pytest.mark.parametrize("failure", [False, True])
def test_reranking_preserves_payload_limit_order_and_fallback(failure):
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.reranker_type = "cross-encoder"
    pipeline.reranker = Mock()
    pipeline.reranker.predict.return_value = [0.1, 0.9, 0.5]
    if failure:
        pipeline.reranker.predict.side_effect = RuntimeError("unavailable")
    docs = [{"id": i, "content": str(i) * 600} for i in range(3)]
    result = pipeline._crossencoder_rerank(docs, "query", 2)
    assert result == (docs[:2] if failure else [docs[1], docs[2]])
    pipeline.reranker.predict.assert_called_once_with([["query", doc["content"][:512]] for doc in docs])


def test_reranking_short_results_do_not_call_model():
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.reranker = Mock()
    docs = [{"content": "source"}]
    assert pipeline._crossencoder_rerank(docs, "query", 2) is docs
    pipeline.reranker.predict.assert_not_called()


def test_database_dispatch_preserves_instance_overrides(tmp_path):
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.backend_type = "qdrant"
    pipeline.config = {"persist_dir": str(tmp_path)}
    pipeline._load_existing_database = Mock()
    pipeline._index_documents = Mock()
    pipeline.vector_db = Mock()
    pipeline._load_or_create_database()
    pipeline._load_existing_database.assert_called_once_with()
    pipeline._index_documents.assert_not_called()
    pipeline.config["force_reindex"] = True
    pipeline.vector_db.clear.side_effect = RuntimeError("clear failed")
    pipeline._load_or_create_database()
    pipeline._index_documents.assert_called_once_with()
