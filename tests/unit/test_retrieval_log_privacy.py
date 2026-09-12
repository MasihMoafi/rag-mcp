"""Application-owned retrieval logs must not echo user/provider payloads."""
from unittest.mock import Mock

import pytest

from rag.core import RAGPipeline


def test_fetch_failure_logs_omit_chained_provider_bodies(monkeypatch, capsys):
    from rag import fetch

    cause = ValueError("private-response-cause")
    error = RuntimeError("private-provider-response-marker")
    error.__cause__ = cause
    pipeline = Mock()
    pipeline.search.side_effect = error
    monkeypatch.setattr(fetch, "get_rag_pipeline_v2", Mock(return_value=pipeline))
    with pytest.raises(RuntimeError) as raised:
        fetch.query_documents("private-query-marker")
    assert raised.value is error
    assert raised.value.__cause__ is cause
    pipeline.search.assert_called_once_with("private-query-marker")
    captured = capsys.readouterr()
    assert "private-" not in captured.out + captured.err


@pytest.mark.parametrize("failure", [False, True])
def test_search_logs_omit_payloads_without_changing_results_or_errors(capsys, failure):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.config = {"reranker_type": "disabled", "rerank_top_k": 1}
    pipeline.backend_type = "lancedb"
    pipeline.embedding_fn = None
    pipeline.vector_db = Mock()
    query = "private-query-marker"
    error = RuntimeError("private-provider-response-marker")
    documents = [{"id": "1", "content": "private-document-marker", "score": 1}]
    pipeline.vector_db.search_fts.return_value = documents
    if failure:
        pipeline.vector_db.search_fts.side_effect = error
        with pytest.raises(RuntimeError) as raised:
            pipeline._search(query)
        assert raised.value is error
    else:
        assert pipeline._search(query) == documents
    pipeline.vector_db.search_fts.assert_called_once_with(query, top_k=20)
    captured = capsys.readouterr()
    logs = captured.out + captured.err
    for marker in (query, str(error), documents[0]["content"]):
        assert marker not in logs
    if failure:
        assert "RuntimeError" in logs


def test_reranker_failure_logs_omit_provider_body_and_preserve_fallback(capsys):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.reranker_type = "cross-encoder"
    pipeline.reranker = Mock()
    pipeline.reranker.predict.side_effect = RuntimeError("private-provider-response-marker")
    documents = [{"content": "private-document-marker"}, {"content": "second"}]
    assert pipeline._crossencoder_rerank(documents, "private-query-marker", 1) == documents[:1]
    captured = capsys.readouterr()
    logs = captured.out + captured.err
    assert "private-" not in logs
    assert "RuntimeError" in logs
