"""Application-owned retrieval logs must not echo user/provider payloads."""
from unittest.mock import MagicMock, Mock

import pytest

from rag.core import RAGPipeline


@pytest.mark.parametrize('operation', ['batch', 'query'])
def test_ollama_status_failure_is_private_and_does_not_fan_out(operation):
    from rag.embeddings import DirectOllamaGPUWrapper
    from rag.lancedb_backend import LanceDBBackend

    embedder = DirectOllamaGPUWrapper('fixture', 'http://localhost:11434')
    embedder._requests = Mock()
    embedder._requests.post.return_value = Mock(status_code=503, text='private-provider-marker')
    backend = LanceDBBackend.__new__(LanceDBBackend)
    backend.embedding_fn = embedder
    with pytest.raises(RuntimeError) as caught:
        if operation == 'batch':
            backend._batch_embed(['private-document-marker'] * 10)
        else:
            embedder.embed_query('private-query-marker')
    assert embedder._requests.post.call_count == 1
    assert '503' in str(caught.value)
    assert 'private-' not in str(caught.value)


@pytest.mark.parametrize("operation", ["init", "embed", "fts", "create_fts", "clear"])
def test_storage_errors_omit_payloads_and_preserve_failure_contract(capsys, operation):
    from rag.lancedb_backend import LanceDBBackend

    backend = LanceDBBackend.__new__(LanceDBBackend)
    backend.db = Mock()
    backend.table = MagicMock()
    backend.table.__len__.return_value = 1
    backend.table_name = "fixture"
    backend.embedding_fn = Mock()
    error = RuntimeError("private-storage-response-marker")
    if operation == "init":
        backend.db.table_names.side_effect = error
        action = backend._init_table
    elif operation == "embed":
        backend.embedding_fn.embed_documents.side_effect = error
        backend.embedding_fn.return_value = [[1.0]]
        action = lambda: backend._batch_embed(["private-document-marker"])
    elif operation == "fts":
        backend.table.search.side_effect = error
        action = lambda: backend.search_fts("private-query-marker")
    elif operation == "create_fts":
        backend.embedding_fn = None
        backend.db.create_table.return_value = backend.table
        backend.table.create_fts_index.side_effect = error
        action = lambda: backend.add_documents_batch([{"id": "1", "content": "private-document-marker"}])
    else:
        backend.db.drop_table.side_effect = error
        action = backend.clear
    if operation in {"init", "clear"}:
        previous_table = backend.table
        action()
        assert backend.table is previous_table
    elif operation == "embed":
        assert action() == [[1.0]]
        backend.embedding_fn.assert_called_once_with(["private-document-marker"])
    else:
        with pytest.raises(RuntimeError) as raised:
            action()
        assert raised.value is error
    captured = capsys.readouterr()
    logs = captured.out + captured.err
    assert "private-" not in logs
    assert "RuntimeError" in logs


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
