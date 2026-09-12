from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest

from rag.core import RAGPipeline
from rag.embeddings import OpenAICompatibleWrapper
from rag.query_lifecycle import QueryLifecycle
from rag import core


def test_pipeline_close_releases_actual_sdk_client_without_network():
    embedder = OpenAICompatibleWrapper('unused-model', api_key='unused-test-key')
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._queries = QueryLifecycle()
    pipeline.backend_type = 'lancedb'
    pipeline.embedding_fn = embedder
    pipeline.vector_db = SimpleNamespace()
    try:
        assert not embedder.client.is_closed()
        pipeline.close()
        assert embedder.client.is_closed()
        pipeline.close()
    finally:
        embedder.client.close()


def test_api_client_stays_open_until_active_search_finishes():
    embedder = OpenAICompatibleWrapper('unused-model', api_key='unused-test-key')
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._queries = QueryLifecycle()
    pipeline.backend_type = 'lancedb'
    pipeline.embedding_fn = embedder
    entered, finish = Event(), Event()

    def search(query):
        entered.set()
        assert finish.wait(5)
        assert not embedder.client.is_closed()
        return []

    pipeline._search = search
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            searching = workers.submit(pipeline.search, 'query')
            try:
                assert entered.wait(5)
                closing = workers.submit(pipeline.close)
                with pytest.raises(TimeoutError):
                    closing.result(timeout=0.05)
                assert not embedder.client.is_closed()
            finally:
                finish.set()
            searching.result(timeout=5)
            closing.result(timeout=5)
        assert embedder.client.is_closed()
    finally:
        embedder.client.close()


@pytest.mark.parametrize('stage', ['probe', 'storage', 'reranker', 'index'])
@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_failed_startup_releases_embedding_client(stage, cleanup_fails, monkeypatch, tmp_path):
    embedder = OpenAICompatibleWrapper('unused-model', api_key='unused-test-key')
    failure = RuntimeError(f'{stage} failed')

    if cleanup_fails:
        def close_then_fail():
            embedder.client.close()
            raise OSError('cleanup failed')
        monkeypatch.setattr(embedder, 'close', close_then_fail)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(core, 'build_embedder', lambda *a, **k: embedder)
    monkeypatch.setattr(core, 'select_device', lambda: 'cpu')
    monkeypatch.setattr(core, 'probe_dimensions', fail if stage == 'probe' else lambda e: 3)
    if stage == 'storage':
        monkeypatch.setattr(core, 'LanceDBBackend', fail)
    elif stage == 'reranker':
        monkeypatch.setattr(core.reranking, 'initialize_reranker', fail)
    elif stage == 'index':
        monkeypatch.setattr(RAGPipeline, '_load_or_create_database', fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            RAGPipeline(dict(persist_dir=str(tmp_path/'index'), document_paths=[],
                             backend='lancedb', embed_provider='openai_compatible',
                             reranker_type='off'))
        assert caught.value is failure
        assert embedder.client.is_closed()
    finally:
        embedder.client.close()
