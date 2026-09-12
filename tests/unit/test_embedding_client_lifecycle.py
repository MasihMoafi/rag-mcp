from types import SimpleNamespace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest

from rag.core import RAGPipeline
from rag.embeddings import OpenAICompatibleWrapper
from rag.query_lifecycle import QueryLifecycle
from rag import core
from rag import index_lifecycle


def test_closed_pipeline_does_not_retain_models_or_indexes():
    import weakref

    class Model:
        pass

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._queries = QueryLifecycle()
    pipeline.backend_type = 'lancedb'
    pipeline.embedding_fn = None
    references = []
    for name in ('reranker', 'reranker_model', 'reranker_tokenizer',
                 'embedding_fn', 'vector_db', 'bm25_index'):
        model = Model()
        references.append(weakref.ref(model))
        setattr(pipeline, name, model)
    del model
    assert all(reference() is not None for reference in references)
    pipeline.close()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize('startup_fails', [False, True])
@pytest.mark.parametrize('injected', [False, True])
def test_contextual_client_ownership(monkeypatch, tmp_path, startup_fails, injected):
    from unittest.mock import Mock
    from rag.contextual_retrieval import AnthropicContextualizer

    client = Mock()
    monkeypatch.setattr(AnthropicContextualizer, '_build_client', staticmethod(lambda timeout: client))
    context = AnthropicContextualizer(model='unused', max_tokens=8, timeout=1)
    monkeypatch.setattr(core, 'AnthropicContextualizer', lambda **kwargs: context)
    failure = RuntimeError('embedding setup failed')

    def embedder(*args, **kwargs):
        if startup_fails:
            raise failure
        return None

    monkeypatch.setattr(core, 'build_embedder', embedder)
    monkeypatch.setattr(RAGPipeline, '_load_or_create_database', lambda self: None)
    config = dict(persist_dir=str(tmp_path), _staged_index=True, use_contextual=True,
                  embed_provider='none', reranker_type='disabled', backend='lancedb')
    if injected:
        config['contextualizer'] = context
    if startup_fails:
        with pytest.raises(RuntimeError) as caught:
            RAGPipeline(config)
        assert caught.value is failure
    else:
        pipeline = RAGPipeline(config)
        pipeline.close()
        pipeline.close()
    assert client.close.call_count == (0 if injected else 1)


def test_contextual_cleanup_still_runs_when_embedding_close_fails():
    from unittest.mock import Mock

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.backend_type = 'lancedb'
    pipeline.embedding_fn = Mock()
    pipeline.embedding_fn.close.side_effect = RuntimeError('embedding close failed')
    pipeline._owned_contextualizer = Mock()
    with pytest.raises(RuntimeError, match='embedding close failed'):
        pipeline._release_resources()
    pipeline._owned_contextualizer.close.assert_called_once_with()


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


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_failed_publication_closes_client_and_preserves_original_error(cleanup_fails, monkeypatch, tmp_path):
    embedder = OpenAICompatibleWrapper('unused-model', api_key='unused-test-key')
    failure = OSError('publication failed')
    old = tmp_path / 'generation-old'
    old.mkdir()
    pointer = tmp_path / 'active-index.json'
    pointer.write_text('"generation-old"')
    built = []

    def build(config):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.config = config
        pipeline._queries = QueryLifecycle()
        pipeline.backend_type = 'lancedb'
        pipeline.embedding_fn = embedder
        pipeline.vector_db = None
        built.append(pipeline)
        return pipeline

    def fail(*args):
        raise failure

    if cleanup_fails:
        def close_then_fail():
            embedder.client.close()
            raise RuntimeError('cleanup failed')
        monkeypatch.setattr(embedder, 'close', close_then_fail)
    monkeypatch.setattr(index_lifecycle, 'write_json_atomic', fail)
    try:
        with pytest.raises(OSError) as caught:
            index_lifecycle.open_or_build(dict(persist_dir=str(tmp_path), force_reindex=True), build)
        assert caught.value is failure
        assert embedder.client.is_closed()
        assert pointer.read_text() == '"generation-old"'
        assert old.is_dir()
        assert Path(built[0].config['persist_dir']).exists() is cleanup_fails
        if cleanup_fails:
            assert caught.value.__notes__
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
