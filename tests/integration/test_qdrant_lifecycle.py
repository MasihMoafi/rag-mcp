"""Local Qdrant persistence checks with controlled vectors, no model calls."""

from rag.qdrant_backend import QdrantVectorDB
from rag import core, fetch
from pathlib import Path
import pytest


class Embedder:
    def embed_documents(self, texts):
        return [[1., 0.] if 'Cedar' in text else [0., 1.] for text in texts]

    def embed_query(self, text):
        return [1., 0.]


@pytest.mark.parametrize('backend', ['lancedb', 'qdrant'])
def test_repeated_cache_churn_preserves_each_scope_without_reindex(tmp_path, monkeypatch, backend):
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'none')
    monkeypatch.setenv('RAG_MCP_RERANKER_TYPE', 'off')
    monkeypatch.setenv('RAG_MCP_CACHE_SIZE', '2')
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, 'backend', backend)
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', {})
    monkeypatch.setattr(Embedder, 'model_name', 'fixture', raising=False)
    monkeypatch.setattr(core, 'build_embedder', lambda config, device: Embedder())
    sources = [tmp_path / f'note-{index}.txt' for index in range(6)]
    for index, source in enumerate(sources):
        source.write_text(f'Cedar evidence for scope {index}.')
    try:
        for source in sources:
            fetch.query_documents('Cedar', str(source))
        generations = sorted((tmp_path / 'index').glob('*/generation-*'))
        assert len(generations) == len(sources)
        def unexpected_reindex(*args):
            pytest.fail('Cache churn must not reindex unchanged source documents')
        monkeypatch.setattr(core.RAGPipeline, '_index_documents', unexpected_reindex)
        for _ in range(3):
            for index, source in enumerate(sources):
                result = fetch.query_documents('Cedar', str(source))
                assert result[0]['content'] == f'Cedar evidence for scope {index}.'
                assert result[0]['source'] == str(source)
                assert len(fetch._rag_system_v2_instances) <= 2
        assert sorted((tmp_path / 'index').glob('*/generation-*')) == generations
    finally:
        fetch.reset_pipelines()


def test_failed_collection_initialization_releases_storage_lock(tmp_path, monkeypatch):
    def fail_initialization(self):
        raise RuntimeError('collection initialization failed')
    with monkeypatch.context() as fault:
        fault.setattr(QdrantVectorDB, '_init_collection', fail_initialization)
        with pytest.raises(RuntimeError, match='collection initialization failed') as failure:
            QdrantVectorDB('recovery', Embedder(), vector_size=2, persist_path=str(tmp_path))
    # Keep the exception traceback alive: recovery must not depend on garbage collection.
    assert failure.value.__traceback__ is not None
    reopened = QdrantVectorDB('recovery', Embedder(), vector_size=2, persist_path=str(tmp_path))
    try:
        reopened.add_documents_batch([{'content': 'Cedar evidence'}])
        assert reopened.search('Cedar')[0]['content'] == 'Cedar evidence'
    finally:
        reopened.client.close()


@pytest.mark.parametrize('change', ['reset', 'settings', 'eviction'])
def test_pipeline_replacement_releases_storage_even_with_external_reference(tmp_path, monkeypatch, change):
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'none')
    monkeypatch.setenv('RAG_MCP_RERANKER_TYPE', 'off')
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, 'backend', 'qdrant')
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', {})
    monkeypatch.setattr(Embedder, 'model_name', 'fixture', raising=False)
    monkeypatch.setattr(core, 'build_embedder', lambda config, device: Embedder())
    source = tmp_path / 'note.txt'
    source.write_text('Cedar evidence')
    previous = fetch.get_rag_pipeline_v2(str(source))
    previous_storage = previous.vector_db
    original_add = QdrantVectorDB.add_documents_batch
    try:
        def unexpected_reindex(*args, **kwargs):
            pytest.fail('Reset or ranking-only changes must not re-embed documents')
        monkeypatch.setattr(QdrantVectorDB, 'add_documents_batch', unexpected_reindex)
        if change == 'reset':
            fetch.reset_pipelines()
        elif change == 'eviction':
            monkeypatch.setenv('RAG_MCP_CACHE_SIZE', '1')
            other = tmp_path / 'other.txt'
            other.write_text('Other evidence')
            # The first scope must release its lock even with storage held externally.
            monkeypatch.setattr(QdrantVectorDB, 'add_documents_batch', original_add)
            fetch.get_rag_pipeline_v2(str(other))
            assert previous._queries.closed
            monkeypatch.setattr(QdrantVectorDB, 'add_documents_batch', unexpected_reindex)
        else:
            monkeypatch.setitem(fetch.RAG_CONFIG_V2, 'rrf_k', 12)
        reopened = fetch.get_rag_pipeline_v2(str(source))
        try:
            assert reopened is not previous
            assert reopened.search('Cedar')[0]['content'] == 'Cedar evidence'
        finally:
            reopened.close()
    finally:
        previous_storage.client.close()
        previous.close()
        fetch.reset_pipelines()


def test_qdrant_retrieval_and_reopen_preserve_source_metadata(tmp_path):
    docs = [{'content': 'Cedar evidence', 'id': 'notes.md::chunk:0', 'location': 'L3'},
            {'content': 'Other evidence', 'id': 'other.md::chunk:0', 'location': 'L8'}]
    for reopen in [False, True]:
        backend = QdrantVectorDB('lifecycle', Embedder(), vector_size=2, persist_path=str(tmp_path))
        try:
            if not reopen:
                backend.add_documents_batch(docs, batch_size=1)
            assert backend.count() == 2
            result = backend.search('Cedar', top_k=1)[0]
            assert result['id'] == 'notes.md::chunk:0'
            assert result['location'] == 'L3'
        finally:
            backend.client.close()


def test_failed_qdrant_refresh_preserves_published_index_on_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr(Embedder, 'model_name', 'fixture', raising=False)
    monkeypatch.setattr(core, 'build_embedder', lambda config, device: Embedder())
    source = tmp_path / 'notes.md'
    source.write_text('Cedar evidence')
    config = dict(persist_dir=str(tmp_path / 'index'), backend='qdrant',
                  document_paths=[str(source)], reranker_type='off',
                  chunk_size=700, chunk_overlap=100, force_reindex=True)
    previous = core.RAGPipeline(config)
    previous.vector_db.client.close()
    old_metadata = (Path(previous.config['persist_dir']) / 'docs.json').read_bytes()
    source.write_text('Replacement evidence')

    def fail_embedding(self, texts):
        raise RuntimeError('provider unavailable')

    with monkeypatch.context() as fault:
        fault.setattr(QdrantVectorDB, '_batch_embed', fail_embedding)
        with pytest.raises(RuntimeError, match='provider unavailable'):
            core.RAGPipeline(config)

    reopened = core.RAGPipeline({**config, 'force_reindex': False})
    try:
        assert reopened.vector_db.count() == 1
        assert reopened.search('Cedar')[0]['content'] == 'Cedar evidence'
        assert (Path(previous.config['persist_dir']) / 'docs.json').read_bytes() == old_metadata
    finally:
        reopened.vector_db.client.close()
