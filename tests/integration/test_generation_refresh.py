"""Published CLI/MCP indexes survive failures throughout a replacement build."""

import json
import inspect
from pathlib import Path
import subprocess
import sys

import pytest

from rag import core, fetch, ingestion, index_lifecycle


@pytest.mark.parametrize('phase', ['storage', 'before', 'after'])
@pytest.mark.parametrize('backend', ['lancedb', 'qdrant'])
def test_process_exit_around_publication_keeps_complete_searchable_index(tmp_path, phase, backend, monkeypatch):
    if backend == 'qdrant':
        monkeypatch.setattr(core, 'build_embedder', lambda config, device: Embedder())
    source = tmp_path / 'note.txt'
    source.write_text('Meeting: Friday.')
    config = dict(persist_dir=str(tmp_path / 'index'), document_paths=[str(source)],
                  backend=backend, embed_provider='none', reranker_type='off',
                  force_reindex=True, chunk_size=700, chunk_overlap=100,
                  top_k=5, rerank_top_k=5)
    previous = core.RAGPipeline(config)
    previous.close()
    source.write_text('Meeting: Monday.')
    script = inspect.getsource(Embedder) + '''
import json, os, socket, sys
def blocked(*args, **kwargs):
    raise RuntimeError('network forbidden in crash regression')
socket.socket.connect = blocked
from rag import core, index_lifecycle
if json.loads(sys.argv[1])['backend'] == 'qdrant':
    core.build_embedder = lambda config, device: Embedder()
if sys.argv[2] == 'storage':
    from rag.lancedb_backend import LanceDBBackend
    from rag.qdrant_backend import QdrantVectorDB
    backend = QdrantVectorDB if json.loads(sys.argv[1])['backend'] == 'qdrant' else LanceDBBackend
    add = backend.add_documents_batch
    def write_then_exit(self, *args, **kwargs):
        add(self, *args, **kwargs)
        os._exit(73)
    backend.add_documents_batch = write_then_exit
write = index_lifecycle.write_json_atomic
def interrupted(path, value):
    if path.name == 'active-index.json' and sys.argv[2] == 'before':
        os._exit(73)
    write(path, value)
    if path.name == 'active-index.json' and sys.argv[2] == 'after':
        os._exit(73)
index_lifecycle.write_json_atomic = interrupted
core.RAGPipeline(json.loads(sys.argv[1]))
'''
    result = subprocess.run([sys.executable, '-c', script, json.dumps(config), phase],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 73, result.stderr
    def unexpected_rebuild(self):
        pytest.fail('crash recovery must reopen persisted data without reindexing')
    monkeypatch.setattr(core.RAGPipeline, '_index_documents', unexpected_rebuild)
    reopened = core.RAGPipeline({**config, 'force_reindex': False})
    try:
        expected = 'Monday' if phase == 'after' else 'Friday'
        passage = reopened.search('Meeting')[0]
        assert passage['content'] == f'Meeting: {expected}.'
        assert passage['source'] == str(source)
        assert passage['location'] == 'L1'
    finally:
        reopened.close()


class Embedder:
    model_name = 'controlled-vectors'

    def embed_documents(self, texts):
        return [[1., 0.] for _ in texts]

    def embed_query(self, text):
        return [1., 0.]


@pytest.mark.parametrize('backend', ['lancedb', 'qdrant'])
def test_live_readers_keep_their_generation_after_successful_refreshes(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(core, 'build_embedder', lambda config, device: Embedder())
    source = tmp_path / 'note.txt'
    root = tmp_path / 'index'
    readers = []
    days = ['Friday', 'Monday', 'Tuesday']
    try:
        for day in days:
            source.write_text(f'Meeting: {day}.')
            readers.append(core.RAGPipeline(dict(
                persist_dir=str(root), document_paths=[str(source)], backend=backend,
                embed_provider='none', reranker_type='off', force_reindex=True,
                chunk_size=700, chunk_overlap=100, top_k=5, rerank_top_k=5)))
        assert index_lifecycle.active_directory(root) == Path(readers[-1].config['persist_dir'])
        assert len({reader.config['persist_dir'] for reader in readers}) == 3
        for reader, day in zip(readers, days):
            result = reader.search('Meeting')[0]
            assert result['content'] == f'Meeting: {day}.'
            assert result['source'] == str(source)
    finally:
        for reader in readers:
            reader.close()


@pytest.mark.parametrize('backend', ['lancedb', 'qdrant'])
@pytest.mark.parametrize('entry_point', ['fetch', 'engine'])
@pytest.mark.parametrize('failure', ['embedding', 'metadata', 'manifest', 'publication'])
def test_failed_generation_keeps_old_storage_and_can_retry(tmp_path, monkeypatch, backend, failure, entry_point):
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'none')
    monkeypatch.setenv('RAG_MCP_RERANKER_TYPE', 'off')
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, 'backend', backend)
    monkeypatch.setattr(core, 'build_embedder', lambda config, device: Embedder())
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', {})
    source = tmp_path / 'note.txt'
    source.write_text('Meeting: Friday.')
    def open_pipeline():
        if entry_point == 'fetch':
            return fetch.get_rag_pipeline_v2(str(source))
        return core.RAGPipeline({**fetch.RAG_CONFIG_V2,
            'persist_dir': str(tmp_path / 'engine-index'), 'document_paths': [str(source)],
            'force_reindex': True, 'reranker_type': 'off', 'source_stamp': {'fixture': True}})

    previous = open_pipeline()
    directory = Path(previous.config['persist_dir'])
    pointer = directory.parent / 'active-index.json'
    old_pointer = pointer.read_bytes()
    old_metadata = (directory / 'docs.json').read_bytes()
    old_manifest = (directory / 'source-manifest.json').read_bytes()
    source.write_text('Meeting: Monday.')
    with monkeypatch.context() as fault:
        def unavailable(*args, **kwargs):
            raise RuntimeError('injected refresh failure')

        if failure == 'embedding':
            fault.setattr(type(previous.vector_db), '_batch_embed', unavailable)
        elif failure == 'metadata':
            fault.setattr(ingestion, 'write_json_atomic', unavailable)
        else:
            original_write = index_lifecycle.write_json_atomic
            def fail_selected_file(path, value):
                if Path(path).name == ('source-manifest.json' if failure == 'manifest' else 'active-index.json'):
                    unavailable()
                original_write(path, value)
            fault.setattr(index_lifecycle, 'write_json_atomic', fail_selected_file)
        with pytest.raises(RuntimeError, match='injected refresh failure'):
            open_pipeline()

    assert pointer.read_bytes() == old_pointer
    assert (directory / 'docs.json').read_bytes() == old_metadata
    assert (directory / 'source-manifest.json').read_bytes() == old_manifest
    assert list(directory.parent.glob('generation-*')) == [directory]
    assert previous.search('Meeting')[0]['content'] == 'Meeting: Friday.'
    if backend == 'qdrant':
        previous.vector_db.client.close()
    fetch.reset_pipelines()
    reopened = core.RAGPipeline({**previous.config, 'force_reindex': False})
    try:
        assert reopened.search('Meeting')[0]['content'] == 'Meeting: Friday.'
    finally:
        if backend == 'qdrant':
            reopened.vector_db.client.close()

    replacement = open_pipeline()
    try:
        assert replacement.search('Meeting')[0]['content'] == 'Meeting: Monday.'
        assert json.loads(pointer.read_text()) == Path(replacement.config['persist_dir']).name
        assert replacement.config['persist_dir'] != previous.config['persist_dir']
    finally:
        if backend == 'qdrant':
            replacement.vector_db.client.close()
        fetch.reset_pipelines()


@pytest.mark.parametrize('backend', ['lancedb', 'qdrant'])
@pytest.mark.parametrize('entry_point', ['fetch', 'engine'])
def test_failed_metadata_recovery_does_not_replace_stored_evidence(tmp_path, monkeypatch, backend, entry_point):
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'none')
    monkeypatch.setenv('RAG_MCP_RERANKER_TYPE', 'off')
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, 'backend', backend)
    monkeypatch.setattr(core, 'build_embedder', lambda config, device: Embedder())
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', {})
    source = tmp_path / 'note.txt'
    source.write_text('Meeting: Friday.')
    previous = fetch.get_rag_pipeline_v2(str(source))
    directory = Path(previous.config['persist_dir'])
    pointer = directory.parent / 'active-index.json'
    published = pointer.read_bytes()
    metadata = directory / 'docs.json'
    metadata.write_text('[interrupted')
    previous_storage = previous.vector_db
    if backend == 'qdrant':
        previous.vector_db.client.close()
    fetch.reset_pipelines()

    def recover():
        if entry_point == 'fetch':
            return fetch.get_rag_pipeline_v2(str(source))
        return core.RAGPipeline({**previous.config, 'force_reindex': False})

    with monkeypatch.context() as fault:
        def failed_save(*args):
            raise OSError('recovery metadata failed')
        fault.setattr(core.RAGPipeline, '_extract_content', lambda self, path: 'Replacement from recovery.')
        fault.setattr(ingestion, 'write_json_atomic', failed_save)
        with pytest.raises(OSError, match='recovery metadata failed'):
            recover()
    assert pointer.read_bytes() == published
    assert metadata.read_text() == '[interrupted'
    assert list(directory.parent.glob('generation-*')) == [directory]
    if backend == 'qdrant':
        from rag.qdrant_backend import QdrantVectorDB
        stored = QdrantVectorDB(previous_storage.collection_name, Embedder(), vector_size=2,
                               persist_path=str(directory / 'qdrant_storage'))
        try:
            assert stored.search('Meeting')[0]['content'] == 'Meeting: Friday.'
        finally:
            stored.client.close()
    else:
        from rag.lancedb_backend import LanceDBBackend
        stored = LanceDBBackend(previous_storage.table_name, Embedder(), vector_size=2,
                                persist_path=str(directory / 'lancedb_storage'))
        assert stored.search_fts('Meeting')[0]['content'] == 'Meeting: Friday.'
    recovered = recover()
    try:
        assert recovered.search('Meeting')[0]['content'] == 'Meeting: Friday.'
        assert recovered.config['persist_dir'] != previous.config['persist_dir']
        assert pointer.read_bytes() != published
    finally:
        if backend == 'qdrant':
            recovered.vector_db.client.close()
        fetch.reset_pipelines()
