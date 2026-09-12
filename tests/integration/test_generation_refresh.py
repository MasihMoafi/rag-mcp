"""Published CLI/MCP indexes survive failures throughout a replacement build."""

import json
from pathlib import Path

import pytest

from rag import core, fetch, ingestion


class Embedder:
    model_name = 'controlled-vectors'

    def embed_documents(self, texts):
        return [[1., 0.] for _ in texts]

    def embed_query(self, text):
        return [1., 0.]


@pytest.mark.parametrize('backend', ['lancedb', 'qdrant'])
@pytest.mark.parametrize('failure', ['embedding', 'metadata', 'manifest', 'publication'])
def test_failed_generation_keeps_old_storage_and_can_retry(tmp_path, monkeypatch, backend, failure):
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
            original_write = fetch.write_json_atomic
            def fail_selected_file(path, value):
                if Path(path).name == ('source-manifest.json' if failure == 'manifest' else 'active-index.json'):
                    unavailable()
                original_write(path, value)
            fault.setattr(fetch, 'write_json_atomic', fail_selected_file)
        with pytest.raises(RuntimeError, match='injected refresh failure'):
            fetch.get_rag_pipeline_v2(str(source))

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

    replacement = fetch.get_rag_pipeline_v2(str(source))
    try:
        assert replacement.search('Meeting')[0]['content'] == 'Meeting: Monday.'
        assert json.loads(pointer.read_text()) == Path(replacement.config['persist_dir']).name
        assert replacement.config['persist_dir'] != previous.config['persist_dir']
    finally:
        if backend == 'qdrant':
            replacement.vector_db.client.close()
        fetch.reset_pipelines()
