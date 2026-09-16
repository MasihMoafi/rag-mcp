"""Editing one file must re-embed that file only, and nothing else."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import server
from rag import core, fetch


class RecordingEmbedder:
    """Fake provider that records every document text it is asked to embed."""

    model_name = 'fixture'

    def __init__(self):
        self.documents = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


PASSAGES = {'first.txt': 'Meeting: Friday.', 'second.txt': 'Meeting: Monday.',
            'third.txt': 'Meeting: Tuesday.'}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'none')
    monkeypatch.setenv('RAG_MCP_CONTEXTUAL_RETRIEVAL', 'off')
    monkeypatch.setenv('RAG_MCP_RERANKER_TYPE', 'off')
    monkeypatch.setenv('RAG_MCP_RERANK_TOP_K', '10')
    embedder = RecordingEmbedder()
    monkeypatch.setattr(core, 'build_embedder', lambda config, device: embedder)

    folder = tmp_path / 'repo'
    folder.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=folder, check=True)
    # The ignore rule exists before the first scan: server caches it per folder.
    (folder / '.gitignore').write_text('ignored.txt\n')
    (folder / 'ignored.txt').write_text('Meeting: ignored data.')
    for name, passage in PASSAGES.items():
        (folder / name).write_text(passage)

    def search(query='Meeting'):
        response = json.loads(server.query_knowledge_base(query, str(folder)))
        assert 'error' not in response, response
        return response['result']

    server._validated_scope_cache.clear()
    fetch.reset_pipelines()
    yield SimpleNamespace(folder=folder, embedder=embedder, index=tmp_path / 'index', search=search)
    fetch.reset_pipelines()
    server._validated_scope_cache.clear()


def published_documents(index_root):
    """Passages the active generation serves, from its own metadata."""
    pointers = list(Path(index_root).rglob('active-index.json'))
    assert len(pointers) == 1, pointers  # One folder keeps one index across file changes.
    generation = json.loads(pointers[0].read_text())
    return json.loads((pointers[0].parent / generation / 'docs.json').read_text())


def test_changed_file_is_the_only_text_embedded_again(workspace):
    assert {passage['content'] for passage in workspace.search()} == set(PASSAGES.values())
    assert sorted(workspace.embedder.documents) == sorted(PASSAGES.values())

    workspace.embedder.documents.clear()
    (workspace.folder / 'second.txt').write_text('Meeting: Wednesday.')
    contents = {passage['content'] for passage in workspace.search()}

    assert workspace.embedder.documents == ['Meeting: Wednesday.']
    assert contents == {'Meeting: Friday.', 'Meeting: Wednesday.', 'Meeting: Tuesday.'}

    # A refreshed index must stay refreshable, not fall back to a full rebuild.
    workspace.embedder.documents.clear()
    (workspace.folder / 'first.txt').write_text('Meeting: Sunday.')
    contents = {passage['content'] for passage in workspace.search()}

    assert workspace.embedder.documents == ['Meeting: Sunday.']
    assert contents == {'Meeting: Sunday.', 'Meeting: Wednesday.', 'Meeting: Tuesday.'}


def test_deleted_file_loses_its_passages_without_embedding_again(workspace):
    assert len(workspace.search()) == 3
    workspace.embedder.documents.clear()

    (workspace.folder / 'third.txt').unlink()
    passages = workspace.search()

    assert workspace.embedder.documents == []
    assert {passage['content'] for passage in passages} == {'Meeting: Friday.', 'Meeting: Monday.'}
    assert all('third.txt' not in passage['source'] for passage in passages)


def test_ignored_and_unsupported_files_cause_no_work(workspace, monkeypatch):
    assert len(workspace.search()) == 3
    workspace.embedder.documents.clear()
    pointer = next(workspace.index.rglob('active-index.json'))
    published = pointer.read_bytes()

    (workspace.folder / 'ignored.txt').write_text('Meeting: still ignored.')
    (workspace.folder / 'tool.exe').write_bytes(b'\x00binary')
    for name in ('_index_documents', '_refresh_documents'):
        monkeypatch.setattr(core.RAGPipeline, name,
                            lambda *args, **kwargs: pytest.fail('Reindexed an unread change'))
    contents = {passage['content'] for passage in workspace.search()}

    assert workspace.embedder.documents == []
    assert contents == set(PASSAGES.values())
    assert pointer.read_bytes() == published


def test_incremental_result_matches_a_rebuild_of_the_same_corpus(workspace):
    workspace.search()
    (workspace.folder / 'second.txt').write_text('Meeting: Wednesday and a longer agenda.')
    (workspace.folder / 'third.txt').unlink()
    (workspace.folder / 'fourth.txt').write_text('Meeting: Thursday.')
    incremental = workspace.search()
    incremental_documents = published_documents(workspace.index)

    fetch.reset_pipelines()
    shutil.rmtree(workspace.index)
    rebuilt = workspace.search()

    key = lambda document: document['id']
    assert sorted(incremental_documents, key=key) == sorted(published_documents(workspace.index), key=key)
    assert ({(passage['content'], passage['source'], passage['location']) for passage in incremental}
            == {(passage['content'], passage['source'], passage['location']) for passage in rebuilt})
