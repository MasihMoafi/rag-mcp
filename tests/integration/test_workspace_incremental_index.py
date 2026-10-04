"""A workspace edit must embed only the documents it adds or changes."""

import shutil
from types import SimpleNamespace

import pytest

import server
from rag import core, execution, fetch, openrouter_pipeline
from rag.workspaces import WorkspaceStore
from tests.integration.test_incremental_index import RecordingEmbedder

DOCUMENTS = {'first': 'Meeting: Friday.', 'second': 'Meeting: Monday.', 'third': 'Meeting: Tuesday.'}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name, value in {'RAG_MCP_DATA_DIR': tmp_path / 'index', 'RAG_MCP_CONFIG': tmp_path / 'config.json',
                        'RAG_MCP_TRACE_DIR': tmp_path / 'traces', 'RAG_MCP_WORKSPACE_DIR': tmp_path / 'workspaces',
                        'RAG_MCP_EMBED_PROVIDER': 'none', 'RAG_MCP_CONTEXTUAL_RETRIEVAL': 'off',
                        'RAG_MCP_RERANKER_TYPE': 'off', 'RAG_MCP_RERANK_TOP_K': '10'}.items():
        monkeypatch.setenv(name, str(value))
    embedder = RecordingEmbedder()
    monkeypatch.setattr(core, 'build_embedder', lambda config, device: embedder)
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', lambda query, passages, config: 'Fixture answer.')
    store = WorkspaceStore()

    def save(documents):
        try:
            revision = store.get('fixture')['revision']
        except OSError:
            revision = None
        store.replace('fixture', 'Fixture', [{'id': key, 'name': key, 'source_type': 'text', 'content': text}
                                              for key, text in documents.items()], expected_revision=revision)

    def search():
        embedder.documents.clear()
        result = execution.execute_run('Meeting', workspace_id='fixture', workspace_store=store)
        assert result['status'] == 'success', result['error']
        return {row['source_id']: row['content'] for row in result['evidence']}

    def rebuild():
        fetch.reset_pipelines()
        server._validated_scope_cache.clear()
        shutil.rmtree(tmp_path / 'index')

    server._validated_scope_cache.clear()
    fetch.reset_pipelines()
    yield SimpleNamespace(save=save, search=search, rebuild=rebuild, embedder=embedder,
                          files=tmp_path / 'workspaces' / 'fixture', index=tmp_path / 'index')
    fetch.reset_pipelines()
    server._validated_scope_cache.clear()


def test_an_added_document_is_the_only_text_embedded(workspace):
    workspace.save(DOCUMENTS)
    workspace.search()
    added = {**DOCUMENTS, 'fourth': 'Meeting: Sunday.'}
    workspace.save(added)
    assert workspace.search() == added
    assert workspace.embedder.documents == ['Meeting: Sunday.']


def test_a_removed_document_embeds_nothing_and_is_not_found(workspace):
    workspace.save(DOCUMENTS)
    workspace.search()
    remaining = {key: text for key, text in DOCUMENTS.items() if key != 'second'}
    workspace.save(remaining)
    assert workspace.search() == remaining
    assert workspace.embedder.documents == []


def test_a_changed_document_is_the_only_text_embedded_again(workspace):
    workspace.save(DOCUMENTS)
    workspace.search()
    changed = {**DOCUMENTS, 'third': 'Meeting: Thursday.'}
    workspace.save(changed)
    assert workspace.search() == changed
    assert workspace.embedder.documents == ['Meeting: Thursday.']


def test_the_updated_index_matches_a_full_rebuild(workspace):
    workspace.save(DOCUMENTS)
    workspace.search()
    edited = {'first': 'Meeting: Friday.', 'third': 'Meeting: Thursday.', 'fourth': 'Meeting: Sunday.'}
    workspace.save(edited)
    updated = workspace.search()
    workspace.rebuild()
    assert workspace.search() == updated == edited


def test_only_the_last_two_revisions_keep_files_and_indexes(workspace):
    for documents in (DOCUMENTS, {**DOCUMENTS, 'fourth': 'Meeting: Sunday.'},
                      {'first': 'Meeting: Saturday.'}, {'first': 'Meeting: Wednesday.'}):
        workspace.save(documents)
        workspace.search()
    stored = {path.read_text() for path in workspace.files.rglob('*')
              if path.is_file() and path.parent.name == 'documents'}
    assert stored == {'Meeting: Saturday.', 'Meeting: Wednesday.'}
    assert len([path for path in workspace.index.rglob('generation-*') if path.is_dir()]) <= 2
