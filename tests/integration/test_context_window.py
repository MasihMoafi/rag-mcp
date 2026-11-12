"""Real lexical index, exact expanded evidence, no model requests."""
import io
import json
from pathlib import Path
import pytest
from rag import cli, control, fetch, openrouter_pipeline, settings
from rag.execution import execute_run
from rag.workspaces import WorkspaceStore


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for key, value in {
        'RAG_MCP_CONFIG': str(tmp_path / 'config.json'), 'RAG_MCP_DATA_DIR': str(tmp_path / 'indexes'),
        'RAG_MCP_TRACE_DIR': str(tmp_path / 'traces'), 'RAG_MCP_WORKSPACE_DIR': str(tmp_path / 'workspaces'),
        'RAG_MCP_EMBED_PROVIDER': 'none', 'RAG_MCP_RERANKER_TYPE': 'off',
        'RAG_MCP_CONTEXTUAL_RETRIEVAL': 'off', 'OPENROUTER_API_KEY': 'test-only',  # synthesis is replaced
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('RAG_MCP_CONTEXT_RADIUS', raising=False)
    monkeypatch.setenv('RAG_MCP_WHOLE_DOCUMENT_CHARS', '0')  # neighbour windows alone, unless a test turns it on
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, 'chunk_size', 60)
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, 'chunk_overlap', 0)
    fetch.reset_pipelines()
    yield tmp_path
    fetch.reset_pipelines()


def test_answer_uses_window_but_search_rank_and_index_generation_stay_unchanged(isolated, monkeypatch):
    text = 'Owner: Ada. Budget currency: euros. 🌿\n\nCedar meeting takes place on Friday.\n\nBring the approval note for the project.'
    store = WorkspaceStore()
    saved = store.replace('fixture', 'Fixture', [
        {'id': 'facts', 'name': 'facts.txt', 'sourceType': 'text', 'content': text},
        {'id': 'excluded', 'name': 'hidden.txt', 'sourceType': 'text', 'content': 'Cedar private excluded secret.'}])
    settings.update_settings({'top_k': '1', 'context_radius': '0'})
    arguments = dict(workspace_id='fixture', source_ids=['facts'], workspace_revision=saved['revision'])
    small = execute_run('Cedar', mode='retrieve', **arguments)
    assert small['status'] == 'success', small
    assert small['evidence'][0]['content'] == 'Cedar meeting takes place on Friday.'
    generation = next(iter(fetch._rag_system_v2_instances.values()))
    settings.update_settings({'context_radius': '1'})
    captured = []
    def answer(question, passages, config):
        captured.extend(passages)
        return 'Ada owns it. [1]'
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', answer)
    result = execute_run('Cedar', **arguments)
    assert result['status'] == 'success', result
    assert next(iter(fetch._rag_system_v2_instances.values())) is generation
    assert captured[0]['original_content'] == text and 'excluded secret' not in captured[0]['content']
    assert captured[0]['score'] == small['retrieved_passages'][0]['score']
    evidence = result['evidence'][0]
    assert evidence['content'] == text and evidence['start_offset'] == 0
    assert evidence['end_offset'] == len(text.encode('utf-16-le')) // 2
    again = execute_run('Cedar', mode='retrieve', **arguments)
    assert again['retrieved_passages'] == small['retrieved_passages']
    assert len(captured) == 1
    traces = list((isolated / 'traces').rglob('*.json'))
    assert any('selection.context' in path.read_text() for path in traces)


def test_short_source_is_read_whole_once_and_long_source_keeps_its_passage(isolated, monkeypatch):
    short = ('Cedar budget: euros.\n\nOwner: Ada and the finance team of the north office.\n\n'
             'Reviews happen at the end of each quarter.\n\nCedar meets on Friday.')
    long = ('Filler line about nothing in particular.\n\n' * 12) + 'Cedar archive moved to the vault.'
    store = WorkspaceStore()
    saved = store.replace('fixture', 'Fixture', [
        {'id': 'short', 'name': 'short.txt', 'sourceType': 'text', 'content': short},
        {'id': 'long', 'name': 'long.txt', 'sourceType': 'text', 'content': long}])
    settings.update_settings({'top_k': '3', 'context_radius': '0'})
    monkeypatch.setenv('RAG_MCP_WHOLE_DOCUMENT_CHARS', str(len(short)))
    arguments = dict(workspace_id='fixture', source_ids=['short', 'long'], workspace_revision=saved['revision'])
    retrieved = execute_run('Cedar', mode='retrieve', **arguments)
    names = [p['name'] for p in retrieved['retrieved_passages']]
    assert names.count('short.txt') == 2 and 'long.txt' in names, names
    captured = []
    def answer(question, passages, config):
        captured.extend(passages)
        return 'Ada owns it. [1]'
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', answer)
    assert execute_run('Cedar', **arguments)['status'] == 'success'
    texts = [p['original_content'] for p in captured]
    assert texts.count(short) == 1 and len(captured) == 2, texts
    assert texts[1 - texts.index(short)] == 'Cedar archive moved to the vault.'


def test_cli_control_and_environment_share_bounded_revisioned_setting(isolated, monkeypatch):
    assert control.read_config()['retrieval']['context_radius'] == '0'
    output = io.StringIO()
    assert cli.main(['config', 'context_radius', '2'], stdout=output, stderr=io.StringIO()) == 0
    assert control.read_config()['retrieval']['context_radius'] == '2'
    before = control.read_config()['revision']
    result = control.patch_config({'context_radius': 1, 'revision': before})
    assert result['retrieval']['context_radius'] == '1'
    with pytest.raises(settings.SettingsConflict):
        control.patch_config({'context_radius': 3, 'revision': before})
    monkeypatch.setenv('RAG_MCP_CONTEXT_RADIUS', '3')
    assert control.read_config()['retrieval']['context_radius'] == '3'
    with pytest.raises(ValueError, match='environment'):
        control.patch_config({'context_radius': 1})


@pytest.mark.parametrize('radius', [-1, 4, True, 0.5, '1'])
def test_control_rejects_invalid_radius_without_changing_config(isolated, radius):
    before = control.read_config()
    with pytest.raises(ValueError):
        control.patch_config({'context_radius': radius})
    assert control.read_config() == before
