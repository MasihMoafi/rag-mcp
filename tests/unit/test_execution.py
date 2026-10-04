import json
import types
from pathlib import Path

import pytest

import server
from rag.execution import execute_run, retrieval_response
from rag.workspaces import WorkspaceStore


def stored_path(store, document_id, workspace_id='book'):
    """The local file of one saved document, as a query reads it."""
    return next(Path(path) for path, document in store.local_snapshot(workspace_id)[2].items()
                if document['id'] == document_id)

@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path / 'traces'))
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(tmp_path / 'workspaces'))
    store = WorkspaceStore()
    store.replace('book', 'Book', [{'id': 'chapter', 'name': 'chapter.txt', 'sourceType': 'text', 'content': 'Cedar Lantern opens Friday.'}])
    return store


def test_public_and_private_clients_use_same_execution(workspace, monkeypatch):
    from rag import openrouter_pipeline
    source = next(workspace.resolve('book').iterdir())
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        return [{'content': 'Cedar Lantern opens Friday.', 'file_path': str(source), 'score': .7}]
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(fetchExternalKnowledge=fetch))
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', lambda *_: 'It opens Friday.')
    private = execute_run('When?', workspace_id='book', client='studio')
    public = json.loads(server.query_knowledge_base('When?', workspace_id='book'))
    assert private['status'] == 'success'
    assert private['answer'] == public['result'] == 'It opens Friday.'
    assert private['evidence'][0]['source_id'] == 'chapter'
    assert private['evidence'][0]['start_offset'] == 0
    assert calls[0] == calls[1]
    assert set(public) == {'result'}
    from rag.trace_reader import read_trace
    trace = read_trace(private['run_id'])
    assert trace is not None
    assert 'Cedar Lantern' not in json.dumps(trace)
    assert trace['spans'][0]['attributes']['configuration_revision'] == private['configuration_revision']
    assert trace['spans'][0]['attributes']['workspace_revision'] == private['workspace_revision']


def test_first_saved_run_uses_gateway_request_fingerprint(workspace, monkeypatch):
    from rag.run_results import RunResultStore

    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(
        fetchExternalKnowledge=lambda **kwargs: []))
    result = execute_run('When?', workspace_id='book', client='studio', mode='retrieve',
                         request_hash_override='f' * 64)
    assert result['request_hash'] == 'f' * 64
    assert RunResultStore().read(result['run_id'])['request_hash'] == 'f' * 64


def test_retrieval_projection_keeps_private_content_opt_in():
    result = {'status': 'success', 'run_id': 'run', 'workspace_id': 'book',
              'workspace_revision': 'workspace-revision', 'configuration_revision': 'config-revision',
              'evidence': [{'rank': 1, 'source_id': 'chapter', 'name': 'chapter.md',
                            'source_type': 'markdown', 'start_offset': 3, 'end_offset': 10,
                            'content': 'Private passage', 'score': 0.7}]}
    metadata = retrieval_response(result)
    assert metadata['passage_count'] == 1 and metadata['workspace_revision'] == 'workspace-revision'
    assert 'Private passage' not in json.dumps(metadata) and 'score' not in metadata['passages'][0]
    assert retrieval_response(result, include_content=True)['passages'][0]['content'] == 'Private passage'
    assert retrieval_response({'status': 'error', 'run_id': 'failed', 'error': 'Stale', 'code': 'workspace_conflict'}) == {
        'error': 'Stale', 'run_id': 'failed', 'code': 'workspace_conflict'}


def test_runtime_manifest_is_private_and_failure_does_not_break_search(workspace, monkeypatch):
    from rag import provenance
    from rag.trace_reader import read_trace
    source = next(workspace.resolve('book').iterdir())
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(
        fetchExternalKnowledge=lambda **kwargs: [{'content': 'Cedar Lantern opens Friday.', 'file_path': str(source)}]))
    private_manifest = {'sha256': 'a' * 64, 'dependencies': {'packages': [{'name': 'private-package', 'version': '1'}]}}
    monkeypatch.setattr(provenance, 'runtime_manifest', lambda: private_manifest)
    result = execute_run('When?', workspace_id='book', mode='retrieve')
    assert result['status'] == 'success' and result['runtime_manifest'] == private_manifest
    from rag.run_results import RunResultStore
    assert RunResultStore().read(result['run_id'])['runtime_manifest'] == private_manifest
    trace = read_trace(result['run_id'])
    assert trace['spans'][0]['attributes']['runtime_manifest_sha256'] == 'a' * 64
    assert 'private-package' not in json.dumps(trace)

    def fail():
        raise OSError('private path')
    monkeypatch.setattr(provenance, 'runtime_manifest', fail)
    available = execute_run('When?', workspace_id='book', mode='retrieve')
    assert available['status'] == 'success' and available['runtime_manifest_status'] == 'unavailable'
    assert read_trace(available['run_id'])['spans'][0]['attributes']['runtime_manifest_status'] == 'unavailable'
    assert 'private path' not in json.dumps(available)


def test_cancelled_run_does_not_start_retrieval(workspace, monkeypatch):
    from threading import Event
    cancelled = Event(); cancelled.set()
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: pytest.fail('Cancelled run must not start retrieval'))
    result = execute_run('When?', workspace_id='book', client='studio', cancelled=cancelled)
    assert result['status'] == 'cancelled'
    assert result['answer'] is None
    from rag.trace_reader import read_trace
    assert read_trace(result['run_id'])['spans'][0]['status'] == 'cancelled'


def test_revision_and_source_selection_fail_before_retrieval(workspace, monkeypatch):
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: pytest.fail('Invalid scope retrieved'))
    rejected = execute_run('When?', workspace_id='book', workspace_revision='stale')
    assert rejected['code'] == 'workspace_conflict'
    from rag.trace_reader import read_trace
    assert 'workspace_revision' not in read_trace(rejected['run_id'])['spans'][0]['attributes']
    assert execute_run('When?', workspace_id='book', source_ids=['absent'])['code'] == 'invalid_sources'


def test_selected_source_and_unicode_location(workspace, monkeypatch):
    from rag import openrouter_pipeline
    old = workspace.snapshot('book')
    workspace.replace('book', 'Book', [
        {'id': 'chapter', 'name': 'chapter.txt', 'sourceType': 'text', 'content': '🌲 Friday.\nFriday.'},
        {'id': 'other', 'name': 'other.txt', 'sourceType': 'text', 'content': 'Monday.'},
    ], expected_revision=old['revision'])
    source = stored_path(workspace, 'chapter')
    def fetch(**kwargs):
        assert kwargs['doc_path'] == [str(source)]
        return [{'content': 'Friday.', 'file_path': str(source), 'start_line': 2}]
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(fetchExternalKnowledge=fetch))
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', lambda *_: 'Friday. [1]')
    result = execute_run('When?', workspace_id='book', source_ids=['chapter'], client='studio')
    assert result['status'] == 'success'
    assert result['evidence'][0]['start_offset'] == 11
    assert result['evidence'][0]['end_offset'] == 18


def test_repeated_passage_uses_verified_index_offsets(workspace, monkeypatch):
    old = workspace.snapshot('book')
    content = '🌲 Cedar. Cedar.'
    workspace.replace('book', 'Book', [
        {'id': 'chapter', 'name': 'chapter.txt', 'sourceType': 'text', 'content': content},
    ], expected_revision=old['revision'])
    source = stored_path(workspace, 'chapter')
    start = content.rfind('Cedar.')
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(
        fetchExternalKnowledge=lambda **kwargs: [{
            'content': 'Cedar.', 'original_content': 'Cedar.', 'file_path': str(source),
            'start_line': 1, 'start_offset': start, 'end_offset': start + len('Cedar.'),
        }]))
    result = execute_run('Cedar?', workspace_id='book', client='studio', mode='retrieve')
    assert result['status'] == 'success'
    assert len(result['evidence']) == 1
    assert result['evidence'][0]['content'] == 'Cedar.'
    assert result['evidence'][0]['start_offset'] == len(content[:start].encode('utf-16-le')) // 2


@pytest.mark.parametrize('offsets', [{}, {'start_offset': 1, 'end_offset': 7}])
def test_repeated_passage_without_valid_offsets_stays_unlinked(workspace, monkeypatch, offsets):
    old = workspace.snapshot('book')
    workspace.replace('book', 'Book', [
        {'id': 'chapter', 'name': 'chapter.txt', 'sourceType': 'text', 'content': 'Cedar. Cedar.'},
    ], expected_revision=old['revision'])
    source = stored_path(workspace, 'chapter')
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(
        fetchExternalKnowledge=lambda **kwargs: [{
            'content': 'Cedar.', 'file_path': str(source), 'start_line': 1, **offsets,
        }]))
    result = execute_run('Cedar?', workspace_id='book', client='studio', mode='retrieve')
    assert result['status'] == 'success'
    assert result['evidence'] == []
    search = retrieval_response(result)
    assert search['passage_count'] == 1
    assert search['passages'][0]['source_id'] == 'chapter'
    assert search['passages'][0]['start_offset'] is None
    assert search['passages'][0]['location_verified'] is False
    assert retrieval_response(result, include_content=True)['passages'][0]['content'] == 'Cedar.'
    from rag.run_results import RunResultStore
    assert RunResultStore().read(result['run_id'])['retrieved_passages'][0]['location_verified'] is False


def test_cancel_after_retrieval_never_calls_generation(workspace, monkeypatch):
    from threading import Event
    from rag import openrouter_pipeline
    cancelled = Event()
    def fetch(**kwargs):
        cancelled.set()
        return []
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(fetchExternalKnowledge=fetch))
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', lambda *_: pytest.fail('Cancelled run generated'))
    assert execute_run('When?', workspace_id='book', cancelled=cancelled)['status'] == 'cancelled'


def test_retrieval_mode_saves_evidence_without_generation(workspace, monkeypatch):
    from rag import openrouter_pipeline
    source = next(workspace.resolve('book').iterdir())
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(
        fetchExternalKnowledge=lambda **kwargs: [{'content': 'Cedar Lantern opens Friday.', 'file_path': str(source)}]))
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', lambda *_: pytest.fail('Retrieval-only run must not generate'))
    result = execute_run('When?', workspace_id='book', client='studio', mode='retrieve')
    assert result['status'] == 'success' and result['answer'] is None
    assert result['mode'] == 'retrieve'
    assert result['evidence'][0]['source_id'] == 'chapter'
    from rag.run_results import RunResultStore
    assert RunResultStore().read(result['run_id']) == result


def test_comparison_model_override_is_private_to_run_and_rejects_stale_config(workspace, monkeypatch):
    from rag import openrouter_pipeline, settings
    source = next(workspace.resolve('book').iterdir())
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(
        fetchExternalKnowledge=lambda **kwargs: [{'content': 'Cedar Lantern opens Friday.', 'file_path': str(source)}]))
    models = []
    def synthesize(query, passages, config):
        models.append(config['answer_model'])
        return 'Friday.'
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', synthesize)
    original = settings.retrieval_config()['answer_model']
    first = execute_run('When?', workspace_id='book', client='studio')
    second = execute_run('When?', workspace_id='book', client='studio', answer_model='fixture/alternative',
                         configuration_revision=first['configuration_revision'])
    assert second['status'] == 'success'
    from rag.trace_reader import read_trace
    overridden = read_trace(second['run_id'])['spans'][0]['attributes']
    assert overridden['configuration_revision'] == first['configuration_revision']
    assert overridden['model'] == 'fixture/alternative'
    assert models == [original, 'fixture/alternative']
    assert settings.retrieval_config()['answer_model'] == original
    rejected = execute_run('When?', workspace_id='book', configuration_revision='0' * 64)
    assert rejected['code'] == 'configuration_conflict'
    assert len(models) == 2


def test_verified_passages_report_source_lines_but_only_for_the_file_s_own_text(workspace, monkeypatch):
    old = workspace.snapshot('book')
    workspace.replace('book', 'Book', [
        {'id': 'notes', 'name': 'notes.md', 'sourceType': 'markdown', 'content': '🌲 One.\nTwo.\nThree.\nFour.'},
        {'id': 'paper', 'name': 'paper.html', 'sourceType': 'html', 'content': 'Alpha.\nBeta.'},
    ], expected_revision=old['revision'])
    def fetch(**kwargs):
        return [
            {'content': 'Two.\nThree.', 'file_path': str(stored_path(workspace, 'notes')), 'start_line': 2},
            {'content': 'Beta.', 'file_path': str(stored_path(workspace, 'paper')), 'start_line': 2},
        ]
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(fetchExternalKnowledge=fetch))
    result = execute_run('Which lines?', workspace_id='book', client='studio', mode='retrieve')
    notes, paper = result['retrieved_passages']
    assert (notes['start_line'], notes['end_line']) == (2, 3)
    assert 'start_line' not in paper and 'end_line' not in paper
    projected = retrieval_response(result)['passages']
    assert (projected[0]['start_line'], projected[0]['end_line']) == (2, 3) and 'start_line' not in projected[1]
    # The persisted evidence contract that Studio reads is unchanged.
    assert all('start_line' not in item and 'end_line' not in item for item in result['evidence'])


ABSTENTION = "I don't have enough information to answer that question."


@pytest.mark.parametrize('scope', ['workspace', 'file'])
@pytest.mark.parametrize('found, reason', [
    (0, 'Search found no passages in the selected sources.'),
    (2, 'Search found 2 passages, but none of them answers the question.'),
])
def test_a_refusal_says_why(workspace, monkeypatch, scope, found, reason):
    from rag import openrouter_pipeline
    from rag.answer_contract import decode_public_answer
    source = stored_path(workspace, 'chapter')
    passages = [{'content': 'Cedar Lantern opens Friday.', 'file_path': str(source)}] * found
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(
        fetchExternalKnowledge=lambda **kwargs: passages))
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', lambda *_: ABSTENTION)
    target = {'workspace_id': 'book'} if scope == 'workspace' else {'doc_path': str(source)}
    public = json.loads(server.query_knowledge_base('Who owns it?', **target))
    assert public == {'result': ABSTENTION, 'reason': reason}
    assert decode_public_answer(json.dumps(public)) == (f'{ABSTENTION}\n{reason}', False)
    assert decode_public_answer(json.dumps({**public, 'reason': 5}))[1] is True
