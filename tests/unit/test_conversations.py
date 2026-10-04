import json

import pytest

from rag.conversations import ConversationStore, run_conversation_query
from rag.run_results import RunResultStore
from rag.workspaces import WorkspaceConflict, WorkspaceStore


@pytest.fixture
def saved(tmp_path):
    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    workspaces.replace('book', 'Book', [{'id': 'source', 'name': 'facts.txt', 'sourceType': 'text', 'content': '🌲 Friday'}])
    results = RunResultStore(tmp_path / 'results')
    run_id = 'b' * 32
    results.save({'schema_version': 1, 'workspace_id': 'book', 'run_id': run_id, 'status': 'success',
                  'answer': 'The meeting is Friday. [1]', 'workspace_revision': 'rev',
                  'configuration_revision': 'c' * 64,
                  'evidence': [{'rank': 1, 'source_id': 'source', 'name': 'facts.txt', 'content': '🌲 Friday',
                                'start_offset': 0, 'end_offset': 9, 'score': None}]})
    return ConversationStore(workspaces, results), results, run_id


def test_private_conversation_append_read_list_and_idempotent_retry(saved):
    store, _, run_id = saved
    conversation_id = 'a' * 32
    first = store.append('book', conversation_id, run_id, 'When?', expected_revision=None)
    assert first['turns'][0]['answer'] == 'The meeting is Friday. [1]'
    assert first['turns'][0]['evidence'][0]['content'] == '🌲 Friday'
    assert first['turns'][0]['question'] == 'When?'
    assert store.append('book', conversation_id, run_id, 'When?', expected_revision=None) == first
    assert store.get('book', conversation_id) == first
    listed = store.list('book')
    assert listed['conversations'][0]['conversation_id'] == conversation_id
    assert listed['conversations'][0]['turn_count'] == 1
    assert store.path('book', conversation_id).stat().st_mode & 0o777 == 0o600
    assert store.path('book', conversation_id).parent.stat().st_mode & 0o777 == 0o700


def test_conflicts_and_forged_or_expired_runs_are_rejected(saved):
    store, results, run_id = saved
    conversation_id = 'a' * 32
    first = store.append('book', conversation_id, run_id, 'When?', expected_revision=None)
    second_id = 'c' * 32
    results.save({'schema_version': 1, 'workspace_id': 'book', 'run_id': second_id, 'status': 'success',
                  'answer': 'Saturday', 'evidence': []})
    with pytest.raises(WorkspaceConflict):
        store.append('book', conversation_id, second_id, 'Again?', expected_revision=None)
    second = store.append('book', conversation_id, second_id, 'Again?', expected_revision=first['revision'])
    assert len(second['turns']) == 2
    with pytest.raises(ValueError):
        store.append('book', conversation_id, run_id, 'Different question?', expected_revision=second['revision'])
    with pytest.raises(FileNotFoundError):
        store.append('book', 'd' * 32, 'e' * 32, 'Missing?', expected_revision=None)
    results._path(run_id).unlink()
    with pytest.raises(FileNotFoundError):
        store.append('book', 'd' * 32, run_id, 'Expired?', expected_revision=None)


def test_corrupt_and_symlinked_records_are_rejected(saved, tmp_path):
    store, _, run_id = saved
    conversation_id = 'a' * 32
    store.append('book', conversation_id, run_id, 'When?', expected_revision=None)
    path = store.path('book', conversation_id)
    path.write_text(json.dumps({'schema_version': 1, 'turns': ['invalid']}))
    with pytest.raises(OSError):
        store.get('book', conversation_id)
    path.unlink()
    outside = tmp_path / 'outside.json'
    outside.write_text('{}')
    path.symlink_to(outside)
    with pytest.raises(OSError):
        store.get('book', conversation_id)
    assert outside.read_text() == '{}'


def test_malformed_saved_evidence_is_not_silently_dropped(saved):
    store, results, _ = saved
    run_id = 'd' * 32
    results.save({'schema_version': 1, 'workspace_id': 'book', 'run_id': run_id,
                  'status': 'success', 'answer': 'Friday', 'evidence': [None]})
    with pytest.raises(ValueError):
        store.append('book', 'e' * 32, run_id, 'When?', expected_revision=None)


def test_query_uses_recorded_context_and_appends_success(saved):
    store, results, _ = saved
    calls = []
    def execute(query, **kwargs):
        calls.append((query, kwargs['client']))
        result = {'schema_version': 1, 'workspace_id': 'book', 'run_id': kwargs['run_id'],
                  'status': 'success', 'answer': 'Friday. [1]', 'evidence': []}
        kwargs['result_store'].save(result)
        return result
    conversation_id = 'a' * 32
    first = run_conversation_query('When?', 'book', conversation_id, client='cli', store=store,
                                   result_store=results, executor=execute)
    assert first['result'] == 'Friday. [1]' and first['conversation_revision']
    second = run_conversation_query('What time?', 'book', conversation_id, client='mcp', store=store,
                                    result_store=results, executor=execute)
    assert calls == [('When?', 'cli'), (
        'Previous question (context only, do not answer): When?\n'
        'Current question (answer this question only): What time?', 'mcp')]
    assert second['conversation_revision'] != first['conversation_revision']
    assert [turn['question'] for turn in store.get('book', conversation_id)['turns']] == ['When?', 'What time?']
    with pytest.raises(WorkspaceConflict):
        run_conversation_query('Stale?', 'book', conversation_id, client='mcp',
                               expected_revision=first['conversation_revision'], store=store,
                               result_store=results, executor=execute)
    assert len(calls) == 2


def test_failed_query_does_not_append_and_save_failure_is_reported(saved):
    store, results, _ = saved
    def failed(query, **kwargs):
        return {'run_id': kwargs['run_id'], 'workspace_id': 'book', 'status': 'error', 'error': 'Offline failure',
                'answer': None, 'evidence': []}
    conversation_id = 'a' * 32
    result = run_conversation_query('When?', 'book', conversation_id, store=store,
                                    result_store=results, executor=failed)
    assert result['error'] == 'Offline failure'
    with pytest.raises(FileNotFoundError):
        store.get('book', conversation_id)
    def unsaved(query, **kwargs):
        return {'run_id': kwargs['run_id'], 'workspace_id': 'book', 'status': 'success',
                'answer': 'Friday', 'evidence': []}
    result = run_conversation_query('When?', 'book', conversation_id, store=store,
                                    result_store=results, executor=unsaved)
    assert result['result'] == 'Friday' and result['conversation_error']
