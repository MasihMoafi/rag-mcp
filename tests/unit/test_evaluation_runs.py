import json
from threading import Event

import pytest
from filelock import FileLock

from rag.evaluation_sets import EvaluationSetStore
from rag.evaluation_runs import EvaluationRunStore, execute_evaluation
from rag.workspaces import WorkspaceConflict, WorkspaceStore


def setup_workspace(tmp_path):
    workspaces = WorkspaceStore(tmp_path)
    snapshot = workspaces.replace('book', 'Book', [{'id': 'source', 'name': 'facts.txt', 'sourceType': 'text', 'content': 'Friday.'}])
    questions = EvaluationSetStore(workspaces).replace('book', [
        {'id': 'one', 'question': 'When?'}, {'id': 'two', 'question': 'Where?'}], expected_revision=None)
    return workspaces, snapshot, questions


def test_batch_freezes_scope_and_keeps_completed_rows_on_error(tmp_path):
    workspaces, snapshot, questions = setup_workspace(tmp_path)
    store = EvaluationRunStore(workspaces)
    calls = []
    def execute(query, **kwargs):
        calls.append((query, kwargs))
        assert store.get('book', 'a' * 32)['status'] == 'running'
        return {'run_id': kwargs['run_id'], 'status': 'success' if len(calls) == 1 else 'error',
                'error': None if len(calls) == 1 else 'Provider unavailable',
                'evidence': [{'source_id': 'source'}] if len(calls) == 1 else []}
    result = execute_evaluation('book', batch_id='a' * 32, store=store, executor=execute,
                                configuration_revision='c' * 64)
    assert result['status'] == 'error'
    assert result['workspace_revision'] == snapshot['revision']
    assert result['question_revision'] == questions['revision']
    assert [row['status'] for row in result['rows']] == ['success', 'error']
    assert result['rows'][0]['passage_count'] == 1
    assert result['rows'][0]['source_count'] == 1
    assert all(call[1]['mode'] == 'retrieve' and call[1]['configuration_revision'] == 'c' * 64 for call in calls)
    assert store.get('book', result['batch_id']) == result
    assert store.list('book')['batches'][0]['completed_count'] == 2
    assert store.path('book', result['batch_id']).stat().st_mode & 0o777 == 0o600


def test_cancel_preserves_progress_and_stale_revisions_do_not_run(tmp_path):
    workspaces, snapshot, questions = setup_workspace(tmp_path)
    store = EvaluationRunStore(workspaces)
    cancelled = Event()
    def execute(query, **kwargs):
        cancelled.set()
        return {'run_id': kwargs['run_id'], 'status': 'success', 'error': None, 'evidence': []}
    result = execute_evaluation('book', store=store, executor=execute, cancelled=cancelled,
                                configuration_revision='c' * 64)
    assert result['status'] == 'cancelled' and len(result['rows']) == 1
    with pytest.raises(WorkspaceConflict):
        execute_evaluation('book', store=store, question_revision='stale', executor=execute)
    with pytest.raises(WorkspaceConflict):
        execute_evaluation('book', store=store, workspace_revision='stale', executor=execute)


def test_interrupted_batch_is_distinguished_from_live_lock_and_retries_are_idempotent(tmp_path):
    workspaces, _, _ = setup_workspace(tmp_path)
    store = EvaluationRunStore(workspaces)
    def execute(query, **kwargs):
        return {'run_id': kwargs['run_id'], 'status': 'success', 'error': None, 'evidence': []}
    result = execute_evaluation('book', batch_id='a' * 32, store=store, executor=execute, configuration_revision='c' * 64)
    assert execute_evaluation('book', batch_id='a' * 32, store=store, executor=lambda *a, **kw: pytest.fail('reran'),
                              configuration_revision='c' * 64) == result
    result['status'] = 'running'
    store.save(result)
    with FileLock(str(store.path('book', 'a' * 32)) + '.lock'):
        assert store.get('book', 'a' * 32)['status'] == 'running'
    assert store.get('book', 'a' * 32)['status'] == 'interrupted'


def test_batch_reader_rejects_invalid_ids_corruption_and_symlinks(tmp_path):
    workspaces, _, _ = setup_workspace(tmp_path)
    store = EvaluationRunStore(workspaces)
    with pytest.raises(ValueError):
        store.get('book', '../outside')
    path = store.path('book', 'a' * 32)
    path.parent.mkdir()
    path.write_text('{broken')
    with pytest.raises(OSError):
        store.get('book', 'a' * 32)
    path.unlink()
    external = tmp_path / 'external.json'
    external.write_text(json.dumps({}))
    path.symlink_to(external)
    with pytest.raises(OSError):
        store.get('book', 'a' * 32)


def test_retry_reads_original_batch_after_questions_and_sources_change(tmp_path):
    workspaces, snapshot, questions = setup_workspace(tmp_path)
    store = EvaluationRunStore(workspaces)
    def execute(query, **kwargs):
        return {'run_id': kwargs['run_id'], 'status': 'success', 'error': None, 'evidence': []}
    options = dict(batch_id='a' * 32, store=store, executor=execute,
                   question_revision=questions['revision'], workspace_revision=snapshot['revision'], source_ids=['source'])
    result = execute_evaluation('book', **options)
    EvaluationSetStore(workspaces).replace('book', [], expected_revision=questions['revision'])
    workspaces.replace('book', 'Book', [], expected_revision=snapshot['revision'])
    assert execute_evaluation('book', **options) == result
    with pytest.raises(WorkspaceConflict):
        execute_evaluation('book', **{**options, 'source_ids': ['different']})


def test_killed_process_keeps_completed_rows_and_reports_interruption(tmp_path):
    import subprocess
    import sys
    import time
    workspaces, _, _ = setup_workspace(tmp_path / 'workspaces')
    marker = tmp_path / 'second-started'
    script = '''
import sys, time
from pathlib import Path
from rag.evaluation_runs import EvaluationRunStore, execute_evaluation
from rag.workspaces import WorkspaceStore
def execute(query, **kwargs):
    if query == 'Where?':
        Path(sys.argv[2]).touch()
        time.sleep(30)
    return {'run_id': kwargs['run_id'], 'status': 'success', 'error': None, 'evidence': []}
execute_evaluation('book', batch_id='b'*32, store=EvaluationRunStore(WorkspaceStore(sys.argv[1])), executor=execute, configuration_revision='c'*64)
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(workspaces.root), str(marker)])
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists()
        assert EvaluationRunStore(workspaces).get('book', 'b' * 32)['status'] == 'running'
        process.terminate()
        process.wait(timeout=2)
        saved = EvaluationRunStore(workspaces).get('book', 'b' * 32)
        assert saved['status'] == 'interrupted' and len(saved['rows']) == 1
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=2)
