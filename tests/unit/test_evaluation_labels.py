import json
import os

import pytest

from rag.evaluation_labels import EvaluationLabelStore
from rag.evaluation_runs import EvaluationRunStore
from rag.run_results import RunResultStore
from rag.workspaces import WorkspaceConflict, WorkspaceStore


@pytest.fixture
def saved(tmp_path):
    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    workspaces.replace('book', 'Book', [{'id': 'source', 'name': 'facts.txt', 'sourceType': 'text', 'content': '🌲 Friday'}])
    batch_id, run_id = 'a' * 32, 'b' * 32
    EvaluationRunStore(workspaces).save({
        'schema_version': 1, 'workspace_id': 'book', 'batch_id': batch_id, 'status': 'success',
        'started_unix': 1, 'finished_unix': 2, 'error': None, 'question_revision': 'q1',
        'workspace_revision': 'w1', 'configuration_revision': 'c' * 64, 'source_ids': ['source'],
        'questions': [{'id': 'q', 'question': 'When?'}],
        'rows': [{'question_id': 'q', 'run_id': run_id, 'status': 'success',
                  'passage_count': 1, 'source_count': 1, 'duration_ms': 4, 'error': None}],
    })
    results = RunResultStore(tmp_path / 'results')
    results.save({'schema_version': 1, 'workspace_id': 'book', 'run_id': run_id, 'status': 'success',
                  'evidence': [{'rank': 1, 'source_id': 'source', 'content': '🌲 Friday',
                                'start_offset': 0, 'end_offset': 9}]})
    return EvaluationLabelStore(workspaces, results), batch_id, run_id, results


def test_labels_are_private_revisioned_and_clearable(saved):
    store, batch_id, run_id, _ = saved
    initial = store.get('book', batch_id)
    assert initial['revision'] is None and initial['labels'] == []
    marked = store.set('book', batch_id, run_id, 1, 'relevant', expected_revision=None)
    assert marked['labels'][0]['question_id'] == 'q'
    assert marked['labels'][0]['source_id'] == 'source'
    assert marked['labels'][0]['content_sha256'] and 'content' not in marked['labels'][0]
    assert store.get('book', batch_id) == marked
    path = store.path('book', batch_id)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(WorkspaceConflict):
        store.set('book', batch_id, run_id, 1, 'irrelevant', expected_revision=None)
    cleared = store.set('book', batch_id, run_id, 1, None, expected_revision=marked['revision'])
    assert cleared['labels'] == [] and cleared['revision'] != marked['revision']


def test_forged_rank_run_and_expired_evidence_are_rejected(saved):
    store, batch_id, run_id, results = saved
    for invalid_run, rank in [('c' * 32, 1), (run_id, 2)]:
        with pytest.raises(ValueError):
            store.set('book', batch_id, invalid_run, rank, 'relevant', expected_revision=None)
    results._path(run_id).unlink()
    with pytest.raises(FileNotFoundError):
        store.set('book', batch_id, run_id, 1, 'relevant', expected_revision=None)
    assert store.get('book', batch_id)['labels'] == []


def test_corrupt_and_symlinked_label_files_are_rejected(saved, tmp_path):
    store, batch_id, _, _ = saved
    path = store.path('book', batch_id)
    path.write_text(json.dumps({'schema_version': 1, 'labels': ['invalid']}))
    with pytest.raises(OSError):
        store.get('book', batch_id)
    path.unlink()
    outside = tmp_path / 'outside.json'
    outside.write_text('{}')
    path.symlink_to(outside)
    with pytest.raises(OSError):
        store.get('book', batch_id)
    assert json.loads(outside.read_text()) == {}
