import pytest
import io
import json

from rag.evaluation_sets import EvaluationSetStore
from rag.workspaces import WorkspaceConflict, WorkspaceStore
from rag.cli import main


def test_question_set_is_private_versioned_and_independent_of_sources(tmp_path):
    workspaces = WorkspaceStore(tmp_path)
    source = workspaces.replace('book', 'Book', [{'id': 'source', 'name': 'facts.txt', 'sourceType': 'text', 'content': 'Friday.'}])
    sets = EvaluationSetStore(workspaces)
    assert sets.get('book') == {'schema_version': 1, 'workspace_id': 'book', 'revision': None, 'questions': []}
    question = {'id': 'q1', 'question': 'When is the meeting?'}
    saved = sets.replace('book', [question], expected_revision=None)
    assert saved['revision'] and sets.get('book') == saved
    assert workspaces.get('book')['revision'] == source['revision']
    assert sets.replace('book', [question], expected_revision=saved['revision']) == saved
    with pytest.raises(WorkspaceConflict):
        sets.replace('book', [], expected_revision=None)
    assert (tmp_path / 'book' / 'evaluation-set.json').stat().st_mode & 0o777 == 0o600


def test_question_set_rejects_invalid_and_missing_workspaces(tmp_path):
    workspaces = WorkspaceStore(tmp_path)
    sets = EvaluationSetStore(workspaces)
    with pytest.raises(OSError):
        sets.get('missing')
    workspaces.replace('book', 'Book', [])
    for questions in ([{'id': 'q', 'question': ''}], [{'id': 'q', 'question': 'One'}, {'id': 'q', 'question': 'Two'}], [{}]):
        with pytest.raises(ValueError):
            sets.replace('book', questions, expected_revision=None)
    assert sets.get('book')['questions'] == []


def test_question_set_never_reads_corrupt_or_linked_file(tmp_path):
    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    workspaces.replace('book', 'Book', [])
    sets = EvaluationSetStore(workspaces)
    path = workspaces.root / 'book' / 'evaluation-set.json'
    path.write_text('{broken')
    with pytest.raises(OSError):
        sets.get('book')
    path.unlink()
    external = tmp_path / 'external.json'
    external.write_text(json.dumps({'schema_version': 1, 'workspace_id': 'book', 'revision': 'r', 'questions': []}))
    path.symlink_to(external)
    with pytest.raises(OSError):
        sets.get('book')


def test_cli_reads_and_updates_the_same_question_set(tmp_path):
    workspaces = WorkspaceStore(tmp_path)
    workspaces.replace('book', 'Book', [])
    environ = {'RAG_MCP_WORKSPACE_DIR': str(tmp_path)}
    output = io.StringIO()
    assert main(['evaluations', 'book', '--add', 'When?'], environ=environ, stdout=output) == 0
    saved = EvaluationSetStore(workspaces).get('book')
    assert saved['questions'][0]['question'] == 'When?'
    output = io.StringIO()
    assert main(['evaluations', 'book', '--json'], environ=environ, stdout=output) == 0
    assert json.loads(output.getvalue()) == saved
    assert main(['evaluations', 'book', '--remove', saved['questions'][0]['id']], environ=environ, stdout=io.StringIO()) == 0
    assert EvaluationSetStore(workspaces).get('book')['questions'] == []


def test_large_unicode_set_cannot_overwrite_a_readable_saved_set(tmp_path):
    workspaces = WorkspaceStore(tmp_path)
    workspaces.replace('book', 'Book', [])
    sets = EvaluationSetStore(workspaces)
    saved = sets.replace('book', [{'id': 'one', 'question': 'When?'}], expected_revision=None)
    with pytest.raises(ValueError, match='storage limit'):
        sets.replace('book', [{'id': str(i), 'question': 'پ' * 4000} for i in range(200)], expected_revision=saved['revision'])
    assert sets.get('book') == saved
