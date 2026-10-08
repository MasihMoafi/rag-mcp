import pytest

from rag.evaluation_labels import EvaluationLabelStore
from rag.evaluation_runs import EvaluationRunStore
from rag.evaluation_scores import score_batch
from rag.run_results import RunResultStore
from rag.workspaces import WorkspaceStore

TEXT = 'Alpha facts here. ' * 20 + 'The meeting is on Friday in Paris. ' * 4 + 'Noise words. ' * 20


@pytest.fixture
def stores(tmp_path):
    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    snapshot = workspaces.replace('book', 'Book', [{'id': 'doc', 'name': 'doc.txt', 'sourceType': 'text', 'content': TEXT}])
    results = RunResultStore(tmp_path / 'results')
    return workspaces, snapshot['revision'], results, EvaluationLabelStore(workspaces, results)


def save_batch(stores, batch_id, passages_by_question, *, revision=None, questions=('q1', 'q2')):
    workspaces, current, results, labels = stores
    rows = []
    for index, question in enumerate(questions):
        run_id = f'{batch_id[:30]}{index:02d}'
        evidence = [{'rank': rank, 'source_id': 'doc', 'content': TEXT[start:end], 'start_offset': start, 'end_offset': end}
                    for rank, (start, end) in enumerate(passages_by_question.get(question, []), 1)]
        results.save({'schema_version': 1, 'run_id': run_id, 'workspace_id': 'book', 'status': 'success', 'evidence': evidence})
        rows.append({'question_id': question, 'run_id': run_id, 'status': 'success', 'passage_count': len(evidence),
                     'source_count': 1 if evidence else 0, 'duration_ms': 1.0, 'error': None})
    labels.batches.save({'schema_version': 1, 'workspace_id': 'book', 'batch_id': batch_id, 'status': 'success',
        'started_unix': 1.0, 'finished_unix': 2.0, 'error': None, 'question_revision': 'r1',
        'workspace_revision': revision or current, 'configuration_revision': '0' * 64, 'source_ids': ['doc'],
        'questions': [{'id': item, 'question': f'{item}?'} for item in questions], 'rows': rows})
    return {row['question_id']: row['run_id'] for row in rows}


def mark(stores, batch_id, run_id, rank, label):
    labels = stores[3]
    labels.set('book', batch_id, run_id, rank, label, expected_revision=labels.get('book', batch_id)['revision'])


ANSWER = (360, 500)       # the Friday-in-Paris sentences
NEAR_ANSWER = (380, 520)  # a different chunking of the same text
NOISE = (0, 100)
OTHER_NOISE = (560, 700)


def test_right_answers_carry_to_another_configuration_on_the_same_sources(stores):
    labelled = save_batch(stores, 'a' * 32, {'q1': [NOISE, ANSWER], 'q2': [OTHER_NOISE]})
    mark(stores, 'a' * 32, labelled['q1'], 2, 'relevant')
    mark(stores, 'a' * 32, labelled['q1'], 1, 'irrelevant')
    save_batch(stores, 'b' * 32, {'q1': [NEAR_ANSWER, NOISE], 'q2': [OTHER_NOISE]})

    first = score_batch('book', 'a' * 32, stores[3])
    second = score_batch('book', 'b' * 32, stores[3])

    assert first['summary'] == {'judged_questions': 1, 'unjudged_questions': 1, 'hit_rate': 1.0, 'recall': 1.0,
                                'mrr': 0.5, 'ndcg': pytest.approx(0.6309, abs=1e-4), 'precision': 0.5,
                                'unjudged_passages': 1, 'missing_evidence': 0}
    assert second['summary']['mrr'] == 1.0 and second['summary']['ndcg'] == 1.0
    q1 = next(item for item in second['questions'] if item['question_id'] == 'q1')
    assert q1['relevant_ranks'] == [1] and q1['gold_count'] == 1
    q2 = next(item for item in second['questions'] if item['question_id'] == 'q2')
    assert q2['gold_count'] == 0 and q2['hit'] is None


def test_a_miss_scores_zero_and_unlabelled_passages_are_counted(stores):
    labelled = save_batch(stores, 'a' * 32, {'q1': [ANSWER]})
    mark(stores, 'a' * 32, labelled['q1'], 1, 'relevant')
    save_batch(stores, 'b' * 32, {'q1': [NOISE, OTHER_NOISE]})

    summary = score_batch('book', 'b' * 32, stores[3])['summary']

    assert summary['hit_rate'] == 0.0 and summary['recall'] == 0.0 and summary['mrr'] == 0.0 and summary['ndcg'] == 0.0
    assert summary['unjudged_passages'] == 2


def test_labels_from_different_sources_are_not_right_answers(stores):
    labelled = save_batch(stores, 'a' * 32, {'q1': [ANSWER]}, revision='older-sources')
    mark(stores, 'a' * 32, labelled['q1'], 1, 'relevant')
    save_batch(stores, 'b' * 32, {'q1': [ANSWER]})

    summary = score_batch('book', 'b' * 32, stores[3])['summary']

    assert summary['judged_questions'] == 0 and summary['hit_rate'] is None


def test_no_labels_means_no_scores(stores):
    save_batch(stores, 'a' * 32, {'q1': [ANSWER]})
    summary = score_batch('book', 'a' * 32, stores[3])['summary']
    assert summary['judged_questions'] == 0 and summary['recall'] is None and summary['unjudged_passages'] == 1


def test_expired_run_evidence_is_reported_not_scored(stores):
    labelled = save_batch(stores, 'a' * 32, {'q1': [ANSWER]})
    mark(stores, 'a' * 32, labelled['q1'], 1, 'relevant')
    stores[2]._path(labelled['q1']).unlink()

    summary = score_batch('book', 'a' * 32, stores[3])['summary']

    assert summary['missing_evidence'] == 1 and summary['judged_questions'] == 0
