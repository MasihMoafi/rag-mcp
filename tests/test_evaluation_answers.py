import json
from threading import Event

import pytest

from rag import evaluation_answers
from rag.evaluation_answers import EvaluationAnswerStore, check_answers, judge_answer
from rag.evaluation_runs import EvaluationRunStore
from rag.passage_answers import ABSTENTION
from rag.run_results import RunResultStore
from rag.workspaces import WorkspaceStore

BATCH = 'a' * 32
PASSAGE = {'rank': 1, 'source_id': 'doc', 'content': 'The meeting is on Friday.', 'start_offset': 0, 'end_offset': 25}


@pytest.fixture
def stores(tmp_path):
    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    snapshot = workspaces.replace('book', 'Book', [{'id': 'doc', 'name': 'doc.txt', 'sourceType': 'text', 'content': 'x'}])
    results = RunResultStore(tmp_path / 'results')
    evidence = {'q1': [PASSAGE], 'q2': []}
    rows = []
    for index, (question, passages) in enumerate(evidence.items()):
        run_id = f'{index:032d}'
        results.save({'schema_version': 1, 'run_id': run_id, 'workspace_id': 'book', 'status': 'success', 'evidence': passages})
        rows.append({'question_id': question, 'run_id': run_id, 'status': 'success', 'passage_count': len(passages),
                     'source_count': 1 if passages else 0, 'duration_ms': 1.0, 'error': None})
    EvaluationRunStore(workspaces).save({'schema_version': 1, 'workspace_id': 'book', 'batch_id': BATCH, 'status': 'success',
        'started_unix': 1.0, 'finished_unix': 2.0, 'error': None, 'question_revision': 'r1',
        'workspace_revision': snapshot['revision'], 'configuration_revision': '0' * 64, 'source_ids': ['doc'],
        'questions': [{'id': 'q1', 'question': 'When is the meeting?'}, {'id': 'q2', 'question': 'Where?'}], 'rows': rows})
    return EvaluationAnswerStore(workspaces, results)


def run(store, *, answer=lambda question, passages: 'Friday, in Paris. [1]', judge=None, cancelled=None):
    calls = {'answer': [], 'judge': []}
    def fake_answer(question, passages):
        calls['answer'].append((question, passages))
        return answer(question, passages)
    def fake_judge(question, reply, passages):
        calls['judge'].append((question, reply, passages))
        return judge(question, reply, passages) if judge else {
            'claims': [{'claim': 'The meeting is on Friday.', 'supported': True},
                       {'claim': 'The meeting is in Paris.', 'supported': False}],
            'relevance': 'partial'}
    value = check_answers('book', BATCH, store=store, answer=fake_answer, judge=fake_judge,
                          cancelled=cancelled, models={'answer_model': 'm/answer', 'judge_model': 'm/judge'})
    return value, calls


def test_answers_are_written_from_saved_passages_and_judged_claim_by_claim(stores):
    value, calls = run(stores)

    assert calls['answer'] == [('When is the meeting?', [PASSAGE])]  # no passages: no paid call
    assert calls['judge'] == [('When is the meeting?', 'Friday, in Paris. [1]', [PASSAGE])]
    first, second = value['items']
    assert first == {'question_id': 'q1', 'run_id': '0' * 32, 'status': 'success', 'answer': 'Friday, in Paris. [1]',
                     'abstained': False, 'claims': [{'claim': 'The meeting is on Friday.', 'supported': True},
                                                    {'claim': 'The meeting is in Paris.', 'supported': False}],
                     'faithfulness': 0.5, 'relevance': 0.5, 'error': None}
    assert second['abstained'] is True and second['answer'] == ABSTENTION and second['faithfulness'] is None
    assert value['status'] == 'success' and value['answer_model'] == 'm/answer' and value['judge_model'] == 'm/judge'
    assert value['summary'] == {'checked_questions': 2, 'answered': 1, 'abstained': 1, 'faithfulness': 0.5,
                                'relevance': 0.5, 'unsupported_claims': 1, 'missing_evidence': 0}
    assert stores.get('book', BATCH) == value


def test_abstentions_are_not_judged_and_unchecked_batches_read_empty(stores):
    assert stores.get('book', BATCH)['status'] is None and stores.get('book', BATCH)['items'] == []
    value, calls = run(stores, answer=lambda question, passages: ABSTENTION)
    assert calls['judge'] == [] and value['summary']['abstained'] == 2 and value['summary']['faithfulness'] is None


def test_answer_without_claims_has_no_faithfulness_but_keeps_relevance(stores):
    value, _ = run(stores, judge=lambda *_: {'claims': [], 'relevance': 'none'})
    assert value['items'][0]['faithfulness'] is None and value['items'][0]['relevance'] == 0.0


def test_first_model_failure_stops_further_paid_calls_and_keeps_completed_items(stores):
    def failing(question, passages):
        raise RuntimeError('OpenRouter request failed with HTTP 402.')
    value, calls = run(stores, answer=failing)
    assert value['status'] == 'error' and len(value['items']) == 1 and calls['judge'] == []
    assert value['items'][0]['status'] == 'error' and 'HTTP 402' in value['items'][0]['error']
    assert stores.get('book', BATCH)['status'] == 'error'


def test_cancel_stops_before_the_next_question(stores):
    cancelled = Event()
    value, _ = run(stores, answer=lambda question, passages: cancelled.set() or 'Friday. [1]', cancelled=cancelled)
    assert value['status'] == 'cancelled' and len(value['items']) == 1


def test_expired_evidence_is_skipped_without_model_calls(stores):
    (stores.results.root / f"{'0' * 32}.json").unlink()
    value, calls = run(stores)
    assert calls['answer'] == [] and value['items'][0]['status'] == 'missing_evidence'
    assert value['summary']['missing_evidence'] == 1


def test_judge_sends_numbered_untrusted_passages_and_reads_strict_json(monkeypatch):
    sent = []
    def post(path, payload, **kwargs):
        sent.append(payload)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
            {'claims': [{'claim': 'Friday.', 'supported': True}], 'relevance': 'full'})}}]}
    monkeypatch.setattr(evaluation_answers, '_post', post)
    verdict = judge_answer('When?', 'Friday. [1]', [PASSAGE], {'judge_model': 'm/judge', 'embed_api_key': 'k',
                                                               'embed_base_url': 'https://openrouter.ai/api/v1'})
    assert verdict == {'claims': [{'claim': 'Friday.', 'supported': True}], 'relevance': 'full'}
    payload = sent[0]
    assert payload['model'] == 'm/judge' and payload['temperature'] == 0
    assert payload['response_format']['json_schema']['strict'] is True
    assert '<passage 1>\nThe meeting is on Friday.\n</passage 1>' in payload['messages'][1]['content']

    def bad(path, payload, **kwargs):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"claims": "x"}'}}]}
    monkeypatch.setattr(evaluation_answers, '_post', bad)
    with pytest.raises(RuntimeError, match='invalid verdict'):
        judge_answer('When?', 'Friday.', [PASSAGE], {'embed_api_key': 'k', 'embed_base_url': 'https://openrouter.ai/api/v1'})
