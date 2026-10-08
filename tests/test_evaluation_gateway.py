import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from rag.companion import create_companion_server
from rag.evaluation_sets import EvaluationSetStore
from rag.run_results import RunResultStore
from rag.workspaces import WorkspaceStore


def request(url, *, payload=None, token='secret', method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    with urlopen(Request(url, data=data, headers={'Authorization': f'Bearer {token}',
                 'Origin': 'http://127.0.0.1:3000', 'Content-Type': 'application/json'}, method=method), timeout=3) as response:
        return json.load(response)


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    snapshot = workspaces.replace('book', 'Book', [{'id': 'source', 'name': 'facts.txt', 'sourceType': 'text', 'content': 'Friday'}])
    questions = EvaluationSetStore(workspaces).replace('book', [
        {'id': 'one', 'question': 'When?'}, {'id': 'two', 'question': 'Where?'}], expected_revision=None)
    calls = []
    gate = {'wait': False, 'started': Event()}
    def execute(query, **kwargs):
        calls.append((query, kwargs))
        if gate['wait']:
            gate['started'].set()
            assert kwargs['cancelled'].wait(2)
        value = {'schema_version': 1, 'run_id': kwargs['run_id'], 'workspace_id': 'book',
                'status': 'cancelled' if kwargs['cancelled'].is_set() else 'success',
                'error': 'Cancelled' if kwargs['cancelled'].is_set() else None,
                'evidence': [] if kwargs['cancelled'].is_set() else [{'rank': 1, 'source_id': 'source', 'content': 'Friday',
                    'start_offset': 0, 'end_offset': 6}]}
        kwargs['result_store'].save(value)
        return value
    server = create_companion_server(token='secret', workspace_store=workspaces, run_executor=execute,
                                    result_store=RunResultStore(tmp_path / 'results'), metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    payload = {'batch_id': 'a' * 32, 'workspace_revision': snapshot['revision'],
               'question_revision': questions['revision'], 'source_ids': ['source']}
    try:
        yield f'http://127.0.0.1:{server.server_port}', payload, calls, gate, workspaces
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_batch_api_persists_retries_and_enforces_authentication(gateway):
    base, payload, calls, _, workspaces = gateway
    url = f'{base}/v1/workspaces/book/evaluation-runs'
    with pytest.raises(HTTPError) as denied:
        request(url, payload=payload, token='wrong')
    assert denied.value.code == 403 and not calls
    saved = request(url, payload=payload)
    assert saved['status'] == 'success' and len(saved['rows']) == 2
    assert all(options['mode'] == 'retrieve' for _, options in calls)
    assert len({options['configuration_revision'] for _, options in calls}) == 1
    assert request(f"{url}/{payload['batch_id']}") == saved
    listed = request(url)
    assert listed['batches'][0]['batch_id'] == saved['batch_id']
    assert listed['batches'][0]['completed_count'] == 2
    assert 'questions' not in listed['batches'][0]
    workspaces.replace('book', 'Changed', [], expected_revision=payload['workspace_revision'])
    assert request(url, payload=payload) == saved and len(calls) == 2
    with pytest.raises(HTTPError) as denied:
        request(f"{url}/{payload['batch_id']}", token='wrong')
    assert denied.value.code == 403


def test_batch_api_rejects_bad_requests_and_stale_scope(gateway):
    base, payload, calls, _, _ = gateway
    url = f'{base}/v1/workspaces/book/evaluation-runs'
    for invalid in ({}, {**payload, 'source_ids': [1]}, {**payload, 'batch_id': '../outside'}):
        with pytest.raises(HTTPError) as error:
            request(url, payload=invalid)
        assert error.value.code == 400
    with pytest.raises(HTTPError) as stale:
        request(url, payload={**payload, 'question_revision': 'stale'})
    assert stale.value.code == 409 and not calls


def test_batch_progress_cancel_and_query_exclusion(gateway):
    base, payload, calls, gate, _ = gateway
    gate['wait'] = True
    url = f'{base}/v1/workspaces/book/evaluation-runs'
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(request, url, payload=payload)
        assert gate['started'].wait(1)
        assert request(f"{url}/{payload['batch_id']}")['status'] == 'running'
        with pytest.raises(HTTPError) as busy:
            request(f'{base}/v1/runs', payload={'run_id': 'b' * 32, 'query': 'Other?', 'workspace_id': 'book'})
        assert busy.value.code == 409
        assert request(f"{url}/{payload['batch_id']}/cancel", payload={})['status'] == 'cancelling'
        saved = running.result(timeout=2)
    assert saved['status'] == 'cancelled' and len(calls) == 1
    assert request(f"{url}/{payload['batch_id']}") == saved


def test_label_api_requires_pairing_and_saved_evidence(gateway):
    base, payload, _, _, workspaces = gateway
    batch = request(f'{base}/v1/workspaces/book/evaluation-runs', payload=payload)
    run_id = batch['rows'][0]['run_id']
    url = f"{base}/v1/workspaces/book/evaluation-runs/{payload['batch_id']}/labels"
    assert request(url)['labels'] == []
    change = {'revision': None, 'run_id': run_id, 'rank': 1, 'label': 'relevant'}
    with pytest.raises(HTTPError) as denied:
        request(url, payload=change, token='wrong', method='PUT')
    assert denied.value.code == 403
    saved = request(url, payload=change, method='PUT')
    assert saved['labels'][0]['label'] == 'relevant'
    assert 'content' not in saved['labels'][0]
    assert request(url) == saved
    with pytest.raises(HTTPError) as stale:
        request(url, payload=change, method='PUT')
    assert stale.value.code == 409
    with pytest.raises(HTTPError) as forged:
        request(url, payload={**change, 'revision': saved['revision'], 'rank': 2}, method='PUT')
    assert forged.value.code == 400
    cleared = request(url, payload={**change, 'revision': saved['revision'], 'label': None}, method='PUT')
    assert cleared['labels'] == []


def test_score_api_scores_a_batch_against_marked_passages(gateway):
    base, payload, _, _, _ = gateway
    batch = request(f'{base}/v1/workspaces/book/evaluation-runs', payload=payload)
    url = f"{base}/v1/workspaces/book/evaluation-runs/{payload['batch_id']}"
    assert request(f'{url}/scores')['summary']['judged_questions'] == 0
    request(f'{url}/labels', payload={'revision': None, 'run_id': batch['rows'][0]['run_id'], 'rank': 1, 'label': 'relevant'}, method='PUT')
    scores = request(f'{url}/scores')
    assert scores['summary']['judged_questions'] == 1 and scores['summary']['mrr'] == 1.0
    with pytest.raises(HTTPError) as denied:
        request(f'{url}/scores', token='wrong')
    with pytest.raises(HTTPError) as missing:
        request(f"{base}/v1/workspaces/book/evaluation-runs/{'b' * 32}/scores")
    assert denied.value.code == 403 and missing.value.code == 404


def test_conversation_api_uses_private_saved_runs_and_revisions(gateway):
    base, _, _, _, workspaces = gateway
    run_id, conversation_id = 'b' * 32, 'c' * 32
    RunResultStore(workspaces.root.parent / 'results').save({
        'schema_version': 1, 'workspace_id': 'book', 'run_id': run_id, 'status': 'success',
        'answer': 'Friday. [1]', 'workspace_revision': 'rev', 'configuration_revision': 'd' * 64,
        'evidence': [{'rank': 1, 'source_id': 'source', 'name': 'facts.txt', 'content': 'Friday',
                      'start_offset': 0, 'end_offset': 6, 'score': None}],
    })
    url = f'{base}/v1/workspaces/book/conversations'
    change = {'revision': None, 'run_id': run_id, 'question': 'When?'}
    with pytest.raises(HTTPError) as denied:
        request(f'{url}/{conversation_id}/turns', payload=change, token='wrong')
    assert denied.value.code == 403
    first = request(f'{url}/{conversation_id}/turns', payload=change)
    assert first['turns'][0]['answer'] == 'Friday. [1]'
    assert request(f'{url}/{conversation_id}') == first
    listed = request(url)
    assert listed['conversations'][0]['conversation_id'] == conversation_id
    assert 'answer' not in listed['conversations'][0]
    assert request(f'{url}/{conversation_id}/turns', payload=change) == first
    with pytest.raises(HTTPError) as stale:
        request(f'{url}/{conversation_id}/turns', payload={**change, 'run_id': 'e' * 32})
    assert stale.value.code == 409
    with pytest.raises(HTTPError) as forged:
        request(f"{url}/{'f' * 32}/turns", payload={**change, 'run_id': 'e' * 32})
    assert forged.value.code == 404
