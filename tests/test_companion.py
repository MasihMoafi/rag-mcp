import json
from types import SimpleNamespace
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from rag.companion import GenerationResult, create_companion_server, run_codex_generation
from rag.run_results import RunResultStore
from rag.workspaces import WorkspaceStore


def request(url, token="secret", origin="https://rag.masihmoafi.com", body=None, method=None):
    headers = {"Authorization": f"Bearer {token}", "Origin": origin}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    return urlopen(Request(url, data=data, headers=headers, method=method), timeout=2)


def test_companion_accepts_only_the_launchers_extra_origin_with_its_token(tmp_path):
    server = create_companion_server(token='secret', allowed_origins={'http://127.0.0.1:3016'},
                                     workspace_store=WorkspaceStore(tmp_path / 'workspaces'), metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f'http://127.0.0.1:{server.server_port}/v1/workspaces'
    try:
        with request(url, origin='http://127.0.0.1:3016') as response:
            assert response.status == 200
            assert response.headers['Access-Control-Allow-Origin'] == 'http://127.0.0.1:3016'
        for origin, token in [('http://127.0.0.1:3017', 'secret'), ('http://127.0.0.1:3016', 'wrong')]:
            with pytest.raises(HTTPError) as denied:
                request(url, origin=origin, token=token)
            assert denied.value.code == 403
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_companion_chunk_preview_is_private_revisioned_and_paged(tmp_path):
    store = WorkspaceStore(tmp_path / 'workspaces')
    saved = store.replace('book', 'Book', [
        {'id': 'chapter/one', 'name': 'book.md', 'source_type': 'markdown', 'content': '# First\n\n' + 'Cedar Lantern. ' * 100},
        {'id': 'private', 'name': 'private.txt', 'source_type': 'text', 'content': 'Private passage'},
    ])
    server = create_companion_server(token='secret', workspace_store=store, metrics_path=tmp_path / 'metrics.json',
                                     embedder_factory=lambda: (_ for _ in ()).throw(AssertionError('must not load a model')))
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}/v1/workspaces/book/chunks'
    url = f'{base}?document_id=chapter%2Fone&revision={saved["revision"]}&limit=1&offset=0'
    try:
        with request(url) as response:
            first = json.load(response)
            assert response.headers['Cache-Control'] == 'no-store'
        assert first['document_id'] == 'chapter/one'
        assert first['total_chunks'] > 1
        assert len(first['chunks']) == 1
        assert first['next_offset'] == 1
        assert 'Private passage' not in str(first)
        with request(f'{base}?document_id=chapter%2Fone&revision={saved["revision"]}&limit=1&offset=1') as response:
            second = json.load(response)
        assert second['chunks'][0]['ordinal'] == 1
        with pytest.raises(HTTPError) as stale:
            request(f'{base}?document_id=chapter%2Fone&revision=old')
        assert stale.value.code == 409
        with pytest.raises(HTTPError) as missing:
            request(f'{base}?document_id=missing&revision={saved["revision"]}')
        assert missing.value.code == 404
        with pytest.raises(HTTPError) as invalid:
            request(f'{base}?document_id=chapter%2Fone&revision={saved["revision"]}&limit=101')
        assert invalid.value.code == 400
        with pytest.raises(HTTPError) as denied:
            request(url, token='wrong')
        assert denied.value.code == 403
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_paired_evaluation_set_has_independent_revision_and_rejects_stale_edit(tmp_path):
    store = WorkspaceStore(tmp_path / 'workspaces')
    source = store.replace('book', 'Book', [])
    server = create_companion_server(token='secret', workspace_store=store, metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f'http://127.0.0.1:{server.server_port}/v1/workspaces/book/evaluations'
    try:
        with request(url) as response:
            assert json.load(response)['revision'] is None
        with request(url, body={'revision': None, 'questions': [{'id': 'q1', 'question': 'When?'}]}, method='PUT') as response:
            saved = json.load(response)
        assert saved['questions'][0]['question'] == 'When?'
        with request(url) as response:
            assert json.load(response) == saved
        assert store.get('book')['revision'] == source['revision']
        with pytest.raises(HTTPError) as stale:
            request(url, body={'revision': None, 'questions': []}, method='PUT')
        assert stale.value.code == 409
        with pytest.raises(HTTPError) as denied:
            request(url, token='wrong')
        assert denied.value.code == 403
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_engine_gateway_saves_results_and_retries_do_not_repeat_execution(tmp_path):
    calls = []
    def execute(query, **kwargs):
        calls.append(query)
        return {'schema_version': 1, 'run_id': kwargs['run_id'], 'status': 'success', 'answer': 'Friday.', 'error': None, 'evidence': []}
    server = create_companion_server(token='secret', run_executor=execute,
                                    result_store=RunResultStore(tmp_path / 'results'), metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    payload = {'run_id': 'a' * 32, 'query': 'When?', 'workspace_id': 'book', 'source_ids': ['chapter']}
    try:
        with pytest.raises(HTTPError) as denied:
            request(f'{base}/v1/runs', token='wrong', body=payload)
        assert denied.value.code == 403
        for _ in range(2):
            with request(f'{base}/v1/runs', body=payload) as response:
                assert json.load(response)['answer'] == 'Friday.'
        assert calls == ['When?']
        with request(f"{base}/v1/runs/{payload['run_id']}/result") as response:
            assert json.load(response)['answer'] == 'Friday.'
        with pytest.raises(HTTPError) as reused:
            request(f'{base}/v1/runs', body={**payload, 'query': 'Different?'})
        assert reused.value.code == 409
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_gateway_conversation_query_saves_answer_in_one_request(tmp_path):
    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    snapshot = workspaces.replace('book', 'Book', [
        {'id': 'chapter', 'name': 'chapter.txt', 'sourceType': 'text', 'content': 'Cedar meets Friday.'},
    ])
    results = RunResultStore(tmp_path / 'results')
    calls = []
    def execute(query, **kwargs):
        calls.append((query, kwargs['source_ids'], kwargs['workspace_revision']))
        return {'schema_version': 1, 'run_id': kwargs['run_id'], 'workspace_id': 'book',
                'workspace_revision': snapshot['revision'], 'status': 'success',
                'answer': 'Friday. [1]', 'error': None, 'evidence': []}
    server = create_companion_server(token='secret', workspace_store=workspaces,
                                     result_store=results, run_executor=execute,
                                     metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    conversation_id = 'c' * 32
    payload = {'run_id': 'a' * 32, 'query': 'When?', 'workspace_id': 'book',
               'workspace_revision': snapshot['revision'], 'source_ids': ['chapter'],
               'conversation_id': conversation_id, 'conversation_revision': None}
    try:
        with request(f'{base}/v1/runs', body=payload) as response:
            first = json.load(response)
        assert first['status'] == 'success' and first['conversation_revision']
        with request(f'{base}/v1/workspaces/book/conversations/{conversation_id}') as response:
            saved = json.load(response)
        assert saved['turns'][0]['question'] == 'When?'
        assert saved['turns'][0]['run_id'] == payload['run_id']
        with request(f'{base}/v1/runs', body=payload) as response:
            assert json.load(response)['conversation_revision'] == first['conversation_revision']
        second = {**payload, 'run_id': 'b' * 32, 'query': 'What time?',
                  'conversation_revision': first['conversation_revision']}
        with request(f'{base}/v1/runs', body=second) as response:
            assert json.load(response)['conversation_revision'] != first['conversation_revision']
        assert calls == [('When?', ['chapter'], snapshot['revision']),
                         ('Previous question (context only, do not answer): When?\n'
                          'Current question (answer this question only): What time?', ['chapter'], snapshot['revision'])]
        with pytest.raises(HTTPError) as stale:
            request(f'{base}/v1/runs', body={**second, 'run_id': 'd' * 32})
        assert stale.value.code == 409
        assert len(calls) == 2
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_completed_run_releases_engine_lock_before_response_is_consumed(tmp_path, monkeypatch):
    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    snapshot = workspaces.replace('book', 'Book', [])
    server = create_companion_server(token='secret', workspace_store=workspaces,
                                     result_store=RunResultStore(tmp_path / 'results'),
                                     run_executor=lambda query, **kwargs: {
                                         'schema_version': 1, 'run_id': kwargs['run_id'],
                                         'workspace_id': 'book', 'workspace_revision': snapshot['revision'],
                                         'status': 'success', 'answer': 'Friday.', 'error': None, 'evidence': []},
                                     metrics_path=tmp_path / 'metrics.json')
    original_send = server.RequestHandlerClass._send_json
    response_sent = Event()
    release_response = Event()

    def delay_after_first_response(handler, status, value):
        original_send(handler, status, value)
        if handler.path == '/v1/runs' and status == 200 and value.get('run_id') == 'a' * 32:
            response_sent.set()
            release_response.wait(timeout=5)

    monkeypatch.setattr(server.RequestHandlerClass, '_send_json', delay_after_first_response)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}/v1/runs'
    payload = {'run_id': 'a' * 32, 'query': 'When?', 'workspace_id': 'book',
               'workspace_revision': snapshot['revision']}
    try:
        with request(base, body=payload) as response:
            assert json.load(response)['status'] == 'success'
        assert response_sent.wait(timeout=2)
        with request(base, body={**payload, 'run_id': 'b' * 32}) as response:
            assert json.load(response)['status'] == 'success'
    finally:
        release_response.set()
        server.shutdown(); server.server_close(); thread.join(timeout=2)


@pytest.mark.parametrize('operation', ['config', 'batch'])
def test_completed_control_action_releases_engine_lock_before_response(tmp_path, monkeypatch, operation):
    from rag import companion as companion_module, control

    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    snapshot = workspaces.replace('book', 'Book', [])
    monkeypatch.setattr(control, 'patch_config', lambda payload: {'saved': True})
    monkeypatch.setattr(companion_module, 'execute_evaluation', lambda *args, **kwargs: {'status': 'complete'})
    server = create_companion_server(token='secret', workspace_store=workspaces,
                                     result_store=RunResultStore(tmp_path / 'results'),
                                     run_executor=lambda query, **kwargs: {
                                         'schema_version': 1, 'run_id': kwargs['run_id'],
                                         'workspace_id': 'book', 'workspace_revision': snapshot['revision'],
                                         'status': 'success', 'answer': 'Friday.', 'error': None, 'evidence': []},
                                     metrics_path=tmp_path / 'metrics.json')
    base = f'http://127.0.0.1:{server.server_port}'
    path = '/v1/config' if operation == 'config' else '/v1/workspaces/book/evaluation-runs'
    original_send = server.RequestHandlerClass._send_json
    response_sent = Event()
    release_response = Event()

    def delay_control_response(handler, status, value):
        original_send(handler, status, value)
        if handler.path == path and status == 200:
            response_sent.set()
            release_response.wait(timeout=5)

    monkeypatch.setattr(server.RequestHandlerClass, '_send_json', delay_control_response)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    action_payload = {'top': 3} if operation == 'config' else {
        'batch_id': 'e' * 32, 'question_revision': 'revision',
        'workspace_revision': snapshot['revision'], 'source_ids': ['chapter']}
    try:
        with request(f'{base}{path}', body=action_payload,
                     method='PATCH' if operation == 'config' else 'POST') as response:
            assert response.status == 200
        assert response_sent.wait(timeout=2)
        with request(f'{base}/v1/runs', body={'run_id': 'b' * 32, 'query': 'When?',
                                            'workspace_id': 'book'}) as response:
            assert json.load(response)['status'] == 'success'
    finally:
        release_response.set()
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_gateway_retry_repairs_unsaved_conversation_without_reexecuting(tmp_path, monkeypatch):
    from rag.conversations import ConversationStore
    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    snapshot = workspaces.replace('book', 'Book', [])
    results = RunResultStore(tmp_path / 'results')
    calls = []
    original_append = ConversationStore.append
    def fail_once(self, *args, **kwargs):
        monkeypatch.setattr(ConversationStore, 'append', original_append)
        raise OSError('fixture write failure')
    monkeypatch.setattr(ConversationStore, 'append', fail_once)
    def execute(query, **kwargs):
        calls.append(query)
        return {'schema_version': 1, 'run_id': kwargs['run_id'], 'workspace_id': 'book',
                'workspace_revision': snapshot['revision'], 'status': 'success',
                'answer': 'Friday.', 'error': None, 'evidence': []}
    server = create_companion_server(token='secret', workspace_store=workspaces,
                                     result_store=results, run_executor=execute,
                                     metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    payload = {'run_id': 'a' * 32, 'query': 'When?', 'workspace_id': 'book',
               'conversation_id': 'c' * 32, 'conversation_revision': None}
    try:
        with request(f'{base}/v1/runs', body=payload) as response:
            first = json.load(response)
        assert first['conversation_error'] and 'conversation_revision' not in first
        with request(f'{base}/v1/runs', body=payload) as response:
            second = json.load(response)
        assert second['conversation_revision'] and 'conversation_error' not in second
        assert calls == ['When?']
        assert ConversationStore(workspaces, results).get('book', payload['conversation_id'])['turns'][0]['run_id'] == payload['run_id']
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
    restarted = create_companion_server(token='secret', workspace_store=workspaces,
                                        result_store=results, run_executor=execute,
                                        metrics_path=tmp_path / 'metrics.json')
    restarted_thread = Thread(target=restarted.serve_forever, daemon=True); restarted_thread.start()
    try:
        with request(f'http://127.0.0.1:{restarted.server_port}/v1/runs', body=payload) as response:
            assert json.load(response)['conversation_revision'] == second['conversation_revision']
        assert calls == ['When?']
    finally:
        restarted.shutdown(); restarted.server_close(); restarted_thread.join(timeout=2)


def test_gateway_retry_after_restart_recovers_first_saved_run(tmp_path):
    from rag.conversations import ConversationStore
    from rag.execution import request_fingerprint

    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    snapshot = workspaces.replace('book', 'Book', [])
    results = RunResultStore(tmp_path / 'results')
    calls = []
    original_save = results.save
    saves = 0

    def fail_second_save(result):
        nonlocal saves
        saves += 1
        if saves == 2:
            raise OSError('fixture second-save failure')
        original_save(result)

    results.save = fail_second_save

    def execute(query, **kwargs):
        calls.append(query)
        result = {'schema_version': 1, 'run_id': kwargs['run_id'], 'workspace_id': 'book',
                  'workspace_revision': snapshot['revision'], 'status': 'success',
                  'answer': 'Friday.', 'error': None, 'evidence': [],
                  'request_hash': kwargs.get('request_hash_override') or request_fingerprint(
                      query, workspace_id='book')}
        results.save(result)
        return result

    payload = {'run_id': 'a' * 32, 'query': 'When?', 'workspace_id': 'book',
               'conversation_id': 'c' * 32, 'conversation_revision': None}
    server = create_companion_server(token='secret', workspace_store=workspaces,
                                     result_store=results, run_executor=execute,
                                     metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        with request(f'http://127.0.0.1:{server.server_port}/v1/runs', body=payload) as response:
            first = json.load(response)
        assert first['persistence_error'] and 'conversation_revision' not in first
        assert calls == ['When?']
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)

    restarted = create_companion_server(token='secret', workspace_store=workspaces,
                                        result_store=RunResultStore(tmp_path / 'results'),
                                        run_executor=execute, metrics_path=tmp_path / 'metrics.json')
    restarted_thread = Thread(target=restarted.serve_forever, daemon=True); restarted_thread.start()
    try:
        with request(f'http://127.0.0.1:{restarted.server_port}/v1/runs', body=payload) as response:
            repaired = json.load(response)
        assert repaired['conversation_revision'] and 'conversation_error' not in repaired
        assert calls == ['When?']
        assert ConversationStore(workspaces, RunResultStore(tmp_path / 'results')).get(
            'book', payload['conversation_id'])['turns'][0]['run_id'] == payload['run_id']
    finally:
        restarted.shutdown(); restarted.server_close(); restarted_thread.join(timeout=2)


def test_engine_gateway_can_cancel_and_never_queues_parallel_queries(tmp_path):
    from threading import Event
    from concurrent.futures import ThreadPoolExecutor
    started = Event()
    def execute(query, **kwargs):
        started.set()
        assert kwargs['cancelled'].wait(timeout=2)
        return {'schema_version': 1, 'run_id': kwargs['run_id'], 'status': 'cancelled', 'answer': None, 'error': 'Query cancelled.', 'evidence': []}
    server = create_companion_server(token='secret', run_executor=execute,
                                    result_store=RunResultStore(tmp_path / 'results'), metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    payload = {'run_id': 'b' * 32, 'query': 'When?', 'workspace_id': 'book'}
    def run():
        with request(f'{base}/v1/runs', body=payload) as response:
            return json.load(response)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(run)
            assert started.wait(timeout=1)
            with pytest.raises(HTTPError) as busy:
                request(f'{base}/v1/runs', body={**payload, 'run_id': 'c' * 32})
            assert busy.value.code == 409
            with request(f"{base}/v1/runs/{payload['run_id']}/cancel", method='POST') as response:
                assert json.load(response)['status'] == 'cancelling'
            assert pending.result(timeout=2)['status'] == 'cancelled'
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_engine_config_is_paired_and_reads_cli_changes(monkeypatch, tmp_path):
    from rag import settings
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.delenv('RAG_MCP_RERANK_TOP_K', raising=False)
    server = create_companion_server(token='secret', codex_ready=lambda: False, metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with pytest.raises(HTTPError) as denied:
            request(f'{base}/v1/config', token='wrong', body={'top': 3}, method='PATCH')
        assert denied.value.code == 403
        with request(f'{base}/v1/config', body={'top': 3}, method='PATCH') as response:
            assert json.load(response)['retrieval']['top_k'] == '3'
        settings.update_settings({'top_k': '8', 'embed_api_key': 'private-secret'})
        with request(f'{base}/v1/config') as response:
            config = json.load(response)
            assert config['retrieval']['top_k'] == '8'
            assert 'private-secret' not in str(config)
        with request(f'{base}/v1/capabilities') as response:
            assert len(json.load(response)['stages']) == 4
        with request(f'{base}/v1/health') as response:
            assert json.load(response)['ready'] is True
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


class FakeEmbedder:
    model_name = "fake-embedder"

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 2.0]


def test_codex_generation_reads_reported_usage(monkeypatch):
    monkeypatch.setattr("rag.companion.shutil.which", lambda _name: "/usr/bin/codex")

    def run(command, **_kwargs):
        assert "--json" in command
        output = command[command.index("--output-last-message") + 1]
        with open(output, "w") as stream:
            stream.write("Grounded answer [1]\n")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"turn.completed","usage":{"input_tokens":41,"output_tokens":9}}\n',
        )

    monkeypatch.setattr("rag.companion.subprocess.run", run)
    result = run_codex_generation({"question": "What?", "history": [], "chunks": []})
    assert result == GenerationResult("Grounded answer [1]", 41, 9)


def test_companion_rejects_unpaired_requests_and_streams_grounded_answers(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.delenv("RAG_MCP_EMBED_PROVIDER", raising=False)
    calls = []
    server = create_companion_server(
        token="secret",
        run_codex=lambda payload: calls.append(payload) or GenerationResult(
            answer="Cedar Lantern [1]", input_tokens=123, output_tokens=17
        ),
        codex_ready=lambda: True,
        embedder_factory=lambda: None,
        metrics_path=tmp_path / "metrics.json",
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with request(f"{base}/health") as response:
            assert json.load(response) == {
                "ready": True,
                "model": "gpt-5.6-luna",
                "embedding": {"provider": "none", "model": ""},
            }
            assert response.headers["Access-Control-Allow-Origin"] == "https://rag.masihmoafi.com"

        try:
            request(f"{base}/embed", body={"texts": ["hello"]})
        except HTTPError as error:
            assert error.code == 409
            assert "rag setup" in json.load(error)["error"]
        else:
            raise AssertionError("embedding without a model must be refused")

        for token, origin in (("wrong", "https://rag.masihmoafi.com"), ("secret", "https://evil.example")):
            try:
                request(f"{base}/generate", token=token, origin=origin, body={"question": "Fact?", "chunks": []})
            except HTTPError as error:
                assert error.code == 403
            else:
                raise AssertionError("unpaired companion request was accepted")

        payload = {
            "question": "What is the codename?",
            "history": [],
            "chunks": [{
                "id": "1",
                "documentId": "facts",
                "documentTitle": "facts.md",
                "content": "The codename is Cedar Lantern.",
                "rank": 1,
            }],
        }
        with request(f"{base}/generate", body=payload) as response:
            events = [json.loads(line) for line in response.read().decode().splitlines()]
        assert events == [
            {"type": "meta", "mode": "provider", "provider": "Codex · Local", "model": "gpt-5.6-luna"},
            {"type": "delta", "text": "Cedar Lantern [1]"},
            {"type": "done"},
        ]
        assert calls == [payload]
        with request(f"{base}/v1/dashboard") as response:
            dashboard = json.load(response)
        assert dashboard["operations"]["generation"]["requests"] == 1
        assert dashboard["operations"]["generation"]["successes"] == 1
        assert dashboard["operations"]["generation"]["failures"] == 0
        assert dashboard["operations"]["generation"]["average_latency_ms"] >= 0
        assert dashboard["models"]["generation"] == "gpt-5.6-luna"
        assert dashboard["usage"] == {"input_tokens": 123, "output_tokens": 17, "cost_usd": None}
        assert dashboard["gates"]["native_operation_metrics"] is True
        assert dashboard["gates"]["durable_operation_metrics"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    restarted = create_companion_server(
        token="secret",
        codex_ready=lambda: True,
        embedder_factory=lambda: None,
        metrics_path=tmp_path / "metrics.json",
    )
    thread = Thread(target=restarted.serve_forever, daemon=True)
    thread.start()
    try:
        with request(f"http://127.0.0.1:{restarted.server_port}/v1/dashboard") as response:
            dashboard = json.load(response)
        assert dashboard["usage"]["input_tokens"] == 123
        assert dashboard["usage"]["output_tokens"] == 17
    finally:
        restarted.shutdown()
        restarted.server_close()
        thread.join(timeout=2)


def test_companion_embeds_with_the_configured_model(tmp_path):
    built = []

    def factory():
        built.append(True)
        return FakeEmbedder()

    metrics_path = tmp_path / "metrics.json"
    server = create_companion_server(token="secret", codex_ready=lambda: False, embedder_factory=factory, metrics_path=metrics_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with request(f"{base}/health") as response:
            assert response.status == 200
            assert json.load(response)["ready"] is False
        with request(f"{base}/embed", body={"texts": ["ab", "abcd"]}) as response:
            assert json.load(response) == {
                "vectors": [[2.0, 1.0], [4.0, 1.0]],
                "model": "fake-embedder",
                "dimensions": 2,
            }
        with request(f"{base}/embed", body={"texts": ["x"]}) as response:
            assert json.load(response)["dimensions"] == 2
        assert built == [True], "the embedder is built once and reused"

        with request(f"{base}/embed", body={"texts": ["ab"], "kind": "query"}) as response:
            assert json.load(response)["vectors"] == [[2.0, 2.0]]

        with request(f"{base}/v1/dashboard") as response:
            dashboard = json.load(response)
        assert dashboard["operations"]["embedding"]["requests"] == 3
        assert dashboard["operations"]["embedding"]["successes"] == 3
        assert dashboard["models"]["embedding"] == "fake-embedder"

        for bad in ({"texts": []}, {"texts": "not a list"}, {"texts": ["", "ok"]}, {"texts": ["ok"], "kind": "invalid"}, {}):
            try:
                request(f"{base}/embed", body=bad)
            except HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError(f"invalid embed payload accepted: {bad}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    restarted = create_companion_server(token="secret", codex_ready=lambda: False, embedder_factory=factory, metrics_path=metrics_path)
    thread = Thread(target=restarted.serve_forever, daemon=True)
    thread.start()
    try:
        with request(f"http://127.0.0.1:{restarted.server_port}/v1/dashboard") as response:
            dashboard = json.load(response)
        assert dashboard["operations"]["embedding"]["requests"] == 3
        assert dashboard["operations"]["embedding"]["successes"] == 3
    finally:
        restarted.shutdown()
        restarted.server_close()
        thread.join(timeout=2)


def test_companion_reads_shared_runs_only_when_paired(monkeypatch, tmp_path):
    from rag import tracing
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path / 'traces'))
    monkeypatch.delenv('RAG_MCP_TRACE_CONTENT', raising=False)
    with tracing.request_trace():
        with tracing.span('retrieval.lexical', candidate_limit=50):
            pass
    server = create_companion_server(token='secret', codex_ready=lambda: False,
                                     metrics_path=tmp_path / 'metrics.json')
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        for token, origin in [('wrong', 'http://localhost:3000'), ('secret', 'https://evil.example')]:
            try:
                request(f'{base}/v1/runs', token=token, origin=origin)
            except HTTPError as error:
                assert error.code == 403
            else:
                raise AssertionError('unpaired trace access allowed')
        with request(f'{base}/v1/runs?limit=1') as response:
            assert response.headers['Cache-Control'] == 'no-store'
            listing = json.load(response)
        trace_id = listing['traces'][0]['trace_id']
        with request(f'{base}/v1/runs/{trace_id}') as response:
            assert json.load(response)['spans'][1]['operation'] == 'retrieval.lexical'
        for path, status in [('/v1/runs?limit=101', 400), ('/v1/runs?offset=-1', 400),
                             ('/v1/runs/../../secret', 404), ('/v1/runs/' + 'a' * 32, 404)]:
            try:
                request(base + path)
            except HTTPError as error:
                assert error.code == status
            else:
                raise AssertionError('invalid request accepted')
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_companion_replaces_and_lists_private_workspaces_only_when_paired(tmp_path):
    from rag.workspaces import WorkspaceStore

    store = WorkspaceStore(tmp_path / "workspaces")
    server = create_companion_server(
        token="secret",
        codex_ready=lambda: False,
        metrics_path=tmp_path / "metrics.json",
        workspace_store=store,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    payload = {
        "name": "Book research",
        "documents": [{
            "id": "rag-book",
            "name": "rag.md",
            "sourceType": "markdown",
            "content": "A shared document marker",
        }],
    }
    try:
        for token, origin in (("wrong", "https://rag.masihmoafi.com"), ("secret", "https://evil.example")):
            with pytest.raises(HTTPError) as rejected:
                request(f"{base}/v1/workspaces/book", token=token, origin=origin, body=payload, method="PUT")
            assert rejected.value.code == 403

        with request(f"{base}/v1/workspaces/book", body=payload, method="PUT") as response:
            stored = json.load(response)
        assert stored["id"] == "book"
        assert stored["document_count"] == 1
        assert all("content" not in item for item in stored["documents"])

        with request(f"{base}/v1/workspaces") as response:
            assert json.load(response)["workspaces"] == [{
                "id": "book", "name": "Book research", "document_count": 1,
                "total_bytes": len("A shared document marker"),
            }]
        with request(f"{base}/v1/workspaces/book") as response:
            detail = json.load(response)
        assert detail["documents"][0]["name"] == "rag.md"
        assert all("content" not in item for item in detail["documents"])
        assert {path.read_text() for path in store.resolve("book").iterdir()} == {"A shared document marker"}
        with request(f"{base}/v1/workspaces/book/snapshot") as response:
            snapshot = json.load(response)
        assert snapshot["documents"] == payload["documents"]
        with request(f"{base}/v1/workspaces/book/snapshot?revision={stored['revision']}") as response:
            assert response.status == 204
            assert response.read() == b""
        with pytest.raises(HTTPError) as stale:
            request(f"{base}/v1/workspaces/book", body={**payload, "documents": []}, method="PUT")
        assert stale.value.code == 409
        with request(f"{base}/v1/workspaces/book", body={**payload, "revision": stored["revision"]}, method="PUT") as response:
            assert json.load(response)["revision"] == stored["revision"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
