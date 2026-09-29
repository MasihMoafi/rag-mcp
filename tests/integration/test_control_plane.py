"""Browser HTTP, CLI, and MCP use real retrieval with an in-memory provider."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from http.client import HTTPConnection
from threading import Thread

from rag import cli, fetch, openrouter_pipeline, server, settings
from rag.companion import create_companion_server
from rag.conversations import ConversationStore
from rag.run_results import RunResultStore
from rag.trace_reader import list_traces, read_trace
from rag.workspaces import WorkspaceStore


def test_cli_and_mcp_search_same_workspace_without_provider(tmp_path, monkeypatch):
    for key, value in {
        'RAG_MCP_CONFIG': str(tmp_path / 'config.json'), 'RAG_MCP_DATA_DIR': str(tmp_path / 'indexes'),
        'RAG_MCP_TRACE_DIR': str(tmp_path / 'traces'), 'RAG_MCP_WORKSPACE_DIR': str(tmp_path / 'workspaces'),
        'RAG_MCP_EMBED_PROVIDER': 'none', 'RAG_MCP_RERANKER_TYPE': 'off',
        'RAG_MCP_CONTEXTUAL_RETRIEVAL': 'off', 'OPENROUTER_API_KEY': '', 'OPENAI_API_KEY': '',
    }.items():
        monkeypatch.setenv(key, value)
    store = WorkspaceStore()
    saved = store.replace('fixture', 'Fixture', [{'id': 'facts', 'name': 'facts.txt', 'sourceType': 'text',
                                                'content': 'The Cedar Lantern meeting happens Friday.'}])
    query = 'When is the Cedar Lantern meeting?'
    output = io.StringIO()
    assert cli.main(['search', query, '--workspace', 'fixture', '--revision', saved['revision'], '--json'],
                    stdout=output, stderr=io.StringIO()) == 0
    local = json.loads(output.getvalue())
    assert local['passages'][0]['source_id'] == 'facts'
    assert 'Cedar Lantern' in local['passages'][0]['content']
    hidden = io.StringIO()
    assert cli.main(['search', query, '--workspace', 'fixture', '--json', '--no-content'],
                    stdout=hidden, stderr=io.StringIO()) == 0
    assert 'content' not in json.loads(hidden.getvalue())['passages'][0]
    stale = io.StringIO()
    assert cli.main(['search', query, '--workspace', 'fixture', '--revision', 'stale', '--json'],
                    stdout=stale, stderr=io.StringIO()) == 1
    assert json.loads(stale.getvalue())['code'] == 'workspace_conflict'
    calls = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'search_workspace',
            'arguments': {'workspace_id': 'fixture', 'query': query, 'workspace_revision': saved['revision']}}},
        {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'search_workspace',
            'arguments': {'workspace_id': 'fixture', 'query': query, 'workspace_revision': saved['revision'],
                          'include_content': True}}},
        {'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': {'name': 'search_workspace',
            'arguments': {'workspace_id': 'fixture', 'query': query, 'workspace_revision': 'stale'}}},
    ]
    process = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'server.py')],
                             input=''.join(json.dumps(call) + '\n' for call in calls).encode(),
                             capture_output=True, env=os.environ.copy(), cwd=tmp_path, timeout=30)
    assert process.returncode == 0, process.stderr.decode()
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert any(tool['name'] == 'search_workspace' for tool in replies[1]['result']['tools'])
    metadata = json.loads(replies[2]['result']['content'][0]['text'])
    content = json.loads(replies[3]['result']['content'][0]['text'])
    assert replies[2]['result']['isError'] is False
    assert metadata['passages'][0]['source_id'] == content['passages'][0]['source_id'] == local['passages'][0]['source_id']
    assert metadata['workspace_revision'] == content['workspace_revision'] == local['workspace_revision']
    assert 'content' not in metadata['passages'][0] and 'score' not in metadata['passages'][0]
    assert content['passages'][0]['content'] == local['passages'][0]['content']
    assert replies[4]['result']['isError'] is True
    assert json.loads(replies[4]['result']['content'][0]['text'])['code'] == 'workspace_conflict'
    assert 'Cedar Lantern' not in replies[2]['result']['content'][0]['text']


def test_browser_cli_and_mcp_share_retrieval_settings_and_saved_runs(tmp_path, monkeypatch):
    for key, value in {
        'RAG_MCP_CONFIG': str(tmp_path / 'config.json'), 'RAG_MCP_DATA_DIR': str(tmp_path / 'indexes'),
        'RAG_MCP_TRACE_DIR': str(tmp_path / 'traces'), 'RAG_MCP_WORKSPACE_DIR': str(tmp_path / 'workspaces'),
        'RAG_MCP_EMBED_PROVIDER': 'none', 'RAG_MCP_RERANKER_TYPE': 'off',
        'RAG_MCP_CONTEXTUAL_RETRIEVAL': 'off', 'OPENROUTER_API_KEY': 'fixture-only',
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('RAG_MCP_TRACE_CONTENT', raising=False)
    evidence_requests = []
    def provider(request, **kwargs):
        payload = json.loads(request.data)
        evidence_requests.append(payload['messages'][1]['content'])
        answer = 'Friday. [1]' if 'Cite supporting passages' in payload['messages'][0]['content'] else 'Friday.'
        return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': answer}}],
                                     'usage': {'prompt_tokens': 20, 'completion_tokens': 3}}).encode())
    monkeypatch.setattr(openrouter_pipeline.urllib.request, 'urlopen', provider)
    store = WorkspaceStore()
    stored = store.replace('fixture', 'Fixture', [{'id': 'facts', 'name': 'facts.txt', 'sourceType': 'text',
                                                 'content': 'The Cedar Lantern meeting happens Friday.'}])
    gateway = create_companion_server(token='fixture-token', workspace_store=store, codex_ready=lambda: False,
                                      metrics_path=tmp_path / 'metrics.json')
    worker = Thread(target=gateway.serve_forever, daemon=True); worker.start()
    fetch.reset_pipelines()
    connection = HTTPConnection('127.0.0.1', gateway.server_port, timeout=15)
    try:
        payload = {'run_id': 'f' * 32, 'query': 'When is the Cedar Lantern meeting?', 'workspace_id': 'fixture',
                   'workspace_revision': stored['revision'], 'source_ids': ['facts']}
        connection.request('POST', '/v1/runs', json.dumps(payload),
                           {'Authorization': 'Bearer fixture-token', 'Origin': 'http://localhost:3000', 'Content-Type': 'application/json'})
        response = connection.getresponse()
        private = json.loads(response.read())
        assert response.status == 200 and private['status'] == 'success'
        assert private['evidence'][0]['source_id'] == 'facts'
        assert private['evidence'][0]['content'] == store.snapshot('fixture')['documents'][0]['content']
        stdout = io.StringIO()
        assert cli.main(['query', payload['query'], '--workspace', 'fixture', '--format', 'json'],
                        stdin=io.StringIO(), stdout=stdout, stderr=io.StringIO()) == 0
        assert json.loads(stdout.getvalue()) == {'result': 'Friday.'}
        assert json.loads(server.query_knowledge_base(payload['query'], workspace_id='fixture')) == {'result': 'Friday.'}
        assert len(evidence_requests) == 3 and len(set(evidence_requests)) == 1
        traces = [read_trace(item['trace_id']) for item in list_traces()['traces']]
        assert len(traces) == 3
        assert {trace['spans'][0]['attributes']['client'] for trace in traces} == {'studio', 'cli', 'mcp'}
        stages = [{span['operation'] for span in trace['spans'] if span['operation'].startswith(('retrieval.', 'selection.', 'synthesis'))} for trace in traces]
        assert stages[0] == stages[1] == stages[2]
        assert all(RunResultStore().read(trace['trace_id'])['evidence'][0]['source_id'] == 'facts' for trace in traces)
        assert 'fixture-only' not in json.dumps(traces) + json.dumps(private)
        for run_id, extra in [('a' * 32, {'mode': 'retrieve'}),
                              ('b' * 32, {'answer_model': 'fixture/alternative', 'configuration_revision': private['configuration_revision']})]:
            connection.request('POST', '/v1/runs', json.dumps({**payload, 'run_id': run_id, **extra}),
                               {'Authorization': 'Bearer fixture-token', 'Origin': 'http://localhost:3000', 'Content-Type': 'application/json'})
            response = connection.getresponse()
            result = json.loads(response.read())
            assert response.status == 200 and result['status'] == 'success'
            assert result['evidence'][0]['source_id'] == 'facts'
            if extra.get('mode') == 'retrieve':
                assert result['answer'] is None and len(evidence_requests) == 3
                assert not any(span['operation'] == 'synthesis' for span in read_trace(run_id)['spans'])
            else:
                assert result['configuration']['retrieval']['answer_model'] == 'fixture/alternative'
                assert settings.retrieval_config()['answer_model'] == private['configuration']['retrieval']['answer_model']
                assert len(evidence_requests) == 4
        conversation_id = 'c' * 32
        conversation_revision = None
        for run_id, question in [('d' * 32, 'When is the meeting?'), ('e' * 32, 'Which day?')]:
            connection.request('POST', '/v1/runs', json.dumps({**payload, 'run_id': run_id,
                               'query': question, 'conversation_id': conversation_id,
                               'conversation_revision': conversation_revision}),
                               {'Authorization': 'Bearer fixture-token', 'Origin': 'http://localhost:3000',
                                'Content-Type': 'application/json'})
            response = connection.getresponse()
            answer = json.loads(response.read())
            assert response.status == 200 and answer['status'] == 'success'
            conversation_revision = answer['conversation_revision']
        turns = ConversationStore(store, RunResultStore()).get('fixture', conversation_id)['turns']
        assert [turn['question'] for turn in turns] == ['When is the meeting?', 'Which day?']
        assert [turn['run_id'] for turn in turns] == ['d' * 32, 'e' * 32]
    finally:
        connection.close(); gateway.shutdown(); gateway.server_close(); worker.join(timeout=2)
        fetch.reset_pipelines()
