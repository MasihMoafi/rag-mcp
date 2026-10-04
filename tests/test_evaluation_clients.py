import io
import json

from filelock import FileLock
import pytest

from rag.cli import main
from rag.evaluation_sets import EvaluationSetStore
from rag.evaluation_runs import EvaluationRunStore
from rag.run_results import RunResultStore
from rag.workspaces import WorkspaceStore


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(tmp_path / 'workspaces'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    workspaces = WorkspaceStore()
    workspaces.replace('book', 'Book', [{'id': 'source', 'name': 'facts.txt', 'sourceType': 'text', 'content': 'Friday'}])
    EvaluationSetStore(workspaces).replace('book', [{'id': 'q1', 'question': 'When?'}], expected_revision=None)
    calls = []
    def execute(query, **kwargs):
        calls.append(kwargs)
        print('Fixture retrieval diagnostic')
        return {'run_id': kwargs['run_id'], 'status': 'success', 'error': None, 'evidence': [{'source_id': 'source'}]}
    monkeypatch.setattr('rag.execution.execute_run', execute)
    return workspaces, calls


def test_cli_runs_and_reads_one_durable_batch_without_polluting_json(workspace):
    workspaces, calls = workspace
    output, errors = io.StringIO(), io.StringIO()
    assert main(['evaluations', 'book', '--run', '--source', 'source', '--json'], stdout=output, stderr=errors) == 0
    saved = json.loads(output.getvalue())
    assert saved['status'] == 'success' and calls[0]['client'] == 'cli'
    assert calls[0]['mode'] == 'retrieve'
    assert 'Fixture retrieval diagnostic' in errors.getvalue()
    output = io.StringIO()
    assert main(['evaluations', 'book', '--batch', saved['batch_id'], '--json'], stdout=output) == 0
    assert json.loads(output.getvalue()) == saved == EvaluationRunStore(workspaces).get('book', saved['batch_id'])
    output = io.StringIO()
    assert main(['evaluations', 'book', '--runs', '--json'], stdout=output) == 0
    assert json.loads(output.getvalue())['batches'][0]['batch_id'] == saved['batch_id']
    assert len(calls) == 1
    assert main(['evaluations', 'book', '--runs', '--source', 'source'], stderr=io.StringIO()) == 2


def test_mcp_uses_same_store_and_handles_busy_engine(workspace, monkeypatch, capsys):
    from rag import server
    workspaces, calls = workspace
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    replies.clear()
    def call(name, **arguments):
        server._handle_request({'id': 1, 'method': 'tools/call', 'params': {'name': name, 'arguments': arguments}})
        return json.loads(replies[-1]['content'][0]['text'])
    saved = call('run_evaluation', workspace_id='book')
    assert not replies[-1]['isError'] and calls[0]['client'] == 'mcp'
    captured = capsys.readouterr()
    assert not captured.out and not captured.err
    assert call('get_evaluation_runs', workspace_id='book', batch_id=saved['batch_id']) == saved
    assert call('get_evaluation_runs', workspace_id='book')['batches'][0]['batch_id'] == saved['batch_id']
    with FileLock(workspaces.root / '.evaluation.lock'):
        assert 'running' in call('run_evaluation', workspace_id='book')['error']
        assert replies[-1]['isError']
    assert len(calls) == 1


def test_cli_and_mcp_share_human_labels(workspace, monkeypatch):
    from rag import server
    workspaces, _ = workspace
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(workspaces.root.parent / 'traces'))
    batch_id, run_id = 'a' * 32, 'b' * 32
    EvaluationRunStore(workspaces).save({
        'schema_version': 1, 'workspace_id': 'book', 'batch_id': batch_id, 'status': 'success',
        'started_unix': 1, 'finished_unix': 2, 'error': None, 'question_revision': 'q1',
        'workspace_revision': 'w1', 'configuration_revision': 'c' * 64, 'source_ids': ['source'],
        'questions': [{'id': 'q1', 'question': 'When?'}],
        'rows': [{'question_id': 'q1', 'run_id': run_id, 'status': 'success', 'passage_count': 1,
                  'source_count': 1, 'duration_ms': 4, 'error': None}],
    })
    RunResultStore().save({'schema_version': 1, 'workspace_id': 'book', 'run_id': run_id, 'status': 'success',
                           'evidence': [{'rank': 1, 'source_id': 'source', 'content': 'Friday',
                                         'start_offset': 0, 'end_offset': 6}]})
    output = io.StringIO()
    assert main(['labels', 'book', batch_id, '--set', f'{run_id}:1:relevant', '--json'], stdout=output) == 0
    marked = json.loads(output.getvalue())
    assert marked['labels'][0]['label'] == 'relevant' and 'content' not in marked['labels'][0]
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    def call(name, **arguments):
        server._handle_request({'id': 1, 'method': 'tools/call', 'params': {'name': name, 'arguments': arguments}})
        return json.loads(replies[-1]['content'][0]['text'])
    assert call('get_evaluation_labels', workspace_id='book', batch_id=batch_id) == marked
    cleared = call('set_evaluation_label', workspace_id='book', batch_id=batch_id, revision=marked['revision'],
                   run_id=run_id, rank=1, label=None)
    assert cleared['labels'] == [] and not replies[-1]['isError']
    output = io.StringIO()
    assert main(['labels', 'book', batch_id, '--json'], stdout=output) == 0
    assert json.loads(output.getvalue()) == cleared


def test_cli_and_mcp_share_conversation_turns(workspace, monkeypatch):
    from rag import server
    workspaces, _ = workspace
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(workspaces.root.parent / 'traces'))
    first_id, second_id, conversation_id = 'a' * 32, 'b' * 32, 'c' * 32
    for run_id, answer in [(first_id, 'Friday. [1]'), (second_id, 'At noon. [1]')]:
        RunResultStore().save({'schema_version': 1, 'workspace_id': 'book', 'run_id': run_id,
                               'status': 'success', 'answer': answer, 'evidence': []})
    output = io.StringIO()
    assert main(['conversations', 'book', conversation_id, '--append-run', first_id,
                 '--question', 'When?', '--json'], stdout=output) == 0
    first = json.loads(output.getvalue())
    assert first['turns'][0]['answer'] == 'Friday. [1]'
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    def call(name, **arguments):
        server._handle_request({'id': 1, 'method': 'tools/call', 'params': {'name': name, 'arguments': arguments}})
        return json.loads(replies[-1]['content'][0]['text'])
    assert call('get_conversations', workspace_id='book', conversation_id=conversation_id) == first
    second = call('append_conversation_turn', workspace_id='book', conversation_id=conversation_id,
                  revision=first['revision'], run_id=second_id, question='What time?')
    assert len(second['turns']) == 2 and not replies[-1]['isError']
    output = io.StringIO()
    assert main(['conversations', 'book', conversation_id, '--json'], stdout=output) == 0
    assert json.loads(output.getvalue()) == second
    assert call('get_conversations', workspace_id='book')['conversations'][0]['turn_count'] == 2


def test_cli_query_and_mcp_query_continue_one_recorded_conversation(workspace, monkeypatch):
    from rag import server
    workspaces, _ = workspace
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(workspaces.root.parent / 'traces'))
    calls = []
    def execute(query, **kwargs):
        calls.append((query, kwargs['client']))
        result = {'schema_version': 1, 'workspace_id': 'book', 'run_id': kwargs['run_id'],
                  'status': 'success', 'answer': 'Friday. [1]', 'evidence': []}
        kwargs['result_store'].save(result)
        return result
    monkeypatch.setattr('rag.execution.execute_run', execute)
    conversation_id = 'a' * 32
    output = io.StringIO()
    code = main(['query', 'When?', '--workspace', 'book', '--conversation', conversation_id,
                 '--format', 'json'], stdin=io.StringIO(), stdout=output, stderr=io.StringIO())
    assert code == 0, output.getvalue()
    first = json.loads(output.getvalue())
    assert first['result'] == 'Friday. [1]' and first['conversation_revision']
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    def call(**arguments):
        server._handle_request({'id': 1, 'method': 'tools/call', 'params': {
            'name': 'query_knowledge_base', 'arguments': arguments}})
        return json.loads(replies[-1]['content'][0]['text'])
    second = call(query='What time?', workspace_id='book', conversation_id=conversation_id,
                  conversation_revision=first['conversation_revision'])
    assert second['result'] == 'Friday. [1]' and second['conversation_revision'] != first['conversation_revision']
    assert calls == [('When?', 'cli'), ('Previous question (context only, do not answer): When?\n'
                                       'Current question (answer this question only): What time?', 'mcp')]
    stale = call(query='Stale?', workspace_id='book', conversation_id=conversation_id,
                 conversation_revision=first['conversation_revision'])
    assert 'changed' in stale['error'] and replies[-1]['isError'] and len(calls) == 2
    output = io.StringIO()
    assert main(['conversations', 'book', conversation_id, '--json'], stdout=output) == 0
    assert [turn['question'] for turn in json.loads(output.getvalue())['turns']] == ['When?', 'What time?']
    assert main(['query', 'Wrong scope', '--path', '.', '--conversation', conversation_id],
                stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO()) == 2
