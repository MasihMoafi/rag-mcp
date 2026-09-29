from pathlib import Path
import sys
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import _TOOL, _TOOLS


def test_stdio_imports_a_source_into_the_shared_workspace(tmp_path):
    import json
    import os
    import subprocess
    from rag.workspaces import WorkspaceStore

    source = tmp_path / 'chapter.md'
    source.write_text('Private chapter for shared import.')
    store = WorkspaceStore(tmp_path / 'workspaces')
    calls = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'add_workspace_sources',
            'arguments': {'workspace_id': 'book', 'path': str(source), 'revision': None}}},
        {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'get_workspaces',
            'arguments': {'workspace_id': 'book'}}},
        {'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': {'name': 'add_workspace_sources',
            'arguments': {'workspace_id': 'book', 'path': str(source), 'revision': None}}},
    ]
    env = {**os.environ, 'RAG_MCP_WORKSPACE_DIR': str(store.root),
           'RAG_MCP_CONFIG': str(tmp_path / 'config.json')}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'server.py')],
                            input=''.join(json.dumps(call) + '\n' for call in calls).encode(),
                            capture_output=True, env=env, cwd=tmp_path, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    tool = next(item for item in replies[1]['result']['tools'] if item['name'] == 'add_workspace_sources')
    assert tool['annotations']['readOnlyHint'] is False
    added = json.loads(replies[2]['result']['content'][0]['text'])
    listed = json.loads(replies[3]['result']['content'][0]['text'])
    assert added['document_count'] == listed['document_count'] == 1
    assert added['revision'] == listed['revision']
    assert replies[4]['result']['isError'] is True
    assert store.snapshot('book')['documents'][0]['content'] == 'Private chapter for shared import.'
    assert 'Private chapter for shared import.' not in result.stdout.decode()


def test_stdio_chunk_preview_is_advertised_and_text_is_opt_in(tmp_path):
    import json
    import os
    import subprocess
    from rag.workspaces import WorkspaceStore

    store = WorkspaceStore(tmp_path / 'workspaces')
    saved = store.replace('book', 'Book', [
        {'id': 'chapter', 'name': 'chapter.txt', 'source_type': 'text', 'content': '😀 Cedar Lantern. ' * 95},
        {'id': 'private', 'name': 'private.txt', 'source_type': 'text', 'content': 'Private passage'},
    ])
    calls = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'get_workspaces', 'arguments': {}}},
        {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'get_workspaces',
            'arguments': {'workspace_id': 'book'}}},
        {'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': {'name': 'get_workspace_chunks',
            'arguments': {'workspace_id': 'book', 'source_id': 'chapter', 'revision': saved['revision'], 'limit': 1, 'offset': 1}}},
        {'jsonrpc': '2.0', 'id': 6, 'method': 'tools/call', 'params': {'name': 'get_workspace_chunks',
            'arguments': {'workspace_id': 'book', 'source_id': 'chapter', 'revision': saved['revision'],
                          'limit': 1, 'offset': 1, 'include_content': True}}},
    ]
    env = {**os.environ, 'RAG_MCP_WORKSPACE_DIR': str(store.root), 'RAG_MCP_CONFIG': str(tmp_path / 'config.json')}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'server.py')],
                            input=''.join(json.dumps(call) + '\n' for call in calls).encode(),
                            capture_output=True, env=env, cwd=tmp_path, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    tools = replies[1]['result']['tools']
    assert any(item['name'] == 'get_workspace_chunks' and item['annotations']['readOnlyHint'] for item in tools)
    assert any(item['name'] == 'get_workspaces' and item['annotations']['readOnlyHint'] for item in tools)
    listed = json.loads(replies[2]['result']['content'][0]['text'])
    details = json.loads(replies[3]['result']['content'][0]['text'])
    assert listed['workspaces'][0]['id'] == details['id'] == 'book'
    assert details['documents'][0]['id'] == 'chapter' and details['revision'] == saved['revision']
    assert 'stored_file' not in str(details) and 'content_sha256' not in str(details)
    metadata = json.loads(replies[4]['result']['content'][0]['text'])
    content = json.loads(replies[5]['result']['content'][0]['text'])
    assert metadata['revision'] == content['revision'] == saved['revision']
    assert metadata['chunks'][0]['start_offset'] > 500
    assert 'content' not in metadata['chunks'][0] and 'heading' not in metadata['chunks'][0]
    assert 'source_content' not in metadata
    assert content['chunks'][0]['content'] in '😀 Cedar Lantern. ' * 95
    assert 'Private passage' not in result.stdout.decode()


def test_stdio_reads_and_updates_shared_evaluation_questions(tmp_path):
    import json
    import os
    import subprocess
    from rag.workspaces import WorkspaceStore

    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    workspaces.replace('book', 'Book', [])
    calls = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': 'get_evaluation_questions', 'arguments': {'workspace_id': 'book'}}},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'set_evaluation_questions', 'arguments': {
            'workspace_id': 'book', 'revision': None, 'questions': [{'id': 'q1', 'question': 'When?'}]}}},
        {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'get_evaluation_questions', 'arguments': {'workspace_id': 'book'}}},
    ]
    env = {**os.environ, 'RAG_MCP_WORKSPACE_DIR': str(workspaces.root)}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'server.py')],
                            input=''.join(json.dumps(call) + '\n' for call in calls).encode(),
                            capture_output=True, env=env, cwd=tmp_path, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    get = lambda reply: json.loads(reply['result']['content'][0]['text'])
    assert get(replies[1])['revision'] is None
    assert get(replies[2])['questions'] == [{'id': 'q1', 'question': 'When?'}]
    assert get(replies[3]) == get(replies[2])


def test_stdio_reads_saved_evaluation_batch_without_passages(tmp_path):
    import json
    import os
    import subprocess
    from rag.evaluation_runs import EvaluationRunStore
    from rag.workspaces import WorkspaceStore

    workspaces = WorkspaceStore(tmp_path / 'workspaces')
    workspaces.replace('book', 'Book', [])
    batch_id, run_id = 'a' * 32, 'b' * 32
    EvaluationRunStore(workspaces).save({
        'schema_version': 1, 'workspace_id': 'book', 'batch_id': batch_id,
        'status': 'success', 'started_unix': 1, 'finished_unix': 2, 'error': None,
        'question_revision': 'q1', 'workspace_revision': 'w1',
        'configuration_revision': 'c' * 64, 'source_ids': ['source'],
        'questions': [{'id': 'q', 'question': 'When?'}],
        'rows': [{'question_id': 'q', 'run_id': run_id, 'status': 'success',
                  'passage_count': 1, 'source_count': 1, 'duration_ms': 4, 'error': None}],
    })
    calls = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': 'get_evaluation_runs', 'arguments': {'workspace_id': 'book'}}},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'get_evaluation_runs', 'arguments': {'workspace_id': 'book', 'batch_id': batch_id}}},
    ]
    env = {**os.environ, 'RAG_MCP_WORKSPACE_DIR': str(workspaces.root), 'RAG_MCP_CONFIG': str(tmp_path / 'config.json')}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'server.py')],
                            input=''.join(json.dumps(call) + '\n' for call in calls).encode(),
                            capture_output=True, env=env, cwd=tmp_path, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    listed = json.loads(replies[1]['result']['content'][0]['text'])
    detail = json.loads(replies[2]['result']['content'][0]['text'])
    assert listed['batches'][0]['batch_id'] == batch_id
    assert listed['batches'][0]['completed_count'] == 1
    assert detail['rows'][0]['run_id'] == run_id
    assert 'evidence' not in result.stdout.decode() and 'passage' not in detail['rows'][0]


@pytest.mark.parametrize('framed', [False, True])
def test_stdio_process_recovers_from_invalid_call(tmp_path, framed):
    import hashlib
    import io
    import json
    import os
    import subprocess

    requests = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
         'params': {'name': ['نام نامعتبر']}},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 4, 'method': 'ping'},
        {'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call',
         'params': {'name': 'get_rag_settings', 'arguments': {}}},
    ]
    payload = b''
    for request in requests:
        body = json.dumps(request, ensure_ascii=False).encode('utf-8')
        payload += (f'Content-Length: {len(body)}\r\n\r\n'.encode() + body
                    if framed else body + b'\n')
    env = {key: value for key, value in os.environ.items()
           if key not in {'PYTHONPATH', 'PYTHONHOME'}}
    env['RAG_MCP_CONFIG'] = str(tmp_path / 'config.json')
    server_path = Path(__file__).resolve().parents[2] / 'server.py'
    result = subprocess.run([sys.executable, str(server_path)], input=payload,
                            capture_output=True, cwd=tmp_path, env=env, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    assert not result.stderr
    if framed:
        stream = io.BytesIO(result.stdout)
        replies = []
        while header := stream.readline():
            assert header.startswith(b'Content-Length: ')
            length = int(header.split(b':', 1)[1])
            assert stream.readline() == b'\r\n'
            replies.append(json.loads(stream.read(length)))
    else:
        replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert [reply['id'] for reply in replies] == [1, 2, 3, 4, 5]
    assert replies[1]['error']['code'] == -32602
    search_tool = next(tool for tool in replies[2]['result']['tools'] if tool['name'] == _TOOL['name'])
    description = search_tool['description']
    for instruction in ('Answer a question',
                        'explicit supported document',
                        'do not scan a repository or workspace root by default',
                        'evidence is insufficient',
                        'coverage or absence claims',
                        'narrow doc_path and continue'):
        assert instruction in description
    assert replies[3]['result'] == {}
    settings_result = replies[4]['result']
    assert not settings_result.get('isError', False)
    report = json.loads(settings_result['content'][0]['text'])
    assert report['runtime']['pid'] > 0
    assert report['runtime']['pid'] != os.getpid()
    implementation_path = server_path.parent / 'rag' / 'server.py'
    assert report['runtime']['server_source_sha256_at_startup'] == hashlib.sha256(implementation_path.read_bytes()).hexdigest()
    assert not (tmp_path / 'config.json').exists()


@pytest.mark.parametrize('params', [
    ['invalid'], [], 'invalid', 1, 0, False,
    {'name': []}, {'name': {}},
    {'name': 'query_knowledge_base', 'arguments': []},
    {'name': 'query_knowledge_base', 'arguments': False},
    {'name': 'query_knowledge_base', 'arguments': [['query', 'unexpected']]},
    {'name': 'set_rag_settings', 'arguments': [['top', 99]]},
    {'name': 'get_rag_settings', 'arguments': []},
    {'name': 'query_knowledge_base', 'arguments': None},
])
def test_malformed_tool_params_do_not_terminate_server(monkeypatch, capsys, params):
    import json
    import server

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail('Malformed request reached a tool handler')

    monkeypatch.setattr(server, 'query_knowledge_base', unexpected_dispatch)
    monkeypatch.setattr(server, '_apply_settings', unexpected_dispatch)
    monkeypatch.setattr(server, '_settings_report', unexpected_dispatch)

    requests = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': params},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'ping'},
    ]
    payloads = iter([*(json.dumps(request) for request in requests), None])
    monkeypatch.setattr(server, '_read_payload', lambda: next(payloads))
    monkeypatch.setattr(server, '_initialized', False)
    monkeypatch.setattr(server, '_use_content_length_framing', False)
    server.main()
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [reply['id'] for reply in replies] == [1, 2, 3]
    assert replies[1]['error']['code'] == -32602
    assert replies[2]['result'] == {}


def test_rag_tool_advertises_saved_run_side_effects() -> None:
    assert _TOOL["name"] == "query_knowledge_base"
    assert _TOOL["annotations"] == {
        "title": "Answer from scoped documents",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }


def test_tool_descriptions_route_content_search_to_query_tool() -> None:
    # An agent asked to "find X in my files" listed workspaces instead of querying.
    tools = {tool['name']: tool['description'] for tool in _TOOLS}
    assert 'files or folders on disk' in tools['query_knowledge_base']
    assert 'no import' in tools['query_knowledge_base']
    for name in ('search_workspace', 'get_workspaces'):
        assert 'query_knowledge_base' in tools[name]


def test_rag_tool_teaches_bounded_escalating_search() -> None:
    description = _TOOL["description"]
    path_description = _TOOL["inputSchema"]["properties"]["doc_path"]["description"]

    assert "canonical file" in description
    assert "do not scan a repository or workspace root by default" in description
    assert "Increase search complexity only when evidence is insufficient" in description
    assert "coverage or absence claims" in description
    assert "narrow doc_path and continue" in description
    assert "smallest deliberate directory scope" in path_description
