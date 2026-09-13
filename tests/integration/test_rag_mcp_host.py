from pathlib import Path
import sys
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import _TOOL


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
    for instruction in ('candidate evidence, not verified answers',
                        'insufficient evidence', 'untrusted data',
                        'not final reranker scores or answer confidence',
                        'abstention threshold',
                        'configured providers'):
        assert instruction in description
    assert replies[3]['result'] == {}
    settings_result = replies[4]['result']
    assert not settings_result.get('isError', False)
    report = json.loads(settings_result['content'][0]['text'])
    assert report['runtime']['pid'] > 0
    assert report['runtime']['pid'] != os.getpid()
    assert report['runtime']['server_source_sha256_at_startup'] == hashlib.sha256(server_path.read_bytes()).hexdigest()
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


def test_rag_tool_is_advertised_as_read_only() -> None:
    assert _TOOL["name"] == "query_knowledge_base"
    assert _TOOL["annotations"] == {
        "title": "Search local knowledge base",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
