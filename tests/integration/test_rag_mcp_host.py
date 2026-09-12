from pathlib import Path
import sys
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import _TOOL


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
