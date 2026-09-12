from pathlib import Path
import sys
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import _TOOL


@pytest.mark.parametrize('params', [['invalid'], [], 'invalid', 1, 0, False])
def test_malformed_tool_params_do_not_terminate_server(monkeypatch, capsys, params):
    import json
    import server

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
