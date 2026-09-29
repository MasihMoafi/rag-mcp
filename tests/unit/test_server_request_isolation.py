"""A failed tool request must not terminate the MCP message loop."""

import io
import json
import os
import sys

import pytest

from rag import server


SECRET = 'sk-or-v1-SECRET /home/private/notes.md'


def _run(monkeypatch, capsys, messages):
    lines = ''.join(json.dumps(message) + '\n' for message in messages)
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(lines.encode())))
    monkeypatch.setattr(server, '_use_content_length_framing', False)
    monkeypatch.setattr(server, '_initialized', False)
    server.main()
    output = capsys.readouterr()
    return {reply['id']: reply for reply in map(json.loads, output.out.splitlines())}, output


@pytest.mark.parametrize('error', [ImportError, RuntimeError])
def test_tool_failure_is_isolated_to_its_request(monkeypatch, capsys, error):
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error(f'cannot import name frozen_settings from rag.settings: {SECRET}')
        return json.dumps({'result': 'healthy answer'})

    monkeypatch.setattr(server, 'query_knowledge_base', query)
    call = {'jsonrpc': '2.0', 'method': 'tools/call',
            'params': {'name': 'query_knowledge_base', 'arguments': {'query': 'q'}}}
    replies, output = _run(monkeypatch, capsys, [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}},
        {**call, 'id': 2},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'ping'},
        {**call, 'id': 4},
    ])
    failed = replies[2]['result']
    assert failed['isError'] is True
    text = failed['content'][0]['text']
    assert error.__name__ in text and 'restart' in text.lower()
    assert replies[3]['result'] == {}
    assert replies[4]['result'] == {'content': [{'type': 'text', 'text': 'healthy answer'}], 'isError': False}
    streams = output.out + output.err
    assert SECRET not in streams and 'frozen_settings' not in streams
    assert len(calls) == 2


def test_source_import_diagnostics_do_not_corrupt_or_leak_through_stdio(monkeypatch, capsys):
    from rag import workspace_import
    def noisy_import(*_args, **_kwargs):
        print('PRIVATE PARSER OUTPUT')
        return {'schema_version': 1, 'workspace_id': 'book', 'revision': 'r1',
                'document_count': 1, 'total_bytes': 10, 'imported_count': 1, 'source_ids': ['source']}
    monkeypatch.setattr(workspace_import, 'import_workspace_sources', noisy_import)
    replies, output = _run(monkeypatch, capsys, [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': 'add_workspace_sources',
            'arguments': {'workspace_id': 'book', 'path': '/tmp/chapter.md', 'revision': None}}},
    ])
    assert replies[2]['result']['isError'] is False
    assert json.loads(replies[2]['result']['content'][0]['text'])['revision'] == 'r1'
    assert 'PRIVATE PARSER OUTPUT' not in output.out + output.err


def test_workspace_search_diagnostics_stay_out_of_mcp_streams(monkeypatch, capfd):
    from rag import execution

    def noisy_search(*_args, **_kwargs):
        print(SECRET)
        print(SECRET, file=sys.stderr)
        os.write(1, (SECRET + '\n').encode())
        os.write(2, (SECRET + '\n').encode())
        return object()

    monkeypatch.setattr(execution, 'execute_run', noisy_search)
    monkeypatch.setattr(execution, 'retrieval_response', lambda _value, **_kwargs: {
        'schema_version': 1, 'workspace_id': 'personal', 'passages': []})
    replies, output = _run(monkeypatch, capfd, [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': 'search_workspace',
            'arguments': {'workspace_id': 'personal', 'query': 'Where?'}}},
    ])
    assert replies[2]['result']['isError'] is False
    assert json.loads(replies[2]['result']['content'][0]['text'])['workspace_id'] == 'personal'
    assert SECRET not in output.out + output.err
