"""MCP clients get invalid-params errors before malformed calls reach tools."""

import json
import os
import subprocess
import sys
from typing import get_args, get_origin, get_type_hints
import types

import pytest

from rag import server


INVALID_CALLS = [
    ('query_knowledge_base', {'query': 42}),
    ('search_workspace', {'workspace_id': 'book', 'query': False}),
    ('get_rag_settings', {'extra': 1}),
    ('get_rag_capabilities', {'extra': 1}),
    ('get_rag_models', {'extra': 1}),
    ('set_rag_settings', {'top': True}),
    ('get_evaluation_questions', {}),
    ('set_evaluation_questions', {'workspace_id': 'book', 'revision': None, 'questions': [{'id': 'q'}]}),
    ('run_evaluation', {'workspace_id': 'book', 'source_ids': [1]}),
    ('get_evaluation_runs', {'workspace_id': 'book', 'limit': '1'}),
    ('get_evaluation_labels', {'workspace_id': 'book'}),
    ('get_evaluation_scores', {'workspace_id': 1, 'batch_id': 'batch'}),
    ('get_evaluation_answers', {'workspace_id': 'book'}),
    ('set_evaluation_label', {'workspace_id': 'book', 'batch_id': 'b', 'revision': None,
                              'run_id': 'r', 'rank': False, 'label': None}),
    ('get_conversations', {'workspace_id': 'book', 'offset': None}),
    ('append_conversation_turn', {'workspace_id': 'book', 'conversation_id': 'c', 'revision': None,
                                  'run_id': 'r'}),
    ('get_workspaces', {'workspace_id': []}),
    ('add_workspace_sources', {'workspace_id': 'book', 'path': 'file.md'}),
    ('get_workspace_chunks', {'workspace_id': 'book', 'source_id': 's', 'include_content': 1}),
]


def test_each_tool_rejects_invalid_arguments_over_stdio(tmp_path):
    calls = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        *({'jsonrpc': '2.0', 'id': i, 'method': 'tools/call',
           'params': {'name': name, 'arguments': arguments}}
          for i, (name, arguments) in enumerate(INVALID_CALLS, 2)),
        {'jsonrpc': '2.0', 'id': len(INVALID_CALLS) + 2, 'method': 'ping'},
    ]
    env = {**os.environ, 'RAG_MCP_CONFIG': str(tmp_path / 'config.json'),
           'RAG_MCP_DATA_DIR': str(tmp_path / 'index'),
           'RAG_MCP_WORKSPACE_DIR': str(tmp_path / 'workspaces'),
           'RAG_MCP_TRACE_DIR': str(tmp_path / 'traces')}
    process = subprocess.run([sys.executable, str(server.repo_root / 'server.py')],
                             input=''.join(json.dumps(call) + '\n' for call in calls),
                             text=True, capture_output=True, env=env, timeout=20, check=True)
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(replies) == len(calls)
    for (name, _), reply in zip(INVALID_CALLS, replies[1:-1]):
        assert reply['error']['code'] == -32602, (name, reply)
        assert 'arguments.' in reply['error']['message'], (name, reply)
    assert replies[-1]['result'] == {}
    assert not (tmp_path / 'workspaces').exists()


def _model_shape(annotation):
    origin = get_origin(annotation)
    if origin in (types.UnionType,):
        return frozenset(_model_shape(item) for item in get_args(annotation))
    if origin is list:
        return ('array', _model_shape(get_args(annotation)[0]))
    if isinstance(annotation, type) and hasattr(annotation, '__required_keys__'):
        return ('object', _model_fields(annotation))
    return {str: 'string', int: 'integer', bool: 'boolean', type(None): 'null'}[annotation]


def _model_fields(model):
    return tuple(sorted((name, _model_shape(annotation), name in model.__required_keys__)
                        for name, annotation in get_type_hints(model).items()))


def _schema_shape(schema):
    kind = schema['type']
    if isinstance(kind, list):
        return frozenset(kind)
    if kind == 'array':
        return ('array', _schema_shape(schema['items']))
    if kind == 'object':
        return ('object', _schema_fields(schema))
    return kind


def _schema_fields(schema):
    return tuple(sorted((name, _schema_shape(field), name in schema.get('required', []))
                        for name, field in schema['properties'].items()))


@pytest.mark.parametrize('tool', server._TOOLS, ids=lambda tool: tool['name'])
def test_request_model_matches_advertised_fields(tool):
    model = server._REQUEST_MODELS[tool['name']]
    assert _model_fields(model) == _schema_fields(tool['inputSchema'])


def test_non_object_tool_response_fails_closed_and_server_continues(monkeypatch, capsys):
    monkeypatch.setattr(server, '_settings_report', lambda: ['unexpected private data'])
    monkeypatch.setattr(server, '_initialized', True)
    monkeypatch.setattr(server, '_use_content_length_framing', False)
    server._handle_request({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
                            'params': {'name': 'get_rag_settings', 'arguments': {}}})
    server._handle_request({'jsonrpc': '2.0', 'id': 3, 'method': 'ping'})
    output = capsys.readouterr().out
    first, second = map(json.loads, output.splitlines())
    assert first['result'] == {'content': [{'type': 'text',
        'text': 'RAG tool returned an invalid response.'}], 'isError': True}
    assert 'unexpected private data' not in output
    assert second['result'] == {}
