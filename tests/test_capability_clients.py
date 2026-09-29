import io
import json
from http.client import HTTPConnection
from threading import Thread

from rag import server, settings
from rag.cli import main
from rag.companion import create_companion_server
from rag.control import capabilities, model_inventory


def test_http_cli_mcp_share_one_capability_report(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(tmp_path / 'workspaces'))
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    settings.update_settings({'embed_provider': 'none', 'reranker_type': 'off'})
    expected = capabilities()
    output = io.StringIO()
    assert main(['capabilities', '--json'], stdout=output) == 0
    assert json.loads(output.getvalue()) == expected
    output = io.StringIO()
    assert main(['capabilities'], stdout=output) == 0
    assert 'generation' in output.getvalue() and 'unavailable' in output.getvalue()

    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    server._handle_request({'id': 1, 'method': 'tools/call',
                            'params': {'name': 'get_rag_capabilities', 'arguments': {}}})
    assert not replies[-1]['isError']
    assert json.loads(replies[-1]['content'][0]['text']) == expected

    gateway = create_companion_server(token='fixture-token', codex_ready=lambda: False,
                                      metrics_path=tmp_path / 'metrics.json')
    worker = Thread(target=gateway.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection('127.0.0.1', gateway.server_port, timeout=5)
    try:
        connection.request('GET', '/v1/capabilities', headers={'Authorization': 'Bearer fixture-token',
                           'Origin': 'http://localhost:3000'})
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == expected
    finally:
        connection.close()
        gateway.shutdown()
        gateway.server_close()
        worker.join(timeout=2)


def test_http_cli_mcp_share_one_model_inventory(tmp_path, monkeypatch):
    from rag import detect, model_catalog
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(tmp_path / 'workspaces'))
    settings.update_settings({'embed_provider': 'none', 'reranker_type': 'off'})
    monkeypatch.setattr(detect, 'detect_ollama', lambda url, max_models: {
        'running': True, 'embedding_models': ['bge-m3'], 'truncated': False})
    monkeypatch.setattr(detect, 'detect_local_runtimes', lambda: [])
    monkeypatch.setattr(detect, 'detect_cached_models', lambda _: [])
    monkeypatch.setattr(model_catalog, 'cached_cpu_models', lambda _: [])
    expected = model_inventory()
    output = io.StringIO()
    assert main(['models', '--json'], stdout=output) == 0
    assert json.loads(output.getvalue()) == expected
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    server._handle_request({'id': 1, 'method': 'tools/call',
                            'params': {'name': 'get_rag_models', 'arguments': {}}})
    assert not replies[-1]['isError']
    assert json.loads(replies[-1]['content'][0]['text']) == expected
    gateway = create_companion_server(token='fixture-token', codex_ready=lambda: False,
                                      metrics_path=tmp_path / 'metrics.json')
    worker = Thread(target=gateway.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection('127.0.0.1', gateway.server_port, timeout=5)
    try:
        connection.request('GET', '/v1/models', headers={'Origin': 'http://localhost:3000'})
        denied = connection.getresponse()
        assert denied.status == 403
        denied.read()
        connection.request('GET', '/v1/models', headers={'Authorization': 'Bearer fixture-token',
                           'Origin': 'http://localhost:3000'})
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == expected
    finally:
        connection.close()
        gateway.shutdown()
        gateway.server_close()
        worker.join(timeout=2)
