import json
from http.client import HTTPConnection
from threading import Thread

from rag import settings
from rag.companion import create_companion_server


def test_gateway_rejects_stale_config_without_losing_cli_edit(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    settings.update_settings({'top_k': '5'})
    gateway = create_companion_server(token='fixture-token', codex_ready=lambda: False,
                                      metrics_path=tmp_path / 'metrics.json')
    worker = Thread(target=gateway.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection('127.0.0.1', gateway.server_port, timeout=5)
    headers = {'Authorization': 'Bearer fixture-token', 'Origin': 'http://localhost:3000',
               'Content-Type': 'application/json'}
    try:
        connection.request('GET', '/v1/config', headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        revision = json.loads(response.read())['revision']
        settings.update_settings({'top_k': '7'})
        connection.request('PATCH', '/v1/config', json.dumps({'top': 4, 'revision': revision}), headers)
        response = connection.getresponse()
        assert response.status == 409
        assert 'changed' in json.loads(response.read())['error'].lower()
        assert settings.retrieval_config()['top_k'] == '7'
    finally:
        connection.close()
        gateway.shutdown()
        gateway.server_close()
        worker.join(timeout=2)


def test_mcp_setting_edit_rejects_stale_revision(tmp_path, monkeypatch):
    from rag import server
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    settings.update_settings({'top_k': '5'})
    stale = settings.config_revision()
    settings.update_settings({'top_k': '7'})
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    server._handle_request({'id': 1, 'method': 'tools/call', 'params': {
        'name': 'set_rag_settings', 'arguments': {'top': 4, 'revision': stale}}})
    assert replies[-1]['isError']
    assert 'changed' in replies[-1]['content'][0]['text'].lower()
    assert settings.retrieval_config()['top_k'] == '7'


def test_http_and_mcp_select_the_same_detected_local_runtime(tmp_path, monkeypatch):
    from rag import cli, control, detect, server
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    settings.update_settings({'embed_api_key': 'private-hosted-key'})
    discovered = {'ollama': {'embedding_models': []}, 'local_runtimes': [{
        'label': 'LM Studio', 'base_url': 'http://localhost:1234/v1',
        'embedding_models': ['local-embedding']}], 'api_keys': []}
    monkeypatch.setattr(detect, 'detect_all', lambda *_: discovered)
    monkeypatch.setattr(control, 'model_inventory', lambda: {'models': [{
        'stage': 'embedding', 'provider': 'openai_compatible', 'model': 'local-embedding',
        'location': 'local', 'discovery_source': 'local_runtime', 'runtime': 'LM Studio'}]})
    monkeypatch.setattr(cli, '_probe_embedding', lambda _: 384)
    gateway = create_companion_server(token='fixture-token', codex_ready=lambda: False,
                                      metrics_path=tmp_path / 'metrics.json')
    worker = Thread(target=gateway.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection('127.0.0.1', gateway.server_port, timeout=5)
    headers = {'Authorization': 'Bearer fixture-token', 'Origin': 'http://localhost:3000',
               'Content-Type': 'application/json'}
    try:
        connection.request('GET', '/v1/config', headers=headers)
        revision = json.loads(connection.getresponse().read())['revision']
        connection.request('PATCH', '/v1/config', json.dumps({
            'embedding_provider': 'openai_compatible', 'embedding_model': 'local-embedding',
            'embedding_source': 'local_runtime', 'embedding_runtime': 'LM Studio',
            'revision': revision}), headers)
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())['embedding']['model'] == 'local-embedding'
        assert settings.load_settings()['embed_base_url'] == 'http://localhost:1234/v1'
        assert settings.load_settings()['embed_api_key'] == ''
    finally:
        connection.close()
        gateway.shutdown()
        gateway.server_close()
        worker.join(timeout=2)

    settings.update_settings({'embed_provider': 'none', 'embed_model': '', 'embed_api_key': 'private-hosted-key'})
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    server._handle_request({'id': 1, 'method': 'tools/call', 'params': {
        'name': 'set_rag_settings', 'arguments': {
            'embedding_provider': 'openai_compatible', 'embedding_model': 'local-embedding',
            'embedding_runtime': 'LM Studio', 'revision': settings.config_revision()}}})
    assert not replies[-1]['isError']
    assert settings.load_settings()['embed_base_url'] == 'http://localhost:1234/v1'
    assert settings.load_settings()['embed_api_key'] == ''
    assert 'embedding_runtime' in server._SETTINGS_SET_TOOL['inputSchema']['properties']
