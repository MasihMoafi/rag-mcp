"""Run with the isolated wheel's interpreter, from outside the source checkout."""
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix='rag-package-smoke-') as temp:
    root = Path(temp)
    os.environ.update(RAG_MCP_CONFIG=str(root / 'config.json'),
                      RAG_MCP_DATA_DIR=str(root / 'indexes'),
                      RAG_MCP_TRACE_DIR=str(root / 'traces'),
                      RAG_MCP_EMBED_PROVIDER='none', RAG_MCP_RERANKER_TYPE='openrouter',
                      RAG_MCP_CONTEXTUAL_RETRIEVAL='off', OPENROUTER_API_KEY='fixture',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import server
    from rag import openrouter_pipeline
    assert str(Path(sys.prefix)) in server.__file__, server.__file__
    cli = subprocess.run([str(Path(sys.executable).with_name('rag')), '--help'],
                         capture_output=True, text=True, check=True)
    assert 'query' in cli.stdout
    messages = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': 'offline-smoke', 'version': '1'}}},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}},
    ]
    mcp = subprocess.run([sys.executable, server.__file__],
                         input=''.join(json.dumps(m) + '\n' for m in messages),
                         capture_output=True, text=True, check=True)
    responses = [json.loads(line) for line in mcp.stdout.splitlines()]
    assert [r['id'] for r in responses] == [1, 2]
    assert any(t['name'] == 'query_knowledge_base' for t in responses[1]['result']['tools'])
    calls = []
    def provider(request, **kwargs):
        calls.append(request.full_url)
        payload = json.loads(request.data)
        if request.full_url.endswith('/rerank'):
            value = {'results': [{'index': i, 'relevance_score': 1.0} for i in range(len(payload['documents']))]}
        else:
            assert 'Meeting: Friday.' in payload['messages'][-1]['content']
            value = {'choices': [{'message': {'content': 'Friday.'}}]}
        return io.BytesIO(json.dumps(value).encode())
    openrouter_pipeline.urllib.request.urlopen = provider
    source = root / 'fixture.txt'
    source.write_text('Meeting: Friday.')
    assert json.loads(server.query_knowledge_base('Meeting', str(source))) == {'result': 'Friday.'}
    assert len(calls) == 2
    assert len(list((root / 'traces').glob('trace-*.json'))) == 1
    print(json.dumps({'installed_server': server.__file__, 'cli_help': 'passed',
                      'mcp_initialize_and_tools': 'passed', 'real_lexical_index_with_fake_hosted_providers': 'passed',
                      'provider_requests': 'simulated only',
                      'versions': {name: importlib.metadata.version(name) for name in ('ragmcp', 'lancedb', 'openai', 'filelock')}}, indent=2))
