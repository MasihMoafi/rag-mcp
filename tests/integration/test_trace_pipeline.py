"""Trace real parsing/index refresh/retrieval with simulated hosted providers."""
import io
import json
import urllib.error

import pytest

import server
from rag import core, fetch, openrouter_pipeline


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    for name, value in {
        'RAG_MCP_CONFIG': str(tmp_path / 'config.json'),
        'RAG_MCP_DATA_DIR': str(tmp_path / 'indexes'),
        'RAG_MCP_TRACE_DIR': str(tmp_path / 'traces'),
        'RAG_MCP_EMBED_PROVIDER': 'none',
        'RAG_MCP_CONTEXTUAL_RETRIEVAL': 'off',
        'RAG_MCP_RERANKER_TYPE': 'openrouter',
        'RAG_MCP_RERANKER_MODEL': 'cohere/rerank-4-fast',
        'RAG_MCP_RERANK_TOP_K': '5',
        'OPENROUTER_API_KEY': 'private-key-marker',
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('RAG_MCP_TRACE_CONTENT', raising=False)
    # Offline exact-word retrieval isolates tracing from embedding service behavior.
    monkeypatch.setattr(core, 'build_embedder', lambda *a, **k: None)
    source = tmp_path / 'private-source-marker.txt'
    source.write_text('Meeting private-document-marker Friday.')
    requests = []

    def provider(request, **kwargs):
        payload = json.loads(request.data)
        requests.append(payload)
        if request.full_url.endswith('/rerank'):
            return io.BytesIO(json.dumps({'results': [
                {'index': i, 'relevance_score': 1 / (i + 1)}
                for i in range(len(payload['documents']))
            ], 'usage': {'search_units': 1, 'cost': 0.002}}).encode())
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'Friday.'}}],
                                    'usage': {'prompt_tokens': 40, 'completion_tokens': 3,
                                              'cost': 0.0001}}).encode())

    monkeypatch.setattr(openrouter_pipeline.urllib.request, 'urlopen', provider)
    fetch.reset_pipelines()
    server._validated_scope_cache.clear()
    yield source, tmp_path / 'traces', requests
    fetch.reset_pipelines()
    server._validated_scope_cache.clear()


def test_real_pipeline_trace_matches_synthesis_and_refresh(pipeline, monkeypatch):
    source, directory, requests = pipeline
    assert json.loads(server.query_knowledge_base('Meeting', str(source))) == {'result': 'Friday.'}
    first_path, = directory.glob('trace-*.json')
    first = json.loads(first_path.read_text())
    spans = {s['operation']: s for s in first['spans']}
    for operation in ('scope.discovery', 'index.lookup', 'index.build', 'ingestion.parse',
                      'ingestion.chunk', 'retrieval.lexical', 'retrieval.rrf',
                      'selection.rerank', 'rerank.provider', 'synthesis'):
        assert operation in spans
    assert spans['query']['status'] == 'success'
    assert spans['index.lookup']['attributes']['cache_state'] == 'build'
    selected = spans['selection.rerank']['attributes']['selected']
    assert spans['synthesis']['attributes']['selected'] == selected
    assert spans['synthesis']['attributes']['usage']['input_tokens'] == 40
    assert spans['rerank.provider']['attributes']['usage']['cost_usd'] == 0.002
    assert 'private-' not in first_path.read_text()
    assert 'private-document-marker' in requests[-1]['messages'][-1]['content']

    source.write_text('Meeting private-document-marker Sunday.')
    server.query_knowledge_base('Meeting', str(source))
    second = json.loads(next(p for p in directory.glob('trace-*.json') if p != first_path).read_text())
    assert any(s['operation'] == 'index.refresh' for s in second['spans'])
    assert 'Sunday' in requests[-1]['messages'][-1]['content']

    monkeypatch.setenv('RAG_MCP_TRACE_DIR', 'off')
    before = list(requests)
    assert json.loads(server.query_knowledge_base('Meeting', str(source))) == {'result': 'Friday.'}
    assert len(list(directory.glob('trace-*.json'))) == 2
    assert requests[-2:] == before[-2:]


def test_provider_failure_stays_private_and_marks_root(pipeline, monkeypatch):
    source, directory, _ = pipeline
    def fail(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 429, 'private-provider-body', {}, io.BytesIO(b'private-body'))
    monkeypatch.setattr(openrouter_pipeline.urllib.request, 'urlopen', fail)
    response = json.loads(server.query_knowledge_base('Meeting', str(source)))
    assert set(response) == {'error'}
    record = json.loads(next(directory.glob('trace-*.json')).read_text())
    assert record['spans'][0]['status'] == 'error'
    provider = next(s for s in record['spans'] if s['operation'] == 'rerank.provider')
    assert provider['status'] == 'error'
    assert provider['attributes']['usage']['cost_usd'] is None
    assert 'private-' not in json.dumps(record) + json.dumps(response)
