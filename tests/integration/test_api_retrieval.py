"""Real SDK serialization, vector storage and retrieval; HTTP stays in memory."""

import json

import httpx
import openai
import pytest

from rag import core


@pytest.mark.parametrize('status', [401, 429, 503])
def test_query_provider_failure_is_reported_and_pipeline_can_recover(tmp_path, monkeypatch, status, capsys):
    import server
    from rag import fetch

    source = tmp_path / 'facts.txt'
    source.write_text('Cedar owns a bicycle.')
    failing = False
    requests = []

    def respond(request):
        requests.append(request)
        if failing:
            return httpx.Response(status, json={'error': {'message': 'private-provider-body-marker'}})
        inputs = json.loads(request.content)['input']
        return httpx.Response(200, json={'data': [
            {'index': index, 'embedding': [1., 0.]} for index in range(len(inputs))]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        real_client = openai.OpenAI
        monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: real_client(http_client=transport, **kwargs))
        pipeline = core.RAGPipeline(dict(document_paths=[str(source)], persist_dir=str(tmp_path / 'index'),
            embed_provider='openai_compatible', embed_model='fixture', embed_api_key='test-only',
            backend='lancedb', reranker_type='off', use_contextual=False,
            chunk_size=700, chunk_overlap=100, top_k=2, rerank_top_k=1, force_reindex=True))
        try:
            failing = True
            requests.clear()
            with pytest.raises(openai.APIStatusError) as caught:
                pipeline.search('Cedar')
            assert caught.value.status_code == status
            assert len(requests) == (1 if status == 401 else 2)
            monkeypatch.setattr(fetch, 'get_rag_pipeline_v2', lambda **kwargs: pipeline)
            responses = []
            monkeypatch.setattr(server, '_result', lambda request_id, result: responses.append(result))
            monkeypatch.setattr(server, '_initialized', False)
            server._handle_request({'jsonrpc': '2.0', 'id': 0, 'method': 'initialize'})
            responses.clear()
            requests.clear()
            server._handle_request({
                'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                'params': {'name': server._TOOLS[0]['name'], 'arguments': {
                    'query': 'private-query-marker', 'doc_path': str(source)}}})
            assert len(responses) == 1
            assert responses[0]['isError'] is True
            assert len(requests) == (1 if status == 401 else 2)
            output = capsys.readouterr()
            evidence = json.dumps(responses) + output.out + output.err
            assert 'private-provider-body-marker' not in evidence
            assert 'private-query-marker' not in evidence
            failing = False
            assert pipeline.search('Cedar')[0]['content'] == 'Cedar owns a bicycle.'
        finally:
            pipeline.close()


def test_api_embeddings_retrieve_semantic_fixture_and_reopen(tmp_path, monkeypatch):
    source = tmp_path / 'facts.md'
    source.write_text('# Transport\nThe bicycle belongs to Cedar.\n\n# Cooking\nSoup needs salt.\n')
    calls = []
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        vectors = []
        for index, text in enumerate(body['input']):
            # Controlled vectors verify plumbing, not model understanding.
            vector = [1., 0.] if ('bicycle' in text or 'pedal' in text) else [0., 1.]
            vectors.append(dict(object='embedding', index=index, embedding=vector))
        return httpx.Response(200, json=dict(object='list', model='fixture', data=vectors[::-1],
                                            usage=dict(prompt_tokens=1, total_tokens=1)))
    config = dict(document_paths=[str(source)], persist_dir=str(tmp_path / 'index'),
                  embed_provider='openai_compatible', embed_model='fixture', embed_api_key='test-only',
                  backend='lancedb', reranker_type='off',
                  use_contextual=False, chunk_size=700, chunk_overlap=100,
                  top_k=2, rerank_top_k=1, force_reindex=True)
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        real_client = openai.OpenAI
        monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: real_client(http_client=transport, **kwargs))
        for reopen in [False, True]:
            config['force_reindex'] = not reopen
            previous_batches = sum(len(call['input']) > 1 for call in calls)
            pipeline = core.RAGPipeline(config.copy())
            query = 'pedal ownership'
            assert pipeline.vector_db.search_fts(query) == []
            result = pipeline.search(query)[0]
            assert 'Cedar' in result['original_content']
            assert result['file_path'] == str(source)
            assert result['location'] == 'L1-L2'
            assert pipeline.vector_db.count() == 2
            assert sum(len(call['input']) > 1 for call in calls) == previous_batches + (not reopen)
    assert all(call['model'] == 'fixture' for call in calls)
