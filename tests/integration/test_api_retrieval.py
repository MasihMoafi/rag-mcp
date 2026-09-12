"""Real SDK serialization, vector storage and retrieval; HTTP stays in memory."""

import json

import httpx
import openai

from rag import core


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
