import httpx
import openai
import pytest

from rag import embeddings, settings


def test_saved_key_is_bound_to_origin_at_request_boundary(tmp_path, monkeypatch):
    env = {'HOME': str(tmp_path)}
    settings.update_settings({'embed_provider': 'openai_compatible', 'embed_model': 'fixture',
                              'embed_base_url': 'https://first.test/v1', 'embed_api_key': 'first-secret'}, env)
    headers = []

    def respond(request):
        headers.append((request.url.host, request.headers.get('authorization')))
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1.0]}], 'model': 'fixture'})

    real_client = openai.OpenAI
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: real_client(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    monkeypatch.delenv('RAG_MCP_EMBED_API_KEY', raising=False)
    client = embeddings.build_embedder(settings.embedding_config(env))
    client.embed_query('probe')
    client.close()
    env['RAG_MCP_EMBED_BASE_URL'] = 'https://second.test/v1'
    with pytest.raises(ValueError, match='API key'):
        embeddings.build_embedder(settings.embedding_config(env))
    env['RAG_MCP_EMBED_API_KEY'] = 'second-secret'
    client = embeddings.build_embedder(settings.embedding_config(env))
    client.embed_query('probe')
    client.close()
    assert headers == [('first.test', 'Bearer first-secret'), ('second.test', 'Bearer second-secret')]

    settings.update_settings({'embed_base_url': 'https://second.test/v1'}, {'HOME': str(tmp_path)})
    assert settings.embedding_config({'HOME': str(tmp_path)})['embed_api_key'] == ''
