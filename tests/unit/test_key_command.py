import io

from rag import keys, settings
from rag.openrouter_pipeline import openrouter_key


def run(action, env, value='', check=lambda _: (True, 'accepted')):
    out, err = io.StringIO(), io.StringIO()
    result = keys.key_command(action, stdin=io.StringIO(value), stdout=out,
                              stderr=err, environ=env, check=check)
    return result, out.getvalue(), err.getvalue()


def test_openrouter_key_is_isolated_and_removed_without_touching_embedding(tmp_path, monkeypatch):
    env = {'RAG_MCP_CONFIG': str(tmp_path / 'config.json')}
    monkeypatch.setenv('RAG_MCP_CONFIG', env['RAG_MCP_CONFIG'])
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    original = {'embed_base_url': 'https://api.openai.com/v1', 'embed_api_key': 'openai-secret'}
    settings.save_settings(original, env)
    assert run('set', env, 'router-secret\n')[0] == 0
    saved = settings.load_settings(env)
    assert saved == {**original, 'openrouter_api_key': 'router-secret'}
    assert openrouter_key(settings.embedding_config()) == 'router-secret'
    assert settings.retrieval_config(env)['reranker_type'] == 'openrouter'
    assert run('remove', env)[0] == 0
    assert settings.load_settings(env) == {**original, 'openrouter_api_key': ''}
    assert settings.retrieval_config(env)['reranker_type'] == 'off'


def test_rejected_openrouter_key_does_not_replace_saved_value(tmp_path):
    env = {'RAG_MCP_CONFIG': str(tmp_path / 'config.json')}
    settings.save_settings({'openrouter_api_key': 'old-secret'}, env)
    result, _, _ = run('set', env, 'bad-secret\n', lambda _: (False, 'rejected'))
    assert result != 0
    assert settings.load_settings(env) == {'openrouter_api_key': 'old-secret'}


def test_openrouter_key_only_reaches_openrouter_embedding_endpoint(monkeypatch):
    from rag import embeddings
    calls = []
    monkeypatch.setattr(embeddings, 'OpenAICompatibleWrapper', lambda *args, **kwargs: calls.append(kwargs))
    config = {'embed_provider': 'openai_compatible', 'openrouter_api_key': 'router-secret',
              'embed_api_key': 'openai-secret', 'embed_base_url': 'https://api.openai.com/v1'}
    embeddings.build_embedder(config)
    assert calls[-1]['api_key'] == 'openai-secret'
    config.update(embed_api_key='', embed_base_url='https://openrouter.ai/api/v1')
    embeddings.build_embedder(config)
    assert calls[-1]['api_key'] == 'router-secret'
    config['embed_base_url'] = 'https://openrouter.ai.evil.invalid/v1'
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'environment-router-secret')
    embeddings.build_embedder(config)
    assert calls[-1]['api_key'] is None
