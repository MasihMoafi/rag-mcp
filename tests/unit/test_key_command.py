import io

import pytest

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


def test_rotation_and_removal_migrate_legacy_openrouter_key(tmp_path, monkeypatch):
    from rag import embeddings
    env = {'RAG_MCP_CONFIG': str(tmp_path / 'config.json')}
    monkeypatch.setenv('RAG_MCP_CONFIG', env['RAG_MCP_CONFIG'])
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    settings.save_settings({'embed_provider': 'openai_compatible',
                            'embed_base_url': 'https://openrouter.ai/api/v1',
                            'embed_api_key': 'legacy-secret'}, env)
    supplied = []
    monkeypatch.setattr(embeddings, 'OpenAICompatibleWrapper', lambda *a, **kw: supplied.append(kw))
    assert run('set', env, 'new-secret\n')[0] == 0
    config = settings.embedding_config(env)
    assert config['embed_api_key'] == ''
    embeddings.build_embedder(config)
    assert supplied[-1]['api_key'] == openrouter_key(config) == 'new-secret'
    assert run('remove', env)[0] == 0
    with pytest.raises(ValueError):
        openrouter_key(settings.embedding_config(env))
    assert not settings.load_settings(env)['embed_api_key']


def test_removing_legacy_key_clears_it_without_a_dedicated_key(tmp_path, monkeypatch):
    env = {'RAG_MCP_CONFIG': str(tmp_path / 'config.json')}
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    settings.save_settings({'embed_base_url': 'https://openrouter.ai/api/v1',
                            'embed_api_key': 'legacy-secret'}, env)
    assert run('remove', env)[0] == 0
    assert settings.load_settings(env)['embed_api_key'] == ''


def test_saved_key_is_available_to_setup_config_and_capabilities(tmp_path, monkeypatch):
    from rag import cli, control, detect, setup_ui
    env = {'RAG_MCP_CONFIG': str(tmp_path / 'config.json')}
    monkeypatch.setenv('RAG_MCP_CONFIG', env['RAG_MCP_CONFIG'])
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setattr(detect, 'detect_all', lambda _: {'api_keys': []})
    settings.save_settings({'openrouter_api_key': 'router-secret'}, env)
    assert setup_ui._api_key_available('OpenRouter', None, '', env)
    assert setup_ui._provider_key('OpenRouter', env) == 'router-secret'
    choice = cli._resolve_embedding_choice('qwen/test', 'openai_compatible', env)
    probes = []
    assert settings.finish_setup(choice, stdout=io.StringIO(), stderr=io.StringIO(), environ=env,
                                 verify=lambda config: probes.append(openrouter_key(config)) or 8) == 0
    assert probes == ['router-secret']
    settings.update_settings(choice, env)
    stage = next(s for s in control.capabilities()['stages'] if s['stage'] == 'embedding')
    assert stage['availability'] == 'unknown'
    assert run('remove', env)[0] == 0
    assert not setup_ui._api_key_available('OpenRouter', None, '', env)
    with pytest.raises(ValueError, match='No API key'):
        cli._resolve_embedding_choice('qwen/test', 'openai_compatible', env)
    stage = next(s for s in control.capabilities()['stages'] if s['stage'] == 'embedding')
    assert stage['availability'] == 'unavailable'
