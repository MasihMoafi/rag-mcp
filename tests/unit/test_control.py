import pytest

from rag import settings
from rag.control import read_config, patch_config, capabilities, model_inventory


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    for name in [*settings.ENV_KEYS.values(), *settings.RETRIEVAL_ENV.values(),
                 'OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY']:
        monkeypatch.delenv(name, raising=False)


def test_config_is_live_and_never_returns_credentials():
    settings.update_settings({'embed_api_key': 'private-secret', 'top_k': '7'})
    assert read_config()['retrieval']['top_k'] == '7'
    assert 'private-secret' not in str(read_config()) + str(capabilities())
    result = patch_config({'top': 3, 'answer_model': 'provider/example'})
    assert result['retrieval']['top_k'] == '3'
    assert settings.retrieval_config()['answer_model'] == 'provider/example'


def test_invalid_patch_is_atomic_and_environment_override_is_explicit(monkeypatch):
    before = settings.load_settings()
    with pytest.raises(ValueError):
        patch_config({'top': 4, 'answer_model': ''})
    assert settings.load_settings() == before
    with pytest.raises(ValueError):
        patch_config({'embed_api_key': 'do-not-store'})
    monkeypatch.setenv('RAG_MCP_RERANK_TOP_K', '9')
    with pytest.raises(ValueError, match='environment'):
        patch_config({'top': 2})
    assert read_config()['overrides'] == ['top_k']


def test_stale_config_revision_rejects_a_browser_edit():
    initial = read_config()['revision']
    settings.update_settings({'top_k': '7'})
    with pytest.raises(settings.SettingsConflict):
        patch_config({'top': 4, 'revision': initial})
    assert settings.retrieval_config()['top_k'] == '7'
    fresh = read_config()['revision']
    saved = patch_config({'top': 4, 'revision': fresh})
    assert saved['retrieval']['top_k'] == '4'
    assert saved['revision'] != fresh


def test_discovered_local_embedding_is_probed_before_save(monkeypatch):
    from rag import cli, control, detect
    settings.update_settings({'embed_api_key': 'private-preference', 'ollama_url': 'http://127.0.0.1:11435'})
    monkeypatch.setattr(control, 'model_inventory', lambda: {'models': [{
        'stage': 'embedding', 'provider': 'fastembed', 'model': 'cached/model',
        'location': 'local', 'discovery_source': 'fastembed_cache', 'runtime': 'FastEmbed'}]})
    monkeypatch.setattr(detect, 'detect_all', lambda *_: {'ollama': {'embedding_models': []}})
    probed = []
    monkeypatch.setattr(cli, '_probe_embedding', lambda config: probed.append(dict(config)) or 384)
    choice = {'embedding_provider': 'fastembed', 'embedding_model': 'cached/model',
              'embedding_source': 'fastembed_cache', 'embedding_runtime': 'FastEmbed',
              'revision': read_config()['revision']}
    saved = patch_config(choice)
    assert saved['embedding'] == {'provider': 'fastembed', 'model': 'cached/model'}
    assert len(probed) == 1 and probed[0]['embed_model'] == 'cached/model'
    assert settings.load_settings()['embed_api_key'] == 'private-preference'
    assert settings.load_settings()['ollama_url'] == 'http://127.0.0.1:11435'


def test_local_embedding_activation_rejects_failed_probe_and_missing_model(monkeypatch):
    from rag import cli, control, detect
    monkeypatch.setattr(control, 'model_inventory', lambda: {'models': [{
        'stage': 'embedding', 'provider': 'fastembed', 'model': 'cached/model',
        'location': 'local', 'discovery_source': 'fastembed_cache', 'runtime': 'FastEmbed'}]})
    monkeypatch.setattr(detect, 'detect_all', lambda *_: {'ollama': {'embedding_models': []}})
    before = settings.load_settings()
    choice = {'embedding_provider': 'fastembed', 'embedding_model': 'cached/model',
              'embedding_source': 'fastembed_cache', 'embedding_runtime': 'FastEmbed',
              'revision': read_config()['revision']}
    monkeypatch.setattr(cli, '_probe_embedding', lambda _: (_ for _ in ()).throw(RuntimeError('private/path')))
    with pytest.raises(ValueError, match='verification failed'):
        patch_config(choice)
    assert settings.load_settings() == before
    with pytest.raises(ValueError, match='discovered'):
        patch_config({**choice, 'embedding_source': []})
    with pytest.raises(ValueError, match='no longer available'):
        patch_config({**choice, 'embedding_model': 'not-cached'})
    assert settings.load_settings() == before


def test_local_runtime_activation_uses_its_endpoint_and_clears_old_key(monkeypatch):
    from rag import cli, control, detect
    settings.update_settings({'embed_provider': 'openai_compatible', 'embed_model': 'old-model',
                              'embed_base_url': 'https://api.openai.com/v1', 'embed_api_key': 'private-key'})
    monkeypatch.setattr(control, 'model_inventory', lambda: {'models': [{
        'stage': 'embedding', 'provider': 'openai_compatible', 'model': 'local-embed',
        'location': 'local', 'discovery_source': 'local_runtime', 'runtime': 'LM Studio'}]})
    monkeypatch.setattr(detect, 'detect_all', lambda *_: {'ollama': {'embedding_models': []},
        'local_runtimes': [{'label': 'LM Studio', 'base_url': 'http://localhost:1234/v1',
                            'embedding_models': ['local-embed']}], 'api_keys': []})
    probed = []
    monkeypatch.setattr(cli, '_probe_embedding', lambda config: probed.append(dict(config)) or 768)
    saved = patch_config({'embedding_provider': 'openai_compatible', 'embedding_model': 'local-embed',
                          'embedding_source': 'local_runtime', 'embedding_runtime': 'LM Studio',
                          'revision': read_config()['revision']})
    assert saved['embedding']['model'] == 'local-embed'
    assert probed[0]['embed_base_url'] == 'http://localhost:1234/v1'
    assert probed[0]['embed_api_key'] == ''
    assert settings.load_settings()['embed_api_key'] == ''


def test_local_runtime_never_inherits_hosted_environment_key(monkeypatch):
    from rag import embeddings
    monkeypatch.setenv('OPENAI_API_KEY', 'private-hosted-key')
    assert embeddings.api_key_from_environment('http://localhost:1234/v1') is None
    supplied = []
    monkeypatch.setattr(embeddings, 'OpenAICompatibleWrapper', lambda *args, **kwargs: supplied.append(kwargs))
    embeddings.build_embedder({'embed_provider': 'openai_compatible', 'embed_model': 'local-embed',
                               'embed_base_url': 'http://localhost:1234/v1', 'embed_api_key': 'private-hosted-key'})
    assert supplied[0]['api_key'] == 'local'


def test_capabilities_distinguish_missing_credentials_and_unverified_model(monkeypatch):
    answer = next(item for item in capabilities()['stages'] if item['stage'] == 'generation')
    assert answer['availability'] == 'unavailable'
    monkeypatch.setenv('OPENROUTER_API_KEY', 'private-secret')
    answer = next(item for item in capabilities()['stages'] if item['stage'] == 'generation')
    assert answer['availability'] == 'unknown'
    assert answer['discovery_source'] == 'engine_configuration'


def test_embedding_capability_uses_endpoint_credentials_without_exposing_them(monkeypatch):
    settings.update_settings({'embed_provider': 'openai_compatible', 'embed_model': 'test-embedding',
                              'embed_base_url': 'https://api.openai.com/v1'})
    embedding = next(item for item in capabilities()['stages'] if item['stage'] == 'embedding')
    assert embedding['availability'] == 'unavailable'
    assert embedding['required_credential'] == 'openai_key'
    settings.update_settings({'embed_api_key': 'private-saved-key'})
    embedding = next(item for item in capabilities()['stages'] if item['stage'] == 'embedding')
    assert embedding['availability'] == 'unknown'
    assert 'private-saved-key' not in str(capabilities())
    settings.update_settings({'embed_api_key': '', 'embed_base_url': 'http://127.0.0.1:1234/v1'})
    embedding = next(item for item in capabilities()['stages'] if item['stage'] == 'embedding')
    assert embedding['availability'] == 'unknown'
    assert embedding['location'] == 'local' and embedding['required_credential'] is None
    settings.update_settings({'embed_base_url': 'https://openrouter.ai/api/v1'})
    embedding = next(item for item in capabilities()['stages'] if item['stage'] == 'embedding')
    assert embedding['availability'] == 'unavailable'
    monkeypatch.setenv('OPENROUTER_API_KEY', 'private-environment-key')
    embedding = next(item for item in capabilities()['stages'] if item['stage'] == 'embedding')
    assert embedding['availability'] == 'unknown'
    assert 'private-environment-key' not in str(capabilities())
    monkeypatch.delenv('OPENROUTER_API_KEY')
    settings.update_settings({'embed_base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/'})
    embedding = next(item for item in capabilities()['stages'] if item['stage'] == 'embedding')
    assert embedding['availability'] == 'unavailable' and embedding['required_credential'] == 'gemini_key'
    monkeypatch.setenv('GOOGLE_API_KEY', 'private-gemini-key')
    assert next(item for item in capabilities()['stages'] if item['stage'] == 'embedding')['availability'] == 'unknown'


def test_model_inventory_is_bounded_and_keeps_configured_models(monkeypatch):
    from rag import detect, model_catalog
    settings.update_settings({'embed_provider': 'ollama', 'embed_model': 'model-0',
                              'answer_model': 'private/answer', 'reranker_type': 'off',
                              'embed_api_key': 'private-secret'})
    monkeypatch.setattr(detect, 'detect_ollama', lambda url, max_models: {
        'running': True, 'embedding_models': [f'model-{n}' for n in range(20)], 'truncated': True})
    monkeypatch.setattr(detect, 'detect_local_runtimes', lambda: [{'label': 'LM Studio',
        'base_url': 'http://127.0.0.1:1234/v1', 'embedding_models': [f'local-{n}' for n in range(25)]}])
    monkeypatch.setattr(detect, 'detect_cached_models', lambda _: [{'label': f'cached-{n}',
        'path': f'/private/cache/{n}'} for n in range(25)])
    monkeypatch.setattr(model_catalog, 'cached_cpu_models', lambda _: [f'cpu-{n}' for n in range(25)])
    report = model_inventory()
    assert report['schema_version'] == 1
    assert len(report['models']) == 81  # 20 each from four sources, plus generation; embedding is deduped.
    assert [item for item in report['models'] if item['model'] == 'model-0'][0]['configured'] is True
    assert [item for item in report['models'] if item['stage'] == 'generation'][0]['model'] == 'private/answer'
    assert all(item['stage'] in {'embedding', 'generation'} for item in report['models'])
    assert all(source['status'] == 'truncated' for source in report['sources'])
    serialized = str(report)
    assert 'private-secret' not in serialized and '/private/cache' not in serialized


def test_model_inventory_marks_failed_source_without_guessing(monkeypatch):
    from rag import detect
    monkeypatch.setattr(detect, 'detect_ollama', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('private-path')))
    monkeypatch.setattr(detect, 'detect_local_runtimes', lambda: [])
    monkeypatch.setattr(detect, 'detect_cached_models', lambda _: [])
    monkeypatch.setattr('rag.model_catalog.cached_cpu_models', lambda _: [])
    report = model_inventory()
    assert report['sources'][0] == {'source': 'ollama', 'status': 'error'}
    assert 'private-path' not in str(report)


def test_a_run_keeps_its_settings_when_cli_changes_the_file():
    settings.update_settings({'top_k': '3'})
    with settings.frozen_settings():
        settings.update_settings({'top_k': '8'})
        assert settings.retrieval_config()['top_k'] == '3'
    assert settings.retrieval_config()['top_k'] == '8'


def test_run_settings_read_one_atomic_file_snapshot(monkeypatch):
    reads = []
    def changing_file(environ=None):
        reads.append(True)
        return {'embed_model': str(len(reads)), 'top_k': str(len(reads))}
    monkeypatch.setattr(settings, 'load_settings', changing_file)
    with settings.frozen_settings():
        assert settings.embedding_config()['embed_model'] == settings.retrieval_config()['top_k']
    assert len(reads) == 1
