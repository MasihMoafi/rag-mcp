import pytest

from rag import settings


def test_environment_variable_names_remain_compatible():
    assert list(settings.ENV_KEYS.items()) == [
        ('embed_provider', 'RAG_MCP_EMBED_PROVIDER'),
        ('embed_model', 'RAG_MCP_EMBED_MODEL'),
        ('embed_base_url', 'RAG_MCP_EMBED_BASE_URL'),
        ('embed_api_key', 'RAG_MCP_EMBED_API_KEY'),
        ('ollama_url', 'RAG_MCP_OLLAMA_URL'),
    ]
    assert list(settings.CONTEXTUAL_ENV.items()) == [
        ('contextual_retrieval', 'RAG_MCP_CONTEXTUAL_RETRIEVAL'),
        ('contextual_model', 'RAG_MCP_CONTEXTUAL_MODEL'),
        ('contextual_max_tokens', 'RAG_MCP_CONTEXTUAL_MAX_TOKENS'),
        ('contextual_timeout', 'RAG_MCP_CONTEXTUAL_TIMEOUT'),
        ('contextual_source_max_chars', 'RAG_MCP_CONTEXTUAL_SOURCE_MAX_CHARS'),
        ('contextual_start_chunks', 'RAG_MCP_CONTEXTUAL_START_CHUNKS'),
        ('contextual_prev_chunks', 'RAG_MCP_CONTEXTUAL_PREV_CHUNKS'),
    ]


@pytest.mark.parametrize('saved, embedding, retrieval', [
    ('  custom  ', 'custom', 'custom'),
    ('   ', 'none', '5'),
    (17, 'none', '17'),
    (True, 'none', 'True'),
    (None, 'none', '5'),
    (1.5, 'none', '5'),
    ([], 'none', '5'),
])
def test_saved_value_types_and_defaults(monkeypatch, saved, embedding, retrieval):
    monkeypatch.setattr(settings, 'load_settings', lambda env: {'embed_provider': saved, 'top_k': saved})
    assert settings.embedding_config({})['embed_provider'] == embedding
    assert settings.retrieval_config({})['top_k'] == retrieval
    assert settings.DEFAULTS['embed_provider'] == 'none'
    assert settings.RETRIEVAL_DEFAULTS['top_k'] == '5'


@pytest.mark.parametrize('override, expected', [('  override  ', 'override'), ('  ', 'saved')])
def test_environment_has_final_precedence(monkeypatch, override, expected):
    monkeypatch.setattr(settings, 'load_settings', lambda env: {'embed_provider': 'saved', 'top_k': 'saved'})
    env = {'RAG_MCP_EMBED_PROVIDER': override, 'RAG_MCP_RERANK_TOP_K': override}
    assert settings.embedding_config(env)['embed_provider'] == expected
    assert settings.retrieval_config(env)['top_k'] == expected


def test_process_environment_is_used_only_when_argument_is_none(monkeypatch):
    monkeypatch.setattr(settings, 'load_settings', lambda env: {})
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'process')
    monkeypatch.setenv('RAG_MCP_RERANK_TOP_K', '12')
    assert settings.embedding_config()['embed_provider'] == 'process'
    assert settings.embedding_config({})['embed_provider'] == 'none'
    assert settings.retrieval_config()['top_k'] == '12'
    assert settings.retrieval_config({})['top_k'] == '5'
