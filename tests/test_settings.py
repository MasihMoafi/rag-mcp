import io
import json
import os
import stat
from pathlib import Path
from threading import Event, Thread
import pytest

from rag import settings, model_catalog


def env_for(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"RAG_MCP_CONFIG": str(tmp_path / "rag" / "config.json"), **extra}


def test_failed_settings_publication_preserves_old_configuration(tmp_path, monkeypatch):
    from utils import persistence

    environ = env_for(tmp_path)
    path = settings.save_settings({'top_k': '5'}, environ)
    original = path.read_bytes()
    def fail_replace(source, destination):
        assert Path(destination) == path
        assert stat.S_IMODE(os.stat(source).st_mode) == 0o600
        raise OSError('publication failed')
    monkeypatch.setattr(persistence.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='publication failed'):
        settings.save_settings({'top_k': '9'}, environ)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_concurrent_cli_setting_updates_keep_both_fields(tmp_path, monkeypatch):
    environ = env_for(tmp_path)
    settings.save_settings({'top_k': '5'}, environ)
    entered, release, second_started = Event(), Event(), Event()
    publish = settings.write_json_atomic
    def delayed(path, value):
        if value.get('answer_model') == 'first/model':
            entered.set()
            assert release.wait(5)
        publish(path, value)
    monkeypatch.setattr(settings, 'write_json_atomic', delayed)
    errors = []
    def change(values, started=None):
        if started:
            started.set()
        try:
            settings.update_settings(values, environ)
        except Exception as error:
            errors.append(error)
    first = Thread(target=change, args=({'answer_model': 'first/model'},))
    second = Thread(target=change, args=({'candidate_k': '17'}, second_started))
    first.start()
    assert entered.wait(5)
    second.start()
    assert second_started.wait(5)
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not errors and not first.is_alive() and not second.is_alive()
    assert settings.load_settings(environ) == {'top_k': '5', 'answer_model': 'first/model', 'candidate_k': '17'}


def test_embedding_choice_keeps_existing_endpoint_and_credential_preferences(tmp_path, monkeypatch):
    from rag import cli, detect
    environ = env_for(tmp_path)
    settings.update_settings({'embed_provider': 'openai_compatible', 'embed_model': 'old-model',
                              'embed_base_url': 'http://127.0.0.1:1234/v1',
                              'embed_api_key': 'private-key', 'ollama_url': 'http://127.0.0.1:11435'}, environ)
    monkeypatch.setattr(detect, 'detect_all', lambda *_: {'ollama': {'embedding_models': []}})
    choice = cli._resolve_embedding_choice('BAAI/bge-small-en-v1.5', 'fastembed', environ)
    assert choice['embed_provider'] == 'fastembed'
    assert choice['embed_base_url'] == 'http://127.0.0.1:1234/v1'
    assert choice['embed_api_key'] == 'private-key'
    assert choice['ollama_url'] == 'http://127.0.0.1:11435'
    with_environment_key = {**environ, 'RAG_MCP_EMBED_API_KEY': 'environment-only-secret'}
    assert cli._resolve_embedding_choice('BAAI/bge-small-en-v1.5', 'fastembed', with_environment_key)['embed_api_key'] == 'private-key'
    settings.update_settings({'embed_api_key': ''}, environ)
    assert cli._resolve_embedding_choice('BAAI/bge-small-en-v1.5', 'fastembed', with_environment_key)['embed_api_key'] == ''


def test_defaults_are_exact_words_only_until_a_choice_is_saved(tmp_path: Path):
    environ = env_for(tmp_path)
    config = settings.embedding_config(environ)
    assert config["embed_provider"] == "none"
    assert not settings.has_embedding(config)
    assert "BM25" in settings.describe_embedding(config)


def test_fresh_reranker_default_requires_an_openrouter_key(tmp_path: Path, monkeypatch):
    no_key = env_for(tmp_path)
    assert settings.retrieval_config(no_key)['reranker_type'] == 'off'
    assert settings.retrieval_config(no_key)['reranker_model'] == ''
    assert settings.retrieval_config({**no_key, 'OPENROUTER_API_KEY': 'fixture-key'})['reranker_type'] == 'openrouter'
    settings.save_settings({'embed_base_url': 'https://openrouter.ai/api/v1', 'embed_api_key': 'fixture-key'}, no_key)
    assert settings.retrieval_config(no_key)['reranker_type'] == 'openrouter'

    explicit = env_for(tmp_path / 'explicit')
    settings.save_settings({'reranker_type': 'openrouter'}, explicit)
    assert settings.retrieval_config(explicit)['reranker_type'] == 'openrouter'
    assert settings.retrieval_config({**explicit, 'RAG_MCP_RERANKER_TYPE': 'off'})['reranker_type'] == 'off'
    assert settings.retrieval_config({**explicit, 'RAG_MCP_RERANKER_TYPE': 'off'})['reranker_model'] == ''

    monkeypatch.setenv('RAG_MCP_CONFIG', env_for(tmp_path / 'empty')['RAG_MCP_CONFIG'])
    monkeypatch.setenv('RAG_MCP_EMBED_API_KEY', '')
    monkeypatch.setenv('OPENROUTER_API_KEY', '')
    with settings.frozen_settings():
        assert settings.retrieval_config()['reranker_type'] == 'off'
    monkeypatch.setenv('RAG_MCP_CONFIG', no_key['RAG_MCP_CONFIG'])
    with settings.frozen_settings():
        assert settings.retrieval_config()['reranker_type'] == 'openrouter'


@pytest.mark.parametrize('content', ['{"embed_provider":', '[]', 'null', '"private-marker"'])
def test_damaged_settings_are_not_replaced_by_an_update(tmp_path, content):
    environ = env_for(tmp_path)
    path = settings.config_path(environ)
    path.parent.mkdir(parents=True)
    path.write_text(content)
    with pytest.raises(ValueError, match='settings') as failure:
        settings.update_settings({'top_k': '9'}, environ)
    assert path.read_text() == content
    assert 'private-marker' not in str(failure.value)


def test_unreadable_settings_do_not_silently_select_defaults(tmp_path, monkeypatch):
    failure = PermissionError('settings unavailable')
    def unreadable(*args, **kwargs):
        raise failure
    monkeypatch.setattr(Path, 'read_text', unreadable)
    with pytest.raises(PermissionError) as caught:
        settings.embedding_config(env_for(tmp_path))
    assert caught.value is failure


def test_saved_choice_is_owner_only_and_environment_overrides_it(tmp_path: Path):
    environ = env_for(tmp_path)
    path = settings.save_settings(
        {"embed_provider": "fastembed", "embed_model": "BAAI/bge-small-en-v1.5"}, environ
    )
    assert path == tmp_path / "rag" / "config.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    config = settings.embedding_config(environ)
    assert (config["embed_provider"], config["embed_model"]) == ("fastembed", "BAAI/bge-small-en-v1.5")

    overridden = settings.embedding_config(env_for(tmp_path, RAG_MCP_EMBED_PROVIDER="ollama", RAG_MCP_EMBED_MODEL="nomic-embed-text"))
    assert (overridden["embed_provider"], overridden["embed_model"]) == ("ollama", "nomic-embed-text")


def test_interactive_setup_lists_every_choice_and_saves_only_after_verification(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("rag.detect.detect_all", lambda *_: {})
    monkeypatch.setattr(model_catalog, "api_models", lambda *args: ["qwen/qwen3-embedding-4b"])
    environ = env_for(tmp_path)
    probes = []
    stdout = io.StringIO()

    code = settings.run_setup(
        stdin=io.StringIO("1\n3\nsk-test\n1\n"),
        stdout=stdout,
        stderr=io.StringIO(),
        environ=environ,
        verify=lambda config: probes.append(dict(config)) or 2560,
    )

    assert code == 0
    assert "No local embedding models found" in stdout.getvalue()
    assert "1) Choose an API model" in stdout.getvalue()
    assert "Works: 2560 dimensions." in stdout.getvalue()
    assert probes == [
        {
            "embed_provider": "openai_compatible",
            "embed_model": "qwen/qwen3-embedding-4b",
            "embed_base_url": "https://openrouter.ai/api/v1",
            "embed_api_key": "sk-test",
            "ollama_url": "http://localhost:11434",
        }
    ]
    assert json.loads(settings.config_path(environ).read_text()) == probes[0]


def test_failed_verification_saves_nothing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("rag.detect.detect_all", lambda *_: {"ollama": {"embedding_models": ["installed-model"]}})
    environ = env_for(tmp_path)
    stderr = io.StringIO()

    def failing(_config):
        raise RuntimeError("model not found")

    code = settings.run_setup(
        stdin=io.StringIO("1\n\n"),
        stdout=io.StringIO(),
        stderr=stderr,
        environ=environ,
        verify=failing,
    )

    assert code == 1
    assert "model not found" in stderr.getvalue()
    assert not settings.config_path(environ).exists()


def test_exact_words_only_choice_needs_no_verification(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("rag.detect.detect_all", lambda *_: {})
    env = env_for(tmp_path)
    def never(_config):
        raise AssertionError("BM25 must not embed anything")
    code = settings.run_setup(stdin=io.StringIO("3\n"), stdout=io.StringIO(), stderr=io.StringIO(), environ=env, verify=never)
    assert code == 0
    assert settings.embedding_config(env)["embed_provider"] == "none"


def test_contextual_retrieval_is_off_by_default_and_can_be_enabled(tmp_path: Path):
    environ = env_for(tmp_path)

    defaults = settings.retrieval_config(environ)
    assert defaults["contextual_retrieval"] == "off"
    assert not settings.contextual_retrieval_enabled(defaults)

    settings.save_settings({"contextual_retrieval": "on", "contextual_model": "claude-sonnet"}, environ)
    saved = settings.retrieval_config(environ)
    assert settings.contextual_retrieval_enabled(saved)
    assert settings.describe_contextual_retrieval(saved) == "on · claude-sonnet"

    overridden = settings.retrieval_config(
        env_for(tmp_path, RAG_MCP_CONTEXTUAL_RETRIEVAL="off")
    )
    assert not settings.contextual_retrieval_enabled(overridden)
