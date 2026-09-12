import io
import json
import os
import stat
from pathlib import Path

from rag import settings, model_catalog


def env_for(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"RAG_MCP_CONFIG": str(tmp_path / "rag" / "config.json"), **extra}


def test_defaults_are_exact_words_only_until_a_choice_is_saved(tmp_path: Path):
    environ = env_for(tmp_path)
    config = settings.embedding_config(environ)
    assert config["embed_provider"] == "none"
    assert not settings.has_embedding(config)
    assert "BM25" in settings.describe_embedding(config)


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
