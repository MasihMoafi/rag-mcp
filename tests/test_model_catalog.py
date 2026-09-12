import io
import json
import urllib.error

import pytest

from rag import detect, model_catalog, prompts, settings


def test_ollama_uses_capabilities_instead_of_model_name(monkeypatch):
    def get(url, **kwargs):
        if url.endswith("/api/tags"):
            return {"models": [{"name": name} for name in ("my-private-vector-model", "bge-chat", "offline")]}
        return {"my-private-vector-model": {"capabilities": ["embedding"]},
                "bge-chat": {"capabilities": ["completion"]}}.get(kwargs["payload"]["model"])
    monkeypatch.setattr(detect, "_get_json", get)
    assert detect.detect_ollama()["embedding_models"] == ["my-private-vector-model"]


def test_cached_model_requires_local_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(detect.importlib.util, "find_spec", lambda _: object())
    snapshot = tmp_path / "models--owner--private-model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "modules.json").write_text("[]")
    assert detect.detect_cached_models({"HF_HUB_CACHE": str(tmp_path)}) == []
    (snapshot / "model.safetensors").write_bytes(b"test weights")
    assert detect.detect_cached_models({"HF_HUB_CACHE": str(tmp_path)}) == [{"label": "owner/private-model", "path": str(snapshot)}]


@pytest.mark.parametrize("provider,payload,expected", [
    ("OpenRouter", {"data": [{"id": "brand/new-vector"}]}, ["brand/new-vector"]),
    ("OpenAI", {"data": [{"id": "gpt-chat"}, {"id": "text-embedding-new"}]}, ["text-embedding-new"]),
    ("Gemini", {"models": [{"name": "models/new-vector", "supportedGenerationMethods": ["embedContent"]},
                             {"name": "models/chat", "supportedGenerationMethods": ["generateContent"]}]}, ["new-vector"]),
])
def test_catalog_returns_live_ids_and_filters_chat(monkeypatch, provider, payload, expected):
    requests = []
    def open_request(request, **kwargs):
        requests.append(request)
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(model_catalog.urllib.request, "urlopen", open_request)
    result = model_catalog.api_models(provider, model_catalog.PROVIDERS[provider][0], "private-key")
    assert result == expected
    assert "private-key" not in requests[0].full_url
    if provider == "OpenRouter":
        assert requests[0].full_url.endswith("/embeddings/models")


def test_catalog_failure_never_invents_model_choices(monkeypatch):
    def reject(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)
    monkeypatch.setattr(model_catalog.urllib.request, "urlopen", reject)
    with pytest.raises(ValueError, match="rejected this API key"):
        model_catalog.api_models("OpenAI", "https://api.openai.com/v1", "wrong-key")


def test_no_models_shows_sources_without_download_presets(monkeypatch, tmp_path):
    monkeypatch.setattr(detect, "detect_all", lambda *_: {})
    output = io.StringIO()
    assert settings.run_setup(stdin=io.StringIO("3\n"), stdout=output, stderr=io.StringIO(),
                              environ={"RAG_MCP_CONFIG": str(tmp_path / "config.json")}) == 0
    assert "No local embedding models found" in output.getvalue()
    assert "BGE small" not in output.getvalue()
    assert "MiniLM" not in output.getvalue()


def test_failed_catalog_returns_to_provider_and_escape_without_saving(monkeypatch, tmp_path):
    monkeypatch.setattr(detect, "detect_all", lambda *_: {})
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    selections = iter([0, 0, prompts.Cancelled(), prompts.Cancelled()])
    def select(*args, **kwargs):
        value = next(selections)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(prompts, "select", select)
    def fail(*args):
        raise ValueError("Could not load the model catalog")
    monkeypatch.setattr(model_catalog, "api_models", fail)
    env = {"RAG_MCP_CONFIG": str(tmp_path / "config.json"), "OPENAI_API_KEY": "test"}
    output = io.StringIO()
    with pytest.raises(prompts.Cancelled):
        settings.run_setup(stdin=io.StringIO(), stdout=output, stderr=io.StringIO(), environ=env)
    assert "Could not load" in output.getvalue()
    assert not settings.config_path(env).exists()
