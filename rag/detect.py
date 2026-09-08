"""What is already available on this machine for meaning-based search.

Used by `rag setup` (to recommend and prefill a choice), `rag setup --auto`
(to pick without questions) and `rag doctor`. Every probe is local and fast.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any
from urllib.request import Request, urlopen

EMBEDDING_NAME = re.compile(r"embed|bge|e5|nomic|gte|minilm|arctic|instructor", re.IGNORECASE)
# Preferred Ollama embedding models, best first.
OLLAMA_PREFERENCE = ("qwen3-embedding", "bge-m3", "nomic-embed", "embeddinggemma", "mxbai-embed", "all-minilm")

API_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}
API_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1", "text-embedding-3-small"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-embedding-001"),
    "openrouter": ("https://openrouter.ai/api/v1", "qwen/qwen3-embedding-4b"),
}
LOCAL_OPENAI_RUNTIMES = (("LM Studio", "http://localhost:1234/v1"), ("vLLM", "http://localhost:8000/v1"))


def _get_json(url: str, timeout: float = 1.5) -> Any:
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def detect_ollama(url: str = "http://localhost:11434") -> dict[str, Any]:
    payload = _get_json(f"{url.rstrip('/')}/api/tags")
    if not isinstance(payload, dict):
        return {"running": False, "embedding_models": []}
    names = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
    embedding = [name for name in names if EMBEDDING_NAME.search(name)]
    embedding.sort(key=lambda name: next((i for i, p in enumerate(OLLAMA_PREFERENCE) if p in name.lower()), len(OLLAMA_PREFERENCE)))
    return {"running": True, "embedding_models": embedding}


def detect_local_runtimes() -> list[dict[str, Any]]:
    found = []
    for label, base in LOCAL_OPENAI_RUNTIMES:
        payload = _get_json(f"{base}/models")
        if isinstance(payload, dict):
            models = [m.get("id", "") for m in payload.get("data", []) if isinstance(m, dict)]
            found.append({"label": label, "base_url": base, "embedding_models": [m for m in models if EMBEDDING_NAME.search(m)]})
    return found


def detect_api_keys(environ: Mapping[str, str] | None = None) -> list[str]:
    environ = os.environ if environ is None else environ
    return [provider for provider, names in API_KEY_ENV.items() if any(environ.get(name, "").strip() for name in names)]


def detect_all(environ: Mapping[str, str] | None = None, ollama_url: str = "http://localhost:11434") -> dict[str, Any]:
    return {"ollama": detect_ollama(ollama_url), "local_runtimes": detect_local_runtimes(), "api_keys": detect_api_keys(environ)}


def describe(detection: Mapping[str, Any]) -> list[str]:
    lines = []
    ollama = detection.get("ollama") or {}
    if ollama.get("running"):
        models = ollama.get("embedding_models") or []
        lines.append("Ollama is running" + (f" with embedding models: {', '.join(models)}" if models else " (no embedding model pulled yet)"))
    for runtime in detection.get("local_runtimes") or []:
        lines.append(f"{runtime['label']} is running at {runtime['base_url']}")
    keys = detection.get("api_keys") or []
    if keys:
        lines.append("API keys in the environment: " + ", ".join({"openai": "OpenAI", "gemini": "Gemini", "openrouter": "OpenRouter"}[k] for k in keys))
    return lines


def recommend(detection: Mapping[str, Any]) -> dict[str, str]:
    """The best embedding config given what runs here; keys stay in the environment."""
    ollama = detection.get("ollama") or {}
    if ollama.get("running") and ollama.get("embedding_models"):
        return {"embed_provider": "ollama", "embed_model": ollama["embedding_models"][0]}
    for runtime in detection.get("local_runtimes") or []:
        if runtime.get("embedding_models"):
            return {"embed_provider": "openai_compatible", "embed_base_url": runtime["base_url"], "embed_model": runtime["embedding_models"][0], "embed_api_key": "local"}
    for provider in ("openai", "gemini", "openrouter"):
        if provider in (detection.get("api_keys") or []):
            base_url, model = API_ENDPOINTS[provider]
            return {"embed_provider": "openai_compatible", "embed_base_url": base_url, "embed_model": model}
    return {"embed_provider": "fastembed", "embed_model": "BAAI/bge-small-en-v1.5"}
