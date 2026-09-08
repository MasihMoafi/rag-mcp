"""Where the user's embedding choice lives, and the chooser that sets it.

Exact-word search (BM25) is always on. Meaning-based search needs an embedding
model, and the user picks one here: `rag setup` writes the choice to a small
JSON file; RAG_MCP_* environment variables override it per process.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO

# Every choice the interactive chooser offers, in display order.
EMBEDDING_CHOICES: tuple[dict[str, str], ...] = (
    {
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
        "label": "Small local model (English)",
        "detail": "Downloads about 70 MB once, runs on the CPU, no account.",
    },
    {
        "provider": "fastembed",
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "label": "Small local multilingual model",
        "detail": "English, Persian and about 50 other languages. About 220 MB.",
    },
    {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "label": "Ollama on this machine",
        "detail": "Uses an embedding model you already pulled into Ollama.",
    },
    {
        "provider": "openai_compatible",
        "model": "text-embedding-3-small",
        "label": "API key",
        "detail": "OpenAI, Gemini, OpenRouter, Voyage or any OpenAI-compatible endpoint. Text leaves this machine.",
    },
    {
        "provider": "sentencetransformer",
        "model": "all-MiniLM-L6-v2",
        "label": "Full local install (sentence-transformers, GPU)",
        "detail": "Needs the [full] extra with PyTorch installed.",
    },
    {
        "provider": "none",
        "model": "",
        "label": "Exact words only (BM25)",
        "detail": "No meaning-based search until you pick a model later.",
    },
)

PROVIDERS = tuple(dict.fromkeys(choice["provider"] for choice in EMBEDDING_CHOICES))

API_ENDPOINTS = {
    "OpenAI": "https://api.openai.com/v1",
    "Gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "OpenRouter": "https://openrouter.ai/api/v1",
}

DEFAULTS = {
    "embed_provider": "none",
    "embed_model": "",
    "embed_base_url": "",
    "embed_api_key": "",
    "ollama_url": "http://localhost:11434",
}

ENV_KEYS = {
    "embed_provider": "RAG_MCP_EMBED_PROVIDER",
    "embed_model": "RAG_MCP_EMBED_MODEL",
    "embed_base_url": "RAG_MCP_EMBED_BASE_URL",
    "embed_api_key": "RAG_MCP_EMBED_API_KEY",
    "ollama_url": "RAG_MCP_OLLAMA_URL",
}

# Retrieval preferences saved next to the embedding choice; RAG_MCP_* overrides per process.
RETRIEVAL_DEFAULTS = {"top_k": "5", "reranker_type": "auto", "reranker_model": ""}
RETRIEVAL_ENV = {
    "top_k": "RAG_MCP_RERANK_TOP_K",
    "reranker_type": "RAG_MCP_RERANKER_TYPE",
    "reranker_model": "RAG_MCP_RERANKER_MODEL",
}
RERANKER_CHOICES = ("off", "auto", "fastembed")
FASTEMBED_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"
ST_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_ONLY_NOTICE = (
    "Exact-word search (BM25) only. Run `rag setup` to add an embedding model "
    "for meaning-based search."
)


def config_path(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    override = environ.get("RAG_MCP_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    base = environ.get("XDG_CONFIG_HOME", "").strip() or os.path.join(
        environ.get("HOME") or str(Path.home()), ".config"
    )
    return Path(base).expanduser() / "rag" / "config.json"


def load_settings(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    try:
        data = json.loads(config_path(environ).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(settings: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> Path:
    path = config_path(environ)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(settings), indent=2) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path


def embedding_config(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Defaults, then the saved file, then RAG_MCP_* environment overrides."""
    environ = os.environ if environ is None else environ
    merged = dict(DEFAULTS)
    saved = load_settings(environ)
    for key in merged:
        value = saved.get(key)
        if isinstance(value, str) and value.strip():
            merged[key] = value.strip()
    for key, name in ENV_KEYS.items():
        value = environ.get(name, "").strip()
        if value:
            merged[key] = value
    return merged


def retrieval_config(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """top_k / reranker: defaults, then the saved file, then RAG_MCP_* overrides."""
    environ = os.environ if environ is None else environ
    merged = dict(RETRIEVAL_DEFAULTS)
    saved = load_settings(environ)
    for key in merged:
        value = saved.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            merged[key] = str(value).strip()
    for key, name in RETRIEVAL_ENV.items():
        value = environ.get(name, "").strip()
        if value:
            merged[key] = value
    return merged


def update_settings(changes: Mapping[str, str], environ: Mapping[str, str] | None = None) -> Path:
    """Change a few keys and keep everything else in the file."""
    data = load_settings(environ)
    data.update({key: str(value) for key, value in changes.items()})
    return save_settings(data, environ)


def describe_reranker(config: Mapping[str, str]) -> str:
    kind = config.get("reranker_type", "auto")
    model = config.get("reranker_model", "")
    if kind in {"disabled", "none", "off"}:
        return "off"
    if kind == "auto":
        return "auto (cross-encoder when sentence-transformers is installed, otherwise off)"
    if kind == "fastembed":
        return f"local ONNX cross-encoder · {model or FASTEMBED_RERANKER}"
    return f"{kind} · {model or ST_RERANKER}"


def describe_settings(environ: Mapping[str, str] | None = None) -> list[str]:
    embedding = embedding_config(environ)
    retrieval = retrieval_config(environ)
    return [
        f"Embedding: {describe_embedding(embedding)}",
        f"Passages per query (top): {retrieval['top_k']}",
        f"Reranker: {describe_reranker(retrieval)}",
        f"Config file: {config_path(environ)}",
    ]


def has_embedding(config: Mapping[str, str]) -> bool:
    return config.get("embed_provider", "none") != "none"


def describe_embedding(config: Mapping[str, str]) -> str:
    provider = config.get("embed_provider", "none")
    if provider == "none":
        return "none · exact-word search (BM25) only"
    names = {
        "fastembed": "local model",
        "ollama": "Ollama",
        "openai_compatible": "API",
        "sentencetransformer": "sentence-transformers",
    }
    return f"{names.get(provider, provider)} · {config.get('embed_model') or 'default model'}"


def default_model(provider: str) -> str:
    for choice in EMBEDDING_CHOICES:
        if choice["provider"] == provider:
            return choice["model"]
    return ""


def _ask(stdin: TextIO, stdout: TextIO, prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    stdout.write(f"{prompt}{suffix}: ")
    stdout.flush()
    line = stdin.readline()
    if not line:
        return default
    return line.strip() or default


def run_setup(
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    environ: Mapping[str, str] | None = None,
    verify: Callable[[Mapping[str, str]], int] | None = None,
) -> int:
    """Interactive chooser. Saves only after the choice embeds a test sentence."""
    from rag import detect

    detection = detect.detect_all(environ)
    recommended = detect.recommend(detection)
    stdout.write(
        "How should rag understand meaning in your documents?\n"
        "Exact-word search (BM25) is always on; this adds meaning-based search.\n\n"
    )
    for line in detect.describe(detection):
        stdout.write(f"  Detected: {line}\n")
    if detect.describe(detection):
        stdout.write("\n")
    default_index = next(
        (i for i, c in enumerate(EMBEDDING_CHOICES, 1) if c["provider"] == recommended["embed_provider"]), 1
    )
    for index, choice in enumerate(EMBEDDING_CHOICES, 1):
        tag = "  (recommended)" if index == default_index else ""
        stdout.write(f"  {index}) {choice['label']}{tag}\n     {choice['detail']}\n")
    stdout.write("\n")
    answer = _ask(stdin, stdout, "Choice", str(default_index))
    try:
        choice = EMBEDDING_CHOICES[int(answer) - 1]
    except (ValueError, IndexError):
        stderr.write(f"Not a listed choice: {answer}\n")
        return 2

    config = dict(DEFAULTS)
    config["embed_provider"] = choice["provider"]
    prefilled = recommended if recommended["embed_provider"] == choice["provider"] else {}
    if choice["provider"] == "none":
        pass
    elif choice["provider"] == "ollama":
        config["embed_model"] = _ask(stdin, stdout, "Ollama embedding model", prefilled.get("embed_model") or choice["model"])
        config["ollama_url"] = _ask(stdin, stdout, "Ollama URL", DEFAULTS["ollama_url"])
    elif choice["provider"] == "openai_compatible":
        stdout.write("Known endpoints: " + " · ".join(f"{name} {url}" for name, url in API_ENDPOINTS.items()) + "\n")
        config["embed_base_url"] = _ask(stdin, stdout, "Endpoint URL", prefilled.get("embed_base_url") or API_ENDPOINTS["OpenAI"])
        config["embed_model"] = _ask(stdin, stdout, "Embedding model", prefilled.get("embed_model") or choice["model"])
        if prefilled:
            stdout.write("Using the API key from your environment variable.\n")
            config["embed_api_key"] = prefilled.get("embed_api_key", "")
        else:
            config["embed_api_key"] = _ask(stdin, stdout, "API key (saved to the config file, readable by you only)")
            if not config["embed_api_key"]:
                stderr.write("An API key is required for this option.\n")
                return 2
    else:
        config["embed_model"] = _ask(stdin, stdout, "Model", choice["model"])

    return finish_setup(config, stdout=stdout, stderr=stderr, environ=environ, verify=verify)


def auto_setup(
    *,
    stdout: TextIO,
    stderr: TextIO,
    environ: Mapping[str, str] | None = None,
    verify: Callable[[Mapping[str, str]], int] | None = None,
) -> int:
    """Pick the best option this machine already has, without questions."""
    from rag import detect

    detection = detect.detect_all(environ)
    config = dict(DEFAULTS)
    config.update(detect.recommend(detection))
    for line in detect.describe(detection):
        stdout.write(f"Detected: {line}\n")
    stdout.write(f"Chosen: {describe_embedding(config)}\n")
    return finish_setup(config, stdout=stdout, stderr=stderr, environ=environ, verify=verify)


def finish_setup(
    config: Mapping[str, str],
    *,
    stdout: TextIO,
    stderr: TextIO,
    environ: Mapping[str, str] | None = None,
    verify: Callable[[Mapping[str, str]], int] | None = None,
) -> int:
    """Embed one probe sentence with the chosen model, then save the choice."""
    if verify is not None and has_embedding(config):
        stdout.write("Checking the embedding model...\n")
        stdout.flush()
        try:
            dimensions = verify(config)
        except Exception as error:  # shown to the user; nothing is saved
            stderr.write(f"Could not embed a test sentence: {error}\nNothing was saved.\n")
            return 1
        stdout.write(f"Works: {dimensions} dimensions.\n")
    path = update_settings(config, environ)
    stdout.write(f"Saved to {path}\nEmbedding: {describe_embedding(config)}\n")
    return 0
