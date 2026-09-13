"""Where the user's embedding choice lives, and the chooser that sets it.

Exact-word search (BM25) is always on. Meaning-based search needs an embedding
model, and the user picks one here: `rag setup` writes the choice to a small
JSON file; RAG_MCP_* environment variables override it per process.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO
from utils.persistence import write_json_atomic

# Every choice the interactive chooser offers, in display order.
# Defaults for explicit command-line provider flags; interactive choices use discovery.
PROVIDER_DEFAULT_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "ollama": "nomic-embed-text",
    "openai_compatible": "text-embedding-3-small",
    "sentencetransformer": "all-MiniLM-L6-v2",
    "none": "",
}
PROVIDERS = tuple(PROVIDER_DEFAULT_MODELS)

DEFAULTS = {
    "embed_provider": "none",
    "embed_model": "",
    "embed_base_url": "",
    "embed_api_key": "",
    "ollama_url": "http://localhost:11434",
}

ENV_KEYS = {key: f"RAG_MCP_{key.upper()}" for key in DEFAULTS}

# Retrieval preferences saved next to the embedding choice; RAG_MCP_* overrides per process.
CONTEXTUAL_DEFAULTS = {
    "contextual_retrieval": "off",
    "contextual_model": "claude-haiku-4-5",
    "contextual_max_tokens": "128",
    "contextual_timeout": "30",
    "contextual_source_max_chars": "12000",
    "contextual_start_chunks": "2",
    "contextual_prev_chunks": "2",
}
CONTEXTUAL_ENV = {key: f"RAG_MCP_{key.upper()}" for key in CONTEXTUAL_DEFAULTS}
RETRIEVAL_DEFAULTS = {
    "top_k": "5",
    "reranker_type": "auto",
    "reranker_model": "",
    **CONTEXTUAL_DEFAULTS,
}
RETRIEVAL_ENV = {
    "top_k": "RAG_MCP_RERANK_TOP_K",
    "reranker_type": "RAG_MCP_RERANKER_TYPE",
    "reranker_model": "RAG_MCP_RERANKER_MODEL",
    **CONTEXTUAL_ENV,
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
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        raise ValueError("Invalid RAG settings JSON; repair the configuration file before continuing.") from None
    if not isinstance(data, dict):
        raise ValueError("Invalid RAG settings: expected a JSON object.")
    return data


def save_settings(settings: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> Path:
    path = config_path(environ)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, dict(settings))
    return path


def _merged_config(defaults, env_keys, value_types, environ):
    environ = os.environ if environ is None else environ
    merged = dict(defaults)
    saved = load_settings(environ)
    for key in merged:
        value = saved.get(key)
        if isinstance(value, value_types) and str(value).strip():
            merged[key] = str(value).strip()
    for key, name in env_keys.items():
        value = environ.get(name, "").strip()
        if value:
            merged[key] = value
    return merged


def embedding_config(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Defaults, then the saved file, then RAG_MCP_* environment overrides."""
    return _merged_config(DEFAULTS, ENV_KEYS, str, environ)


def retrieval_config(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """top_k / reranker: defaults, then the saved file, then RAG_MCP_* overrides."""
    return _merged_config(RETRIEVAL_DEFAULTS, RETRIEVAL_ENV, (str, int), environ)


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


def contextual_retrieval_enabled(config: Mapping[str, str]) -> bool:
    return str(config.get("contextual_retrieval", "off")).strip().lower() in {
        "on",
        "true",
        "1",
        "enabled",
    }


def describe_contextual_retrieval(config: Mapping[str, str]) -> str:
    if not contextual_retrieval_enabled(config):
        return "off"
    return f"on · {config.get('contextual_model') or CONTEXTUAL_DEFAULTS['contextual_model']}"


def describe_settings(environ: Mapping[str, str] | None = None) -> list[str]:
    embedding = embedding_config(environ)
    retrieval = retrieval_config(environ)
    return [
        f"Embedding: {describe_embedding(embedding)}",
        f"Passages per query (top): {retrieval['top_k']}",
        f"Reranker: {describe_reranker(retrieval)}",
        f"Contextual retrieval: {describe_contextual_retrieval(retrieval)}",
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
    model = config.get("embed_model") or "default model"
    parts = Path(model).parts
    if provider == "sentencetransformer" and len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--"):
        model = parts[-3][len("models--"):].replace("--", "/")
    return f"{names.get(provider, provider)} · {model}"


def default_model(provider: str) -> str:
    return PROVIDER_DEFAULT_MODELS.get(provider, "")


def run_setup(*, stdin: TextIO, stdout: TextIO, stderr: TextIO, environ: Mapping[str, str] | None = None, verify: Callable[[Mapping[str, str]], int] | None = None) -> int:
    from rag.setup_ui import run

    try:
        return run(stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, verify=verify)
    except ValueError as error:
        stderr.write(f"{error}\n")
        return 2


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
