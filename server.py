#!/usr/bin/env python
# coding: utf-8
"""Minimal stdio MCP server exposing local hybrid RAG search as a tool.

Standalone fork of Elpis's internal elpis-rag host, generalized so any
MCP-compatible client (Claude Code, Claude Desktop, Codex, etc.) can point it
at an arbitrary workspace.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from fnmatch import fnmatch
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

repo_root = Path(__file__).resolve().parent
workspace_root = Path(
    os.environ.get("RAG_MCP_WORKSPACE_ROOT") or os.getcwd()
).expanduser().resolve()
sys.path.insert(0, str(repo_root))

from rag.settings import BM25_ONLY_NOTICE, embedding_config, has_embedding  # noqa: E402
from rag import settings as user_settings  # noqa: E402
from rag.formats import INDEXABLE_EXTENSIONS  # noqa: E402

_SERVER_NAME = "rag-mcp"
_SERVER_VERSION = "0.2.0"
_use_content_length_framing = False
_initialized = False
_INDEXABLE_EXTENSIONS = set(INDEXABLE_EXTENSIONS)
_UNSAFE_PATH_COMPONENTS = {"node_modules", ".git", ".venv", "venv", "dist", "build", "target", "__pycache__", "rag_db_v2"}
_INDEX_EXCLUDED_PATH_COMPONENTS = {".ipynb_checkpoints", "migrations", "rag_db_v2"}
_INDEX_EXCLUDED_FILE_PATTERNS = ("test_*.py", "*_test.py", "*__pycache__*", "*.pyc")
_MAX_INDEXED_FILES = 500
_INDEX_MAX_DEPTH = 5
_DEFAULT_MAX_DEPTH = 20
_DEFAULT_MAX_TOKENS = 2_000_000


@dataclass(frozen=True)
class _ScopeManifest:
    files: tuple[str, ...]
    watched_paths: tuple[tuple[str, int, int, bool], ...]

    def is_current(self) -> bool:
        for path, mtime_ns, size, is_file in self.watched_paths:
            try:
                stat = Path(path).stat()
            except OSError:
                return False
            if stat.st_mtime_ns != mtime_ns:
                return False
            if is_file and stat.st_size != size:
                return False
        return True


_validated_scope_cache: dict[tuple[str, int, int], _ScopeManifest] = {}

_TOOL = {
    "name": "query_knowledge_base",
    "description": (
        "Run local hybrid RAG (BM25 + vector + reranking) over a workspace or a "
        "supplied file/directory. Use this for broad semantic discovery instead of "
        "loading many files into context. Follow retrieved source paths with exact "
        "search or file reads to get current line positions."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for the knowledge base",
            },
            "doc_path": {
                "type": "string",
                "default": "",
                "description": "Optional file or directory to search. Empty uses the configured workspace root.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "annotations": {
        "title": "Search local knowledge base",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}

_SETTINGS_GET_TOOL = {
    "name": "get_rag_settings",
    "description": "Show the local rag search setup: embedding model, passages per query (top N), reranker, contextual retrieval, and what this machine has (Ollama models, local runtimes, API keys).",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    "annotations": {"title": "Show rag settings", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
}
_SETTINGS_SET_TOOL = {
    "name": "set_rag_settings",
    "description": (
        "Change the local rag search setup from plain words: passages per query (top), the reranker "
        "(off, auto, fastembed or a model name), or the embedding model (guessing where it runs: Ollama, "
        "a small local ONNX model, or an API whose key is already in the environment). Never pass API keys here; "
        "the user enters keys with `rag setup` in a terminal. The next query uses the new settings."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "top": {"type": "integer", "minimum": 1, "description": "How many passages a query returns"},
            "reranker": {"type": "string", "description": "off | auto | fastembed | a reranker model name"},
            "contextual_retrieval": {"type": "string", "enum": ["on", "off"], "description": "Enable or disable Anthropic-style ingestion-time contextual retrieval"},
            "embedding_model": {"type": "string", "description": "Embedding model name, e.g. qwen3-embedding:0.6b or BAAI/bge-small-en-v1.5"},
            "embedding_provider": {"type": "string", "enum": ["fastembed", "ollama", "openai_compatible", "sentencetransformer", "none"], "description": "Where the embedding model runs; guessed when omitted"},
        },
        "additionalProperties": False,
    },
    "annotations": {"title": "Change rag settings", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
}
_TOOLS = [_TOOL, _SETTINGS_GET_TOOL, _SETTINGS_SET_TOOL]


def _settings_report() -> dict[str, Any]:
    from rag import detect

    embedding = embedding_config()
    return {
        "settings": user_settings.describe_settings(),
        "embedding": {"provider": embedding["embed_provider"], "model": embedding["embed_model"]},
        "retrieval": user_settings.retrieval_config(),
        "detected": detect.detect_all(os.environ, embedding["ollama_url"]),
    }


def _apply_settings(arguments: dict[str, Any]) -> dict[str, Any]:
    from rag.cli import _config_command, _probe_embedding
    import argparse
    import io

    changed = []
    with contextlib.redirect_stdout(sys.stderr):
        for name, setting in (("top", "top"), ("reranker", "reranker"), ("contextual_retrieval", "contextual")):
            if name in arguments:
                args = argparse.Namespace(setting=setting, value=str(arguments[name]), provider=None, no_verify=True)
                code = _config_command(args, stdout=io.StringIO(), stderr=(err := io.StringIO()), environ=os.environ, verify=None)
                if code != 0:
                    return {"error": err.getvalue().strip()}
                changed.append(name)
        if "embedding_model" in arguments or arguments.get("embedding_provider") == "none":
            if arguments.get("embedding_provider") == "none":
                user_settings.update_settings({"embed_provider": "none", "embed_model": ""})
            else:
                args = argparse.Namespace(setting="embedding", value=str(arguments["embedding_model"]), provider=arguments.get("embedding_provider"), no_verify=False)
                code = _config_command(args, stdout=io.StringIO(), stderr=(err := io.StringIO()), environ=os.environ, verify=_probe_embedding)
                if code != 0:
                    return {"error": err.getvalue().strip()}
            changed.append("embedding")
    if not changed:
        return {"error": "Nothing to change: pass top, reranker, contextual_retrieval, embedding_model or embedding_provider."}
    try:
        importlib.import_module("rag.fetch").reset_pipelines()
    except Exception:
        pass
    return {"changed": changed, "settings": user_settings.describe_settings()}


def _read_payload() -> str | None:
    global _use_content_length_framing

    first = sys.stdin.buffer.readline()
    if not first:
        return None

    if not first.lower().startswith(b"content-length:"):
        return first.decode("utf-8", errors="replace")

    _use_content_length_framing = True
    content_length = None
    line = first
    while line not in (b"\r\n", b"\n"):
        text = line.decode("ascii", errors="ignore").strip()
        if ":" in text:
            key, value = text.split(":", 1)
            if key.strip().lower() == "content-length":
                try:
                    content_length = int(value.strip())
                except ValueError:
                    return None
        line = sys.stdin.buffer.readline()
        if not line:
            return None

    if content_length is None or content_length < 0:
        return None
    return sys.stdin.buffer.read(content_length).decode("utf-8", errors="replace")


def _write_message(message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False)
    if _use_content_length_framing:
        data = payload.encode("utf-8")
        sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


def _result(request_id: Any, result: dict[str, Any]) -> None:
    _write_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: Any, code: int, message: str) -> None:
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


def _normalize_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    return str(candidate.resolve())


def _configured_limit(name: str, default: int) -> int:
    """Read a positive RAG scan limit, falling back to its safe default."""
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        return default
    return value if value > 0 else default


def _matches_excluded_file(path: Path) -> bool:
    return any(
        fnmatch(str(path), pattern) or fnmatch(path.name, pattern)
        for pattern in _INDEX_EXCLUDED_FILE_PATTERNS
    )


def _validate_rag_scope(path: Path) -> _ScopeManifest | None:
    """Validate a scope and retain its safe file manifest for reuse."""
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if any(part in _UNSAFE_PATH_COMPONENTS for part in path.parts):
        raise ValueError(
            "RAG will not scan dependency or build folders such as node_modules. "
            "Choose the project source folder instead."
        )
    if path.is_file():
        return None

    max_depth = _configured_limit("RAG_MCP_MAX_DEPTH", _DEFAULT_MAX_DEPTH)
    max_tokens = _configured_limit("RAG_MCP_MAX_TOKENS", _DEFAULT_MAX_TOKENS)
    cache_key = (str(path), max_depth, max_tokens)
    cached = _validated_scope_cache.get(cache_key)
    if cached is not None and cached.is_current():
        return cached

    estimated_tokens = 0
    indexed_files: list[str] = []
    watched_paths: dict[str, tuple[int, int, bool]] = {}
    for current, dirs, files in os.walk(path):
        current_path = Path(current)
        try:
            current_stat = current_path.stat()
        except OSError:
            continue
        watched_paths[str(current_path)] = (current_stat.st_mtime_ns, 0, False)
        depth = len(current_path.relative_to(path).parts)
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in _UNSAFE_PATH_COMPONENTS and not directory.startswith(".")
            and not (current_path / directory).is_symlink()
        ]
        if depth >= max_depth and dirs:
            raise ValueError(
                f"RAG scan exceeds the configured {max_depth}-folder depth limit. "
                "Choose a narrower folder or raise RAG_MCP_MAX_DEPTH deliberately."
            )
        for name in files:
            candidate = current_path / name
            if candidate.is_symlink():
                continue
            if candidate.suffix.lower() not in _INDEXABLE_EXTENSIONS:
                continue
            try:
                candidate_stat = candidate.stat()
            except OSError:
                continue
            watched_paths[str(candidate)] = (
                candidate_stat.st_mtime_ns,
                candidate_stat.st_size,
                True,
            )
            estimated_tokens += candidate_stat.st_size // 4 + 1
            if estimated_tokens > max_tokens:
                raise ValueError(
                    f"RAG scan exceeds the configured {max_tokens:,}-token limit. "
                    "Choose a narrower folder or raise RAG_MCP_MAX_TOKENS deliberately."
                )
            if len(indexed_files) >= _MAX_INDEXED_FILES or depth > _INDEX_MAX_DEPTH:
                continue
            relative_parts = candidate.relative_to(path).parts
            if (
                name.startswith(".")
                or any(part in _INDEX_EXCLUDED_PATH_COMPONENTS for part in relative_parts[:-1])
                or _matches_excluded_file(candidate)
            ):
                continue
            indexed_files.append(str(candidate))

    manifest = _ScopeManifest(
        files=tuple(indexed_files),
        watched_paths=tuple(
            (path_string, mtime_ns, size, is_file)
            for path_string, (mtime_ns, size, is_file) in watched_paths.items()
        ),
    )
    _validated_scope_cache[cache_key] = manifest
    return manifest


def query_knowledge_base(query: str, doc_path: str = "") -> str:
    """Load and run RAG only after an explicit tool call."""
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        return json.dumps({"error": "query must be a non-empty string"})

    normalized_doc_path = (
        _normalize_path(doc_path) if doc_path.strip() else str(workspace_root)
    )

    try:
        scope = _validate_rag_scope(Path(normalized_doc_path))
        fetch_doc_path: str | list[str] = normalized_doc_path
        if scope is not None and scope.files:
            fetch_doc_path = list(scope.files)
        # rag.core/fetch/qdrant_backend log via plain print() to stdout. On a stdio
        # MCP transport, stdout carries only JSON-RPC frames, so redirect that noise
        # to stderr for the duration of the call instead of touching every print().
        with contextlib.redirect_stdout(sys.stderr):
            module = importlib.import_module("rag.fetch")
            fetch = getattr(module, "fetchExternalKnowledge", None)
            if not callable(fetch):
                return json.dumps({"error": "rag.fetch has no fetchExternalKnowledge function"})
            value = fetch(query=query, doc_path=fetch_doc_path)
        response: dict[str, Any] = {"result": value}
        if not has_embedding(embedding_config()):
            response["notice"] = BM25_ONLY_NOTICE
        return json.dumps(response)
    except Exception as exc:
        return json.dumps({"error": f"RAG query failed: {exc}"})


def _handle_request(message: dict[str, Any]) -> None:
    global _initialized

    method = message.get("method")
    request_id = message.get("id")

    if request_id is None:
        return
    if method == "initialize":
        params = message.get("params") or {}
        protocol_version = params.get("protocolVersion", "2025-06-18")
        _result(
            request_id,
            {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
            },
        )
        _initialized = True
        return
    if method == "ping":
        _result(request_id, {})
        return
    if not _initialized:
        _error(request_id, -32002, "Server is not initialized")
        return
    if method == "tools/list":
        _result(request_id, {"tools": _TOOLS})
        return
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        if name not in {tool["name"] for tool in _TOOLS}:
            _error(request_id, -32601, "Unknown tool")
            return
        arguments = params.get("arguments") or {}
        try:
            if name == _SETTINGS_GET_TOOL["name"]:
                output = json.dumps(_settings_report())
            elif name == _SETTINGS_SET_TOOL["name"]:
                output = json.dumps(_apply_settings(dict(arguments)))
            else:
                output = query_knowledge_base(
                    query=arguments.get("query", ""),
                    doc_path=arguments.get("doc_path", ""),
                )
        except (AttributeError, TypeError, ValueError) as exc:
            output = json.dumps({"error": f"Invalid arguments: {exc}"})
        _result(
            request_id,
            {
                "content": [{"type": "text", "text": output}],
                "isError": "error" in json.loads(output),
            },
        )
        return
    _error(request_id, -32601, f"Method not found: {method}")


def main() -> None:
    while True:
        payload = _read_payload()
        if payload is None:
            return
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict):
            _handle_request(message)


if __name__ == "__main__":
    main()
