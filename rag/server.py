#!/usr/bin/env python
# coding: utf-8
"""General-purpose stdio MCP server for explicitly scoped document answers."""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import lru_cache
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from filelock import Timeout


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

repo_root = Path(__file__).resolve().parent.parent
workspace_root = Path(
    os.environ.get("RAG_MCP_WORKSPACE_ROOT") or os.getcwd()
).expanduser().resolve()
sys.path.insert(0, str(repo_root))

from rag.settings import embedding_config, retrieval_config  # noqa: E402
from rag import settings as user_settings  # noqa: E402
from rag.answer_contract import decode_public_answer  # noqa: E402
from rag.formats import INDEXABLE_EXTENSIONS  # noqa: E402
from rag.index_lifecycle import selected_file_scope  # noqa: E402
from rag import tracing  # noqa: E402
from rag.workspaces import WorkspaceStore  # noqa: E402

_SERVER_NAME = "rag-mcp"
_SERVER_VERSION = "0.2.0"
_SERVER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_use_content_length_framing = False
_initialized = False
_INDEXABLE_EXTENSIONS = set(INDEXABLE_EXTENSIONS)
_UNSAFE_PATH_COMPONENTS = {"node_modules", ".git", ".venv", "venv", "dist", "build", "target", "__pycache__", "rag_db_v2"}
_INDEX_EXCLUDED_PATH_COMPONENTS = {".ipynb_checkpoints", "migrations", "rag_db_v2"}
_INDEX_EXCLUDED_FILE_PATTERNS = ("test_*.py", "*_test.py", "*__pycache__*", "*.pyc")
_DEFAULT_MAX_INDEXED_FILES = 5_000
_DEFAULT_INDEX_MAX_DEPTH = 12
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


_validated_scope_cache: dict[tuple[Any, ...], _ScopeManifest] = {}

_TOOL = {
    "name": "query_knowledge_base",
    "description": (
        "The main rag tool: search files or folders on disk by content and answer a question from them. "
        "Answer a question from one explicit supported document or deliberately scoped directory "
        "(doc_path; indexed on demand, no import step), or from a named local workspace shared with "
        "Studio and the rag CLI (workspace_id). "
        "Start with the smallest canonical file or subdirectory likely to contain the "
        "answer; do not scan a repository or workspace root by default. Increase search complexity "
        "only when evidence is insufficient: try exact terms, synonyms, and counterclaims before "
        "broadening the scope. For coverage or absence claims, use multiple differently worded "
        "questions. On a scan-limit error, narrow doc_path and continue. With conversation_id, "
        "continue and save a turn in the shared workspace conversation. Returns the answer and "
        "run/conversation IDs for that mode; chunks, paths, scores, and retrieval metadata remain internal."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to answer from the selected document scope",
            },
            "doc_path": {
                "type": "string",
                "description": (
                    "Required supported document or smallest deliberate directory scope. "
                    "Repository and workspace roots are not implicit."
                ),
            },
            "workspace_id": {
                "type": "string",
                "description": (
                    "A local workspace ID listed by `rag workspaces`. Use this instead of doc_path "
                    "to query the same documents as a paired Studio workspace."
                ),
            },
            "conversation_id": {
                "type": "string",
                "description": "Optional shared conversation ID; with workspace_id, continue its recorded context and save this answer there.",
            },
            "conversation_revision": {
                "type": ["string", "null"],
                "description": "Optional expected revision; rejects stale context before a provider call.",
            },
            "extensions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "uniqueItems": True,
                "description": (
                    "Optional file-type filter for a directory, such as ['.md'] or ['pdf', 'docx']. "
                    "Omit it to include every supported document type."
                ),
            },
        },
        "required": ["query"],
        "oneOf": [{"required": ["doc_path"]}, {"required": ["workspace_id"]}],
        "additionalProperties": False,
    },
    "annotations": {
        "title": "Answer from scoped documents",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
}

_SEARCH_WORKSPACE_TOOL = {
    'name': 'search_workspace',
    'description': ('Retrieve ranked passages from a named local workspace without generating an answer. '
                    'Covers only sources already imported into that workspace; to search files or folders '
                    'on disk, use query_knowledge_base with doc_path. '
                    'Returns run/revision IDs and passage locations when verified; unavailable offsets are null. Set include_content=true '
                    'to send private passage text and scores to the MCP host.'),
    'inputSchema': {'type': 'object', 'properties': {
        'workspace_id': {'type': 'string'}, 'query': {'type': 'string'},
        'source_ids': {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                       'items': {'type': 'string'}},
        'workspace_revision': {'type': 'string'},
        'include_content': {'type': 'boolean'}},
        'required': ['workspace_id', 'query'], 'additionalProperties': False},
    'annotations': {'title': 'Search shared workspace', 'readOnlyHint': False,
                    'destructiveHint': False, 'idempotentHint': False, 'openWorldHint': False},
}

_SETTINGS_GET_TOOL = {
    "name": "get_rag_settings",
    "description": "Show the RAG answer pipeline: embedding model, candidate and final passage counts, reranker, answer model, contextual retrieval, and detected runtimes or provider keys.",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    "annotations": {"title": "Show rag settings", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
}
_CAPABILITIES_GET_TOOL = {
    "name": "get_rag_capabilities",
    "description": "Show configured engine stages, providers, models, and credential readiness without making provider requests or exposing keys.",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    "annotations": {"title": "Show rag capabilities", "readOnlyHint": True, "destructiveHint": False,
                    "idempotentHint": True, "openWorldHint": False},
}
_MODELS_GET_TOOL = {
    "name": "get_rag_models",
    "description": "On demand, list configured and locally installed or cached model IDs. Does not query hosted catalogs, download or load models, or expose keys or cache paths.",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    "annotations": {"title": "Discover local rag models", "readOnlyHint": True, "destructiveHint": False,
                    "idempotentHint": True, "openWorldHint": False},
}
_SETTINGS_SET_TOOL = {
    "name": "set_rag_settings",
    "description": (
        "Change the RAG pipeline from plain words: passages supplied to synthesis (top), the reranker "
        "(off, auto, fastembed, openrouter or a model name), or the embedding model (guessing where it runs: Ollama, "
        "a small local ONNX model, or an API whose key is already in the environment). For a detected local "
        "OpenAI-compatible model, pass embedding_runtime from get_rag_models (LM Studio or vLLM). Never pass API keys here; "
        "the user enters keys with `rag setup` in a terminal. The next query uses the new settings."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "top": {"type": "integer", "minimum": 1, "description": "How many reranked passages answer synthesis may use"},
            "candidates": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Candidate passages before reranking"},
            "answer_model": {"type": "string", "description": "OpenRouter answer model ID"},
            "reranker": {"type": "string", "description": "off | auto | fastembed | openrouter | a reranker model name"},
            "contextual_retrieval": {"type": "string", "enum": ["on", "off"], "description": "Enable or disable Anthropic-style ingestion-time contextual retrieval"},
            "embedding_model": {"type": "string", "description": "Embedding model name, e.g. qwen/qwen3-embedding-4b or BAAI/bge-small-en-v1.5"},
            "embedding_provider": {"type": "string", "enum": ["fastembed", "ollama", "openai_compatible", "sentencetransformer", "none"], "description": "Where the embedding model runs; guessed when omitted"},
            "embedding_runtime": {"type": "string", "description": "Detected local OpenAI-compatible runtime label (LM Studio or vLLM) when choosing its embedding model"},
            "revision": {"type": ["string", "null"], "description": "Optional revision from get_rag_settings; reject a stale edit before saving."},
        },
        "additionalProperties": False,
    },
    "annotations": {"title": "Change rag settings", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
}
_EVALUATIONS_GET_TOOL = {
    "name": "get_evaluation_questions",
    "description": "Read the versioned evaluation question set in a shared RAG workspace.",
    "inputSchema": {"type": "object", "properties": {"workspace_id": {"type": "string"}},
                    "required": ["workspace_id"], "additionalProperties": False},
    "annotations": {"title": "Read evaluation questions", "readOnlyHint": True, "destructiveHint": False,
                    "idempotentHint": True, "openWorldHint": False},
}
_EVALUATIONS_SET_TOOL = {
    "name": "set_evaluation_questions",
    "description": "Replace a shared workspace evaluation question set using its current revision. Rejects stale edits.",
    "inputSchema": {"type": "object", "properties": {
        "workspace_id": {"type": "string"}, "revision": {"type": ["string", "null"]},
        "questions": {"type": "array", "maxItems": 200, "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "question": {"type": "string"}},
            "required": ["id", "question"], "additionalProperties": False}}},
        "required": ["workspace_id", "revision", "questions"], "additionalProperties": False},
    "annotations": {"title": "Update evaluation questions", "readOnlyHint": False, "destructiveHint": True,
                    "idempotentHint": True, "openWorldHint": False},
}
_EVALUATION_RUN_TOOL = {
    'name': 'run_evaluation',
    'description': 'Run a shared workspace question set sequentially through engine retrieval and save a durable batch. Uses configured embedding/reranking providers; no answer synthesis. Passage counts are not correctness judgments.',
    'inputSchema': {'type': 'object', 'properties': {
        'workspace_id': {'type': 'string'},
        'source_ids': {'type': 'array', 'minItems': 1, 'maxItems': 500, 'items': {'type': 'string'}},
        'question_revision': {'type': 'string'}, 'workspace_revision': {'type': 'string'}},
        'required': ['workspace_id'], 'additionalProperties': False},
    'annotations': {'title': 'Run evaluation questions', 'readOnlyHint': False, 'destructiveHint': False,
                    'idempotentHint': False, 'openWorldHint': True},
}
_EVALUATION_RUNS_GET_TOOL = {
    'name': 'get_evaluation_runs',
    'description': 'List saved evaluation batches, or read one batch by ID. Includes question snapshots and per-question trace IDs, without private passages.',
    'inputSchema': {'type': 'object', 'properties': {
        'workspace_id': {'type': 'string'}, 'batch_id': {'type': 'string'},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100},
        'offset': {'type': 'integer', 'minimum': 0, 'maximum': 10000}},
        'required': ['workspace_id'], 'additionalProperties': False},
    'annotations': {'title': 'Read evaluation batches', 'readOnlyHint': True, 'destructiveHint': False,
                    'idempotentHint': True, 'openWorldHint': False},
}
_EVALUATION_LABELS_GET_TOOL = {
    'name': 'get_evaluation_labels',
    'description': 'Read human relevance judgments for one saved evaluation batch. Passage text is not included.',
    'inputSchema': {'type': 'object', 'properties': {
        'workspace_id': {'type': 'string'}, 'batch_id': {'type': 'string'}},
        'required': ['workspace_id', 'batch_id'], 'additionalProperties': False},
    'annotations': {'title': 'Read evaluation labels', 'readOnlyHint': True, 'destructiveHint': False,
                    'idempotentHint': True, 'openWorldHint': False},
}
_EVALUATION_LABELS_SET_TOOL = {
    'name': 'set_evaluation_label',
    'description': 'Mark one retained passage relevant or irrelevant, or clear its judgment. Requires the latest label revision.',
    'inputSchema': {'type': 'object', 'properties': {
        'workspace_id': {'type': 'string'}, 'batch_id': {'type': 'string'},
        'revision': {'type': ['string', 'null']}, 'run_id': {'type': 'string'},
        'rank': {'type': 'integer', 'minimum': 1},
        'label': {'type': ['string', 'null'], 'enum': ['relevant', 'irrelevant', None]}},
        'required': ['workspace_id', 'batch_id', 'revision', 'run_id', 'rank', 'label'],
        'additionalProperties': False},
    'annotations': {'title': 'Set evaluation label', 'readOnlyHint': False, 'destructiveHint': False,
                    'idempotentHint': True, 'openWorldHint': False},
}
_CONVERSATIONS_GET_TOOL = {
    'name': 'get_conversations',
    'description': 'List shared workspace conversations, or read one private conversation including its saved answers and evidence.',
    'inputSchema': {'type': 'object', 'properties': {
        'workspace_id': {'type': 'string'}, 'conversation_id': {'type': 'string'},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100},
        'offset': {'type': 'integer', 'minimum': 0, 'maximum': 10000}},
        'required': ['workspace_id'], 'additionalProperties': False},
    'annotations': {'title': 'Read workspace conversations', 'readOnlyHint': True, 'destructiveHint': False,
                    'idempotentHint': True, 'openWorldHint': False},
}
_CONVERSATION_APPEND_TOOL = {
    'name': 'append_conversation_turn',
    'description': 'Append a saved successful answer run to a shared workspace conversation using its current revision.',
    'inputSchema': {'type': 'object', 'properties': {
        'workspace_id': {'type': 'string'}, 'conversation_id': {'type': 'string'},
        'revision': {'type': ['string', 'null']}, 'run_id': {'type': 'string'},
        'question': {'type': 'string'}},
        'required': ['workspace_id', 'conversation_id', 'revision', 'run_id', 'question'],
        'additionalProperties': False},
    'annotations': {'title': 'Append conversation turn', 'readOnlyHint': False, 'destructiveHint': False,
                    'idempotentHint': True, 'openWorldHint': False},
}
_WORKSPACE_CHUNKS_GET_TOOL = {
    'name': 'get_workspace_chunks',
    'description': 'Preview next-build boundaries or inspect published index metadata for one saved workspace source. Published metadata may be stale and is not a vector-table verification. Private chunk text and headings are omitted unless include_content is true; full source text is never returned.',
    'inputSchema': {'type': 'object', 'properties': {
        'workspace_id': {'type': 'string'}, 'source_id': {'type': 'string'},
        'revision': {'type': 'string'},
        'view': {'type': 'string', 'enum': ['preview', 'indexed']},
        'scope_source_ids': {'type': 'array', 'minItems': 1, 'maxItems': 500, 'uniqueItems': True,
                             'items': {'type': 'string'}, 'description': 'Exact source selection used by the published index.'},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100},
        'offset': {'type': 'integer', 'minimum': 0},
        'include_content': {'type': 'boolean', 'description': 'Explicitly return private text for this page of chunks.'}},
        'required': ['workspace_id', 'source_id'], 'additionalProperties': False},
    'annotations': {'title': 'Preview workspace chunks', 'readOnlyHint': True, 'destructiveHint': False,
                    'idempotentHint': True, 'openWorldHint': False},
}
_WORKSPACES_GET_TOOL = {
    'name': 'get_workspaces',
    'description': 'List shared local workspace IDs, or provide workspace_id to read its revision and source IDs. Returns metadata only, without source text, stored paths, or hashes. It does not search content; to find something in files, use query_knowledge_base.',
    'inputSchema': {'type': 'object', 'properties': {'workspace_id': {'type': 'string'}},
                    'additionalProperties': False},
    'annotations': {'title': 'Read shared workspaces', 'readOnlyHint': True, 'destructiveHint': False,
                    'idempotentHint': True, 'openWorldHint': False},
}
_WORKSPACE_SOURCES_ADD_TOOL = {
    'name': 'add_workspace_sources',
    'description': 'Import one explicit local file or bounded folder into a shared workspace. Requires its current revision (null to create); stale edits are rejected. Extracts text locally without indexing or model calls. Returns metadata and source IDs, never document text.',
    'inputSchema': {'type': 'object', 'properties': {
        'workspace_id': {'type': 'string'}, 'path': {'type': 'string'},
        'revision': {'type': ['string', 'null']},
        'extensions': {'type': 'array', 'minItems': 1, 'maxItems': 16,
                       'items': {'type': 'string'}}},
        'required': ['workspace_id', 'path', 'revision'], 'additionalProperties': False},
    'annotations': {'title': 'Import workspace sources', 'readOnlyHint': False, 'destructiveHint': True,
                    'idempotentHint': True, 'openWorldHint': False},
}
_TOOLS = [_TOOL, _SEARCH_WORKSPACE_TOOL, _SETTINGS_GET_TOOL, _CAPABILITIES_GET_TOOL, _MODELS_GET_TOOL, _SETTINGS_SET_TOOL, _EVALUATIONS_GET_TOOL, _EVALUATIONS_SET_TOOL,
          _EVALUATION_RUN_TOOL, _EVALUATION_RUNS_GET_TOOL, _EVALUATION_LABELS_GET_TOOL, _EVALUATION_LABELS_SET_TOOL,
          _CONVERSATIONS_GET_TOOL, _CONVERSATION_APPEND_TOOL, _WORKSPACES_GET_TOOL, _WORKSPACE_SOURCES_ADD_TOOL,
          _WORKSPACE_CHUNKS_GET_TOOL]


def _settings_report() -> dict[str, Any]:
    from rag import detect

    embedding = embedding_config()
    return {
        "revision": user_settings.config_revision(),
        "runtime": {"pid": os.getpid(), "server_source_sha256_at_startup": _SERVER_SOURCE_SHA256},
        "settings": user_settings.describe_settings(),
        "embedding": {"provider": embedding["embed_provider"], "model": embedding["embed_model"]},
        "retrieval": user_settings.retrieval_config(),
        "detected": detect.detect_all(os.environ, embedding["ollama_url"]),
        "diagnostics": tracing.health(),
    }


def _apply_settings(arguments: dict[str, Any], *, expected_revision=user_settings.REVISION_UNSET) -> dict[str, Any]:
    from rag.cli import _config_command, _probe_embedding
    import argparse
    import io
    import tempfile

    changed = []
    original = user_settings.load_settings()
    with tempfile.TemporaryDirectory(prefix="rag-settings-") as temporary, contextlib.redirect_stdout(sys.stderr):
        staged_environ = dict(os.environ, RAG_MCP_CONFIG=str(Path(temporary) / "config.json"))
        user_settings.save_settings(original, staged_environ)
        for name, setting in (("top", "top"), ("candidates", "candidates"), ("answer_model", "answer"), ("reranker", "reranker"), ("contextual_retrieval", "contextual")):
            if name in arguments:
                args = argparse.Namespace(setting=setting, value=str(arguments[name]), provider=None, no_verify=True)
                code = _config_command(args, stdout=io.StringIO(), stderr=(err := io.StringIO()), environ=staged_environ, verify=None)
                if code != 0:
                    return {"error": err.getvalue().strip()}
                changed.append(name)
        if "embedding_model" in arguments or arguments.get("embedding_provider") == "none":
            if arguments.get("embedding_provider") == "none":
                if arguments.get('embedding_runtime') is not None:
                    return {"error": "A disabled embedding has no runtime."}
                user_settings.update_settings({"embed_provider": "none", "embed_model": ""}, staged_environ)
            else:
                args = argparse.Namespace(setting="embedding", value=str(arguments["embedding_model"]),
                                          provider=arguments.get("embedding_provider"),
                                          runtime=arguments.get('embedding_runtime'), no_verify=False)
                code = _config_command(args, stdout=io.StringIO(), stderr=(err := io.StringIO()), environ=staged_environ, verify=_probe_embedding)
                if code != 0:
                    return {"error": err.getvalue().strip()}
            changed.append("embedding")
        if changed:
            staged = user_settings.load_settings(staged_environ)
            updates = {key: value for key, value in staged.items() if original.get(key) != value}
            if updates:
                user_settings.update_settings(updates, expected_data=original,
                                              expected_revision=expected_revision)
    if not changed:
        return {"error": "Nothing to change: pass top, reranker, contextual_retrieval, embedding_model or embedding_provider."}
    try:
        if 'rag.fetch' in sys.modules:
            sys.modules['rag.fetch'].reset_pipelines()
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


@contextlib.contextmanager
def _quiet_private_output():
    """Keep engine and native-library diagnostics off MCP host streams."""
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        try:
            os.dup2(quiet.fileno(), 1)
            os.dup2(quiet.fileno(), 2)
            yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


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


@lru_cache(maxsize=4096)
def _git_ignored_set(directory: str) -> frozenset[str]:
    """Names git ignores in one directory, empty outside a work tree."""
    try:
        listing = os.listdir(directory)
    except OSError:
        return frozenset()
    if not listing:
        return frozenset()
    try:
        completed = subprocess.run(
            ["git", "-C", directory, "check-ignore", "--stdin", "-z"],
            input="\0".join(listing),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    # 0 = something ignored, 1 = nothing ignored, 128 = not a git work tree.
    if completed.returncode not in (0, 1):
        return frozenset()
    return frozenset(name for name in completed.stdout.split("\0") if name)


def _is_git_ignored(candidate: Path) -> bool:
    """Whether git ignores this file, so ignored data cannot consume the budget."""
    return candidate.name in _git_ignored_set(str(candidate.parent))


def _allowed_extensions(extensions: list[str] | None) -> frozenset[str]:
    if extensions is None:
        return frozenset(_INDEXABLE_EXTENSIONS)
    if not isinstance(extensions, list) or not extensions:
        raise ValueError("extensions must be a non-empty list when supplied")
    allowed = set()
    for value in extensions:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("each extension must be a non-empty string")
        extension = value.strip().lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension not in _INDEXABLE_EXTENSIONS:
            raise ValueError(f"Unsupported document extension: {extension}")
        allowed.add(extension)
    return frozenset(allowed)


@tracing.traced('scope.discovery')
def _validate_rag_scope(path: Path, extensions: list[str] | None = None) -> _ScopeManifest | None:
    """Validate a scope and retain its safe file manifest for reuse."""
    allowed_extensions = _allowed_extensions(extensions)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if any(part in _UNSAFE_PATH_COMPONENTS for part in path.parts):
        raise ValueError(
            "RAG will not scan dependency or build folders such as node_modules. "
            "Choose the project source folder instead."
        )
    if path.is_file():
        extension = path.suffix.lower()
        if extension not in _INDEXABLE_EXTENSIONS:
            raise ValueError(f"Unsupported document extension: {extension or '(none)'}")
        if extension not in allowed_extensions:
            raise ValueError(f"The selected file does not match the requested extensions: {extension}")
        return None

    max_depth = _configured_limit("RAG_MCP_MAX_DEPTH", _DEFAULT_MAX_DEPTH)
    max_tokens = _configured_limit("RAG_MCP_MAX_TOKENS", _DEFAULT_MAX_TOKENS)
    max_files = _configured_limit("RAG_MCP_MAX_FILES", _DEFAULT_MAX_INDEXED_FILES)
    index_max_depth = _configured_limit("RAG_MCP_INDEX_MAX_DEPTH", _DEFAULT_INDEX_MAX_DEPTH)
    cache_key = (str(path), tuple(sorted(allowed_extensions)), max_depth, max_tokens, max_files, index_max_depth)
    cached = _validated_scope_cache.get(cache_key)
    if cached is not None and cached.is_current():
        return cached

    estimated_tokens = 0
    indexed_files: list[str] = []
    watched_paths: dict[str, tuple[int, int, bool]] = {}
    def fail_scan(error):
        raise error

    for current, dirs, files in os.walk(path, onerror=fail_scan):
        current_path = Path(current)
        current_stat = current_path.stat()
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
            if candidate.suffix.lower() not in allowed_extensions:
                continue
            candidate_stat = candidate.stat()
            watched_paths[str(candidate)] = (
                candidate_stat.st_mtime_ns,
                candidate_stat.st_size,
                True,
            )
            # Every filter below decides a file will never be indexed, so applying
            # them before the budget keeps ignored data out of the token count.
            # Counting first made a repository with a large ignored data directory
            # unscannable even though its tracked sources fit comfortably.
            if depth > index_max_depth:
                continue
            relative_parts = candidate.relative_to(path).parts
            if (
                name.startswith(".")
                or any(part in _INDEX_EXCLUDED_PATH_COMPONENTS for part in relative_parts[:-1])
                or _matches_excluded_file(candidate)
            ):
                continue
            if _is_git_ignored(candidate):
                continue
            estimated_tokens += candidate_stat.st_size // 4 + 1
            if estimated_tokens > max_tokens:
                raise ValueError(
                    f"RAG scan exceeds the configured {max_tokens:,}-token limit. "
                    "Retry with doc_path set to the canonical file or a smaller folder. "
                    "Raise RAG_MCP_MAX_TOKENS only when the broader corpus is deliberate."
                )
            if len(indexed_files) >= max_files:
                raise ValueError(
                    f"Folder exceeds the {max_files}-document limit; select a smaller folder."
                )
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


def _query_knowledge_base(
    query: str,
    doc_path: str = "",
    extensions: list[str] | None = None,
    workspace_id: str = "",
    *, selected_files: list[str] | None = None, cite_sources: bool = False, retrieval_only: bool = False, answer_model: str | None = None,
) -> str:
    """Load and run RAG only after an explicit tool call."""
    tracing.annotate(server_source_sha256_at_startup=_SERVER_SOURCE_SHA256)
    tracing.content('query', query)
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        return json.dumps({"error": "query must be a non-empty string"})

    has_path = isinstance(doc_path, str) and bool(doc_path.strip())
    has_workspace = isinstance(workspace_id, str) and bool(workspace_id.strip())
    if has_path == has_workspace:
        return json.dumps({"error": "provide exactly one of doc_path or workspace_id; doc_path must name an explicit supported file or scoped folder"})
    if has_workspace:
        try:
            normalized_doc_path = str(WorkspaceStore().resolve(workspace_id.strip()))
            tracing.annotate(workspace_id=workspace_id.strip())
        except (ValueError, OSError):
            return json.dumps({"error": "workspace_id does not name an available local workspace"})
    else:
        normalized_doc_path = _normalize_path(doc_path)

    try:
        scope = _validate_rag_scope(Path(normalized_doc_path), extensions)
    except ValueError as error:
        # Scope validation owns these actionable messages; provider errors remain sanitized.
        return json.dumps({'error': str(error)})
    try:
        fetch_doc_path: str | list[str] = normalized_doc_path
        fetch_scope: str | None = None
        if scope is not None:
            if not scope.files:
                requested = ", ".join(sorted(_allowed_extensions(extensions)))
                raise ValueError(f"No matching documents ({requested}) found in {normalized_doc_path}")
            fetch_doc_path = list(scope.files)
            # Name the folder so its index survives files being added or removed.
            fetch_scope = normalized_doc_path
        if selected_files is not None:
            fetch_doc_path = selected_files
            # An index scope must not reuse a full-corpus manifest for a subset.
            fetch_scope = selected_file_scope(normalized_doc_path, selected_files)
        # rag.core/fetch/qdrant_backend log via plain print() to stdout. On a stdio
        # MCP transport, stdout carries only JSON-RPC frames, so redirect that noise
        # to stderr for the duration of the call instead of touching every print().
        with contextlib.redirect_stdout(sys.stderr):
            module = importlib.import_module("rag.fetch")
            fetch = getattr(module, "fetchExternalKnowledge", None)
            if not callable(fetch):
                return json.dumps({"error": "rag.fetch has no fetchExternalKnowledge function"})
            try:
                from rag.query_decomposition import balanced_passages, decompose_query

                facets = decompose_query(query)
                tracing.annotate(facet_count=len(facets))
                if len(facets) == 1:
                    value = fetch(query=query, doc_path=fetch_doc_path, scope=fetch_scope)
                else:
                    groups = []
                    for facet in facets:
                        facet_value = fetch(
                            query=facet,
                            doc_path=fetch_doc_path,
                            scope=fetch_scope,
                        )
                        if isinstance(facet_value, list):
                            groups.append(facet_value)
                    with tracing.span('selection.balance'):
                        value = balanced_passages(groups, limit=5)
                        tracing.passages('selected', value)
            except Exception as exc:
                return json.dumps({"error": (
                    f"RAG answer pipeline failed ({type(exc).__name__}). "
                    "Check the configured OpenRouter models and index availability."
                )})
        tracing.checkpoint()
        if isinstance(value, str):
            answer = value
        else:
            from rag.openrouter_pipeline import synthesize_answer

            tracing.passages('synthesis_evidence', value)
            if retrieval_only:
                return json.dumps({'result': 'Retrieval completed.'})
            try:
                answer = synthesize_answer(
                    query,
                    value,
                    {**embedding_config(), **retrieval_config(), 'cite_sources': cite_sources,
                     **({'answer_model': answer_model} if answer_model is not None else {})},
                )
            except Exception as exc:
                return json.dumps({'error': (
                    f'RAG answer synthesis failed ({type(exc).__name__}). '
                    'Check the configured OpenRouter model and availability.'
                )})
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Answer synthesis returned invalid output.")
        return json.dumps({"result": answer.strip()})
    except Exception as exc:
        return json.dumps({"error": f"RAG query failed ({type(exc).__name__}). Check the document scope and engine settings."})


def query_knowledge_base(query: str, doc_path: str = "", extensions: list[str] | None = None, workspace_id: str = "") -> str:
    """Public MCP/CLI adapter: execution details stay behind the paired gateway."""
    from rag.execution import execute_run, public_response
    result = execute_run(query, doc_path, extensions, workspace_id, client="mcp")
    return public_response(result)


def _handle_request(message: dict[str, Any]) -> None:
    global _initialized

    method = message.get("method")
    request_id = message.get("id")

    if request_id is None:
        return
    if message.get("params") is not None and not isinstance(message["params"], dict):
        _error(request_id, -32602, "Invalid params: expected an object")
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
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            _error(request_id, -32602, "Invalid params: expected a tool name string and arguments object")
            return
        if name not in {tool["name"] for tool in _TOOLS}:
            _error(request_id, -32601, "Unknown tool")
            return
        try:
            if name == _SETTINGS_GET_TOOL["name"]:
                output = json.dumps(_settings_report())
            elif name == _SEARCH_WORKSPACE_TOOL['name']:
                if (set(arguments) - {'workspace_id', 'query', 'source_ids', 'workspace_revision', 'include_content'}
                        or not isinstance(arguments.get('workspace_id'), str)
                        or not isinstance(arguments.get('query'), str)
                        or not arguments['query'].strip()
                        or ('source_ids' in arguments and
                            (not isinstance(arguments['source_ids'], list) or not arguments['source_ids']
                             or any(not isinstance(source, str) for source in arguments['source_ids'])))
                        or ('workspace_revision' in arguments and not isinstance(arguments['workspace_revision'], str))
                        or type(arguments.get('include_content', False)) is not bool):
                    raise ValueError('workspace_id and nonempty query are required; optional search fields must match the schema.')
                WorkspaceStore.validate_id(arguments['workspace_id'])
                with _quiet_private_output():
                    from rag.execution import execute_run, retrieval_response
                    value = execute_run(arguments['query'], workspace_id=arguments['workspace_id'],
                                        source_ids=arguments.get('source_ids'),
                                        workspace_revision=arguments.get('workspace_revision'),
                                        mode='retrieve', client='mcp')
                output = json.dumps(retrieval_response(value, include_content=arguments.get('include_content', False)),
                                    ensure_ascii=False)
            elif name == _CAPABILITIES_GET_TOOL["name"]:
                from rag.control import capabilities
                output = json.dumps(capabilities(), ensure_ascii=False)
            elif name == _MODELS_GET_TOOL["name"]:
                from rag.control import model_inventory
                output = json.dumps(model_inventory(), ensure_ascii=False)
            elif name == _WORKSPACES_GET_TOOL['name']:
                store = WorkspaceStore()
                try:
                    value = store.describe(arguments['workspace_id']) if 'workspace_id' in arguments else {
                        'schema_version': 1, 'workspaces': store.list_workspaces()}
                    output = json.dumps(value, ensure_ascii=False)
                except FileNotFoundError:
                    output = json.dumps({'error': 'Workspace not found.'})
            elif name == _WORKSPACE_SOURCES_ADD_TOOL['name']:
                if (not {'workspace_id', 'path', 'revision'} <= arguments.keys()
                        or set(arguments) - {'workspace_id', 'path', 'revision', 'extensions'}
                        or not isinstance(arguments['path'], str) or not arguments['path'].strip()):
                    raise ValueError('workspace_id, path, and revision are required; path must be nonempty.')
                with _quiet_private_output():
                    from rag.workspace_import import import_workspace_sources
                    value = import_workspace_sources(
                        WorkspaceStore(), arguments['workspace_id'], arguments['path'], arguments.get('extensions'),
                        expected_revision=arguments['revision'])
                output = json.dumps(value, ensure_ascii=False)
            elif name == _WORKSPACE_CHUNKS_GET_TOOL['name']:
                try:
                    store = WorkspaceStore()
                    view = arguments.get('view', 'preview')
                    if view not in {'preview', 'indexed'}:
                        raise ValueError('Invalid chunk view.')
                    if view != 'indexed' and 'scope_source_ids' in arguments:
                        raise ValueError('Selected source scope requires indexed view.')
                    if view == 'indexed':
                        from rag.index_inspection import inspect_published_chunks
                        revision = arguments.get('revision') or store.get(arguments.get('workspace_id'))['revision']
                        value = inspect_published_chunks(
                            store, arguments.get('workspace_id'), arguments.get('source_id'), revision,
                            limit=arguments.get('limit', 50), offset=arguments.get('offset', 0),
                            include_content=arguments.get('include_content', False),
                            scope_source_ids=arguments.get('scope_source_ids'))
                    else:
                        from rag.chunk_preview import inspect_workspace_chunks
                        value = inspect_workspace_chunks(
                            store, arguments.get('workspace_id'), arguments.get('source_id'),
                            arguments.get('revision'), limit=arguments.get('limit', 50),
                            offset=arguments.get('offset', 0), include_content=arguments.get('include_content', False))
                    output = json.dumps(value, ensure_ascii=False)
                except FileNotFoundError:
                    output = json.dumps({'error': 'Workspace or source not found.'})
            elif name == _SETTINGS_SET_TOOL["name"]:
                changes = dict(arguments)
                revision = changes.pop('revision', user_settings.REVISION_UNSET)
                output = json.dumps(_apply_settings(changes, expected_revision=revision))
            elif name in {_EVALUATION_RUN_TOOL['name'], _EVALUATION_RUNS_GET_TOOL['name']}:
                from rag.evaluation_runs import EvaluationRunStore, execute_evaluation
                try:
                    if name == _EVALUATION_RUN_TOOL['name']:
                        with _quiet_private_output():
                            value = execute_evaluation(arguments.get('workspace_id'), source_ids=arguments.get('source_ids'),
                                question_revision=arguments.get('question_revision'), workspace_revision=arguments.get('workspace_revision'), client='mcp')
                    else:
                        batches = EvaluationRunStore()
                        value = batches.get(arguments.get('workspace_id'), arguments['batch_id']) if arguments.get('batch_id') else batches.list(
                            arguments.get('workspace_id'), limit=arguments.get('limit', 20), offset=arguments.get('offset', 0))
                    output = json.dumps(value, ensure_ascii=False)
                except Timeout:
                    output = json.dumps({'error': 'Another evaluation is running. Read its saved progress and retry after it finishes.'})
            elif name in {_CONVERSATIONS_GET_TOOL['name'], _CONVERSATION_APPEND_TOOL['name']}:
                from rag.conversations import ConversationStore
                store = ConversationStore()
                value = (store.get(arguments.get('workspace_id'), arguments['conversation_id'])
                         if name == _CONVERSATIONS_GET_TOOL['name'] and arguments.get('conversation_id') else
                         store.list(arguments.get('workspace_id'), limit=arguments.get('limit', 20), offset=arguments.get('offset', 0))
                         if name == _CONVERSATIONS_GET_TOOL['name'] else
                         store.append(arguments.get('workspace_id'), arguments.get('conversation_id'),
                                      arguments.get('run_id'), arguments.get('question'),
                                      expected_revision=arguments.get('revision')))
                output = json.dumps(value, ensure_ascii=False)
            elif name in {_EVALUATION_LABELS_GET_TOOL['name'], _EVALUATION_LABELS_SET_TOOL['name']}:
                from rag.evaluation_labels import EvaluationLabelStore
                store = EvaluationLabelStore()
                try:
                    value = (store.get(arguments.get('workspace_id'), arguments.get('batch_id'))
                             if name == _EVALUATION_LABELS_GET_TOOL['name'] else
                             store.set(arguments.get('workspace_id'), arguments.get('batch_id'),
                                       arguments.get('run_id'), arguments.get('rank'), arguments.get('label'),
                                       expected_revision=arguments.get('revision')))
                    output = json.dumps(value, ensure_ascii=False)
                except Timeout:
                    output = json.dumps({'error': 'Evaluation labels are busy. Retry with the latest revision.'})
            elif name in {_EVALUATIONS_GET_TOOL['name'], _EVALUATIONS_SET_TOOL['name']}:
                from rag.evaluation_sets import EvaluationSetStore
                store = EvaluationSetStore()
                value = (store.get(arguments.get('workspace_id'))
                         if name == _EVALUATIONS_GET_TOOL['name'] else
                         store.replace(arguments.get('workspace_id'), arguments.get('questions'),
                                       expected_revision=arguments.get('revision')))
                output = json.dumps(value, ensure_ascii=False)
            else:
                if arguments.get('conversation_id') is not None:
                    if not arguments.get('workspace_id') or arguments.get('doc_path') or arguments.get('extensions'):
                        raise ValueError('A conversation query requires workspace_id and cannot use doc_path or extensions.')
                    from rag.conversations import run_conversation_query
                    with _quiet_private_output():
                        options = ({'expected_revision': arguments['conversation_revision']}
                                   if 'conversation_revision' in arguments else {})
                        value = run_conversation_query(arguments.get('query'), arguments['workspace_id'],
                                                       arguments['conversation_id'], client='mcp', **options)
                    output = json.dumps(value, ensure_ascii=False)
                else:
                    with _quiet_private_output():
                        output = query_knowledge_base(
                            query=arguments.get("query", ""),
                            doc_path=arguments.get("doc_path", ""),
                            extensions=arguments.get("extensions"),
                            workspace_id=arguments.get("workspace_id", ""),
                        )
        except (AttributeError, TypeError, ValueError) as exc:
            output = json.dumps({"error": f"Invalid arguments: {exc}"})
        except FileNotFoundError:
            output = json.dumps({"error": "Source or workspace not found." if name == _WORKSPACE_SOURCES_ADD_TOOL['name']
                                 else "Conversation or saved run not found."})
        except Timeout:
            output = json.dumps({"error": "Workspace storage is busy. Retry with the latest revision."})
        except OSError:
            output = json.dumps({"error": "Workspace import unavailable." if name == _WORKSPACE_SOURCES_ADD_TOOL['name']
                                 else "Evaluation workspace unavailable."})
        except Exception as exc:
            # Isolate failed requests without echoing provider payloads or private paths.
            print(f'[RAG] Tool {name} failed: {type(exc).__name__}', file=sys.stderr)
            output = json.dumps({"error": f"RAG tool failed internally ({type(exc).__name__}). If RAG was updated "
                                 "while this server was running, restart the RAG MCP server and retry."})
        if name == _TOOL["name"] and arguments.get('conversation_id') is not None:
            parsed_output = json.loads(output)
            public_output = output
            is_error = bool(parsed_output.get('error') or parsed_output.get('conversation_error'))
        elif name == _TOOL["name"]:
            public_output, is_error = decode_public_answer(output)
        else:
            parsed_output = json.loads(output)
            public_output = output
            is_error = bool(parsed_output.get("error"))
        _result(
            request_id,
            {
                "content": [{"type": "text", "text": public_output}],
                "isError": is_error,
            },
        )
        return
    _error(request_id, -32601, f"Method not found: {method}")


def main() -> None:
    try:
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
    finally:
        # Do not load retrieval dependencies for a server that never used them.
        retrieval = sys.modules.get('rag.fetch')
        try:
            if retrieval is not None:
                retrieval.reset_pipelines()
        except Exception as error:
            print(f'[RAG] Shutdown cleanup failed: {type(error).__name__}', file=sys.stderr)


if __name__ == "__main__":
    main()
