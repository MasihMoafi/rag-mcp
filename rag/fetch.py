"""Scoped pipeline configuration, cache reuse and document-query orchestration."""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
from threading import Lock, RLock
from weakref import WeakValueDictionary
from .core import RAGPipelineV2
from .discovery import resolve_path, find_all_indexable_files, find_relevant_files
from .formats import INDEXABLE_EXTENSIONS as _INDEXABLE_EXTENSIONS
from .settings import contextual_retrieval_enabled, embedding_config, retrieval_config
from .index_lifecycle import active_directory, open_or_build, source_snapshot

def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Pipeline defaults; provider and retrieval preferences are merged at creation.
RAG_CONFIG_V2 = {
    "version": "v2",
    "persist_dir": "./rag_db_v2",
    "document_paths": ["./files/"],
    # embed_* keys come from rag.settings.embedding_config() at pipeline creation:
    # the user's `rag setup` choice, overridable by RAG_MCP_EMBED_* variables.
    "top_k": _env_int("RAG_MCP_TOP_K", 5),
    "chunk_size": _env_int("RAG_MCP_CHUNK_SIZE", 700),
    "chunk_overlap": _env_int("RAG_MCP_CHUNK_OVERLAP", 100),
    "rrf_k": _env_int("RAG_MCP_RRF_K", 60),  # Reciprocal-rank fusion constant
    # rerank_top_k / reranker_type / reranker_model come from rag.settings.retrieval_config()
    # at pipeline creation (`rag config`, overridable by RAG_MCP_RERANK_TOP_K etc.).
    "force_reindex": False,
    "bm25_k1": 1.5,  # BM25 parameters
    "bm25_b": 0.75,
    "distance_metric": "cosine",  # Vector distance metric
    "backend": _env_str("RAG_MCP_BACKEND", "lancedb"),  # lancedb | qdrant
}

_rag_system_v2_instances: Dict[str, RAGPipelineV2] = {}
_scope_locks = WeakValueDictionary()
_scope_locks_guard = Lock()


def _scope_lock(key):
    with _scope_locks_guard:
        lock = _scope_locks.get(key)
        if lock is None:
            lock = RLock()
            _scope_locks[key] = lock
        return lock


def _close_pipeline(pipeline):
    close = getattr(pipeline, 'close', None)
    if close is not None:
        close()


def _retire_cached_pipeline(key):
    pipeline = _rag_system_v2_instances.get(key)
    if pipeline is not None:
        _close_pipeline(pipeline)
        if _rag_system_v2_instances.get(key) is pipeline:
            del _rag_system_v2_instances[key]


def reset_pipelines() -> None:
    """Drain cached pipelines and release storage before forgetting them."""
    with _scope_locks_guard:
        keys = set(_rag_system_v2_instances) | set(_scope_locks)
    for key in keys:
        with _scope_lock(key):
            _retire_cached_pipeline(key)

def index_root() -> str:
    """Where indexes live: RAG_MCP_DATA_DIR, else the XDG data dir. Never inside the package."""
    root = os.environ.get("RAG_MCP_DATA_DIR", "").strip() or os.path.join(
        os.environ.get("XDG_DATA_HOME", "").strip() or os.path.join(os.path.expanduser("~"), ".local", "share"),
        "rag",
        "index",
    )
    os.makedirs(root, exist_ok=True)
    return root


def _safe_dir_name(path: str, embedding: Optional[Dict[str, str]] = None) -> str:
    abs_path = os.path.abspath(path)
    embedding = embedding or {}
    # The index depends on the embedding model: a different model (or none) needs its own table.
    signature = f"{abs_path}\n{embedding.get('embed_provider', '')}\n{embedding.get('embed_model', '')}"
    h = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    base = os.path.basename(abs_path.rstrip(os.sep)) or "root"
    return f"{base}_{h}"

def get_rag_pipeline_v2(doc_path: Optional[str | list[str]] = None):
    
    # Resolve path
    if isinstance(doc_path, list):
        doc_paths = [resolve_path(path) for path in doc_path]
        if not doc_paths:
            raise ValueError("No indexable files in selected scope")
        doc_path = doc_paths
    elif doc_path:
        doc_path = resolve_path(doc_path)
        if not os.path.exists(doc_path):
            raise ValueError(f"Path does not exist: {doc_path}")
    
    key = "\n".join(doc_path) if isinstance(doc_path, list) else (doc_path if doc_path else "__DEFAULT__")
    with _scope_lock(key):
        return _get_scoped_pipeline(key, doc_path)


def _get_scoped_pipeline(key, doc_path):
    import sys
    sys.stderr.write(f"[RAG V2] Creating new instance for {key}...\n")
    sys.stderr.flush()
    try:
        config = RAG_CONFIG_V2.copy()
        embedding = embedding_config()
        config.update(embedding)
        retrieval = retrieval_config()
        final_count = max(1, int(retrieval["top_k"]))
        config["rerank_top_k"] = final_count
        config["top_k"] = max(final_count, int(config["top_k"]))
        config["reranker_type"] = retrieval["reranker_type"]
        config["reranker_model"] = retrieval["reranker_model"]
        config["use_contextual"] = contextual_retrieval_enabled(retrieval)
        contextual = {name: retrieval[name] for name in (
            "contextual_model",
            "contextual_max_tokens",
            "contextual_timeout",
            "contextual_source_max_chars",
            "contextual_start_chunks",
            "contextual_prev_chunks",
        )}
        config.update(contextual)
        current_dir = index_root()
        if doc_path:
            config["document_paths"] = doc_path if isinstance(doc_path, list) else [doc_path]
            scope_dir = os.path.join(current_dir, _safe_dir_name(key, embedding))
            os.makedirs(scope_dir, exist_ok=True)
            config["persist_dir"] = scope_dir
        else:
            config["persist_dir"] = os.path.join(current_dir, config["persist_dir"])
            config["document_paths"] = [os.path.join(current_dir, path) for path in config["document_paths"]]
        scope_root = Path(config['persist_dir'])
        config['index_scope'] = str(scope_root)
        config['persist_dir'] = str(active_directory(scope_root))
        
        stamp = {
            "version": 3,  # Source-qualified IDs, preserved by the storage backend.
            "parsing_version": 5,  # Static HTML content extraction with physical source-line positions.
            "storage": {name: config[name] for name in ("backend", "distance_metric")},
            "chunking": {
                "version": 2,  # Fenced Markdown code does not create section headings.
                "size": config["chunk_size"],
                "overlap": config["chunk_overlap"],
            },
            "sources": source_snapshot(config['document_paths']),
            "embedding": {
                name: embedding.get(name, "")
                for name in ("embed_provider", "embed_model", "embed_base_url", "ollama_url")
            },
            "contextual": {
                "enabled": config["use_contextual"],
                **{
                    name: value for name, value in contextual.items()
                    if name != "contextual_timeout"
                },
            },
        }
        config["source_stamp"] = stamp
        cached = _rag_system_v2_instances.get(key)
        if cached is not None and all(cached.config.get(name) == config[name] for name in (
            "backend", "distance_metric", "rrf_k", "bm25_k1", "bm25_b", "persist_dir",
            *embedding,
            "top_k",
            "rerank_top_k",
            "reranker_type",
            "reranker_model",
            "use_contextual",
            *contextual,
            "source_stamp",
        )):
            return cached
        manifest = Path(config["persist_dir"]) / "source-manifest.json"
        try:
            previous_stamp = json.loads(manifest.read_text())
        except (OSError, ValueError):
            previous_stamp = None
        config["force_reindex"] = config.get("force_reindex", False) or previous_stamp != stamp
        if not config['force_reindex']:
            # Embedded Qdrant permits one client per storage path.
            _retire_cached_pipeline(key)

        sys.stderr.write("[RAG] About to create RAGPipelineV2...\n")
        sys.stderr.flush()
        instance = open_or_build(config, lambda build_config: RAGPipelineV2(
            config={**build_config, '_staged_index': True}))
        try:
            _retire_cached_pipeline(key)
        except Exception:
            _close_pipeline(instance)
            raise
        _rag_system_v2_instances[key] = instance
        sys.stderr.write("[RAG] V2 initialization complete\n")
        sys.stderr.flush()
        return instance
    except Exception as e:
        sys.stderr.write(f"[RAG] FATAL ERROR: {e}\n")
        sys.stderr.flush()
        raise

def query_documents(
    query: str,
    doc_path: Optional[str | list[str]] = None,
    exclude_global_archive: bool = False,
) -> list[dict] | str:
    """Retrieve evidence from a file, directory or prevalidated file manifest."""
    try:
        if not isinstance(query, str) or not query:
            raise ValueError("Enter a question to search for.")

        # The MCP host may pass a validated manifest so a directory is not
        # recursively enumerated a second time in the same process.
        if isinstance(doc_path, list):
            all_files = doc_path
            if not all_files:
                raise ValueError("No readable documents found in the selected scope.")
            doc_path = all_files
        elif doc_path:
            resolved_path = resolve_path(doc_path)
            if not os.path.exists(resolved_path):
                raise ValueError(f"Path does not exist: {resolved_path}")

            if os.path.isdir(resolved_path):
                # Bound folder scans and exclude generated/test artifacts.
                excluded_paths = set()
                if exclude_global_archive:
                    excluded_paths.add(os.path.realpath(os.path.expanduser("~/.elpis/memories/archive.md")))
                all_files = find_all_indexable_files(
                    resolved_path,
                    max_files=500,
                    exclude_patterns=['test_*.py', '*_test.py', '*__pycache__*', '*.pyc'],
                    excluded_paths=excluded_paths,
                )

                if not all_files:
                    raise ValueError(f"No readable documents found in {resolved_path}")

                print(f"[RAG V2] Indexing {len(all_files)} files from {resolved_path}")

                # Pass the vetted files so RAG cannot silently add global memory artifacts.
                doc_path = all_files

        pipeline = get_rag_pipeline_v2(doc_path=doc_path)
        return pipeline.search(query)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

# Preserve legacy imports and the wrapper's patchable delegation target.
fetchExternalKnowledgeV2 = query_documents

def fetchExternalKnowledge(
    query: str,
    doc_path: Optional[str | list[str]] = None,
    exclude_global_archive: bool = False,
) -> list[dict] | str:
    return fetchExternalKnowledgeV2(query, doc_path=doc_path, exclude_global_archive=exclude_global_archive)
