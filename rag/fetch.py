"""Scoped pipeline configuration, cache reuse and document-query orchestration."""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional
from threading import Lock, RLock
from weakref import WeakValueDictionary
from .core import RAGPipelineV2
from .discovery import resolve_path, find_all_indexable_files, find_relevant_files
from .formats import INDEXABLE_EXTENSIONS as _INDEXABLE_EXTENSIONS
from .settings import contextual_retrieval_enabled, embedding_config, retrieval_config
from .index_lifecycle import active_directory, open_or_build, source_delta
from .index_config import RAG_CONFIG_V2, _env_int, _safe_dir_name, build_source_stamp, index_root_path
from .query_lifecycle import PipelineClosedError
from . import tracing

_rag_system_v2_instances: Dict[str, RAGPipelineV2] = {}
_scope_locks = WeakValueDictionary()
_scope_locks_guard = Lock()
_cache_trim_lock = Lock()


def _trim_cached_pipelines(keep):
    limit = max(1, _env_int('RAG_MCP_CACHE_SIZE', 8))
    if len(_rag_system_v2_instances) <= limit or not _cache_trim_lock.acquire(blocking=False):
        return
    try:
        for key in list(_rag_system_v2_instances):
            if len(_rag_system_v2_instances) <= limit:
                break
            lock = _scope_lock(key)
            if key == keep or not lock.acquire(blocking=False):
                continue
            try:
                pipeline = _rag_system_v2_instances.get(key)
                close = getattr(pipeline, 'close_if_idle', None)
                if close is not None and close():
                    del _rag_system_v2_instances[key]
            except Exception as error:
                print(f'[RAG] Cache retirement failed: {type(error).__name__}', file=sys.stderr)
            finally:
                lock.release()
    finally:
        _cache_trim_lock.release()


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
    first_error = None
    for key in keys:
        try:
            with _scope_lock(key):
                _retire_cached_pipeline(key)
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error

def index_root() -> str:
    """Where indexes live: RAG_MCP_DATA_DIR, else the XDG data dir. Never inside the package."""
    root = index_root_path()
    os.makedirs(root, exist_ok=True)
    return root


def _pipeline_key(doc_path, scope):
    return scope or ("\n".join(doc_path) if isinstance(doc_path, list) else (doc_path or "__DEFAULT__"))


@tracing.traced('index.lookup')
def get_rag_pipeline_v2(doc_path: Optional[str | list[str]] = None, scope: Optional[str] = None):

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

    # The folder a manifest was enumerated from keeps one index across added and
    # removed files; without it every file-list change starts a new index.
    key = _pipeline_key(doc_path, scope)
    with _scope_lock(key):
        instance = _get_scoped_pipeline(key, doc_path)
        # Dict insertion order tracks the last acquisition of each scope.
        _rag_system_v2_instances.pop(key, None)
        _rag_system_v2_instances[key] = instance
    _trim_cached_pipelines(keep=key)
    return instance


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
        candidate_count = max(final_count, int(retrieval.get("candidate_k", 50)))
        config["rerank_top_k"] = final_count
        config["candidate_k"] = candidate_count
        config["top_k"] = max(candidate_count, int(config["top_k"]))
        config["reranker_type"] = retrieval["reranker_type"]
        config["reranker_model"] = retrieval["reranker_model"]
        config["reranker_timeout"] = retrieval.get("reranker_timeout", "60")
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
        scope_root = Path(config['persist_dir']).resolve()
        config['index_scope'] = str(scope_root)
        config['persist_dir'] = str(active_directory(scope_root))
        
        stamp = build_source_stamp(config, embedding, contextual)
        config["source_stamp"] = stamp
        tracing.annotate(manifest_id=tracing.identity(json.dumps(stamp, sort_keys=True)),
                         source_count=len(config['document_paths']),
                         embedding_model=config.get('embed_model'), reranker_model=config.get('reranker_model'),
                         chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'],
                         candidate_limit=config['candidate_k'], final_limit=config['rerank_top_k'])
        cached = _rag_system_v2_instances.get(key)
        if cached is not None and all(cached.config.get(name) == config[name] for name in (
            "backend", "distance_metric", "rrf_k", "bm25_k1", "bm25_b", "persist_dir",
            *embedding,
            "top_k",
            "candidate_k",
            "rerank_top_k",
            "reranker_type",
            "reranker_model",
            "reranker_timeout",
            "use_contextual",
            *contextual,
            "source_stamp",
        )):
            tracing.annotate(cache_state='memory_hit', generation_id=tracing.identity(config['persist_dir']))
            return cached
        manifest = Path(config["persist_dir"]) / "source-manifest.json"
        try:
            previous_stamp = json.loads(manifest.read_text())
        except (OSError, ValueError):
            previous_stamp = None
        # Only LanceDB can carry a published index forward; Qdrant point ids are
        # positional, so a changed file cannot be replaced in place there.
        delta = (source_delta(previous_stamp, stamp)
                 if config['backend'] == 'lancedb' and not config.get('force_reindex', False) else None)
        config['source_delta'] = delta
        config["force_reindex"] = config.get("force_reindex", False) or (
            delta is None and previous_stamp != stamp)
        if not config['force_reindex'] and config['backend'] == 'qdrant':
            # Embedded Qdrant permits one client per storage path.
            _retire_cached_pipeline(key)

        sys.stderr.write("[RAG] About to create RAGPipelineV2...\n")
        sys.stderr.flush()
        instance = open_or_build(config, lambda build_config: RAGPipelineV2(
            config={**build_config, '_staged_index': True}))
        tracing.annotate(cache_state='build' if config['force_reindex'] else ('refresh' if delta else 'reopen'),
                         generation_id=tracing.identity(instance.config['persist_dir']))
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

@tracing.traced('retrieval.facet')
def query_documents(
    query: str,
    doc_path: Optional[str | list[str]] = None,
    exclude_global_archive: bool = False,
    scope: Optional[str] = None,
) -> list[dict] | str:
    """Retrieve evidence from a file, directory or prevalidated file manifest.

    `scope` names the folder a file manifest came from, so that folder keeps one
    index while files are added and removed."""
    tracing.content('query', query)
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
                    max_files=501,  # Detect overflow rather than silently indexing a subset.
                    exclude_patterns=['test_*.py', '*_test.py', '*__pycache__*', '*.pyc'],
                    excluded_paths=excluded_paths,
                )

                if len(all_files) > 500:
                    raise ValueError('Folder exceeds the 500-document limit; select a smaller folder.')
                if not all_files:
                    raise ValueError(f"No readable documents found in {resolved_path}")

                print(f"[RAG V2] Indexing {len(all_files)} files from {resolved_path}")

                # Pass the vetted files so RAG cannot silently add global memory artifacts.
                doc_path = all_files
                scope = resolved_path

        pipeline = get_rag_pipeline_v2(doc_path=doc_path, scope=scope)
        try:
            return pipeline.search(query)
        except PipelineClosedError:
            return get_rag_pipeline_v2(doc_path=doc_path, scope=scope).search(query)
    except Exception as e:
        print(f"[RAG] Retrieval failed: {type(e).__name__}", file=sys.stderr)
        raise

# Preserve legacy imports and the wrapper's patchable delegation target.
fetchExternalKnowledgeV2 = query_documents

def fetchExternalKnowledge(
    query: str,
    doc_path: Optional[str | list[str]] = None,
    exclude_global_archive: bool = False,
    scope: Optional[str] = None,
) -> list[dict] | str:
    return fetchExternalKnowledgeV2(query, doc_path=doc_path,
                                    exclude_global_archive=exclude_global_archive, scope=scope)


def expand_answer_context(passages, doc_path, scope, radius):
    """Use this scope's published chunks without changing its index or rankings."""
    from .context_window import expand_passage_windows
    from .ingestion import read_documents, source_files
    from .index_lifecycle import RebuildRequired
    if radius == 0:
        return passages
    key = _pipeline_key(doc_path, scope)
    try:
        # Reuse the searched generation only; expanding an answer must never build an index.
        with _scope_lock(key):
            pipeline = _rag_system_v2_instances.get(key)
            if pipeline is None:
                return passages
            with pipeline._queries.query(), tracing.span('selection.context', context_radius=radius,
                                                         input_count=len(passages)):
                result = expand_passage_windows(passages, read_documents(pipeline), pipeline._extract_content,
                                                radius, set(source_files(pipeline)))
                tracing.annotate(output_count=len(result), expanded_count=sum(
                    (before.get('start_offset'), before.get('end_offset')) !=
                    (after.get('start_offset'), after.get('end_offset'))
                    for before, after in zip(passages, result)))
                return result
    except (OSError, RebuildRequired, PipelineClosedError):
        return passages
