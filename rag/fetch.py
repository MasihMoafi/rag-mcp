import os
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional
from .core import RAGPipelineV2
from .settings import contextual_retrieval_enabled, embedding_config, retrieval_config

def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Every value here is a default, not a ceiling: override any of them with the
# matching RAG_MCP_* environment variable, no source edit required. The
# shipped defaults are deliberately small and fast (~80MB each for
# embedding and reranking) rather than the largest model available.
RAG_CONFIG_V2 = {
    "version": "v2",
    "persist_dir": "./rag_db_v2",
    "document_paths": ["./files/"],
    # embed_* keys come from rag.settings.embedding_config() at pipeline creation:
    # the user's `rag setup` choice, overridable by RAG_MCP_EMBED_* variables.
    "top_k": _env_int("RAG_MCP_TOP_K", 5),
    "chunk_size": _env_int("RAG_MCP_CHUNK_SIZE", 700),
    "chunk_overlap": _env_int("RAG_MCP_CHUNK_OVERLAP", 100),
    "rrf_k": _env_int("RAG_MCP_RRF_K", 60),  # RRF parameter for fusion (better than weighted ensemble)
    # rerank_top_k / reranker_type / reranker_model come from rag.settings.retrieval_config()
    # at pipeline creation (`rag config`, overridable by RAG_MCP_RERANK_TOP_K etc.).
    "force_reindex": False,
    "bm25_k1": 1.5,  # BM25 parameters
    "bm25_b": 0.75,
    "distance_metric": "cosine",  # Vector distance metric
    "backend": _env_str("RAG_MCP_BACKEND", "lancedb"),  # lancedb | qdrant
}

_rag_system_v2_instances: Dict[str, RAGPipelineV2] = {}

from .formats import INDEXABLE_EXTENSIONS as _INDEXABLE_EXTENSIONS


def reset_pipelines() -> None:
    """Forget loaded pipelines so the next query picks up changed settings."""
    _rag_system_v2_instances.clear()

def resolve_path(path: str) -> str:
    """Resolve user-friendly paths to absolute paths"""
    if not path:
        return None
    
    # Handle common shortcuts
    shortcuts = {
        "desktop": "~/Desktop",
        "documents": "~/Documents",
        "downloads": "~/Downloads",
    }
    
    lower = path.lower().strip()
    if lower in shortcuts:
        path = shortcuts[lower]
    
    # Expand ~ and environment variables
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    
    # Make absolute if relative
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    return path

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
    import sys
    
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
        
        sources = []
        for document_path in config["document_paths"]:
            source = Path(document_path)
            paths = source.rglob("*") if source.is_dir() else [source]
            for path in paths:
                if path.is_file():
                    stat = path.stat()
                    sources.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
        stamp = {
            "version": 3,  # Source-qualified IDs, preserved by the storage backend.
            "chunking": {
                "size": config["chunk_size"],
                "overlap": config["chunk_overlap"],
            },
            "sources": sorted(sources),
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

        sys.stderr.write("[RAG] About to create RAGPipelineV2...\n")
        sys.stderr.flush()
        instance = RAGPipelineV2(config=config)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=manifest.parent, delete=False) as pending:
            json.dump(stamp, pending)
        os.replace(pending.name, manifest)
        _rag_system_v2_instances[key] = instance
        sys.stderr.write("[RAG] V2 initialization complete\n")
        sys.stderr.flush()
        return instance
    except Exception as e:
        sys.stderr.write(f"[RAG] FATAL ERROR: {e}\n")
        sys.stderr.flush()
        raise

def find_all_indexable_files(
    directory: str,
    max_depth: int = 5,
    include_patterns: Optional[list] = None,
    exclude_patterns: Optional[list] = None,
    max_files: Optional[int] = None,
    excluded_paths: Optional[set[str]] = None,
) -> list:
    """Recursively find indexable files with selective filtering"""
    if not os.path.isdir(directory):
        return []

    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build', 'target', '.ipynb_checkpoints', 'migrations', 'rag_db_v2'}

    all_files = []

    def matches_pattern(path: str, patterns: list) -> bool:
        """Check if path matches any glob pattern"""
        import fnmatch
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
        return False

    def walk_dir(path: str, depth: int = 0):
        if depth > max_depth:
            return
        if max_files and len(all_files) >= max_files:
            return

        try:
            for entry in os.listdir(path):
                # Skip hidden files and excluded directories
                if entry.startswith('.') or entry in exclude_dirs:
                    continue

                entry_path = os.path.join(path, entry)

                # Apply exclude patterns
                if exclude_patterns and matches_pattern(entry_path, exclude_patterns):
                    continue

                if os.path.isfile(entry_path):
                    if excluded_paths and os.path.realpath(entry_path) in excluded_paths:
                        continue
                    if entry.lower().endswith(_INDEXABLE_EXTENSIONS):
                        # Apply include patterns
                        if include_patterns:
                            if matches_pattern(entry_path, include_patterns):
                                all_files.append(entry_path)
                        else:
                            all_files.append(entry_path)

                        if max_files and len(all_files) >= max_files:
                            return
                elif os.path.isdir(entry_path):
                    walk_dir(entry_path, depth + 1)
        except PermissionError:
            print(f"[RAG] Permission denied: {path}")

    print(f"[RAG] Scanning: {directory}")
    if include_patterns:
        print(f"[RAG] Include patterns: {include_patterns}")
    if exclude_patterns:
        print(f"[RAG] Exclude patterns: {exclude_patterns}")
    if max_files:
        print(f"[RAG] Max files: {max_files}")

    walk_dir(directory)
    print(f"[RAG] Found {len(all_files)} indexable files")

    return all_files


def find_relevant_files(query: str, directory: str, max_files: int = 5) -> list:
    """Find files in directory whose names match query keywords"""
    if not os.path.isdir(directory):
        return []
    
    # Query should already be cleaned by main.py's Instructor parsing
    print(f"[RAG V2] Searching for files matching: '{query}' in {directory}")
    
    query_words = query.lower().split()
    
    # If no meaningful query, return first few indexable files
    if not query_words:
        all_files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and file.lower().endswith(_INDEXABLE_EXTENSIONS):
                all_files.append(file_path)
        return all_files[:max_files]

    scored_files = []
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if not os.path.isfile(file_path):
            continue
        if not file.lower().endswith(_INDEXABLE_EXTENSIONS):
            continue
        
        # Score based on filename match
        filename_lower = file.lower()
        score = sum(1 for word in query_words if word in filename_lower)
        
        if score > 0:
            scored_files.append((score, file_path))
    
    # Return top matches
    scored_files.sort(reverse=True, key=lambda x: x[0])
    return [path for score, path in scored_files[:max_files]]

def fetchExternalKnowledgeV2(
    query: str,
    doc_path: Optional[str | list[str]] = None,
    exclude_global_archive: bool = False,
) -> str:
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
                # Find ALL indexable files recursively (limit to 100 like v1)
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

# For compatibility with a unified interface
def fetchExternalKnowledge(
    query: str,
    doc_path: Optional[str | list[str]] = None,
    exclude_global_archive: bool = False,
) -> str:
    return fetchExternalKnowledgeV2(query, doc_path=doc_path, exclude_global_archive=exclude_global_archive)
