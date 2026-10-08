"""Lightweight index defaults and identity shared by queries and inspection."""

import hashlib
import os
from typing import Dict, Optional

from .index_lifecycle import source_snapshot


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
    "top_k": _env_int("RAG_MCP_TOP_K", 50),
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


def index_root_path() -> str:
    """Index location without creating it (for read-only inspection)."""
    return os.environ.get("RAG_MCP_DATA_DIR", "").strip() or os.path.join(
        os.environ.get("XDG_DATA_HOME", "").strip() or os.path.join(os.path.expanduser("~"), ".local", "share"),
        "rag",
        "index",
    )


def _safe_dir_name(path: str, embedding: Optional[Dict[str, str]] = None) -> str:
    abs_path = os.path.abspath(path)
    embedding = embedding or {}
    # The index depends on the embedding model: a different model (or none) needs its own table.
    signature = f"{abs_path}\n{embedding.get('embed_provider', '')}\n{embedding.get('embed_model', '')}"
    h = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    base = os.path.basename(abs_path.rstrip(os.sep)) or "root"
    return f"{base}_{h}"


def build_source_stamp(config, embedding, contextual):
    """The exact stamp a workspace query compares with its published index."""
    return {
        "version": 3,
        "parsing_version": 8,
        "storage": {name: config[name] for name in ("backend", "distance_metric")},
        "chunking": {"version": 2, "size": config["chunk_size"], "overlap": config["chunk_overlap"]},
        "sources": source_snapshot(config["document_paths"]),
        "embedding": {
            name: embedding.get(name, "")
            for name in ("embed_provider", "embed_model", "embed_base_url", "ollama_url")
        },
        "contextual": {
            "enabled": config["use_contextual"],
            **{name: value for name, value in contextual.items() if name != "contextual_timeout"},
        },
    }
