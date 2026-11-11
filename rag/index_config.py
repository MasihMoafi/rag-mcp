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
    # On by default since EnterpriseRAG-Bench, 11 Nov 2026: together 66.3% -> 75.2% correct answers
    # (evals/experiments/enterprise-options-2026-11-10/). Set the variable to 0 to turn one off.
    # title_context: the file's title heads every chunk, for both searches.
    # keyword_stemming: English stemming and stop words in keyword search (prose, not code).
    # balanced_candidates: each search keeps an equal share of the candidates before reranking.
    "title_context": _env_int("RAG_MCP_TITLE_CONTEXT", 1) == 1,
    "keyword_stemming": _env_int("RAG_MCP_KEYWORD_STEMMING", 1) == 1,
    "balanced_candidates": _env_int("RAG_MCP_BALANCED_CANDIDATES", 1) == 1,
}

# Index-shaping options; a change rebuilds the index once (the stamp records those that are on).
INDEX_OPTIONS = ("title_context", "keyword_stemming")


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


from .chunking import LONG_CHUNK_OVERLAP, LONG_CHUNK_SIZE, LONG_DOCUMENT_CHARS


def build_source_stamp(config, embedding, contextual):
    """The exact stamp a workspace query compares with its published index."""
    options = {name: True for name in INDEX_OPTIONS if config.get(name)}
    return {
        **({"options": options} if options else {}),
        "version": 3,
        "parsing_version": 9,
        "storage": {name: config[name] for name in ("backend", "distance_metric")},
        "chunking": {"version": 3, "size": config["chunk_size"], "overlap": config["chunk_overlap"],
                     "long_document": [LONG_DOCUMENT_CHARS, LONG_CHUNK_SIZE, LONG_CHUNK_OVERLAP]},
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
