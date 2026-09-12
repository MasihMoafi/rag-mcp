"""Compose the retrieval pipeline while retaining legacy import and method hooks."""

import sys
import os
try:
    import torch
except ImportError:  # light install without PyTorch: CPU only, no sentence-transformers
    torch = None
# Preserve the existing local-provider proxy behavior during engine import.
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]
clear_proxy_settings()

from .embeddings import build_embedder, probe_dimensions
from .contextual_retrieval import AnthropicContextualizer
from .lancedb_backend import LanceDBBackend
from . import ingestion, parsing, reranking
from .bm25 import BM25Index
from .chunking import chunk_text, chunk_document


def select_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps and mps.is_available() else "cpu"


class RAGPipeline:
    """Hybrid local retrieval with optional contextual indexing and reranking."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = select_device()
        print(f"[RAG V2] Using compute device: {self.device}")

        self.contextualizer = self.config.get("contextualizer")
        if self.config.get("use_contextual", False) and self.contextualizer is None:
            self.contextualizer = AnthropicContextualizer(
                model=str(self.config.get("contextual_model", "claude-haiku-4-5")),
                max_tokens=int(self.config.get("contextual_max_tokens", 128)),
                timeout=float(self.config.get("contextual_timeout", 30)),
            )

        # Embeddings come from the user's `rag setup` choice (or RAG_MCP_* env).
        # None means exact-word (BM25) search only.
        embed_provider = str(self.config.get("embed_provider") or "none").lower()
        self.embedding_fn = build_embedder(self.config, device=self.device)
        if self.embedding_fn is None:
            print("[RAG V2] No embedding model configured: exact-word (BM25) search only. Run `rag setup` to add meaning-based search.")
        else:
            print(f"[RAG V2] Using {embed_provider} embeddings with model: {self.embedding_fn.model_name}")

        # Determine vector database backend (LanceDB by default, or Qdrant)
        self.backend_type = self.config.get("backend", os.environ.get("RAG_MCP_BACKEND", "lancedb")).lower()
        print(f"[RAG V2] Using backend: {self.backend_type}")

        # One probe embedding tells the real vector size for any provider/model.
        vector_size = probe_dimensions(self.embedding_fn) if self.embedding_fn is not None else 0

        # Create unique collection name from persist_dir to avoid conflicts
        persist_dir_name = os.path.basename(self.config.get("persist_dir"))
        collection_name = f"rag_v2_{persist_dir_name}" if persist_dir_name != "rag_db_v2" else "rag_v2_default"

        if self.backend_type == "lancedb":
            lance_path = os.path.join(self.config.get("persist_dir"), "lancedb_storage")
            self.vector_db = LanceDBBackend(
                table_name=collection_name,
                embedding_fn=self.embedding_fn,
                vector_size=vector_size,
                distance=self.config.get("distance_metric", "cosine"),
                persist_path=lance_path,
            )
            self.bm25_index = None
        else:
            if self.embedding_fn is None:
                raise ValueError("The Qdrant backend needs an embedding model. Run `rag setup` or use the LanceDB backend.")
            from .qdrant_backend import QdrantVectorDB

            qdrant_path = os.path.join(self.config.get("persist_dir"), "qdrant_storage")
            self.vector_db = QdrantVectorDB(
                collection_name=collection_name,
                embedding_fn=self.embedding_fn,
                vector_size=vector_size,
                distance=self.config.get("distance_metric", "cosine"),
                persist_path=qdrant_path
            )
            # Keep BM25 (still valuable for lexical search)
            self.bm25_index = BM25Index(
                k1=self.config.get("bm25_k1", 1.5),
                b=self.config.get("bm25_b", 0.75)
            )

        reranking.initialize_reranker(self)

        # Load or create database
        self._load_or_create_database()

    _extract_text_from_pdf = parsing._extract_text_from_pdf
    _extract_text_with_unstructured = parsing._extract_text_with_unstructured
    _extract_content = parsing._extract_content
    _extract_text_from_notebook = parsing._extract_text_from_notebook
    _extract_text_from_json = parsing._extract_text_from_json

    _locate_chunk = ingestion.locate_chunk
    _add_context_to_chunk = ingestion.add_chunk_context
    _load_or_create_database = ingestion.load_or_create_database
    _index_documents = ingestion.index_documents
    _save_database = ingestion.save_documents
    _load_existing_database = ingestion.load_existing_database

    _rrf_fusion = reranking.fuse_rankings
    _llm_rerank = reranking.legacy_llm_rerank
    _crossencoder_rerank = reranking.rerank_documents

    def search(self, query: str) -> list[dict] | str:
        """Return ranked passages from the configured local retrieval pipeline."""
        try:
            top_k = self.config.get("top_k", 20)
            rerank_top_k = self.config.get("rerank_top_k", 3)
            
            print(f"[RAG V2] Searching with query: '{query[:50]}...' (backend: {self.backend_type})")
            
            if self.backend_type == "lancedb":
                vector_results = []
                if self.embedding_fn is not None:
                    print("[RAG V2] Step 3: LanceDB Vector search...")
                    vector_results_raw = self.vector_db.search_vector(query, top_k=top_k)
                    vector_results = [(doc, doc.get("score", 0)) for doc in vector_results_raw]
                else:
                    print("[RAG V2] Step 3: skipped (no embedding model; exact-word search only)")
                print("[RAG V2] Step 4: LanceDB Tantivy FTS search...")
                bm25_results_raw = self.vector_db.search_fts(query, top_k=top_k)
                bm25_results = [(doc, doc.get("score", 0)) for doc in bm25_results_raw]
            else:
                print("[RAG V2] Step 3: Vector search...")
                vector_results_qdrant = self.vector_db.search(query, top_k=top_k)
                # Convert to tuple format for RRF
                vector_results = [(doc, doc.get("score", 0)) for doc in vector_results_qdrant]
                print("[RAG V2] Step 4: BM25 search...")
                bm25_results = self.bm25_index.search(query, k=top_k)
            print("[RAG V2] Step 5: RRF fusion...")
            fused_docs = self._rrf_fusion(vector_results, bm25_results, k=self.config.get("rrf_k", 60))
            
            if not fused_docs:
                return "No relevant information found."
            reranker_type = str(self.config.get("reranker_type", "auto")).lower()
            
            if reranker_type in {"disabled", "none", "off"}:
                final_docs = fused_docs[:rerank_top_k]
            elif self.reranker_type != "qwen3" and getattr(self, "reranker", None) is None:
                # No reranker model is loaded (light install): keep the fusion order.
                final_docs = fused_docs[:rerank_top_k]
            elif reranker_type == "llm":
                print(f"[RAG Core] Reranking {len(fused_docs)} initial results using LLM Reranker...")
                final_docs = self._llm_rerank(fused_docs, query, k=rerank_top_k)
            else:
                print(f"[RAG Core] Reranking {len(fused_docs)} initial results using CrossEncoder Reranker...")
                final_docs = self._crossencoder_rerank(fused_docs, query, k=rerank_top_k)
            sys.stderr.write(f"[RAG V2] Returning {len(final_docs)} unique results\n")
            return final_docs
            
        except Exception as e:
            print(f"[RAG V2] Search error: {e}")
            return []


# Existing CLI, MCP callers and serialized class references use this name.
RAGPipelineV2 = RAGPipeline
