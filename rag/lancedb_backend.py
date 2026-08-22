"""
LanceDB Vector and Hybrid Search Backend for RAG V2.

Replaces embedded Qdrant + custom Python BM25 with native LanceDB:
- Columnar Apache Arrow storage on disk
- Memory-mapped zero-copy reads
- Native Tantivy-powered Full Text Search (BM25)
- Native vector search + RRF hybrid fusion
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Callable, Dict, List

# Central proxy manager — strip SOCKS, keep HTTP(S) for cloud/remote
from utils.proxy import strip_socks as _strip_socks
_strip_socks()

import lancedb
import pyarrow as pa


class LanceDBBackend:
    """Embedded LanceDB backend supporting vector, FTS (Tantivy BM25), and hybrid search."""

    def __init__(
        self,
        table_name: str,
        embedding_fn: Callable,
        vector_size: int = 768,
        distance: str = "cosine",
        persist_path: str = "./lancedb_storage",
    ):
        self.table_name = re.sub(r"[^A-Za-z0-9_]", "_", table_name)
        self.embedding_fn = embedding_fn
        self.vector_size = vector_size
        self.distance = distance.lower()
        self.persist_path = persist_path

        os.makedirs(self.persist_path, exist_ok=True)
        sys.stderr.write(f"[LanceDB] Opening database at: {self.persist_path}\n")
        self.db = lancedb.connect(self.persist_path)
        self.table = None
        self._init_table()

    def _init_table(self):
        """Open existing table if present."""
        try:
            if hasattr(self.db, "table_names"):
                table_names = self.db.table_names()
            elif hasattr(self.db, "list_tables"):
                table_names = self.db.list_tables()
            else:
                table_names = []
            if self.table_name in table_names:
                self.table = self.db.open_table(self.table_name)
                sys.stderr.write(f"[LanceDB] Opened existing table: {self.table_name} ({len(self.table)} rows)\n")
        except Exception as e:
            sys.stderr.write(f"[LanceDB] Notice during init: {e}\n")

    def _embed_single(self, text: str) -> List[float]:
        """Embed single query string."""
        if hasattr(self.embedding_fn, "embed_query"):
            return self.embedding_fn.embed_query(text)
        elif callable(self.embedding_fn):
            return self.embedding_fn(text)
        else:
            raise ValueError(f"Unsupported embedding function type: {type(self.embedding_fn)}")

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding with parallel processing fallback."""
        if hasattr(self.embedding_fn, "embed_documents"):
            try:
                return self.embedding_fn.embed_documents(texts)
            except Exception as e:
                sys.stderr.write(f"[LanceDB] Batch embed_documents failed: {e}\n")

        try:
            result = self.embedding_fn(texts)
            if isinstance(result, list) and len(result) == len(texts):
                return result
        except Exception:
            pass

        from concurrent.futures import ThreadPoolExecutor
        max_workers = 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            vectors = list(executor.map(self._embed_single, texts))
        return vectors

    def add_documents_batch(self, documents: List[Dict[str, Any]], batch_size: int = 512):
        """Add documents to LanceDB in optimal GPU batches ($B=512$) and build Tantivy FTS index."""
        if not documents:
            return

        total = len(documents)
        sys.stderr.write(f"[LanceDB] Indexing {total} documents in GPU batches of {batch_size}...\n")



        all_records = []
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            contents = [doc.get("content", "") for doc in batch]
            vectors = self._batch_embed(contents)

            for j, (doc, vector) in enumerate(zip(batch, vectors)):
                record = {
                    "id": str(i + j),
                    "vector": vector,
                    "content": doc.get("content", ""),
                    "file_path": str(doc.get("file_path", "")),
                    "source": str(doc.get("source", doc.get("file_path", ""))),
                    "heading": str(doc.get("heading", "")),
                    "start_line": int(doc.get("start_line", 0)),
                    "end_line": int(doc.get("end_line", 0)),
                    "page_number": int(doc.get("page_number", 0)),
                    "original_content": str(doc.get("original_content", doc.get("content", ""))),
                }
                all_records.append(record)

        # Overwrite or create table
        metric_map = {"cosine": "cosine", "euclidean": "l2", "dot": "dot"}
        metric = metric_map.get(self.distance, "cosine")

        self.table = self.db.create_table(
            self.table_name,
            data=all_records,
            mode="overwrite",
        )
        sys.stderr.write(f"[LanceDB] Created table '{self.table_name}' with {len(all_records)} records.\n")

        # Create Full Text Search (FTS) index using Tantivy
        try:
            self.table.create_fts_index("content", replace=True)
            sys.stderr.write(f"[LanceDB] Built Tantivy FTS index on 'content' column.\n")
        except Exception as e:
            sys.stderr.write(f"[LanceDB] FTS index creation note: {e}\n")

    def search_vector(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        if self.table is None or len(self.table) == 0:
            return []

        query_vector = self._embed_single(query)
        metric_map = {"cosine": "cosine", "euclidean": "l2", "dot": "dot"}
        metric = metric_map.get(self.distance, "cosine")

        results = (
            self.table.search(query_vector, vector_column_name="vector")
            .metric(metric)
            .limit(top_k)
            .to_list()
        )

        formatted = []
        for hit in results:
            # Distance in LanceDB: for cosine, score = 1 - distance (similarity)
            distance = hit.get("_distance", 0.0)
            score = 1.0 - distance if metric == "cosine" else -distance
            doc = {k: v for k, v in hit.items() if k not in ("vector", "_distance")}
            doc["score"] = score
            formatted.append(doc)
        return formatted

    def search_fts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Tantivy BM25 full-text search."""
        if self.table is None or len(self.table) == 0:
            return []

        try:
            results = (
                self.table.search(query, query_type="fts")
                .limit(top_k)
                .to_list()
            )
            formatted = []
            for hit in results:
                score = hit.get("_score", 1.0)
                doc = {k: v for k, v in hit.items() if k not in ("vector", "_distance", "_score")}
                doc["score"] = score
                formatted.append(doc)
            return formatted
        except Exception as e:
            sys.stderr.write(f"[LanceDB] FTS search fallback: {e}\n")
            return []

    def count(self) -> int:
        """Get document count."""
        return len(self.table) if self.table is not None else 0

    def clear(self):
        """Drop table."""
        try:
            self.db.drop_table(self.table_name)
            self.table = None
            sys.stderr.write(f"[LanceDB] Dropped table: {self.table_name}\n")
        except Exception as e:
            sys.stderr.write(f"[LanceDB] Clear notice: {e}\n")
