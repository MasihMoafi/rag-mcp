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

import re

import lancedb
import pyarrow as pa
from .backend_embeddings import embed_batch, embed_single


_IDENTIFIER_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|[_:./\\-]+"
)


class LanceDBBackend:
    """Embedded LanceDB backend supporting vector, FTS (Tantivy BM25), and hybrid search."""

    def __init__(
        self,
        table_name: str,
        embedding_fn: Callable | None,
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
            sys.stderr.write(f"[LanceDB] Notice during init: {type(e).__name__}\n")

    @staticmethod
    def _identifier_words(content: str) -> str:
        """Content with identifier boundaries turned into spaces.

        Tantivy's default tokenizer splits on non-alphanumerics, so
        ``queued_user_messages`` is already three words but ``QueuedUserMessage``
        is one opaque token. Indexing this alongside the raw content lets a
        plain-English query reach a CamelCase symbol.
        """
        return _IDENTIFIER_BOUNDARY.sub(" ", content)

    _embed_single = embed_single

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        return embed_batch(self, texts, "[LanceDB] Batch embed_documents failed")

    def add_documents_batch(self, documents: List[Dict[str, Any]], batch_size: int = 512):
        """Add documents to LanceDB in optimal GPU batches ($B=512$) and build Tantivy FTS index."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not documents:
            return

        total = len(documents)
        sys.stderr.write(f"[LanceDB] Indexing {total} documents in GPU batches of {batch_size}...\n")



        batch_error = None

        def batches():
            nonlocal batch_error
            schema = None
            try:
                for i in range(0, total, batch_size):
                    batch = documents[i : i + batch_size]
                    contents = [doc.get("content", "") for doc in batch]
                    vectors = self._batch_embed(contents) if self.embedding_fn is not None else [None] * len(batch)
                    if len(vectors) != len(batch):
                        raise ValueError(f"Embedding count {len(vectors)} does not match document count {len(batch)}.")
                    records = []
                    for j, (doc, vector) in enumerate(zip(batch, vectors)):
                        record = {
                            "id": str(doc.get("id", i + j)),
                            "content": doc.get("content", ""),
                            "file_path": str(doc.get("file_path", "")),
                            "source": str(doc.get("source", doc.get("file_path", ""))),
                            "heading": str(doc.get("heading", "")),
                            "location": str(doc.get("location", "")),
                            "start_line": int(doc.get("start_line", 0)),
                            "end_line": int(doc.get("end_line", 0)),
                            "page_number": int(doc.get("page_number", 0)),
                            "original_content": str(doc.get("original_content", doc.get("content", ""))),
                            "symbols": self._identifier_words(doc.get("content", "")),
                        }
                        if vector is not None:
                            record["vector"] = vector
                        records.append(record)
                    if schema is None:
                        schema = pa.schema([
                            (key, pa.int64() if key in {"start_line", "end_line", "page_number"} else pa.string())
                            for key in records[0] if key != "vector"
                        ])
                        if self.embedding_fn is not None:
                            schema = schema.append(pa.field("vector", pa.list_(pa.float32(), len(vectors[0]))))
                    arrow_batch = pa.RecordBatch.from_pylist(records, schema=schema)
                    del records, record, vectors, vector
                    yield arrow_batch
            except GeneratorExit:
                raise
            except BaseException as error:
                batch_error = error
                # Arrow wraps iterator errors; do not embed provider details in its message.
                raise RuntimeError("Embedding batch preparation failed.") from None

        try:
            self.table = self.db.create_table(self.table_name, data=batches(), mode="overwrite")
        except Exception:
            if batch_error is not None:
                raise batch_error from None
            raise
        sys.stderr.write(f"[LanceDB] Created table '{self.table_name}' with {total} records.\n")

        # Create Full Text Search (FTS) index using Tantivy
        try:
            # stem/stop-word removal and positionless indexing are wrong for code:
            # they drop "is not", mangle identifiers, and make any quoted phrase
            # query raise instead of returning results.
            # Native FTS indexes one field each, so the identifier column gets its own.
            for column in ("content", "symbols"):
                self.table.create_fts_index(
                    column,
                    replace=True,
                    with_position=True,
                    stem=False,
                    remove_stop_words=False,
                )
            sys.stderr.write("[LanceDB] Built FTS indexes on 'content' and 'symbols'.\n")
        except Exception as e:
            sys.stderr.write(f"[LanceDB] FTS index creation failed: {type(e).__name__}\n")
            raise

    def search_vector(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search. Empty when no embedding model is configured."""
        if self.embedding_fn is None or self.table is None or len(self.table) == 0:
            return []

        query_vector = self._embed_single(query)
        metric_map = {"cosine": "cosine", "euclidean": "l2", "dot": "dot"}
        metric = metric_map.get(self.distance, "cosine")

        results = (
            self.table.search(query_vector, vector_column_name="vector")
            .metric(metric)
            .select([name for name in self.table.schema.names if name != "vector"] + ["_distance"])
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
            schema = getattr(self.table, "schema", None)
            names = list(schema.names) if schema is not None else []
            fts_columns = [
                name for name in ("content", "symbols") if name in names
            ] or ["content"]
            payload = [name for name in names if name not in ("vector", "symbols")]
            results = (
                self.table.search(query, query_type="fts", fts_columns=fts_columns)
                .select(payload + ["_score"])
                .limit(top_k)
                .to_list()
            )
            formatted = []
            for hit in results:
                score = hit.get("_score", 1.0)
                doc = {
                    k: v
                    for k, v in hit.items()
                    if k not in ("vector", "symbols", "_distance", "_score")
                }
                doc["score"] = score
                formatted.append(doc)
            return formatted
        except Exception as e:
            # Deliberately propagate: a retrieval failure must never reach the
            # caller disguised as "no matches". See
            # tests/unit/test_backend_contract.py::test_fts_failure_is_not_reported_as_no_matches.
            sys.stderr.write(f"[LanceDB] FTS search failed: {type(e).__name__}\n")
            raise

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
            sys.stderr.write(f"[LanceDB] Clear notice: {type(e).__name__}\n")
