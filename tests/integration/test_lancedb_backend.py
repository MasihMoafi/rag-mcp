import os
import shutil
import tempfile
import pytest
from rag.lancedb_backend import LanceDBBackend


class DummyEmbedder:
    def embed_query(self, text: str):
        # 4-dim dummy vector based on hash
        h = hash(text) % 1000
        return [float(h % 10), float((h // 10) % 10), 1.0, 0.5]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def test_lancedb_backend_crud_and_search():
    temp_dir = tempfile.mkdtemp()
    try:
        backend = LanceDBBackend(
            table_name="test_table",
            embedding_fn=DummyEmbedder(),
            vector_size=4,
            distance="cosine",
            persist_path=temp_dir,
        )

        docs = [
            {"content": "Napoleon Bonaparte was a French military commander.", "file_path": "napoleon.txt"},
            {"content": "Attention Is All You Need introduced the Transformer.", "file_path": "transformer.txt"},
            {"content": "LanceDB is a serverless vector database built with Rust and Arrow.", "file_path": "lancedb.txt"},
        ]

        backend.add_documents_batch(docs)
        assert backend.count() == 3

        # Vector search
        v_results = backend.search_vector("Who was Napoleon?", top_k=2)
        assert len(v_results) > 0
        assert "content" in v_results[0]
        assert "file_path" in v_results[0]

        # FTS search
        fts_results = backend.search_fts("Transformer", top_k=1)
        assert len(fts_results) == 1
        assert "Transformer" in fts_results[0]["content"]

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
