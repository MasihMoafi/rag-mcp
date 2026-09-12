"""Local Qdrant persistence checks with controlled vectors, no model calls."""

from rag.qdrant_backend import QdrantVectorDB
from rag import ingestion
from types import SimpleNamespace
import pytest


class Embedder:
    def embed_documents(self, texts):
        return [[1., 0.] if 'Cedar' in text else [0., 1.] for text in texts]

    def embed_query(self, text):
        return [1., 0.]


def test_qdrant_retrieval_and_reopen_preserve_source_metadata(tmp_path):
    docs = [{'content': 'Cedar evidence', 'id': 'notes.md::chunk:0', 'location': 'L3'},
            {'content': 'Other evidence', 'id': 'other.md::chunk:0', 'location': 'L8'}]
    for reopen in [False, True]:
        backend = QdrantVectorDB('lifecycle', Embedder(), vector_size=2, persist_path=str(tmp_path))
        try:
            if not reopen:
                backend.add_documents_batch(docs, batch_size=1)
            assert backend.count() == 2
            result = backend.search('Cedar', top_k=1)[0]
            assert result['id'] == 'notes.md::chunk:0'
            assert result['location'] == 'L3'
        finally:
            backend.client.close()


@pytest.mark.xfail(strict=True, reason="Refresh clears the published collection before embedding succeeds")
def test_failed_qdrant_refresh_preserves_published_index_on_reopen(tmp_path, monkeypatch):
    old_doc = {'content': 'Cedar evidence', 'id': 'notes.md::chunk:0', 'location': 'L3'}
    backend = QdrantVectorDB('refresh', Embedder(), vector_size=2, persist_path=str(tmp_path))
    try:
        backend.add_documents_batch([old_doc])

        def fail_embedding(texts):
            raise RuntimeError('provider unavailable')

        monkeypatch.setattr(backend, '_batch_embed', fail_embedding)
        pipeline = SimpleNamespace(
            config={'persist_dir': str(tmp_path), 'force_reindex': True},
            backend_type='qdrant', vector_db=backend,
            _index_documents=lambda: backend.add_documents_batch([{'content': 'Replacement evidence'}]),
        )
        with pytest.raises(RuntimeError, match='provider unavailable'):
            ingestion.load_or_create_database(pipeline)
    finally:
        backend.client.close()

    reopened = QdrantVectorDB('refresh', Embedder(), vector_size=2, persist_path=str(tmp_path))
    try:
        assert reopened.count() == 1
        result = reopened.search('Cedar', top_k=1)[0]
        assert result['id'] == old_doc['id']
        assert result['location'] == 'L3'
    finally:
        reopened.client.close()
