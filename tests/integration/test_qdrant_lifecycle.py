"""Local Qdrant persistence checks with controlled vectors, no model calls."""

from rag.qdrant_backend import QdrantVectorDB


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
