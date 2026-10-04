"""Every embedding batch of an index build appears in the request trace."""

import json

from rag import tracing
from rag.lancedb_backend import LanceDBBackend
from tests.integration.test_incremental_index import RecordingEmbedder


def test_every_embedding_batch_is_traced(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path / 'traces'))
    embedder = RecordingEmbedder()
    backend = LanceDBBackend('fixture', embedder, vector_size=2, persist_path=str(tmp_path / 'db'))
    with tracing.request_trace():
        backend.add_documents_batch([{'id': str(n), 'content': f'Passage {n}.'} for n in range(1100)], batch_size=512)
    [trace] = [json.loads(path.read_text()) for path in (tmp_path / 'traces').glob('trace-*.json')]
    batches = [span['attributes']['input_count'] for span in trace['spans'] if span['operation'] == 'embedding.documents']
    assert len(embedder.documents) == 1100
    assert batches == [512, 512, 76]
