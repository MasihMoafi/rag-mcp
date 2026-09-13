"""Count provider work while building and reopening real local indexes."""

import pytest

from rag import core


@pytest.mark.parametrize('backend', ['lancedb', 'qdrant'])
def test_startup_embeds_only_what_storage_requires(tmp_path, monkeypatch, backend):
    calls = []

    class Embedder:
        model_name = 'counted-fixture'

        def embed_documents(self, texts):
            calls.append(('documents', list(texts)))
            return [[1., 0., 0.] for _ in texts]

        def embed_query(self, text):
            calls.append(('query', text))
            return [1., 0., 0.]

    monkeypatch.setattr(core, 'build_embedder', lambda *args, **kwargs: Embedder())
    monkeypatch.setattr(core, 'select_device', lambda: 'cpu')
    source = tmp_path / 'facts.txt'
    source.write_text('Cedar trees preserve the evidence used to verify retrieval and exact source citations.')
    config = dict(persist_dir=str(tmp_path / 'index'), document_paths=[str(source)],
                  backend=backend, embed_provider='fixture', reranker_type='off',
                  chunk_size=700, chunk_overlap=0)

    results = []
    startup_queries = []
    for _ in range(2):
        start = len(calls)
        pipeline = core.RAGPipeline(config)
        try:
            startup_queries.append([text for kind, text in calls[start:] if kind == 'query'])
            results.append(pipeline.search('Cedar'))
        finally:
            pipeline.close()

    assert results[0] == results[1]
    assert results[0][0]['original_content'] == source.read_text()
    assert len([call for call in calls if call[0] == 'documents']) == 1
    assert startup_queries == ([['dimension probe'], ['dimension probe']]
                               if backend == 'qdrant' else [[], []])
