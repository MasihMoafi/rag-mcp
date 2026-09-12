from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from rag.lancedb_backend import LanceDBBackend
from rag.qdrant_backend import QdrantVectorDB


@pytest.fixture(params=[LanceDBBackend, QdrantVectorDB])
def backend(request):
    return request.param.__new__(request.param)


def test_embedding_method_precedes_callable(backend):
    calls = []
    class Embedder:
        def embed_documents(self, texts):
            calls.append(texts)
            return [[1.0], [2.0]]
        def __call__(self, texts):
            pytest.fail('batch method already succeeded')
    backend.embedding_fn = Embedder()
    assert backend._batch_embed(['a', 'b']) == [[1.0], [2.0]]
    assert calls == [['a', 'b']]


@pytest.mark.parametrize('vector_count', [0, 1, 3])
def test_incomplete_or_extra_vectors_never_write_a_batch(backend, vector_count):
    backend.table_name = backend.collection_name = 'test'
    backend.embedding_fn = SimpleNamespace(embed_documents=lambda texts: [[1.0]] * vector_count)
    def unexpected_write(*args, **kwargs):
        pytest.fail('Mismatched embeddings reached storage')
    backend.db = SimpleNamespace(create_table=unexpected_write)
    backend.client = SimpleNamespace(upsert=unexpected_write)
    with pytest.raises(ValueError, match='Embedding count'):
        backend.add_documents_batch([{'content': 'first'}, {'content': 'second'}])


@pytest.mark.parametrize('batch_size', [0, -1])
def test_invalid_batch_size_is_rejected_before_embedding(backend, batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        backend.add_documents_batch([{'content': 'first'}], batch_size=batch_size)


def test_callable_batch_follows_method_failure(backend, capsys):
    calls = []
    class Embedder:
        def embed_documents(self, texts):
            raise RuntimeError('batch unavailable')
        def __call__(self, texts):
            calls.append(texts)
            return [[1.0], [2.0]]
    backend.embedding_fn = Embedder()
    assert backend._batch_embed(['a', 'b']) == [[1.0], [2.0]]
    assert calls == [['a', 'b']]
    logs = capsys.readouterr().err
    if isinstance(backend, LanceDBBackend):
        assert 'RuntimeError' in logs
        assert 'batch unavailable' not in logs
    else:
        assert 'batch unavailable' in logs


def test_parallel_fallback_keeps_two_workers_and_input_order(backend, monkeypatch):
    import concurrent.futures
    real_executor = concurrent.futures.ThreadPoolExecutor
    workers = []
    def executor(*args, **kwargs):
        workers.append(kwargs['max_workers'])
        return real_executor(*args, **kwargs)
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', executor)
    barrier = Barrier(2, timeout=3)
    lock = Lock()
    seen = []
    class Embedder:
        def embed_documents(self, texts):
            raise RuntimeError('batch unavailable')
        def __call__(self, texts):
            return []  # Wrong batch length must fall back to per-query embedding.
        def embed_query(self, text):
            with lock:
                seen.append(text)
            barrier.wait()
            return [float(text)]
    backend.embedding_fn = Embedder()
    assert backend._batch_embed(['0', '1', '2', '3']) == [[0.0], [1.0], [2.0], [3.0]]
    assert workers == [2]
    assert sorted(seen) == ['0', '1', '2', '3']


def test_parallel_query_failure_propagates(backend):
    def fail(text):
        raise RuntimeError('provider offline')
    backend.embedding_fn = SimpleNamespace(embed_query=fail)
    with pytest.raises(RuntimeError, match='provider offline'):
        backend._batch_embed(['a'])


def test_fts_failure_is_not_reported_as_no_matches():
    backend = LanceDBBackend.__new__(LanceDBBackend)
    class BrokenTable:
        def __len__(self):
            return 1
        def search(self, *args, **kwargs):
            raise RuntimeError('FTS unavailable')
    backend.table = BrokenTable()
    with pytest.raises(RuntimeError, match='FTS unavailable'):
        backend.search_fts('evidence')


@pytest.mark.parametrize('dense', [False, True])
@pytest.mark.parametrize('fts_failure', [False, True])
def test_lancedb_record_batches_and_fts_contract(dense, fts_failure, capsys):
    backend = LanceDBBackend.__new__(LanceDBBackend)
    backend.table_name = 'test'
    backend.distance = 'euclidean'
    batches, writes, indexes = [], [], []
    def embed(texts):
        batches.append(texts)
        return [[float(len(text))] for text in texts]
    backend.embedding_fn = SimpleNamespace(embed_documents=embed) if dense else None
    def create_fts_index(column, **kwargs):
        indexes.append((column, kwargs))
        if fts_failure:
            raise RuntimeError('FTS unavailable')
    table = SimpleNamespace(create_fts_index=create_fts_index)
    def create_table(name, **kwargs):
        writes.append((name, kwargs))
        return table
    backend.db = SimpleNamespace(create_table=create_table)
    docs = [
        dict(id='source::0', content='Cedar', file_path='source.md', location='L1',
             heading='Title', start_line=1, end_line=1, original_content='Original'),
        dict(content='فارسی', source='book.pdf', page_number=2),
        dict(content='Third'),
    ]
    if fts_failure:
        with pytest.raises(RuntimeError, match='FTS unavailable'):
            backend.add_documents_batch(docs, batch_size=2)
    else:
        backend.add_documents_batch(docs, batch_size=2)
    assert batches == ([['Cedar', 'فارسی'], ['Third']] if dense else [])
    expected = [
        dict(id='source::0', content='Cedar', file_path='source.md', source='source.md',
             heading='Title', location='L1', start_line=1, end_line=1, page_number=0,
             original_content='Original'),
        dict(id='1', content='فارسی', file_path='', source='book.pdf', heading='',
             location='', start_line=0, end_line=0, page_number=2, original_content='فارسی'),
        dict(id='2', content='Third', file_path='', source='', heading='', location='',
             start_line=0, end_line=0, page_number=0, original_content='Third'),
    ]
    if dense:
        for record in expected:
            record['vector'] = [float(len(record['content']))]
    assert writes == [('test', dict(data=expected, mode='overwrite'))]
    assert indexes == [('content', {'replace': True})]
    assert backend.table is table
    if fts_failure:
        logs = capsys.readouterr().err
        assert 'RuntimeError' in logs
        assert 'FTS unavailable' not in logs
    backend.add_documents_batch([])
    assert len(writes) == 1


def test_qdrant_payloads_keep_metadata_ids_and_batches():
    backend = QdrantVectorDB.__new__(QdrantVectorDB)
    backend.collection_name = 'test'
    embedded, writes = [], []
    def embed(texts):
        embedded.append(texts)
        return [[float(len(text))] for text in texts]
    backend.embedding_fn = SimpleNamespace(embed_documents=embed)
    backend.client = SimpleNamespace(upsert=lambda **kwargs: writes.append(kwargs))
    docs = [dict(content='Cedar', id='source::0', score=7, location='L1', custom={'x': 1}),
            dict(content='فارسی', id='source::1', page_number=2), dict(content='Third')]
    backend.add_documents_batch(docs, batch_size=2)
    assert embedded == [['Cedar', 'فارسی'], ['Third']]
    assert [write['collection_name'] for write in writes] == ['test', 'test']
    assert [len(write['points']) for write in writes] == [2, 1]
    points = [point for write in writes for point in write['points']]
    assert [point.id for point in points] == [0, 1, 2]
    assert [point.payload for point in points] == docs
    assert [point.vector for point in points] == [[5.0], [5.0], [5.0]]
    docs[0]['content'] = 'changed'
    assert points[0].payload['content'] == 'Cedar'


@pytest.mark.parametrize('api', ['search', 'query_points'])
def test_qdrant_search_preserves_payload_precedence_and_defaults(api):
    backend = QdrantVectorDB.__new__(QdrantVectorDB)
    backend.collection_name = 'test'
    backend.embedding_fn = SimpleNamespace(embed_query=lambda text: [1.0])
    calls = []
    hits = [SimpleNamespace(payload={'content': 'Cedar', 'score': 7, 'id': 'source::0'}, score=0.5),
            SimpleNamespace(payload={'location': 'L2'}, score=0.25)]
    def search(**kwargs):
        calls.append(kwargs)
        return hits if api == 'search' else SimpleNamespace(points=hits)
    backend.client = SimpleNamespace(**{api: search})
    result = backend.search('question', top_k=2)
    assert result == [dict(content='Cedar', score=7, id='source::0'), dict(content='', score=0.25, location='L2')]
    assert calls == [dict(collection_name='test', limit=2,
                          **{'query_vector' if api == 'search' else 'query': [1.0]})]
    result[0]['content'] = 'changed'
    assert hits[0].payload['content'] == 'Cedar'
