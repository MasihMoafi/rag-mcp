import math

import pytest

from rag.core import BM25Index


def test_term_frequency_document_frequency_and_incremental_rebuild():
    index = BM25Index()
    first = {"id": "first", "content": "Cedar cedar کتاب"}
    second = {"id": "second", "content": "کتاب"}
    index.add_document(first)
    index.add_document(second)
    result = index.search("CEDAR", k=2)
    expected_score = math.log(2) * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / 2) + 1e-9)
    assert result == [(first, math.exp(-0.1 * expected_score))]
    assert index._doc_freqs == {"cedar": 1, "کتاب": 2}
    index.add_document({"id": "third", "content": "cedar"})
    updated = index.search("cedar", k=3)
    assert [doc["id"] for doc, _ in updated] == ["third", "first"]
    assert updated[1][1] != result[0][1]


def test_custom_tokenizer_and_stable_ties():
    index = BM25Index(tokenizer=lambda text: text.split("|"))
    docs = [{"id": i, "content": "A|B"} for i in range(3)]
    for doc in docs:
        index.add_document(doc)
    assert [doc for doc, _ in index.search("A", k=2)] == docs[:2]
    assert index.search("a", k=2) == []


@pytest.mark.parametrize("document,error", [(None, TypeError), ({}, ValueError), ({"content": 3}, TypeError)])
def test_invalid_documents(document, error):
    with pytest.raises(error):
        BM25Index().add_document(document)


def test_empty_and_invalid_query_contracts():
    index = BM25Index()
    assert index.search("anything", k=0) == []
    index.add_document({"content": ""})
    assert index.search("anything") == []
    with pytest.raises(ValueError):
        index.search("anything", k=0)
    index.add_document({"content": "word"})
    assert index.search("!!!") == []


def test_sparse_search_scores_only_matching_documents(monkeypatch):
    index = BM25Index()
    for i in range(100):
        index.add_document({'id': i, 'content': f'common term{i}'})
    original = index._compute_bm25_score
    visited = []
    def score(tokens, position):
        visited.append(position)
        return original(tokens, position)
    monkeypatch.setattr(index, '_compute_bm25_score', score)
    assert [doc['id'] for doc, _ in index.search('term8 term8 missing', k=5)] == [8]
    assert visited == [8]


def test_postings_match_exhaustive_ranking_after_incremental_additions():
    index = BM25Index()
    for count in (17, 53):
        for i in range(count):
            index.add_document({'id': len(index.documents), 'content': f'common group{i % 7} ' * (i % 4 + 1)})
        for query in ('group2 group2 common', 'group3 group6', 'missing', 'common'):
            for k in (1, 5, 100):
                actual = index.search(query, k=k)
                scores = [(index._compute_bm25_score(index._tokenizer(query), i), doc)
                          for i, doc in enumerate(index.documents)]
                scores = sorted((x for x in scores if x[0] > 1e-9), key=lambda x: x[0], reverse=True)
                assert actual == [(doc, math.exp(-0.1 * score)) for score, doc in scores[:k]]
