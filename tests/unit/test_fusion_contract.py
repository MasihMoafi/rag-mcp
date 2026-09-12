"""Rank fusion preserves source identity and deterministic ordering."""

import pytest

from rag.core import RAGPipelineV2


def fuse(vector, lexical, weights=None, k=60):
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {} if weights is None else {"ensemble_weights": weights}
    return pipeline._rrf_fusion(vector, lexical, k=k)


def test_shared_id_keeps_first_vector_record_and_combines_ranks():
    first = {"id": "shared", "content": "vector record"}
    replacement = {"id": "shared", "content": "lexical record"}
    other = {"id": "other"}
    result = fuse([(first, 999), (other, 0)], [(replacement, -999)])
    assert result == [first, other]
    assert result[0] is first


def test_duplicate_id_uses_last_rank_but_first_record():
    first = {"id": "duplicate"}
    later = {"id": "duplicate", "content": "later"}
    middle = {"id": "middle"}
    for vector in (True, False):
        ranked = [(first, 0), (middle, 0), (later, 0)]
        result = fuse(ranked if vector else [], [] if vector else ranked)
        assert result == [middle, first]
        assert result[1] is first


@pytest.mark.parametrize("weights, expected", [
    (None, ["vector", "lexical"]),
    ([0, 1], ["lexical"]),
    ([1, 0], ["vector"]),
    ([0, 0], []),
    ([-1, 1], ["lexical"]),
    ([1, 2], ["lexical", "vector"]),
])
def test_weights_filter_nonpositive_scores_and_preserve_ties(weights, expected):
    result = fuse([({"id": "vector"}, 100)], [({"id": "lexical"}, -100)], weights)
    assert [doc["id"] for doc in result] == expected


def test_missing_ids_use_object_identity_not_content():
    first, second = {"content": "same"}, {"content": "same"}
    result = fuse([(first, 0), (second, 0)], [(first, 0)])
    assert len(result) == 2
    assert result[0] is first and result[1] is second


def test_empty_results():
    assert fuse([], []) == []
