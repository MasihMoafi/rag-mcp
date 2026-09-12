"""Offline characterization of the retrieval engine's returned passages."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag.core import RAGPipelineV2


@pytest.fixture
def pipeline():
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {"top_k": 7, "rerank_top_k": 2, "reranker_type": "disabled"}
    pipeline.backend_type = "lancedb"
    pipeline.embedding_fn = None
    pipeline.reranker_type = "cross-encoder"
    pipeline.reranker = None
    pipeline.vector_db = SimpleNamespace(
        search_vector=Mock(return_value=[]), search_fts=Mock(return_value=[])
    )
    return pipeline


def test_search_preserves_duplicate_content_source_identity_and_limit(pipeline, capsys):
    documents = [
        {"id": f"source-{i}", "original_content": "متن یکسان", "content": "context",
         "file_path": f"/{i}.md", "location": f"L{i + 1}"}
        for i in range(3)
    ]
    pipeline.vector_db.search_fts.return_value = documents

    result = pipeline.search("متن")

    assert result == documents[:2]
    assert result[0] is documents[0] and result[1] is documents[1]
    pipeline.vector_db.search_vector.assert_not_called()
    pipeline.vector_db.search_fts.assert_called_once_with("متن", top_k=7)
    assert capsys.readouterr().err == "[RAG V2] Returning 2 unique results\n"


def test_search_uses_both_configured_retrieval_branches(pipeline):
    dense = {"id": "dense", "content": "Dense evidence"}
    lexical = {"id": "lexical", "content": "Keyword evidence"}
    pipeline.embedding_fn = object()
    pipeline.vector_db.search_vector.return_value = [dense]
    pipeline.vector_db.search_fts.return_value = [lexical]

    assert pipeline.search("evidence") == [dense, lexical]
    pipeline.vector_db.search_vector.assert_called_once_with("evidence", top_k=7)
    pipeline.vector_db.search_fts.assert_called_once_with("evidence", top_k=7)


def test_search_empty_result_contract(pipeline):
    assert pipeline.search("missing") == "No relevant information found."


@pytest.mark.parametrize("branch", ["search_fts", "search_vector"])
def test_search_dependency_error_contract(pipeline, branch):
    pipeline.embedding_fn = object()
    getattr(pipeline.vector_db, branch).side_effect = RuntimeError("storage unavailable")
    with pytest.raises(RuntimeError, match="storage unavailable"):
        pipeline.search("evidence")
