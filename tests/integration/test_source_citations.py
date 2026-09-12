"""Real storage/search/reopen checks, without models or external services."""

import fitz
import pytest
import json
from pathlib import Path

from rag import fetch
from rag.core import BM25Index


@pytest.fixture(autouse=True)
def isolated_index(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MCP_DATA_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("RAG_MCP_EMBED_PROVIDER", "none")
    monkeypatch.setenv("RAG_MCP_RERANKER_TYPE", "off")
    monkeypatch.setenv("RAG_MCP_CONTEXTUAL_RETRIEVAL", "off")
    fetch.reset_pipelines()
    yield
    fetch.reset_pipelines()


def test_markdown_citations_survive_search_and_reopening(tmp_path):
    source = tmp_path / "chapter.md"
    source.write_text("Preamble.\n\n# Chapter\n\n## Retrieval\n\nCedar retrieval preserves original evidence.\n")
    for _ in range(2):
        result = fetch.fetchExternalKnowledge("Cedar", str(source))[0]
        assert result["file_path"] == str(source)
        assert result["heading"] == "Chapter > Retrieval"
        assert result["location"] == "L5-L7"
        assert (result["start_line"], result["end_line"]) == (5, 7)
        assert result["original_content"] in source.read_text()
        fetch.reset_pipelines()


def test_same_named_sources_keep_distinct_ids_across_storage_and_fusion(tmp_path):
    corpus = tmp_path / "corpus"
    for name, detail in [("alpha", "Cedar grows in the north."), ("beta", "Cedar grows in the south.")]:
        directory = corpus / name
        directory.mkdir(parents=True)
        (directory / "notes.txt").write_text(detail)
    for _ in range(2):
        pipeline = fetch.get_rag_pipeline_v2([str(path) for path in sorted(corpus.rglob("*.txt"))])
        # Qdrant's lexical arm is rebuilt from this same persisted source cache;
        # LanceDB itself uses its FTS index instead of the in-memory BM25 class.
        documents = json.loads((Path(pipeline.config["persist_dir"]) / "docs.json").read_text())
        lexical_index = BM25Index()
        for document in documents:
            lexical_index.add_document(document)
        lexical = lexical_index.search("Cedar", k=5)
        stored = pipeline.vector_db.search_fts("Cedar", top_k=5)
        assert len({doc["id"] for doc, _score in lexical}) == 2
        assert {doc["id"] for doc in stored} == {doc["id"] for doc, _score in lexical}
        # Feed real stored documents as the dense arm, with controlled ranks;
        # no model is needed to verify fusion identity and score accumulation.
        fused = pipeline._rrf_fusion([(doc, 0.1) for doc in stored], lexical)
        assert len(fused) == 2
        assert len({doc["file_path"] for doc in fused}) == 2
        fetch.reset_pipelines()


def test_pdf_page_citation_survives_search_and_reopening(tmp_path):
    source = tmp_path / "chapter.pdf"
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Introduction.")
        pdf.new_page().insert_text((72, 72), "\n".join(
            ["General background for this chapter."] * 25
            + ["Cedar retrieval preserves source evidence."]
        ))
        pdf.save(source)
    for _ in range(2):
        result = fetch.fetchExternalKnowledge("Cedar", str(source))[0]
        assert result["page_number"] == 2
        assert result["location"] == "p.2"
        assert result["file_path"] == str(source)
        fetch.reset_pipelines()
