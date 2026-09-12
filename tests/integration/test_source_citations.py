"""Real storage/search/reopen checks, without models or external services."""

import fitz
import pytest

from rag import fetch


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
