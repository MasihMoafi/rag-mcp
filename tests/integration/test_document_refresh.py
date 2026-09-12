"""Regressions for stale and missing passages found in the manual CLI review."""

import os

import pytest

from rag import fetch


@pytest.fixture
def local_search(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MCP_DATA_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("RAG_MCP_EMBED_PROVIDER", "none")
    monkeypatch.setenv("RAG_MCP_RERANKER_TYPE", "off")
    monkeypatch.setenv("RAG_MCP_RERANK_TOP_K", "5")
    fetch.reset_pipelines()
    yield lambda path: fetch.fetchExternalKnowledge("Meeting", str(path))
    fetch.reset_pipelines()


def test_short_file_refreshes_in_memory_and_after_reopening(tmp_path, local_search):
    document = tmp_path / "note.txt"
    document.write_text("Meeting: Friday.")
    assert local_search(document)[0]["content"] == "Meeting: Friday."
    original_stat = document.stat()
    document.write_text("Meeting: Monday.")
    os.utime(document, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert local_search(document)[0]["content"] == "Meeting: Monday."
    fetch.reset_pipelines()
    document.write_text("Meeting: Sunday.")
    assert local_search(document)[0]["content"] == "Meeting: Sunday."


def test_emptied_file_cannot_return_old_passages(tmp_path, local_search):
    document = tmp_path / "note.txt"
    document.write_text("Meeting: Friday.")
    assert local_search(document)
    document.write_text("")
    with pytest.raises(ValueError, match="No readable text"):
        local_search(document)


def test_deleted_document_disappears_from_folder_results(tmp_path, local_search):
    folder = tmp_path / "documents"
    folder.mkdir()
    first, second = folder / "one.txt", folder / "two.txt"
    first.write_text("Meeting: Friday.")
    second.write_text("Meeting: Monday.")
    assert len(local_search(folder)) == 2
    second.unlink()
    assert [item["content"] for item in local_search(folder)] == ["Meeting: Friday."]
