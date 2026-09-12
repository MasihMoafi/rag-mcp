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


@pytest.mark.parametrize("failure", ["extraction", "embedding"])
def test_failed_refresh_preserves_previous_index_without_serving_it(tmp_path, local_search, monkeypatch, failure):
    from rag.lancedb_backend import LanceDBBackend
    document = tmp_path / "note.txt"
    document.write_text("Meeting: Friday.")
    assert local_search(document)[0]["content"] == "Meeting: Friday."
    previous = fetch.get_rag_pipeline_v2(str(document))
    manifest = next((tmp_path / "index").rglob("source-manifest.json"))
    original_manifest = manifest.read_bytes()
    document.write_text("" if failure == "extraction" else "Meeting: Monday.")
    original_add = LanceDBBackend.add_documents_batch
    if failure == "embedding":
        def fail_embedding(self, documents, batch_size=512):
            raise RuntimeError("embedding provider unavailable")
        monkeypatch.setattr(LanceDBBackend, "add_documents_batch", fail_embedding)
    error, message = (ValueError, "No readable text") if failure == "extraction" else (RuntimeError, "embedding provider unavailable")
    for _ in range(2):
        with pytest.raises(error, match=message):
            local_search(document)
        assert manifest.read_bytes() == original_manifest
        # Inspect storage directly: a failed rebuild must not destroy recovery data.
        stored = LanceDBBackend(previous.vector_db.table_name, None,
                                persist_path=previous.vector_db.persist_path)
        assert stored.count() == 1
        assert stored.search_fts("Meeting")[0]["content"] == "Meeting: Friday."
        fetch.reset_pipelines()
    monkeypatch.setattr(LanceDBBackend, "add_documents_batch", original_add)
    document.write_text("Meeting: Monday.")
    assert local_search(document)[0]["content"] == "Meeting: Monday."


def test_deleted_document_disappears_from_folder_results(tmp_path, local_search):
    folder = tmp_path / "documents"
    folder.mkdir()
    first, second = folder / "one.txt", folder / "two.txt"
    first.write_text("Meeting: Friday.")
    second.write_text("Meeting: Monday.")
    assert len(local_search(folder)) == 2
    second.unlink()
    assert [item["content"] for item in local_search(folder)] == ["Meeting: Friday."]


def test_mcp_reports_storage_failure_as_error(tmp_path, local_search, monkeypatch):
    import json
    import server
    from rag.lancedb_backend import LanceDBBackend

    document = tmp_path / "note.txt"
    document.write_text("Meeting: Friday.")
    assert local_search(document)
    def unavailable(self, query, top_k=5):
        raise RuntimeError("storage unavailable")
    monkeypatch.setattr(LanceDBBackend, "search_fts", unavailable)
    response = json.loads(server.query_knowledge_base("Meeting", str(document)))
    assert "result" not in response
    assert "storage unavailable" in response["error"]


def test_failed_fts_build_does_not_commit_manifest_and_is_retried(tmp_path, local_search, monkeypatch):
    document = tmp_path / "note.txt"
    document.write_text("Meeting: Friday.")
    assert local_search(document)
    previous = fetch.get_rag_pipeline_v2(str(document))
    manifest = next((tmp_path / "index").rglob("source-manifest.json"))
    original_manifest = manifest.read_bytes()
    table_type = type(previous.vector_db.table)
    original_build = table_type.create_fts_index
    def unavailable(*args, **kwargs):
        raise RuntimeError("keyword index build unavailable")
    document.write_text("Meeting: Monday.")
    monkeypatch.setattr(table_type, "create_fts_index", unavailable)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="keyword index build unavailable"):
            local_search(document)
        assert manifest.read_bytes() == original_manifest
        from rag.lancedb_backend import LanceDBBackend
        stored = LanceDBBackend(previous.vector_db.table_name, None,
                                persist_path=previous.vector_db.persist_path)
        assert stored.search_fts("Meeting")[0]["content"] == "Meeting: Friday."
        fetch.reset_pipelines()
    monkeypatch.setattr(table_type, "create_fts_index", original_build)
    assert local_search(document)[0]["content"] == "Meeting: Monday."
    active = fetch.get_rag_pipeline_v2(str(document))
    from pathlib import Path
    assert (Path(active.config['persist_dir']) / 'source-manifest.json').read_bytes() != original_manifest


@pytest.mark.parametrize('corruption', ['[truncated', '{}', '[{}]', '[]'])
def test_invalid_metadata_rebuilds_from_sources(tmp_path, local_search, corruption):
    import json
    document = tmp_path / 'note.txt'
    document.write_text('Meeting: Friday.')
    assert local_search(document)
    metadata = next((tmp_path / 'index').rglob('docs.json'))
    metadata.write_text(corruption)
    fetch.reset_pipelines()
    assert local_search(document)[0]['content'] == 'Meeting: Friday.'
    from pathlib import Path
    active = fetch.get_rag_pipeline_v2(str(document))
    restored = json.loads((Path(active.config['persist_dir']) / 'docs.json').read_text())
    assert isinstance(restored, list) and restored[0]['content'] == 'Meeting: Friday.'
    assert metadata.read_text() == corruption


def test_source_change_during_indexing_preserves_previous_generation(tmp_path, local_search, monkeypatch):
    from pathlib import Path
    from rag.core import RAGPipeline
    document = tmp_path / 'note.txt'
    document.write_text('Meeting: Friday.')
    assert local_search(document)[0]['content'] == 'Meeting: Friday.'
    previous = fetch.get_rag_pipeline_v2(str(document))
    pointer = Path(previous.config['index_scope']) / 'active-index.json'
    published = pointer.read_bytes()
    document.write_text('Meeting: Monday.')
    save = RAGPipeline._save_database
    def edit_after_indexing(self, docs):
        save(self, docs)
        document.write_text('Meeting: Tuesday.')
    with monkeypatch.context() as fault:
        fault.setattr(RAGPipeline, '_save_database', edit_after_indexing)
        with pytest.raises(RuntimeError, match='Sources changed'):
            local_search(document)
    assert pointer.read_bytes() == published
    assert previous.search('Meeting')[0]['content'] == 'Meeting: Friday.'
    fetch.reset_pipelines()
    assert local_search(document)[0]['content'] == 'Meeting: Tuesday.'
