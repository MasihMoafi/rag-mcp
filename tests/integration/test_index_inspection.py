"""Published workspace chunks are inspectable without reopening a pipeline."""

import json
import io
import shutil
import pytest
from threading import Thread
from urllib.request import Request, urlopen

from rag import cli, fetch, server
from rag.companion import create_companion_server
from rag.execution import execute_run
from rag.index_inspection import inspect_published_chunks
from rag.workspaces import WorkspaceStore


def test_published_chunks_follow_default_shared_query_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MCP_WORKSPACE_DIR", str(tmp_path / "workspaces"))
    monkeypatch.setenv("RAG_MCP_DATA_DIR", str(tmp_path / "indexes"))
    monkeypatch.setenv("RAG_MCP_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("RAG_MCP_EMBED_PROVIDER", "none")
    monkeypatch.setenv("RAG_MCP_RERANKER_TYPE", "off")
    monkeypatch.setenv("RAG_MCP_CONTEXTUAL_RETRIEVAL", "off")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    store = WorkspaceStore(tmp_path / "workspaces")
    saved = store.replace("book", "Book", [
        {"id": "chapter", "name": "chapter.txt", "source_type": "text", "content": "Cedar Lantern meeting Friday."},
        {"id": "other", "name": "other.txt", "source_type": "text", "content": "Ocean marker."},
    ])
    try:
        run = execute_run("When is the Cedar Lantern meeting?", workspace_id="book", mode="retrieve", client="cli")
        assert run["status"] == "success"
        view = inspect_published_chunks(store, "book", "chapter", saved["revision"])
        assert view["status"] == "current" and view["total_chunks"] == 1
        other = inspect_published_chunks(store, "book", "other", saved["revision"])
        assert other["status"] == "current" and other["total_chunks"] == 1
    finally:
        fetch.reset_pipelines()


def test_published_chunks_follow_explicit_source_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MCP_WORKSPACE_DIR", str(tmp_path / "workspaces"))
    monkeypatch.setenv("RAG_MCP_DATA_DIR", str(tmp_path / "indexes"))
    monkeypatch.setenv("RAG_MCP_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("RAG_MCP_EMBED_PROVIDER", "none")
    monkeypatch.setenv("RAG_MCP_RERANKER_TYPE", "off")
    monkeypatch.setenv("RAG_MCP_CONTEXTUAL_RETRIEVAL", "off")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    store = WorkspaceStore(tmp_path / "workspaces")
    saved = store.replace("book", "Book", [
        {"id": "chapter", "name": "chapter.txt", "source_type": "text", "content": "Cedar Lantern meeting Friday."},
        {"id": "other", "name": "other.txt", "source_type": "text", "content": "Ocean marker."},
    ])
    try:
        run = execute_run("Cedar Lantern", workspace_id="book", source_ids=["chapter"], mode="retrieve", client="cli")
        assert run["status"] == "success"
        assert inspect_published_chunks(store, "book", "chapter", saved["revision"])["status"] == "not_indexed"
        selected = inspect_published_chunks(store, "book", "chapter", saved["revision"], scope_source_ids=["chapter"])
        assert selected["status"] == "current" and selected["total_chunks"] == 1
        assert selected["scope_source_ids"] == ["chapter"]
        with pytest.raises(ValueError, match="Selected source scope"):
            inspect_published_chunks(store, "book", "chapter", saved["revision"], scope_source_ids=["other"])
        with pytest.raises(ValueError, match="Invalid selected source scope"):
            inspect_published_chunks(store, "book", "chapter", saved["revision"], scope_source_ids=["chapter", "chapter"])

        cli_output = io.StringIO()
        assert cli.main(["chunks", "book", "chapter", "--view", "indexed", "--scope-source", "chapter", "--json"],
                        stdout=cli_output) == 0
        assert json.loads(cli_output.getvalue()) == selected
        assert cli.main(["chunks", "book", "chapter", "--scope-source", "chapter"],
                        stdout=io.StringIO(), stderr=io.StringIO()) == 2

        replies = []
        monkeypatch.setattr(server, "_result", lambda _request_id, result: replies.append(result))
        monkeypatch.setattr(server, "_initialized", True)
        server._handle_request({"id": 1, "method": "tools/call", "params": {"name": "get_workspace_chunks",
            "arguments": {"workspace_id": "book", "source_id": "chapter", "view": "indexed",
                          "scope_source_ids": ["chapter"]}}})
        assert json.loads(replies[-1]["content"][0]["text"]) == selected
        server._handle_request({"id": 2, "method": "tools/call", "params": {"name": "get_workspace_chunks",
            "arguments": {"workspace_id": "book", "source_id": "chapter", "scope_source_ids": ["chapter"]}}})
        assert replies[-1]["isError"]

        gateway = create_companion_server(token="fixture", workspace_store=store,
                                          metrics_path=tmp_path / "metrics.json")
        worker = Thread(target=gateway.serve_forever, daemon=True)
        worker.start()
        try:
            url = (f"http://127.0.0.1:{gateway.server_port}/v1/workspaces/book/chunks"
                   f"?document_id=chapter&revision={saved['revision']}&view=indexed&scope_source_id=chapter")
            with urlopen(Request(url, headers={"Authorization": "Bearer fixture",
                                               "Origin": "http://localhost:3000"}), timeout=3) as response:
                assert json.load(response) == selected
            from urllib.error import HTTPError
            with pytest.raises(HTTPError) as rejected:
                urlopen(Request(url.replace("view=indexed&", ""), headers={"Authorization": "Bearer fixture",
                                                                    "Origin": "http://localhost:3000"}), timeout=3)
            assert rejected.value.code == 400
        finally:
            gateway.shutdown()
            gateway.server_close()
            worker.join(timeout=2)
    finally:
        fetch.reset_pipelines()


def test_published_chunks_are_source_scoped_model_free_and_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MCP_DATA_DIR", str(tmp_path / "indexes"))
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("RAG_MCP_EMBED_PROVIDER", "none")
    monkeypatch.setenv("RAG_MCP_CONTEXTUAL_RETRIEVAL", "off")
    store = WorkspaceStore(tmp_path / "workspaces")
    content = "😀 Cedar Lantern. " * 95
    saved = store.replace("book", "Book", [
        {"id": "chapter", "name": "chapter.txt", "source_type": "text", "content": content},
        {"id": "private", "name": "private.txt", "source_type": "text", "content": "Private passage"},
    ])
    revision = saved["revision"]
    absent = inspect_published_chunks(store, "book", "chapter", revision)
    assert absent["status"] == "not_indexed" and absent["chunks"] == []
    assert not (tmp_path / "indexes").exists()

    fetch.get_rag_pipeline_v2(str(store.resolve("book")))
    fetch.reset_pipelines()
    monkeypatch.setattr(fetch, "RAGPipelineV2", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pipeline opened")))

    page = inspect_published_chunks(store, "book", "chapter", revision, limit=1, offset=1)
    assert page["status"] == "current" and page["total_chunks"] > 1
    assert page["metadata_kind"] == "published_documents_unverified_storage"
    assert len(page["chunks"]) == 1 and page["chunks"][0]["start_offset"] > 500
    assert "content" not in str(page["chunks"]) and "heading" not in str(page["chunks"])
    assert "Private passage" not in str(page) and str(store.root) not in str(page)
    explicit = inspect_published_chunks(store, "book", "chapter", revision, limit=1, offset=1, include_content=True)
    chunk = explicit["chunks"][0]
    assert chunk["original_content"] in content
    assert content.encode("utf-16-le")[2 * chunk["start_offset"]:2 * chunk["end_offset"]].decode("utf-16-le") == chunk["original_content"]
    assert "Private passage" not in str(explicit)

    monkeypatch.setenv("RAG_MCP_WORKSPACE_DIR", str(store.root))
    cli_output = io.StringIO()
    assert cli.main(["chunks", "book", "chapter", "--revision", revision, "--view", "indexed",
                     "--limit", "1", "--offset", "1", "--json"], stdout=cli_output) == 0
    assert json.loads(cli_output.getvalue()) == page
    replies = []
    monkeypatch.setattr(server, "_result", lambda _request_id, result: replies.append(result))
    monkeypatch.setattr(server, "_initialized", True)
    server._handle_request({"id": 1, "method": "tools/call", "params": {"name": "get_workspace_chunks",
        "arguments": {"workspace_id": "book", "source_id": "chapter", "revision": revision,
                      "view": "indexed", "limit": 1, "offset": 1}}})
    assert json.loads(replies[-1]["content"][0]["text"]) == page

    gateway = create_companion_server(token="fixture", workspace_store=store,
                                      metrics_path=tmp_path / "metrics.json")
    worker = Thread(target=gateway.serve_forever, daemon=True)
    worker.start()
    try:
        url = (f"http://127.0.0.1:{gateway.server_port}/v1/workspaces/book/chunks"
               f"?document_id=chapter&revision={revision}&view=indexed&limit=1&offset=1")
        with urlopen(Request(url, headers={"Authorization": "Bearer fixture",
                                           "Origin": "http://localhost:3000"}), timeout=3) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response) == page
        with urlopen(Request(url + "&include_content=true", headers={"Authorization": "Bearer fixture",
                                                                  "Origin": "http://localhost:3000"}), timeout=3) as response:
            assert json.load(response) == explicit
    finally:
        gateway.shutdown()
        gateway.server_close()
        worker.join(timeout=2)

    monkeypatch.setitem(fetch.RAG_CONFIG_V2, "chunk_size", 123)
    assert inspect_published_chunks(store, "book", "chapter", revision)["status"] == "stale"
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, "chunk_size", 700)

    scope = next((tmp_path / "indexes").rglob("active-index.json")).parent
    published = json.loads((scope / "active-index.json").read_text())
    docs = scope / published / "docs.json"
    records = json.loads(docs.read_text())
    docs.write_text(json.dumps(records[:-1]))
    assert inspect_published_chunks(store, "book", "chapter", revision)["status"] == "unavailable"
    docs.write_text(json.dumps(records))
    for record in records:
        record.pop("start_offset", None)
        record.pop("end_offset", None)
    docs.write_text(json.dumps(records))
    legacy = inspect_published_chunks(store, "book", "chapter", revision)
    assert legacy["status"] == "current" and legacy["chunks"][0]["start_offset"] is None
    records[0]["start_offset"], records[0]["end_offset"] = 0, 2
    docs.write_text(json.dumps(records))
    bad_offset = inspect_published_chunks(store, "book", "chapter", revision)
    assert bad_offset["chunks"][0]["start_offset"] is None
    docs.write_text("invalid json")
    unavailable = inspect_published_chunks(store, "book", "chapter", revision)
    assert unavailable["status"] == "unavailable" and unavailable["chunks"] == []
    docs.write_text(json.dumps(records))
    shutil.rmtree(scope / published / "lancedb_storage")
    assert inspect_published_chunks(store, "book", "chapter", revision)["status"] == "unavailable"
