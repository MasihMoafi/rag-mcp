import json
import stat
from pathlib import Path

import pytest

from rag.workspaces import WorkspaceConflict, WorkspaceStore


def test_workspace_replacement_is_private_atomic_and_metadata_only(tmp_path):
    store = WorkspaceStore(tmp_path / "workspaces")
    summary = store.replace(
        "personal",
        "Personal workspace",
        [
            {
                "id": "book/../chapter one",
                "name": "RAG book.md",
                "source_type": "markdown",
                "content": "# Retrieval\n\nCedar Lantern",
            },
            {
                "id": "notes",
                "name": "notes.txt",
                "source_type": "text",
                "content": "Friday",
            },
        ],
    )

    assert summary == {
        "schema_version": 1,
        "revision": summary["revision"],
        "id": "personal",
        "name": "Personal workspace",
        "document_count": 2,
        "total_bytes": len("# Retrieval\n\nCedar Lantern".encode()) + len("Friday".encode()),
        "documents": summary["documents"],
    }
    assert [item["id"] for item in summary["documents"]] == ["book/../chapter one", "notes"]
    assert all("content" not in item for item in summary["documents"])
    assert all(".." not in item["stored_file"] and "/" not in item["stored_file"] for item in summary["documents"])

    workspace = store.resolve("personal")
    manifest = json.loads((workspace.parent / "manifest.json").read_text())
    assert all("content" not in item for item in manifest["documents"])
    assert {path.read_text() for path in workspace.iterdir()} == {"# Retrieval\n\nCedar Lantern", "Friday"}
    assert stat.S_IMODE(workspace.parent.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in workspace.iterdir())

    replaced = store.replace("personal", "Renamed", [{"id": "notes", "name": "notes.txt", "source_type": "text", "content": "Monday"}], expected_revision=summary["revision"])
    assert replaced["document_count"] == 1
    assert [item["content"] for item in store.snapshot("personal")["documents"]] == ["Monday"]
    # The previous revision keeps its files for a query that started before this save.
    assert {path.read_text() for path in store.resolve("personal").iterdir()} == {
        "# Retrieval\n\nCedar Lantern", "Friday", "Monday"}
    assert store.list_workspaces() == [{"id": "personal", "name": "Renamed", "document_count": 1, "total_bytes": 6}]


@pytest.mark.parametrize("workspace_id", ["", "../escape", "a/b", "has space", "x" * 65])
def test_workspace_ids_cannot_select_paths(tmp_path, workspace_id):
    store = WorkspaceStore(tmp_path / "workspaces")
    with pytest.raises(ValueError):
        store.replace(workspace_id, "name", [])


def test_workspace_limits_are_enforced_before_publication(tmp_path):
    store = WorkspaceStore(tmp_path / "workspaces", max_document_bytes=5, max_total_bytes=8, max_documents=2)
    with pytest.raises(ValueError, match="document exceeds"):
        store.replace("personal", "name", [{"id": "one", "name": "one", "source_type": "text", "content": "123456"}])
    with pytest.raises(ValueError, match="workspace exceeds"):
        store.replace("personal", "name", [
            {"id": "one", "name": "one", "source_type": "text", "content": "12345"},
            {"id": "two", "name": "two", "source_type": "text", "content": "12345"},
        ])
    assert not (tmp_path / "workspaces" / "personal" / "manifest.json").exists()


def test_unchanged_sync_keeps_index_identity_and_stale_writers_cannot_replace(tmp_path):
    store = WorkspaceStore(tmp_path)
    documents = [{"id": "one", "name": "book.md", "source_type": "markdown", "content": "Friday"}]
    first = store.replace("book", "Book", documents)
    original_path = store.resolve("book")
    _, _, original_files = store.local_snapshot("book")
    same = store.replace("book", "Book", documents, expected_revision=first["revision"])
    assert same == first
    assert store.resolve("book") == original_path
    second = store.replace("book", "Book", [{**documents[0], "content": "Monday"}], expected_revision=first["revision"])
    for revision in (None, first["revision"]):
        with pytest.raises(WorkspaceConflict):
            store.replace("book", "Book", [], expected_revision=revision)
    assert store.snapshot("book")["documents"][0]["content"] == "Monday"
    assert store.get("book")["revision"] == second["revision"]
    assert [Path(path).read_text() for path in original_files] == ["Friday"], "in-flight readers keep their files"


def test_two_simultaneous_writers_have_exactly_one_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store = WorkspaceStore(tmp_path)
    first = store.replace("book", "Book", [])
    def replace(name):
        try:
            store.replace("book", name, [], expected_revision=first["revision"])
            return True
        except WorkspaceConflict:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(replace, ["Writer A", "Writer B"])) == [False, True]


def test_workspace_sources_can_be_read_one_page_at_a_time(tmp_path):
    store = WorkspaceStore(tmp_path)
    store.replace("book", "Book", [{"id": str(n), "name": f"{n}.txt", "source_type": "text", "content": "x"}
                                   for n in range(5)])
    first = store.describe("book", limit=2)
    assert [item["id"] for item in first["documents"]] == ["0", "1"]
    assert (first["document_count"], first["next_offset"]) == (5, 2)
    last = store.describe("book", limit=2, offset=4)
    assert [item["id"] for item in last["documents"]] == ["4"] and last["next_offset"] is None
    for limit, offset in ((0, 0), (501, 0), (2, -1), ("2", 0)):
        with pytest.raises(ValueError):
            store.describe("book", limit=limit, offset=offset)


def test_mcp_workspace_read_returns_one_page_by_default(tmp_path, monkeypatch):
    from rag import server
    store = WorkspaceStore(tmp_path)
    store.replace("book", "Book", [{"id": str(n), "name": f"{n}.txt", "source_type": "text", "content": "x"}
                                   for n in range(150)])
    monkeypatch.setenv("RAG_MCP_WORKSPACE_DIR", str(tmp_path))
    replies = []
    monkeypatch.setattr(server, "_result", lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, "_initialized", False)
    server._handle_request({"id": 0, "method": "initialize"})
    for arguments in ({"workspace_id": "book"}, {"workspace_id": "book", "offset": 100}):
        server._handle_request({"id": 1, "method": "tools/call",
                                "params": {"name": "get_workspaces", "arguments": arguments}})
    first, second = (json.loads(reply["content"][0]["text"]) for reply in replies[-2:])
    assert (len(first["documents"]), first["next_offset"]) == (100, 100)
    assert (len(second["documents"]), second["next_offset"]) == (50, None)
