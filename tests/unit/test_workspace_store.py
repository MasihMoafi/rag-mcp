import json
import stat

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
    manifest = json.loads((workspace.parents[2] / "manifest.json").read_text())
    assert all("content" not in item for item in manifest["documents"])
    assert {path.read_text() for path in workspace.iterdir()} == {"# Retrieval\n\nCedar Lantern", "Friday"}
    assert stat.S_IMODE(workspace.parent.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in workspace.iterdir())

    replaced = store.replace("personal", "Renamed", [{"id": "notes", "name": "notes.txt", "source_type": "text", "content": "Monday"}], expected_revision=summary["revision"])
    assert replaced["document_count"] == 1
    assert {path.read_text() for path in store.resolve("personal").iterdir()} == {"Monday"}
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
    same = store.replace("book", "Book", documents, expected_revision=first["revision"])
    assert same == first
    assert store.resolve("book") == original_path
    second = store.replace("book", "Book", [{**documents[0], "content": "Monday"}], expected_revision=first["revision"])
    for revision in (None, first["revision"]):
        with pytest.raises(WorkspaceConflict):
            store.replace("book", "Book", [], expected_revision=revision)
    assert store.snapshot("book")["documents"][0]["content"] == "Monday"
    assert store.get("book")["revision"] == second["revision"]
    assert next(original_path.iterdir()).read_text() == "Friday", "in-flight readers keep an immutable snapshot"


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
