import io
import json

from rag.cli import main
from rag.workspaces import WorkspaceStore


def test_cli_and_mcp_read_same_private_chunk_page(tmp_path, monkeypatch):
    from rag import server

    root = tmp_path / "workspaces"
    monkeypatch.setenv("RAG_MCP_WORKSPACE_DIR", str(root))
    store = WorkspaceStore(root)
    content = "😀 Cedar Lantern. " * 95
    saved = store.replace("book", "Book", [
        {"id": "chapter", "name": "chapter.txt", "source_type": "text", "content": content},
        {"id": "private", "name": "private.txt", "source_type": "text", "content": "Private passage"},
    ])
    details_out = io.StringIO()
    assert main(["workspaces", "book", "--json"], stdout=details_out,
                environ={"RAG_MCP_WORKSPACE_DIR": str(root)}) == 0
    details = json.loads(details_out.getvalue())
    assert details["revision"] == saved["revision"]
    assert [item["id"] for item in details["documents"]] == ["chapter", "private"]
    assert "stored_file" not in str(details) and "content_sha256" not in str(details)
    assert "Private passage" not in str(details)
    output, errors = io.StringIO(), io.StringIO()
    assert main(["chunks", "book", "chapter", "--revision", saved["revision"], "--limit", "1", "--offset", "1", "--json"],
                stdout=output, stderr=errors, environ={"RAG_MCP_WORKSPACE_DIR": str(root)}) == 0
    cli_page = json.loads(output.getvalue())
    assert cli_page["total_chunks"] > 1 and len(cli_page["chunks"]) == 1
    assert cli_page["chunks"][0]["start_offset"] > 500
    assert "content" not in cli_page["chunks"][0] and "heading" not in cli_page["chunks"][0]
    assert "source_content" not in cli_page
    assert "Cedar Lantern" not in output.getvalue() and "Private passage" not in output.getvalue()

    replies = []
    monkeypatch.setattr(server, "_result", lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, "_initialized", True)
    def call(**arguments):
        server._handle_request({"id": 1, "method": "tools/call", "params": {
            "name": "get_workspace_chunks", "arguments": {"workspace_id": "book", "source_id": "chapter", **arguments}}})
        return json.loads(replies[-1]["content"][0]["text"])

    mcp_page = call(revision=saved["revision"], limit=1, offset=1)
    assert mcp_page == cli_page
    assert not replies[-1]["isError"]
    server._handle_request({"id": 2, "method": "tools/call", "params": {"name": "get_workspaces",
                           "arguments": {"workspace_id": "book"}}})
    assert json.loads(replies[-1]["content"][0]["text"]) == {**details, "next_offset": None}  # MCP returns one page.
    server._handle_request({"id": 2, "method": "tools/call", "params": {"name": "get_workspaces",
                           "arguments": {"workspace_id": ""}}})
    assert replies[-1]["isError"]
    cli_content = io.StringIO()
    assert main(["chunks", "book", "chapter", "--revision", saved["revision"], "--limit", "1", "--offset", "1", "--content", "--json"],
                stdout=cli_content, environ={"RAG_MCP_WORKSPACE_DIR": str(root)}) == 0
    with_content = call(revision=saved["revision"], limit=1, offset=1, include_content=True)
    assert with_content == json.loads(cli_content.getvalue())
    assert with_content["chunks"][0]["content"] in content
    assert "source_content" not in with_content and "Private passage" not in str(with_content)
    assert main(["chunks", "book", "chapter", "--revision", "stale", "--json"],
                stdout=io.StringIO(), stderr=io.StringIO(), environ={"RAG_MCP_WORKSPACE_DIR": str(root)}) == 2
    assert "error" in call(revision="stale")
    assert "error" in call(view="unknown")
