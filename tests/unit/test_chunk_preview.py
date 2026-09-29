import pytest

import subprocess
import sys
from pathlib import Path

from rag.chunking import chunk_document
from rag.chunk_preview import preview_workspace_chunks
from rag.workspaces import WorkspaceConflict, WorkspaceStore


def test_cold_chunk_inspection_does_not_import_the_retrieval_pipeline(tmp_path):
    script = '''
import sys
from rag.workspaces import WorkspaceStore
from rag.chunk_preview import preview_workspace_chunks
from rag.index_inspection import inspect_published_chunks
store = WorkspaceStore(sys.argv[1])
saved = store.replace('book', 'Book', [{'id': 'one', 'name': 'one.txt', 'source_type': 'text', 'content': 'Friday'}])
preview_workspace_chunks(store, 'book', 'one', saved['revision'])
assert inspect_published_chunks(store, 'book', 'one', saved['revision'])['status'] == 'not_indexed'
assert not {'rag.fetch', 'rag.core', 'torch'}.intersection(sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_preview_matches_engine_chunker_and_utf16_source_offsets(tmp_path):
    store = WorkspaceStore(tmp_path)
    content = "# First\n\n" + "😀 Cedar Lantern. " * 95 + "\n## Second\n\nCedar Lantern."
    saved = store.replace("book", "Book", [
        {"id": "chapter", "name": "book.md", "source_type": "markdown", "content": content},
        {"id": "secret", "name": "private.txt", "source_type": "text", "content": "Private passage"},
    ])
    result = preview_workspace_chunks(store, "book", "chapter", saved["revision"], limit=2, offset=1)
    expected = chunk_document(content, file_path="book.md", chunk_size=result["chunk_size"], chunk_overlap=result["chunk_overlap"])
    assert result["total_chunks"] == len(expected)
    assert [(item["content"], item["heading"]) for item in result["chunks"]] == [
        (item["content"], item["heading"]) for item in expected[1:3]
    ]
    assert result["source_content"] == content
    assert "Private passage" not in str(result)
    for item in result["chunks"]:
        utf16 = content.encode("utf-16-le")
        selected = utf16[2 * item["start_offset"]:2 * item["end_offset"]].decode("utf-16-le")
        assert selected == item["content"]
    assert result["chunks"][0]["start_offset"] > 500, "repeated text must locate the actual second source slice"
    assert result["chunks"][1]["start_offset"] > result["chunks"][0]["start_offset"]


def test_preview_rejects_stale_revision_missing_document_and_invalid_page(tmp_path):
    store = WorkspaceStore(tmp_path)
    saved = store.replace("book", "Book", [{"id": "one", "name": "one", "source_type": "text", "content": "Friday"}])
    with pytest.raises(WorkspaceConflict):
        preview_workspace_chunks(store, "book", "one", "old")
    with pytest.raises(FileNotFoundError):
        preview_workspace_chunks(store, "book", "other", saved["revision"])
    for limit, offset in [(0, 0), (101, 0), (1, -1), (True, 0)]:
        with pytest.raises(ValueError):
            preview_workspace_chunks(store, "book", "one", saved["revision"], limit=limit, offset=offset)
