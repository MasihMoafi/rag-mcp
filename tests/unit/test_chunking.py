from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.core import chunk_document, chunk_text


def test_markdown_headings_become_breadcrumbs():
    text = (
        "# Title\n\nIntro paragraph that is long enough to survive the fifty "
        "character minimum chunk filter easily.\n\n"
        "## Install\n\nRun `pip install thing` and then verify it worked by "
        "checking the version output on your machine.\n\n"
        "## Usage\n\nCall the function with your query and inspect the ranked "
        "results that come back from the search call.\n"
    )
    chunks = chunk_document(text, file_path="README.md", chunk_size=700, chunk_overlap=50)

    assert chunks, "expected at least one chunk"
    headings = {c["heading"] for c in chunks}
    assert "Title > Install" in headings
    assert "Title > Usage" in headings
    assert all("content" in c for c in chunks)


def test_code_chunk_keeps_whole_function_and_reports_symbol():
    text = (
        "import os\n\n"
        "def helper_one():\n"
        "    return os.getcwd()\n\n"
        "def helper_two():\n"
        "    value = helper_one()\n"
        "    return value.upper() if value else None\n"
    )
    # chunk_size chosen so each function paragraph lands in its own chunk;
    # overlap=0 so a prior function's def-line can't leak into the next
    # chunk and confuse the nearest-symbol lookup.
    chunks = chunk_document(text, file_path="module.py", chunk_size=100, chunk_overlap=0)

    assert chunks
    two_chunk = next(c for c in chunks if "return value.upper" in c["content"])
    assert "def helper_two" in two_chunk["content"]
    assert two_chunk["heading"] == "helper_two"


def test_large_function_falls_back_to_sliding_window():
    long_body = "\n".join(f"    line_{i} = {i}" for i in range(200))
    text = f"def big_function():\n{long_body}\n"
    chunks = chunk_document(text, file_path="module.py", chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c["content"]) <= 220 for c in chunks)


def test_plain_text_without_structure_falls_back_to_chunk_text():
    text = "x" * 5000
    structured = chunk_document(text, file_path="notes.txt", chunk_size=700, chunk_overlap=100)
    raw = chunk_text(text, chunk_size=700, chunk_overlap=100)

    assert [c["content"] for c in structured] == raw
    assert all(c["heading"] == "" for c in structured)


def test_too_short_text_returns_no_chunks():
    assert chunk_document("short", file_path="a.md") == []
