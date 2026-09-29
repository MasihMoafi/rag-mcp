from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.core import chunk_document, chunk_text
from rag.chunking import chunk_document_spans


def test_structural_spans_are_exact_even_when_chunks_repeat():
    for name, source in [
        ("repeat.txt", "😀 Cedar Lantern. " * 95),
        ("chapter.md", "# First\n\n" + "😀 Cedar Lantern. " * 95 + "\n## Second\n\nFinish."),
        ("book.pdf", "😀 Cedar Lantern. " * 95 + "\n\nSecond page."),
    ]:
        spans = chunk_document_spans(source, name)
        assert [{"content": item["content"], "heading": item["heading"]} for item in spans] == chunk_document(source, name)
        assert all(source[item["start_offset"]:item["end_offset"]] == item["content"] for item in spans)
        assert spans[1]["start_offset"] > 500


def test_markdown_code_fences_do_not_create_fake_headings():
    text = '# Guide\n\n```python\n# a code comment\nprint(1)\n```\n\n## Usage\nRun it.'
    chunks = chunk_document(text, file_path='guide.md')
    assert [chunk['heading'] for chunk in chunks] == ['Guide', 'Guide > Usage']
    assert '# a code comment\nprint(1)' in chunks[0]['content']
    assert all(chunk['content'] in text for chunk in chunks)


def test_long_tilde_fence_is_not_closed_by_shorter_fence():
    text = '# Guide\n~~~~text\n## example\n~~~\n## still example\n~~~~\n## Real\nEnd.'
    chunks = chunk_document(text, file_path='guide.md')
    assert [chunk['heading'] for chunk in chunks] == ['Guide', 'Guide > Real']


def test_unclosed_and_mismatched_fences_keep_examples_inside_section():
    for false_close in ['~~~', '```not a closing fence', '``']:
        text = '# Guide\n```python\n# comment\n' + false_close + '\n## still code'
        chunks = chunk_document(text, file_path='guide.md')
        assert chunks == [{'content': text, 'heading': 'Guide'}]


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


def test_short_text_is_searchable_but_whitespace_is_not():
    assert chunk_document("short", file_path="a.md") == [{"content": "short", "heading": ""}]
    assert chunk_document("  \n\t", file_path="a.md") == []
