"""Converter noise stays out of searchable text; cited source text stays exact."""

from rag.core import RAGPipelineV2
from rag.ingestion import build_documents
from rag.parsers.text import clean_text

NOISY_MARKDOWN = "\n".join([
    "---",
    "title: Sensor Fault Diagnosis",
    "---",
    "",
    "**Author:** Masih  ",
    "",
    "------------------------------------------------------------------------",
    "",
    "Results",
    "=======",
    "",
    "| Step        | What was done      |",
    "|:------------|-------------------:|",
    "| **1. Data** | Generate 1000 rows |",
    "",
    "\\> Given four sensors \\* healthy or faulty?",
    "",
    '<span class="math display">y = 1</span>',
    "",
    '<figure><img src="a.png" loading="lazy" alt="Feature distributions." /></figure>',
    '<div class="table-wrap" tabindex="0" aria-label="Confusion matrix counts">',
    "",
    "",
    "",
    "End.",
])


def test_markdown_noise_is_removed_and_words_kept():
    assert clean_text(NOISY_MARKDOWN, markdown=True) == "\n".join([
        "title: Sensor Fault Diagnosis",
        "",
        "**Author:** Masih",
        "",
        "# Results",
        "",
        "| Step | What was done |",
        "| **1. Data** | Generate 1000 rows |",
        "",
        "> Given four sensors * healthy or faulty?",
        "",
        "y = 1",
        "",
        "Feature distributions.",
        "Confusion matrix counts",
        "",
        "End.",
    ])


def test_plain_text_loses_decoration_without_gaining_markdown():
    raw = "# ===== NOTES =====\n# title: report\n# ====================\nSection 1------Part 2\nHeading\n-------\nText.\n\n\n\nNext."
    assert clean_text(raw, markdown=False) == "# NOTES\n# title: report\nSection 1 Part 2\nHeading\nText.\n\nNext."


def test_code_blocks_and_ordinary_text_are_unchanged():
    fenced = "Intro text.\n\n```yaml\nkey:  value\n---\npattern: \\.md$ <span>\n```\n\nA well-known range 40--50 and a -> b."
    assert clean_text(fenced, markdown=True) == fenced
    assert clean_text("One paragraph.\n\nTwo paragraphs, 1,000 rows.", markdown=False) == "One paragraph.\n\nTwo paragraphs, 1,000 rows."


def test_index_searches_clean_text_but_cites_exact_source_lines(tmp_path):
    source = tmp_path / "report.md"
    source.write_text(NOISY_MARKDOWN)
    code = tmp_path / "tool.py"
    code.write_text("x = 1  # ----\n\n\n\ny = 2\n")
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {"chunk_size": 2000, "chunk_overlap": 0}

    [document] = build_documents(pipeline, [str(source)])
    for noise in ("-----", "|:---", "<span", "\\>", "<figure"):
        assert noise not in document["content"]
    assert "Generate 1000 rows" in document["content"]
    lines = source.read_text().splitlines()
    assert document["original_content"] == "\n".join(lines[document["start_line"] - 1:document["end_line"]])

    [code_document] = build_documents(pipeline, [str(code)])
    assert code_document["content"] == code_document["original_content"] == "x = 1  # ----\n\n\n\ny = 2"
