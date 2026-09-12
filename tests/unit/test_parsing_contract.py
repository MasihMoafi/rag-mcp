"""Extraction contracts without downloading optional OCR models."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import fitz
import pytest

from rag.core import RAGPipelineV2


@pytest.fixture
def pipeline():
    return RAGPipelineV2.__new__(RAGPipelineV2)


def test_plain_text_preserves_source_and_read_errors(pipeline, tmp_path, capsys):
    path = tmp_path / "source.py"
    text = "# کتاب\n\ndef run():\n    return 1\n"
    path.write_text(text)
    assert pipeline._extract_content(str(path)) == text
    with pytest.raises(FileNotFoundError):
        pipeline._extract_content(str(tmp_path / "missing"))


@pytest.mark.parametrize("data,expected", [
    ({"nested": {"x": "کتاب"}, "items": [{"n": 2}, "value", None]},
     "nested.x: کتاب\nitems.[0].n: 2\nvalue\nNone"),
    ([], ""), ("scalar", "scalar"), (False, "False"),
])
def test_json_flattening(pipeline, tmp_path, data, expected):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    assert pipeline._extract_content(str(path)) == expected


def test_malformed_json(pipeline, tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        pipeline._extract_content(str(path))


def test_named_json_reader_and_legacy_keyword_call_match(pipeline, tmp_path):
    from rag.parsers.json import extract_json_text
    path = tmp_path / 'source.json'
    path.write_text('{"content": "evidence"}')
    assert pipeline._extract_text_from_json(json_path=str(path)) == extract_json_text(str(path)) == 'content: evidence'


def test_named_notebook_reader_preserves_keyword_call(pipeline, tmp_path):
    from rag.parsers.notebook import extract_notebook_text
    path = tmp_path / 'source.ipynb'
    path.write_text('{"cells": [{"source": "evidence"}]}')
    assert pipeline._extract_text_from_notebook(notebook_path=str(path)) == extract_notebook_text(str(path)) == 'evidence'


def test_pdf_keyword_call_keeps_missing_file_contract(pipeline, tmp_path):
    with pytest.raises(fitz.FileNotFoundError):
        pipeline._extract_text_from_pdf(pdf_path=str(tmp_path / 'missing.pdf'))


def test_notebook_sources_ignore_outputs_and_preserve_cell_separators(pipeline, tmp_path):
    path = tmp_path / "cells.ipynb"
    path.write_text(json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["# Title\n", "text"]},
        {"cell_type": "code", "source": "print(1)", "outputs": [{"text": "output"}]},
        {"cell_type": "code", "source": ["  "]},
    ]}))
    assert pipeline._extract_content(str(path)) == "# Title\ntext\fprint(1)"


def test_pdf_normalization_and_page_separators(pipeline, tmp_path):
    path = tmp_path / "pages.pdf"
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "First  page\nsecond line")
        doc.new_page().insert_text((72, 72), "Next page")
        doc.save(path)
    assert pipeline._extract_content(str(path)) == "First page second line\fNext page"


def test_pdf_ligatures_expand_without_changing_mathematical_symbols(monkeypatch):
    from rag.parsers import pdf
    from types import SimpleNamespace
    class Document:
        def __iter__(self):
            return iter([SimpleNamespace(get_text=lambda: 'ﬁrst eﬀicient ﬂow ﬃ ﬄ ﬅ ﬆ x² √dk')])
        def close(self):
            pass
    monkeypatch.setattr(pdf.fitz, 'open', lambda _: Document())
    assert pdf.extract_pdf_text('ligatures.pdf') == 'first efficient flow ffi ffl st st x² √dk'


def test_pdf_document_is_closed_when_page_extraction_fails(monkeypatch):
    from rag.parsers import pdf
    closed = []
    class BrokenDocument:
        def __iter__(self):
            raise RuntimeError('damaged page')
        def close(self):
            closed.append(True)
    monkeypatch.setattr(pdf.fitz, 'open', lambda _: BrokenDocument())
    with pytest.raises(RuntimeError, match='damaged page'):
        pdf.extract_pdf_text('damaged.pdf')
    assert closed == [True]


@pytest.mark.parametrize("text,expected_calls", [("text", 0), ("", 1), (" \n", 1)])
def test_pdf_dispatch_preserves_overrides_and_fallback(pipeline, text, expected_calls):
    pipeline._extract_text_from_pdf = Mock(return_value=text)
    pipeline._extract_text_with_unstructured = Mock(return_value="OCR")
    result = pipeline._extract_content("input.PDF")
    assert result == ("OCR" if expected_calls else text)
    pipeline._extract_text_from_pdf.assert_called_once_with("input.PDF")
    assert pipeline._extract_text_with_unstructured.call_count == expected_calls


@pytest.mark.parametrize('fallback', ['', '   ', 'recovered text'])
def test_pdf_failure_requires_nonempty_fallback(pipeline, fallback):
    error = RuntimeError('damaged page')
    pipeline._extract_text_from_pdf = Mock(side_effect=error)
    pipeline._extract_text_with_unstructured = Mock(return_value=fallback)
    if fallback.strip():
        assert pipeline._extract_content('damaged.pdf') == fallback
    else:
        with pytest.raises(RuntimeError) as caught:
            pipeline._extract_content('damaged.pdf')
        assert caught.value is error
    pipeline._extract_text_with_unstructured.assert_called_once_with('damaged.pdf')


@pytest.mark.parametrize('empty', [[], ['  ']])
def test_failed_ocr_with_empty_fallback_propagates(pipeline, monkeypatch, empty):
    error = RuntimeError('OCR unavailable')
    monkeypatch.setitem(sys.modules, 'rapidocr_onnxruntime',
                        SimpleNamespace(RapidOCR=Mock(side_effect=error)))
    for name in ('unstructured', 'unstructured.partition', 'unstructured.partition.auto'):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    sys.modules['unstructured.partition.auto'].partition = Mock(return_value=empty)
    with pytest.raises(RuntimeError) as caught:
        pipeline._extract_text_with_unstructured('image.png')
    assert caught.value is error


def test_successful_empty_ocr_and_fallback_remain_empty(pipeline, monkeypatch):
    monkeypatch.setitem(sys.modules, 'rapidocr_onnxruntime',
                        SimpleNamespace(RapidOCR=lambda: Mock(return_value=([], None))))
    for name in ('unstructured', 'unstructured.partition', 'unstructured.partition.auto'):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    sys.modules['unstructured.partition.auto'].partition = Mock(return_value=[])
    assert pipeline._extract_text_with_unstructured('image.png') == ''


def test_image_ocr_short_circuits_fallback(pipeline, monkeypatch):
    engine = Mock(return_value=([[None, "first"], [None, "second"]], None))
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=lambda: engine))
    assert pipeline._extract_content("image.png") == "first\nsecond"
    engine.assert_called_once_with("image.png")


def test_optional_extractor_page_labels_and_ocr_failure(pipeline, monkeypatch, capsys):
    class Element:
        def __init__(self, text, page=None):
            self.text = text
            self.metadata = SimpleNamespace(page_number=page)

        def __str__(self):
            return self.text

    partition = Mock(return_value=[Element(" title ", 2), Element("  "), Element("body")])
    for name in ("unstructured", "unstructured.partition", "unstructured.partition.auto"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["unstructured.partition.auto"].partition = partition
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=Mock(side_effect=RuntimeError("OCR failed"))))
    assert pipeline._extract_content("image.png") == "[Page 2] title\n\nbody"
    partition.assert_called_once_with(filename="image.png")
    assert "[OCR] Warning" in capsys.readouterr().out
    partition.side_effect = RuntimeError("parser failed")
    with pytest.raises(RuntimeError, match="parser failed"):
        pipeline._extract_content("document.docx")
    with pytest.raises(RuntimeError, match="parser failed"):
        pipeline._extract_content("image.png")
