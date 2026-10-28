from types import SimpleNamespace

import pytest

from rag.parsers import pdf
from rag.chunking import chunk_document


def page(*blocks, size=10.5):
    """A page as PyMuPDF's 'dict' extraction returns it: blocks of lines in the body of the page."""
    data = {'height': 600, 'blocks': [{'type': 0, 'lines': [
        {'bbox': (60, 100 + 14 * i, 360, 112 + 14 * i), 'spans': [{'text': line, 'size': size}]}
        for i, line in enumerate(block)]} for block in blocks]}
    return SimpleNamespace(get_text=lambda kind='text': data, rect=SimpleNamespace(height=600))


class Document(list):
    def close(self):
        pass


@pytest.mark.parametrize('label', ['Figure 4:', 'Fig. 2.', 'Table 3:'])
def test_pdf_caption_is_separate_without_discarding_diagram_text(label, monkeypatch):
    lines = ['Repeated diagram labels and tokens.', f'{label} This caption explains the diagram.']
    monkeypatch.setattr(pdf.pymupdf, 'open', lambda path: Document([page(lines)]))
    text = pdf.extract_pdf_text('example.pdf')
    chunks = chunk_document(text, 'example.pdf', 700, 100)
    assert [c['content'] for c in chunks] == lines
    assert ' '.join(text.split()) == ' '.join(' '.join(lines).split())
    assert all(c['content'] in text for c in chunks)


def test_inline_figure_reference_does_not_create_boundary(monkeypatch):
    raw = 'We discuss Figure 4: the attention example in the next section.'
    monkeypatch.setattr(pdf.pymupdf, 'open', lambda path: Document([page([raw])]))
    assert pdf.extract_pdf_text('example.pdf') == raw


def test_caption_does_not_absorb_next_pages_diagram_labels(monkeypatch):
    monkeypatch.setattr(pdf.pymupdf, 'open', lambda path: Document([page(['Figure 4: Caption.']), page(['Next diagram labels.'])]))
    text = pdf.extract_pdf_text('example.pdf')
    assert text.count('\f') == 1
    assert [c['content'] for c in chunk_document(text, 'example.pdf')] == [
        'Figure 4: Caption.', 'Next diagram labels.']
