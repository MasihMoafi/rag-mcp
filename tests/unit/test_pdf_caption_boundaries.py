from types import SimpleNamespace

import pytest

from rag.parsers import pdf
from rag.chunking import chunk_document


@pytest.mark.parametrize('label', ['Figure 4:', 'Fig. 2.', 'Table 3:'])
def test_pdf_caption_is_separate_without_discarding_diagram_text(label, monkeypatch):
    raw = f'Repeated diagram labels and tokens.\n{label} This caption explains the diagram.\n'
    document = [SimpleNamespace(get_text=lambda: raw)]
    class Document(list):
        def close(self):
            pass
    monkeypatch.setattr(pdf.pymupdf, 'open', lambda path: Document(document))
    text = pdf.extract_pdf_text('example.pdf')
    chunks = chunk_document(text, 'example.pdf', 700, 100)
    assert [c['content'] for c in chunks] == [
        'Repeated diagram labels and tokens.', f'{label} This caption explains the diagram.']
    assert ' '.join(text.split()) == ' '.join(raw.split())
    assert all(c['content'] in text for c in chunks)


def test_inline_figure_reference_does_not_create_boundary(monkeypatch):
    raw = 'We discuss Figure 4: the attention example in the next section.'
    class Document(list):
        def close(self):
            pass
    monkeypatch.setattr(pdf.pymupdf, 'open', lambda path: Document([SimpleNamespace(get_text=lambda: raw)]))
    assert pdf.extract_pdf_text('example.pdf') == raw


def test_caption_does_not_absorb_next_pages_diagram_labels(monkeypatch):
    class Document(list):
        def close(self):
            pass
    pages = [SimpleNamespace(get_text=lambda: 'Figure 4: Caption.\n'),
             SimpleNamespace(get_text=lambda: 'Next diagram labels.\n')]
    monkeypatch.setattr(pdf.pymupdf, 'open', lambda path: Document(pages))
    text = pdf.extract_pdf_text('example.pdf')
    assert text.count('\f') == 1
    assert [c['content'] for c in chunk_document(text, 'example.pdf')] == [
        'Figure 4: Caption.', 'Next diagram labels.']
