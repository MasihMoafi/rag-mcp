"""PDFs are parsed into paragraphs and chapter headings, without running headers or page numbers, and
book-length documents are chunked in wide windows (Masih, 28 Oct 2026: "fixed chunking seldom provides the
results we want unless it's parsed properly")."""
import pymupdf
import pytest

from rag.chunking import chunk_document
from rag.parsers.pdf import extract_pdf_text

BODY = 10.5


def make_pdf(path, pages):
    """pages: lists of (kind, text). Kinds: 'header', 'heading', 'para', 'number'."""
    doc = pymupdf.open()
    for page_items in pages:
        page = doc.new_page(width=420, height=600)
        y = 70
        for kind, text in page_items:
            if kind == 'header':
                page.insert_text((60, 30), text, fontsize=8)
            elif kind == 'number':
                page.insert_text((200, 580), text, fontsize=8)
            elif kind == 'heading':
                page.insert_textbox(pymupdf.Rect(60, y, 360, y + 40), text, fontsize=18, fontname='hebo')
                y += 50
            else:
                rect = pymupdf.Rect(60, y, 360, y + 120)
                page.insert_textbox(rect, text, fontsize=BODY)
                y += 140
    doc.save(path)
    doc.close()
    return str(path)


P1 = ('The emperor rose before dawn and read every dispatch himself, marking the margins in a hand '
      'no secretary could follow without long practice.')
P2 = ('By noon the orders had gone out to every corps, and the staff knew better than to ask him to '
      'repeat a single figure he had already given.')


def test_paragraphs_survive_parsing(tmp_path):
    text = extract_pdf_text(make_pdf(tmp_path / 'p.pdf', [[('para', P1), ('para', P2)]]))
    assert ' '.join(P1.split()) in ' '.join(text.split())
    assert '\n\n' in text.strip()  # the two paragraphs stay apart
    first, second = [part for part in text.split('\n\n') if part.strip()][:2]
    assert '\n' not in first.strip() and 'dawn' in first and 'noon' in second  # lines inside a paragraph are joined


def test_running_headers_and_page_numbers_are_dropped(tmp_path):
    pages = [[('header', 'NAPOLEON IN HIS OWN WORDS'), ('para', f'Body of page {i}, which is unique. ' + P1), ('number', str(i + 11))]
             for i in range(6)]
    text = extract_pdf_text(make_pdf(tmp_path / 'h.pdf', pages))
    assert 'NAPOLEON IN HIS OWN WORDS' not in text
    assert not any(line.strip() == str(n) for line in text.splitlines() for n in range(11, 17))
    assert all(f'Body of page {i}' in text for i in range(6))
    assert text.count('\f') == 5  # page markers stay for page citations


def test_chapter_headings_become_sections_with_breadcrumbs(tmp_path):
    pages = [[('heading', 'Chapter One: The Army of Italy'), ('para', P1), ('para', P2)],
             [('heading', 'Chapter Two: Egypt'), ('para', 'The fleet sailed from Toulon in May with the savants aboard.')]]
    text = extract_pdf_text(make_pdf(tmp_path / 'c.pdf', pages))
    assert '# Chapter One: The Army of Italy' in text and '# Chapter Two: Egypt' in text
    chunks = chunk_document(text, 'c.pdf', 700, 100)
    egypt = next(c for c in chunks if 'Toulon' in c['content'])
    assert egypt['heading'] == 'Chapter Two: Egypt'
    assert all('Toulon' not in c['content'] for c in chunks if c['heading'].startswith('Chapter One'))  # no chunk spans two chapters


def test_captions_stay_their_own_chunk(tmp_path):
    pages = [[('para', P1), ('para', 'Figure 4: The attention weights for one head.'), ('para', P2)]]
    text = extract_pdf_text(make_pdf(tmp_path / 'f.pdf', pages))
    chunks = [c['content'] for c in chunk_document(text, 'f.pdf', 700, 100)]
    assert 'Figure 4: The attention weights for one head.' in chunks


def paragraphs(n, words=60):
    sentence = 'The garrison held the bridge through the night while the engineers worked below. '
    return '\n\n'.join(f'Paragraph {i}. ' + sentence * (words // 12) for i in range(n))


def test_book_length_documents_get_wide_windows():
    book = paragraphs(700)  # about 300,000 characters: a book
    sizes = [len(c['content']) for c in chunk_document(book, 'book.txt', 700, 100)]
    assert max(sizes) > 1200 and sum(sizes) / len(sizes) > 1000


def test_short_documents_keep_the_configured_window():
    note = paragraphs(12)
    assert max(len(c['content']) for c in chunk_document(note, 'note.txt', 700, 100)) <= 700


def test_code_keeps_the_configured_window_even_when_long():
    code = '\n\n'.join(f'def f{i}(x):\n    return x + {i}\n' for i in range(12000))
    assert max(len(c['content']) for c in chunk_document(code, 'big.py', 700, 100)) <= 700


def fake_pdf(monkeypatch, *pages):
    """pages: lists of blocks; a block is a list of (text, size, bold) lines in the body of the page."""
    from types import SimpleNamespace
    from rag.parsers import pdf
    def page(blocks):
        data = {'height': 600, 'blocks': [{'type': 0, 'lines': [
            {'bbox': (60, 100 + 20 * i, 360, 116 + 20 * i), 'spans': [{'text': t, 'size': size, 'flags': 16 if bold else 0, 'font': ''}]}
            for i, (t, size, bold) in enumerate(block)]} for block in blocks]}
        return SimpleNamespace(get_text=lambda kind='text': data, rect=SimpleNamespace(height=600))
    class Document(list):
        def close(self):
            pass
    monkeypatch.setattr(pdf.pymupdf, 'open', lambda path: Document([page(b) for b in pages]))
    return pdf.extract_pdf_text('book.pdf')


def test_decorated_initial_letter_rejoins_its_word(monkeypatch):
    text = fake_pdf(monkeypatch, [[('T', 48, False)], [('he king rode north.', 10.5, False)]])
    assert 'The king rode north.' in text and '# T' not in text


def test_bold_capital_section_title_is_a_heading(monkeypatch):
    text = fake_pdf(monkeypatch, [[('ON FRIENDSHIP AND LOVE', 10.5, True)], [('Friendship is but a word.', 10.5, False)] * 3])
    assert '# ON FRIENDSHIP AND LOVE' in text


def test_font_ligature_glyphs_become_letters(monkeypatch):
    text = fake_pdf(monkeypatch, [[('A SelecƟon from His WriƩen Words', 10.5, False)]])
    assert text == 'A Selection from His Written Words'


def test_contents_entries_are_not_headings(monkeypatch):
    text = fake_pdf(monkeypatch, [[('THE RISE OF CAPITALISM 59', 10.5, True)], [('1–37 Aegon I the Conqueror', 18, False)]])
    assert '# ' not in text


def test_decorated_initial_marks_a_chapter_break(monkeypatch):
    text = fake_pdf(monkeypatch, [[('T', 48, False)], [('he king rode north.', 10.5, False)]],
                    [[('A', 48, False)], [('egon sailed.', 10.5, False)]])
    chunks = chunk_document(text, 'book.pdf', 700, 100)
    assert [c['heading'] for c in chunks] == ['Chapter 1', 'Chapter 2']
    assert chunks[1]['content'].endswith('Aegon sailed.')


def test_a_heading_alone_is_not_a_chunk():
    text = '# PART ONE\n\n# CHAPTER ONE\n\nThe army crossed the river at dawn.'
    chunks = chunk_document(text, 'book.pdf', 700, 100)
    assert len(chunks) == 1 and chunks[0]['heading'] == 'CHAPTER ONE'
    assert 'The army crossed' in chunks[0]['content']
