"""PDF text extraction that keeps the document's structure.

A PDF stores characters and their positions, not paragraphs or chapters, so plain text extraction
returns one line per printed line and loses everything a chunker needs. Here each PyMuPDF text block
becomes one paragraph (its lines joined, end-of-line hyphens mended), a block set clearly larger than
the body text becomes a "# " heading line, and running headers and page numbers - short lines in the
top or bottom margin that repeat across pages - are dropped. Paragraphs are separated by a blank line;
pages are joined with a form-feed marker so a chunk's page number can be recovered later.
"""

import re
from collections import Counter

import pymupdf

# Expand typographic Latin ligatures only; compatibility normalization would
# also change meaningful mathematical characters such as superscripts.
# Some fonts (Calibri among them) map their ti and tt ligatures to Ɵ and Ʃ: "SelecƟon", "WriƩen".
_LIGATURES = str.maketrans(dict(zip('ﬀﬁﬂﬃﬄﬅﬆƟƩ', ('ff', 'fi', 'fl', 'ffi', 'ffl', 'st', 'st', 'ti', 'tt'))))
_CAPTION = re.compile(r'^(?:Figure|Fig\.|Table)\s+\d+[.:]\s')
_PAGE_NUMBER = re.compile(r'^(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,7})$', re.IGNORECASE)
_ROMAN = re.compile(r'^[IVXLCDM]+\.?$')
MARGIN = 0.08          # share of the page height, top and bottom, where running headers and page numbers sit
REPEATED_SHARE = 0.3   # a margin line on at least this share of pages (and at least 3) is a running header
HEADING_RATIO = 1.25   # a block this much larger than the body text is a heading
HEADING_MAX_CHARS = 150


def _page_blocks(page):
    """Each text block as a list of lines: (text, top share, bottom share, [(size, chars, bold)])."""
    data = page.get_text('dict')
    height = data.get('height') or page.rect.height or 1
    blocks = []
    for block in data.get('blocks', []):
        if block.get('type', 0) != 0:
            continue
        lines = []
        for line in block.get('lines', []):
            text = ''.join(span['text'] for span in line['spans']).translate(_LIGATURES)
            if text.strip():
                sizes = [(round(span['size'], 1), len(span['text'].strip()), bool(span.get('flags', 0) & 16) or 'Bold' in span.get('font', ''))
                         for span in line['spans'] if span['text'].strip()]
                lines.append((text, line['bbox'][1] / height, line['bbox'][3] / height, sizes))
        if lines:
            blocks.append(lines)
    return blocks


def _margin_key(text):
    return re.sub(r'\d+', '#', ' '.join(text.split()).lower())


def _in_margin(line):
    return line[1] < MARGIN or line[2] > 1 - MARGIN


def _join_lines(lines):
    out = ''
    for text, *_ in lines:
        text = ' '.join(text.split())
        if out.endswith('-') and text[:1].islower():
            out = out[:-1] + text  # a word hyphenated at the end of a printed line
        else:
            out = f'{out} {text}' if out else text
    return out


def extract_pdf_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    try:
        pages = [_page_blocks(page) for page in doc]
    finally:
        doc.close()

    # Running headers and footers: the same short margin line, numbers aside, on many pages.
    seen = Counter()
    for blocks in pages:
        seen.update({_margin_key(line[0]) for lines in blocks for line in lines if _in_margin(line)})
    repeated = {key for key, count in seen.items() if count >= max(3, REPEATED_SHARE * len(pages))}

    def furniture(line):
        return _in_margin(line) and (_margin_key(line[0]) in repeated or _PAGE_NUMBER.match(line[0].strip()))

    sizes = Counter()
    for blocks in pages:
        for lines in blocks:
            for line in lines:
                if not furniture(line):
                    for size, chars, _ in line[3]:
                        sizes[size] += chars
    body = sizes.most_common(1)[0][0] if sizes else 0

    def is_heading(text, mean, bold):
        letters = re.findall(r'[^\W\d_]', text)
        if len(letters) < 3 or len(text) > HEADING_MAX_CHARS or _ROMAN.match(text):
            return False
        larger = body and mean >= body * HEADING_RATIO
        bold_capitals = bold >= 0.9 and ''.join(letters) == ''.join(letters).upper() and not text.rstrip().endswith('.')
        return larger or bold_capitals

    out_pages = []
    drop_cap = ''
    for blocks in pages:
        paragraphs = []
        for lines in blocks:
            lines = [line for line in lines if not furniture(line)]
            if not lines:
                continue
            chars = sum(c for line in lines for _, c, _ in line[3]) or 1
            mean = sum(size * c for line in lines for size, c, _ in line[3]) / chars
            bold = sum(c for line in lines for _, c, b in line[3] if b) / chars
            text = _join_lines(lines)
            if body and len(text) <= 2 and text.isalpha() and mean >= body * 2:
                drop_cap = text  # a decorated initial letter belongs to the first word of the next paragraph
                continue
            if is_heading(text, mean, bold):
                paragraphs.append('# ' + text)
                continue
            if drop_cap:
                lines = [(drop_cap + lines[0][0].lstrip(), *lines[0][1:])] + lines[1:]
                drop_cap = ''
            # A caption starts its own paragraph even when the block runs on into body text.
            current = []
            for line in lines:
                if current and _CAPTION.match(line[0].strip()):
                    paragraphs.append(_join_lines(current))
                    current = []
                current.append(line)
            paragraphs.append(_join_lines(current))
        page_text = '\n\n'.join(p for p in paragraphs if p)
        # A heading that ends a page must not run on into the next page's first line.
        out_pages.append(page_text + '\n' if paragraphs and paragraphs[-1].startswith('# ') else page_text)
    # A heading that opens a page needs its own line, so the chunker sees it after the page marker.
    return '\f'.join('\n' + page if page.startswith('# ') else page for page in out_pages)
