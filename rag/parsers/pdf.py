"""PDF text extraction with preserved page separators."""

import re

import pymupdf

# Expand typographic Latin ligatures only; compatibility normalization would
# also change meaningful mathematical characters such as superscripts.
_LIGATURES = str.maketrans(dict(zip('ﬀﬁﬂﬃﬄﬅﬆ', ('ff', 'fi', 'fl', 'ffi', 'ffl', 'st', 'st'))))


def extract_pdf_text(pdf_path: str) -> str:
    """PDF extraction. Pages are joined with a form-feed marker so a chunk's
    page number can be recovered later; without this the page boundary is
    destroyed and every chunk looks like it came from nowhere."""
    doc = pymupdf.open(pdf_path)
    try:
        pages = []
        for page in doc:
            sections = re.split(r'(?m)^(?=(?:Figure|Fig\.|Table)\s+\d+[.:]\s)',
                                page.get_text().translate(_LIGATURES))
            pages.append('\n\n'.join(re.sub(r'\s+', ' ', part).strip()
                                     for part in sections if part.strip()))
            if len(sections) > 1:
                pages[-1] += '\n\n'  # Do not join a caption to the next page's diagram.
    finally:
        doc.close()
    return "\f".join(pages)
