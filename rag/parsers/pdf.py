"""PDF text extraction with preserved page separators."""

import re

import fitz

# Expand typographic Latin ligatures only; compatibility normalization would
# also change meaningful mathematical characters such as superscripts.
_LIGATURES = str.maketrans(dict(zip('ﬀﬁﬂﬃﬄﬅﬆ', ('ff', 'fi', 'fl', 'ffi', 'ffl', 'st', 'st'))))


def extract_pdf_text(pdf_path: str) -> str:
    """PDF extraction. Pages are joined with a form-feed marker so a chunk's
    page number can be recovered later; without this the page boundary is
    destroyed and every chunk looks like it came from nowhere."""
    try:
        doc = fitz.open(pdf_path)
        pages = [re.sub(r'\s+', ' ', page.get_text().translate(_LIGATURES)).strip() for page in doc]
        doc.close()
        return "\f".join(pages)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
