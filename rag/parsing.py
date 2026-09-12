"""Pipeline extraction dispatch and compatibility adapters.

Readers live in parsers/. These adapters retain the pipeline method override
points used by callers; extraction policy is unchanged.
"""

import os

from .parsers.json import extract_json_text
from .parsers.notebook import extract_notebook_text
from .parsers.office import extract_office_or_image_text
from .parsers.pdf import extract_pdf_text
from .parsers.text import extract_plain_text
from .parsers.html import extract_html_text


def _extract_text_from_pdf(self, pdf_path: str) -> str:
    return extract_pdf_text(pdf_path)


def _extract_text_with_unstructured(self, file_path: str) -> str:
    return extract_office_or_image_text(file_path)


def _extract_text_from_notebook(self, notebook_path: str) -> str:
    return extract_notebook_text(notebook_path)


def _extract_text_from_json(self, json_path: str) -> str:
    return extract_json_text(json_path)


def _extract_content(self, file_path: str) -> str:
    """Extract content from various file types, using Unstructured for images and complex docs."""
    ext = os.path.splitext(file_path)[1].lower()
    unstructured_exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".docx", ".pptx", ".xlsx", ".csv"}
    if ext in unstructured_exts:
        return self._extract_text_with_unstructured(file_path)
    elif ext == ".pdf":
        try:
            text = self._extract_text_from_pdf(file_path)
        except Exception:
            text = self._extract_text_with_unstructured(file_path)
            if not text or not text.strip():
                raise
            return text
        if not text or not text.strip():
            text = self._extract_text_with_unstructured(file_path)
        return text
    elif ext == ".json":
        return self._extract_text_from_json(file_path)
    elif ext == ".ipynb":
        return self._extract_text_from_notebook(file_path)
    elif ext == ".html":
        return extract_html_text(file_path)
    else:
        return extract_plain_text(file_path)
