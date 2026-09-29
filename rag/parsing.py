"""Pipeline extraction dispatch and compatibility adapters.

Readers live in parsers/. These adapters retain the pipeline method override
points used by callers; extraction policy is unchanged.
"""

import os

from .parsers.json import extract_json_text
from .parsers.notebook import extract_notebook_text
from .parsers.office import extract_office_text
from .parsers.pdf import extract_pdf_text
from .parsers.text import extract_plain_text
from .parsers.html import extract_html_text


def _extract_text_from_pdf(self, pdf_path: str) -> str:
    return extract_pdf_text(pdf_path)


def _extract_text_with_unstructured(self, file_path: str) -> str:
    return extract_office_text(file_path)


def _extract_text_from_notebook(self, notebook_path: str) -> str:
    return extract_notebook_text(notebook_path)


def _extract_text_from_json(self, json_path: str) -> str:
    return extract_json_text(json_path)


def _extract_content(self, file_path: str) -> str:
    """Extract content from various file types, using Unstructured for office documents."""
    return extract_content(
        file_path,
        pdf=self._extract_text_from_pdf,
        office=self._extract_text_with_unstructured,
        notebook=self._extract_text_from_notebook,
        json_reader=self._extract_text_from_json,
    )


def extract_content(file_path: str, *, pdf=extract_pdf_text, office=extract_office_text,
                    notebook=extract_notebook_text, json_reader=extract_json_text) -> str:
    """Model-free extraction shared by indexing and workspace source import."""
    ext = os.path.splitext(file_path)[1].lower()
    unstructured_exts = {".docx", ".pptx", ".xlsx", ".csv"}
    if ext in unstructured_exts:
        return office(file_path)
    elif ext == ".pdf":
        try:
            text = pdf(file_path)
        except Exception:
            text = office(file_path)
            if not text or not text.strip():
                raise
            return text
        if not text or not text.strip():
            text = office(file_path)
        return text
    elif ext == ".json":
        return json_reader(file_path)
    elif ext == ".ipynb":
        return notebook(file_path)
    elif ext == ".html":
        return extract_html_text(file_path)
    else:
        return extract_plain_text(file_path)
