"""Format extraction methods bound by RAGPipelineV2; source text stays unchanged."""

import json
import os
import re

import fitz


def _extract_text_from_pdf(self, pdf_path: str) -> str:
    """PDF extraction. Pages are joined with a form-feed marker so a chunk's
    page number can be recovered later; without this the page boundary is
    destroyed and every chunk looks like it came from nowhere."""
    try:
        doc = fitz.open(pdf_path)
        pages = [re.sub(r'\s+', ' ', page.get_text()).strip() for page in doc]
        doc.close()
        return "\f".join(pages)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def _extract_text_with_unstructured(self, file_path: str) -> str:
    """Extract text, tables, and OCR/element descriptions using Unstructured and RapidOCR."""
    ext = os.path.splitext(file_path)[1].lower()
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}

    # For standalone image files, run RapidOCR directly
    if ext in image_exts:
        try:
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()
            result, _ = engine(file_path)
            if result:
                return "\n".join([line[1] for line in result])
        except Exception as e:
            print(f"[OCR] Warning: RapidOCR failed for {file_path}: {e}")

    try:
        from unstructured.partition.auto import partition
        elements = partition(filename=file_path)
        parts = []
        for el in elements:
            text = str(el).strip()
            if text:
                page_num = getattr(getattr(el, 'metadata', None), 'page_number', None)
                page_prefix = f"[Page {page_num}] " if page_num else ""
                parts.append(f"{page_prefix}{text}")
        res = "\n\n".join(parts)
        if res:
            return res
    except Exception as e:
        print(f"[Unstructured] Warning: extraction failed for {file_path}: {e}")

    return ""

def _extract_content(self, file_path: str) -> str:
    """Extract content from various file types, using Unstructured for images and complex docs."""
    ext = os.path.splitext(file_path)[1].lower()
    unstructured_exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".docx", ".pptx", ".xlsx", ".csv"}
    if ext in unstructured_exts:
        return self._extract_text_with_unstructured(file_path)
    elif ext == ".pdf":
        text = self._extract_text_from_pdf(file_path)
        if not text or not text.strip():
            text = self._extract_text_with_unstructured(file_path)
        return text
    elif ext == ".json":
        return self._extract_text_from_json(file_path)
    elif ext == ".ipynb":
        return self._extract_text_from_notebook(file_path)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

def _extract_text_from_notebook(self, notebook_path: str) -> str:
    """Extract text from Jupyter notebook (.ipynb). Cells are joined with a
    form-feed marker, same as PDF pages, so a chunk's originating cell can
    be recovered later instead of collapsing into one undifferentiated blob."""
    try:
        from .notebook_chunker import extract_cells_from_notebook
        cells = extract_cells_from_notebook(notebook_path)
        return "\f".join(cell['content'] for cell in cells)
    except Exception as e:
        print(f"Error extracting text from {notebook_path}: {e}")
        return ""

def _extract_text_from_json(self, json_path: str) -> str:
    """Extract text from JSON - handles various structures"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert JSON to searchable text
        def flatten_json(obj, prefix=''):
            text_parts = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        text_parts.extend(flatten_json(value, f"{prefix}{key}."))
                    else:
                        text_parts.append(f"{prefix}{key}: {value}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    text_parts.extend(flatten_json(item, f"{prefix}[{i}]."))
            else:
                text_parts.append(str(obj))
            return text_parts

        text_parts = flatten_json(data)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error extracting text from {json_path}: {e}")
        return ""
