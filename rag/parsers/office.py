"""Office and image extraction through optional OCR/Unstructured adapters."""

import os


def extract_office_or_image_text(file_path: str) -> str:
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
