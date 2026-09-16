"""Office document extraction through the optional Unstructured adapter."""


def extract_office_text(file_path: str) -> str:
    """Extract text and tables from DOCX/PPTX/XLSX/CSV through Unstructured."""
    from unstructured.partition.auto import partition
    elements = partition(filename=file_path)
    parts = []
    for el in elements:
        text = str(el).strip()
        if text:
            page_num = getattr(getattr(el, 'metadata', None), 'page_number', None)
            page_prefix = f"[Page {page_num}] " if page_num else ""
            parts.append(f"{page_prefix}{text}")
    return "\n\n".join(parts)
