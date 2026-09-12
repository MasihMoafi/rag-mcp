"""UTF-8 source reader for formats without a specialized extractor."""


def extract_plain_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as source:
        return source.read()
