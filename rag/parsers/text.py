"""UTF-8 source reader for formats without a specialized extractor."""


def extract_plain_text(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as source:
            return source.read()
    except Exception as error:
        print(f"Error reading {file_path}: {error}")
        return ""
