"""Source-faithful structural chunking and heading metadata."""

import os
import re
from typing import Dict, List, Tuple


def _chunk_text_slices(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
    """Return content and exact source positions for each structural slice.

    Sizes are character budgets. Only a token longer than the budget requires
    a hard cut; overlap is selected from the source, never stitched onto a chunk.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    if not text or not text.strip():
        return []

    chunks: List[Tuple[str, int, int]] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            # Avoid tiny chunks when a heading/paragraph ends near the start.
            minimum = len(window) // 2
            for pattern in (r"\n\s*\n", r"[.!?؟](?=\s)", r"\n", r"\s+"):
                boundaries = [match.end() for match in re.finditer(pattern, window)
                              if match.end() >= minimum]
                if boundaries:
                    end = start + boundaries[-1]
                    break
            else:
                # A short prefix before a long word still provides a valid
                # boundary; do not split that word just to fill half a chunk.
                spaces = list(re.finditer(r"\s+", window))
                if spaces:
                    end = start + spaces[-1].end()
        raw = text[start:end]
        chunk = raw.strip()

        if chunk.strip():
            chunk_start = start + len(raw) - len(raw.lstrip())
            chunk_end = end - (len(raw) - len(raw.rstrip()))
            chunks.append((chunk, chunk_start, chunk_end))

        if end == len(text):
            break
        next_start = max(start + 1, end - chunk_overlap)
        # Do not prepend a sentence/paragraph fragment merely to meet the
        # overlap budget. Prefer a complete unit within that budget, even when
        # this means no overlap at an existing paragraph boundary.
        suffix = text[next_start:end]
        for pattern in (r"\n\s*\n", r"[.!?؟](?=\s)"):
            boundary = re.search(pattern, suffix)
            if boundary:
                next_start += boundary.end()
                break
        # Move an overlap boundary out of a word. For an oversized unbroken
        # token retain the hard-window fallback rather than dropping its tail.
        candidate = next_start
        while candidate < end and candidate > 0 and not text[candidate - 1].isspace():
            candidate += 1
        start = candidate if candidate < end else end

    return chunks


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> List[str]:
    """Return contiguous source slices, preferring structural/word boundaries."""
    return [content for content, _, _ in _chunk_text_slices(text, chunk_size, chunk_overlap)]


# Book-length prose is cut in wide windows: a narrative's facts sit in scenes and paragraphs that a
# 700-character window splits apart, so book-length text gets 1,600 characters with 250 overlap. Code keeps the
# configured window, since a function is the unit there.
LONG_DOCUMENT_CHARS = int(os.environ.get("RAG_MCP_LONG_DOCUMENT_CHARS", "150000"))
LONG_CHUNK_SIZE = int(os.environ.get("RAG_MCP_LONG_CHUNK_SIZE", "1600"))
LONG_CHUNK_OVERLAP = int(os.environ.get("RAG_MCP_LONG_CHUNK_OVERLAP", "250"))
_CAPTION_RE = re.compile(r"^(?:Figure|Fig\.|Table)\s+\d+[.:]\s")
_UNIT_BREAK_RE = re.compile(r"\n\s*\n|\f")

_MD_EXTS = {".md", ".markdown"}
_CODE_EXTS = {
    ".py", ".rs", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp",
    ".go", ".java", ".kt", ".sh", ".bash", ".zsh", ".sql",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
_CODE_SYMBOL_RE = re.compile(
    r"^\s*(?:pub\s+|pub\(crate\)\s+|async\s+|export\s+|static\s+)*"
    r"(?:def|fn|class|function|struct|impl|interface|type)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def _split_markdown_sections(text: str) -> List[Tuple[str, str, int]]:
    """Split markdown into (heading, section text, source start) on headings."""
    sections: List[Tuple[str, str, int]] = []
    stack: List[Tuple[int, str]] = []
    current_lines: List[str] = []
    current_start = 0
    fence = ""

    def flush():
        if current_lines:
            breadcrumb = " > ".join(title for _, title in stack)
            raw = "\n".join(current_lines)
            body = raw.strip()
            if body:
                sections.append((breadcrumb, body, current_start + len(raw) - len(raw.lstrip())))

    position = 0
    for line in text.split("\n"):
        line_start = position
        position += len(line) + 1
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if not current_lines:
                current_start = line_start
            current_lines.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = ""
            continue
        if marker and (marker[1][0] == "~" or "`" not in marker[2]):
            if not current_lines:
                current_start = line_start
            fence = marker[1]
            current_lines.append(line)
            continue
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_lines = [line]
            current_start = line_start
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = [entry for entry in stack if entry[0] < level]
            stack.append((level, title))
        else:
            if not current_lines:
                current_start = line_start
            current_lines.append(line)
    flush()

    return sections


def _caption_units(text: str) -> List[Tuple[str, int]]:
    """Split a section so each figure or table caption is a unit of its own; the paragraphs between
    captions stay together for the window chunker. Returns (unit text, start offset)."""
    units: List[Tuple[str, int]] = []
    unit_start, position = 0, 0
    for match in list(_UNIT_BREAK_RE.finditer(text)) + [None]:
        end = match.start() if match else len(text)
        paragraph = text[position:end]
        if _CAPTION_RE.match(paragraph.strip()):
            before = text[unit_start:position]
            if before.strip():
                units.append((before, unit_start))
            units.append((paragraph, position))
            unit_start = match.end() if match else len(text)
        position = match.end() if match else len(text)
    rest = text[unit_start:]
    if rest.strip():
        units.append((rest, unit_start))
    return units


def _first_code_symbol(text: str) -> str:
    """Label the first regex-recognized symbol in a chunk, not its enclosing scope."""
    for line in text.split("\n"):
        match = _CODE_SYMBOL_RE.match(line)
        if match:
            return match.group(1)
    return ""


def chunk_document_spans(
    text: str, file_path: str = "", chunk_size: int = 700, chunk_overlap: int = 100
) -> List[Dict[str, str | int]]:
    """Source-faithful chunks and positions with Markdown section boundaries and preferred
    paragraph, sentence, and word boundaries within the character budget.
    Oversized unbroken tokens fall back to hard cuts. Returns a "heading"
    breadcrumb (Markdown section path or first recognized code symbol) for citations.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    if not text or not text.strip():
        return []

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _CODE_EXTS and len(text) >= LONG_DOCUMENT_CHARS:
        chunk_size, chunk_overlap = max(chunk_size, LONG_CHUNK_SIZE), max(chunk_overlap, LONG_CHUNK_OVERLAP)

    if ext == '.pdf':
        # The PDF reader marks chapter headings as "# " lines: no chunk spans two chapters, each carries
        # its chapter as a breadcrumb, and captions stay chunks of their own.
        sections = _split_markdown_sections(text) if _HEADING_RE.search(text) else [("", text, 0)]
        results = []
        for heading, section, section_start in sections:
            for unit, unit_start in _caption_units(section):
                results.extend({'content': content, 'heading': heading, 'start_offset': section_start + unit_start + start,
                                'end_offset': section_start + unit_start + end}
                               for content, start, end in _chunk_text_slices(unit, chunk_size, chunk_overlap)
                               if not all(line.startswith('# ') for line in content.splitlines() if line.strip()))  # a heading alone is no passage
        return results

    if ext in _MD_EXTS and _HEADING_RE.search(text):
        results = []
        for breadcrumb, section, section_start in _split_markdown_sections(text):
            for content, start, end in _chunk_text_slices(section, chunk_size, chunk_overlap):
                results.append({"content": content, "heading": breadcrumb,
                                "start_offset": section_start + start, "end_offset": section_start + end})
        if results:
            return results

    heading_fn = _first_code_symbol if ext in _CODE_EXTS else (lambda _c: "")
    return [{"content": content, "heading": heading_fn(content),
             "start_offset": start, "end_offset": end}
            for content, start, end in _chunk_text_slices(text, chunk_size, chunk_overlap)]


def chunk_document(
    text: str, file_path: str = "", chunk_size: int = 700, chunk_overlap: int = 100
) -> List[Dict[str, str]]:
    """Return the existing content/heading contract from exact source spans."""
    return [{"content": str(item["content"]), "heading": str(item["heading"])}
            for item in chunk_document_spans(text, file_path, chunk_size, chunk_overlap)]
