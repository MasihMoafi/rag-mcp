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

    if ext == '.pdf':
        results = []
        section_start = 0
        for section in text.split('\n\n'):
            results.extend({'content': content, 'heading': '', 'start_offset': section_start + start,
                            'end_offset': section_start + end}
                           for content, start, end in _chunk_text_slices(section, chunk_size, chunk_overlap))
            section_start += len(section) + 2
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
