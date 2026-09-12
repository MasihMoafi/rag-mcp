"""Source-faithful structural chunking and heading metadata."""

import os
import re
from typing import Dict, List, Tuple


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> List[str]:
    """Return contiguous source slices, preferring structural/word boundaries.

    Sizes are character budgets. Only a token longer than the budget requires
    a hard cut; overlap is selected from the source, never stitched onto a chunk.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    if not text or not text.strip():
        return []

    chunks = []
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
        chunk = text[start:end].strip()

        if chunk.strip():
            chunks.append(chunk)

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


def _split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown into (heading_breadcrumb, section_text) pairs on '#' headings."""
    sections: List[Tuple[str, str]] = []
    stack: List[Tuple[int, str]] = []
    current_lines: List[str] = []
    fence = ""

    def flush():
        if current_lines:
            breadcrumb = " > ".join(title for _, title in stack)
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((breadcrumb, body))

    for line in text.split("\n"):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            current_lines.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = ""
            continue
        if marker and (marker[1][0] == "~" or "`" not in marker[2]):
            fence = marker[1]
            current_lines.append(line)
            continue
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_lines = [line]
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = [entry for entry in stack if entry[0] < level]
            stack.append((level, title))
        else:
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


def chunk_document(
    text: str, file_path: str = "", chunk_size: int = 700, chunk_overlap: int = 100
) -> List[Dict[str, str]]:
    """Source-faithful chunks with Markdown section boundaries and preferred
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
        return [{'content': chunk, 'heading': ''}
                for section in text.split('\n\n')
                for chunk in chunk_text(section, chunk_size, chunk_overlap)]

    if ext in _MD_EXTS and _HEADING_RE.search(text):
        results = []
        for breadcrumb, section in _split_markdown_sections(text):
            for chunk in chunk_text(section, chunk_size, chunk_overlap):
                if chunk.strip():
                    results.append({"content": chunk, "heading": breadcrumb})
        if results:
            return results

    heading_fn = _first_code_symbol if ext in _CODE_EXTS else (lambda _c: "")
    return [{"content": chunk, "heading": heading_fn(chunk)}
            for chunk in chunk_text(text, chunk_size, chunk_overlap)]
