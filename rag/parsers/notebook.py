"""Notebook source text and cell metadata; generated outputs are excluded."""

import json
import re
from typing import Any, Dict, List


def extract_notebook_text(notebook_path: str) -> str:
    """Extract text from Jupyter notebook (.ipynb). Cells are joined with a
    form-feed marker, same as PDF pages, so a chunk's originating cell can
    be recovered later instead of collapsing into one undifferentiated blob."""
    try:
        from ..notebook_chunker import extract_cells_from_notebook
        cells = extract_cells_from_notebook(notebook_path)
        parts = []
        previous_index = 0
        for cell in cells:
            index = cell['cell_index']
            parts.append("\f" * (index - previous_index) + cell['content'])
            previous_index = index
        return "".join(parts)
    except Exception as e:
        print(f"Error extracting text from {notebook_path}: {e}")
        return ""


def extract_cells_from_notebook(notebook_path: str) -> List[Dict[str, Any]]:
    """
    Extract individual cells from Jupyter notebook with rich metadata.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        List of dicts with keys: content, source, cell_type, cell_id, exercise_number, chunk_id
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error loading notebook {notebook_path}: {e}")
        return []

    cells = notebook.get('cells', [])
    extracted_cells = []

    for idx, cell in enumerate(cells):
        cell_type = cell.get('cell_type', 'unknown')
        cell_id = cell.get('id', f'cell_{idx}')

        # Join source lines (notebooks store source as list of strings)
        source_lines = cell.get('source', [])
        if isinstance(source_lines, list):
            content = ''.join(source_lines)
        else:
            content = str(source_lines)

        # Skip empty cells
        if not content.strip():
            continue

        # Extract exercise number from content
        exercise_number = extract_exercise_number(content)

        # Build metadata
        metadata = {
            'content': content,
            'source': notebook_path,
            'cell_type': cell_type,
            'cell_id': cell_id,
            'cell_index': idx,
            'exercise_number': exercise_number,
            'chunk_id': idx  # Legacy identifier is the original cell position
        }

        extracted_cells.append(metadata)

    return extracted_cells


_EXERCISE_PATTERNS = [
    re.compile(r'#\s*GRADED\s+CELL:\s*exercise\s+(\d+)', re.IGNORECASE),
    re.compile(r'###\s*Exercise\s+(\d+)', re.IGNORECASE),
    re.compile(r'Exercise\s+(\d+)', re.IGNORECASE),
]


def extract_exercise_number(content: str) -> str | None:
    """
    Extract exercise number from cell content.

    Patterns:
    - "# GRADED CELL: exercise 1"
    - "### Exercise 2:"
    - "Exercise 3 - "
    """
    for pattern in _EXERCISE_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group(1)

    return None
