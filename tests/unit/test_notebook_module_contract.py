import json

from rag import notebook_chunker


def test_notebook_metadata_keeps_original_cell_positions(tmp_path):
    path = tmp_path / 'notes.ipynb'
    path.write_text(json.dumps({'cells': [
        {'cell_type': 'code', 'source': ['  ']},
        {'cell_type': 'markdown', 'id': 'known', 'source': ['### Exercise 2\n', 'متن']},
    ]}))
    assert notebook_chunker.extract_cells_from_notebook(str(path)) == [{
        'content': '### Exercise 2\nمتن', 'source': str(path), 'cell_type': 'markdown',
        'cell_id': 'known', 'cell_index': 1, 'exercise_number': '2', 'chunk_id': 1,
    }]


def test_exercise_patterns_keep_priority():
    text = 'Exercise 8\n### Exercise 2\n# GRADED CELL: exercise 3'
    assert notebook_chunker.extract_exercise_number(text) == '3'
    assert notebook_chunker.extract_exercise_number('no exercise') is None


def test_extracted_text_preserves_empty_cell_positions(tmp_path):
    from rag.parsers.notebook import extract_notebook_text
    path = tmp_path / 'positions.ipynb'
    path.write_text(json.dumps({'cells': [
        {'cell_type': 'code', 'source': []},
        {'cell_type': 'markdown', 'source': 'First evidence'},
        {'cell_type': 'code', 'source': '  ', 'outputs': [{'text': 'JUNK'}]},
        {'cell_type': 'code', 'source': 'second = 42'},
    ]}))
    assert extract_notebook_text(str(path)) == '\fFirst evidence\f\fsecond = 42'


def test_notebook_requires_explicit_cell_list(tmp_path):
    import pytest
    path = tmp_path / 'invalid.ipynb'
    for value in ({}, {'cells': None}, {'cells': {}}, {'cells': 'text'}, []):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match='cells'):
            notebook_chunker.extract_cells_from_notebook(str(path))
    path.write_text('{"cells": []}')
    assert notebook_chunker.extract_cells_from_notebook(str(path)) == []


def test_unreadable_notebook_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        notebook_chunker.extract_cells_from_notebook(str(tmp_path / 'missing.ipynb'))


def test_legacy_notebook_imports_reference_named_parser():
    from rag.parsers import notebook
    assert notebook_chunker.extract_cells_from_notebook is notebook.extract_cells_from_notebook
    assert notebook_chunker.extract_exercise_number is notebook.extract_exercise_number


def test_notebook_reader_import_does_not_load_other_formats():
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, '-c', 'import sys, rag.notebook_chunker; '
         'assert not {"fitz", "torch", "unstructured", "rag.core"}.intersection(sys.modules)'],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
