"""Compatibility imports; notebook extraction lives in parsers.notebook."""

from .parsers.notebook import (
    _EXERCISE_PATTERNS,
    extract_cells_from_notebook,
    extract_exercise_number,
)
