"""Publish complete JSON files without exposing partial writes."""

import json
import os
from pathlib import Path
import tempfile


def write_json_atomic(path, value):
    path = Path(path)
    pending_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as pending:
            pending_path = pending.name
            json.dump(value, pending, indent=2)
            pending.flush()
            os.fsync(pending.fileno())
        os.replace(pending_path, path)
        if os.name == 'posix':
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if pending_path is not None and os.path.exists(pending_path):
            os.unlink(pending_path)
