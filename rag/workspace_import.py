"""Import local files as extracted text into the shared, revisioned workspace."""

from __future__ import annotations

from typing import Any

import hashlib
import os
import stat
import tempfile
from pathlib import Path

from rag.parsing import extract_content
from rag.server import _normalize_path, _validate_rag_scope
from rag.workspaces import WorkspaceConflict, WorkspaceStore


_SOURCE_TYPES = {
    '.md': 'markdown', '.pdf': 'pdf', '.html': 'html', '.json': 'json_nosql',
    '.docx': 'docx', '.py': 'code_python', '.rs': 'code_rust',
    '.ts': 'code_typescript', '.tsx': 'code_typescript',
    '.js': 'code_javascript', '.jsx': 'code_javascript', '.sql': 'sql',
}
_UNSPECIFIED_REVISION = object()


def _open_source(path: Path) -> int:
    """Open every path component without following links, including directory swaps."""
    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY'):
        raise ValueError('Safe file import is unavailable on this platform.')
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            next_directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        return os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)


def import_workspace_sources(store: WorkspaceStore, workspace_id: str, path: Path | str,
                             extensions: list[str] | None = None,
                             *, expected_revision: str | None | object = _UNSPECIFIED_REVISION) -> dict[str, Any]:
    """Extract one file or a bounded folder before publishing a CAS source snapshot."""
    workspace_id = store.validate_id(workspace_id)
    if expected_revision is not _UNSPECIFIED_REVISION and expected_revision is not None and (
        not isinstance(expected_revision, str) or not expected_revision
    ):
        raise ValueError('Workspace revision must be a string or null.')
    manifest_path = store.root / workspace_id / 'manifest.json'
    if expected_revision is not _UNSPECIFIED_REVISION:
        observed = store.get(workspace_id)['revision'] if manifest_path.exists() else None
        if observed != expected_revision:
            raise WorkspaceConflict('Workspace changed in another client. Reload before importing.')
    path = Path(_normalize_path(str(path)))
    scope = _validate_rag_scope(path, extensions)
    files = sorted(scope.files) if scope is not None else [str(path)]
    if not files:
        raise ValueError('No readable documents found in the selected scope.')

    imported, skipped = [], []
    for file in files:
        source = Path(file)
        try:
            with os.fdopen(_open_source(source), 'rb') as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError('Source is not a regular file.')
                if before.st_size > store.max_document_bytes:
                    raise ValueError(f'Source file exceeds the {store.max_document_bytes}-byte import limit.')
                with tempfile.TemporaryDirectory(prefix='rag-import-') as directory:
                    snapshot = Path(directory) / source.name
                    with snapshot.open('wb') as output:
                        remaining = store.max_document_bytes + 1
                        while remaining:
                            block = stream.read(min(65536, remaining))
                            if not block:
                                break
                            output.write(block)
                            remaining -= len(block)
                        if remaining == 0:
                            raise ValueError('Source file exceeds the import limit.')
                    after = os.fstat(stream.fileno())
                    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                        raise ValueError('A source changed during import. Retry with a stable folder.')
                    content = extract_content(str(snapshot))
                    empty = not snapshot.read_bytes().strip() if not content or not content.strip() else False
                current = source.lstat()
                if (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size) != (
                        before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size):
                    raise ValueError('A source changed during import. Retry with a stable folder.')
        except (OSError, ValueError, RuntimeError) as error:
            where = source.relative_to(path) if scope is not None else source.name
            raise ValueError(f'Cannot read {where}: {error}') from error
        if not content or not content.strip():
            # An empty file (a package's __init__.py) loses nothing; text the parser lost does.
            if scope is not None and empty:
                skipped.append(str(source.relative_to(path)))
                continue
            raise ValueError(f'No readable text found in {source.name}.')
        imported.append({
            'id': 'file-' + hashlib.sha256(str(source).encode()).hexdigest(),
            'name': str(source.relative_to(path)) if scope is not None else source.name,
            'sourceType': _SOURCE_TYPES.get(source.suffix.lower(), 'text'),
            'content': content,
        })
    if not imported:
        raise ValueError('No readable text found in the selected scope.')
    if scope is not None and not scope.is_current():
        raise ValueError('The folder changed during import. Retry with a stable folder.')

    if manifest_path.exists():
        current = store.snapshot(workspace_id)
        documents = current['documents']
        revision = current['revision']
        name = current['name']
    else:
        documents, revision, name = [], None, workspace_id
    if expected_revision is not _UNSPECIFIED_REVISION and revision != expected_revision:
        raise WorkspaceConflict('Workspace changed in another client. Reload before importing.')
    positions = {item['id']: index for index, item in enumerate(documents)}
    for document in imported:
        position = positions.get(document['id'])
        if position is None:
            positions[document['id']] = len(documents)
            documents.append(document)
        else:
            documents[position] = document
    saved = store.replace(workspace_id, name, documents, expected_revision=revision)
    return {
        'schema_version': 1,
        'workspace_id': workspace_id,
        'revision': saved['revision'],
        'document_count': saved['document_count'],
        'total_bytes': saved['total_bytes'],
        'imported_count': len(imported),
        'skipped_empty': skipped,
        'source_ids': [item['id'] for item in imported],
    }
