"""Import local files as extracted text into the shared, revisioned workspace."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag.parsing import extract_content
from rag.server import _validate_rag_scope
from rag.workspaces import WorkspaceConflict, WorkspaceStore


_SOURCE_TYPES = {
    '.md': 'markdown', '.pdf': 'pdf', '.html': 'html', '.json': 'json_nosql',
    '.docx': 'docx', '.py': 'code_python', '.rs': 'code_rust',
    '.ts': 'code_typescript', '.tsx': 'code_typescript',
    '.js': 'code_javascript', '.jsx': 'code_javascript', '.sql': 'sql',
}
_UNSPECIFIED_REVISION = object()


def import_workspace_sources(store: WorkspaceStore, workspace_id: str, path: Path | str,
                             extensions: list[str] | None = None,
                             *, expected_revision: str | None | object = _UNSPECIFIED_REVISION) -> dict:
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
    path = Path(path).expanduser().resolve()
    scope = _validate_rag_scope(path, extensions)
    files = sorted(scope.files) if scope is not None else [str(path)]
    if not files:
        raise ValueError('No readable documents found in the selected scope.')

    imported = []
    for file in files:
        source = Path(file)
        before = source.stat()
        if before.st_size > store.max_document_bytes:
            raise ValueError(f'Source file exceeds the {store.max_document_bytes}-byte import limit.')
        content = extract_content(file)
        if not content or not content.strip():
            raise ValueError(f'No readable text found in {source.name}.')
        after = source.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise ValueError('A source changed during import. Retry with a stable folder.')
        imported.append({
            'id': 'file-' + hashlib.sha256(str(source.resolve()).encode()).hexdigest(),
            'name': str(source.relative_to(path)) if scope is not None else source.name,
            'sourceType': _SOURCE_TYPES.get(source.suffix.lower(), 'text'),
            'content': content,
        })
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
        'source_ids': [item['id'] for item in imported],
    }
