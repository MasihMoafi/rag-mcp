"""Bounded read of published workspace document metadata; never opens a pipeline."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from rag.chunk_preview import _utf16_positions
from rag.index_lifecycle import active_directory, selected_file_scope
from rag.workspaces import WorkspaceConflict, WorkspaceStore


_MAX_METADATA_BYTES = 64 * 1024 * 1024


def _read_json(path: Path) -> Any:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_size > _MAX_METADATA_BYTES:
            raise ValueError("Published index metadata is invalid or too large.")
        data = stream.read(_MAX_METADATA_BYTES + 1)
        if len(data) > _MAX_METADATA_BYTES:
            raise ValueError("Published index metadata is too large.")
    return json.loads(data)


def _lancedb_count_matches(active: Path, expected: int) -> bool:
    """Check the published table's row count without constructing a pipeline."""
    storage = active / "lancedb_storage"
    if expected <= 0 or not storage.is_dir() or storage.is_symlink():
        return False
    try:
        import lancedb

        database = lancedb.connect(str(storage))
        tables = database.list_tables().tables
        return len(tables) == 1 and len(database.open_table(tables[0])) == expected
    except Exception:
        return False


def inspect_published_chunks(
    store: WorkspaceStore,
    workspace_id: str,
    document_id: str,
    revision: str,
    *,
    limit: int = 50,
    offset: int = 0,
    include_content: bool = False,
    scope_source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Read one source from the active generation, with an explicit freshness state."""
    if not isinstance(revision, str) or not revision:
        raise ValueError("A workspace revision is required.")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("A document id is required.")
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
        raise ValueError("Invalid chunk page parameters.")
    if type(include_content) is not bool:
        raise ValueError("include_content must be a boolean.")
    if scope_source_ids is not None and (
        not isinstance(scope_source_ids, list) or not 1 <= len(scope_source_ids) <= 500
        or any(not isinstance(item, str) for item in scope_source_ids)
        or len(set(scope_source_ids)) != len(scope_source_ids)
    ):
        raise ValueError("Invalid selected source scope.")

    manifest = store.get(workspace_id)
    if manifest["revision"] != revision:
        raise WorkspaceConflict("Workspace changed in another client. Reload before inspecting chunks.")
    document = next((item for item in manifest["documents"] if item["id"] == document_id), None)
    if document is None:
        raise FileNotFoundError("Source not found.")
    available = {item["id"] for item in manifest["documents"]}
    selected_ids = available if scope_source_ids is None else set(scope_source_ids)
    if document_id not in selected_ids or not selected_ids <= available:
        raise ValueError("Selected source scope must include available sources and the inspected source.")
    selected_documents = [item for item in manifest["documents"] if item["id"] in selected_ids]
    canonical_ids = [item["id"] for item in selected_documents]
    stored_file = store.document_filename(document_id, document["source_type"])
    if document["stored_file"] != stored_file:
        raise OSError("Workspace source metadata is invalid.")
    # Query scope resolves the immutable workspace generation to an absolute path.
    source_root = Path(store.root / store.validate_id(workspace_id)
                       / "generations" / revision / "documents").resolve()
    source_path = source_root / stored_file
    base = {
        "schema_version": 1,
        "metadata_kind": "published_documents_unverified_storage",
        "workspace_id": workspace_id,
        "revision": revision,
        "document_id": document_id,
        "scope_source_ids": canonical_ids,
        "name": document["name"],
        "source_type": document["source_type"],
        "offset": offset,
        "total_chunks": 0,
        "next_offset": None,
        "chunks": [],
    }

    # Import settings only after validating the source; neither lookup creates
    # an index directory nor loads an embedding model.
    from rag.index_config import RAG_CONFIG_V2, _safe_dir_name, build_source_stamp, index_root_path
    from rag.settings import contextual_retrieval_enabled, embedding_config, retrieval_config

    embedding = embedding_config()
    selected_files = [str(source_root / store.document_filename(item["id"], item["source_type"]))
                      for item in selected_documents]
    selected_scope = Path(index_root_path()) / _safe_dir_name(selected_file_scope(str(source_root), selected_files), embedding)
    legacy_scope = Path(index_root_path()) / _safe_dir_name(str(source_root), embedding)
    scope = selected_scope if selected_scope.is_dir() or len(selected_documents) != len(manifest["documents"]) else legacy_scope
    if not scope.is_dir():
        return {**base, "status": "not_indexed"}
    try:
        active = active_directory(scope)
        stamp = _read_json(active / "source-manifest.json")
        docs = _read_json(active / "docs.json")
        if not isinstance(stamp, dict) or not isinstance(docs, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("file_path"), str)
            or not isinstance(item.get("content"), str)
            for item in docs
        ):
            raise ValueError("Published index metadata is invalid.")
        storage = stamp.get("storage")
        if not isinstance(storage, dict) or storage.get("backend") not in {"lancedb", "qdrant"}:
            raise ValueError("Published index storage is invalid.")
        if storage["backend"] == "lancedb" and not _lancedb_count_matches(active, len(docs)):
            raise ValueError("Published index storage does not match its document metadata.")
        retrieval = retrieval_config()
        contextual = {name: retrieval[name] for name in (
            "contextual_model", "contextual_max_tokens", "contextual_timeout",
            "contextual_source_max_chars", "contextual_start_chunks", "contextual_prev_chunks",
        )}
        document_paths = selected_files if scope == selected_scope else [str(source_root)]
        expected = build_source_stamp({
            **RAG_CONFIG_V2,
            "document_paths": document_paths,
            "use_contextual": contextual_retrieval_enabled(retrieval),
        }, embedding, contextual)
        source_content = source_path.read_text(encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError):
        return {**base, "status": "unavailable"}

    selected = [item for item in docs if item["file_path"] == str(source_path)]
    end = min(len(selected), offset + limit)
    page = selected[offset:end]
    positions = {position for item in page for position in (item.get("start_offset"), item.get("end_offset"))
                 if type(position) is int and 0 <= position <= len(source_content)}
    utf16 = _utf16_positions(source_content, positions)
    chunks = []
    for ordinal, item in enumerate(page, start=offset):
        start, stop = item.get("start_offset"), item.get("end_offset")
        original = item.get("original_content")
        exact = (type(start) is int and type(stop) is int and 0 <= start <= stop <= len(source_content)
                 and isinstance(original, str)
                 and source_content[start:stop].replace("\f", " ") == original)
        chunk = {
            "ordinal": ordinal,
            "location": item.get("location") if isinstance(item.get("location"), str) else None,
            "start_offset": utf16[start] if exact else None,
            "end_offset": utf16[stop] if exact else None,
        }
        if include_content:
            chunk.update(
                original_content=original if isinstance(original, str) else None,
                indexed_content=item["content"],
                heading=item.get("heading") if isinstance(item.get("heading"), str) else None,
            )
        chunks.append(chunk)
    return {
        **base,
        "status": "current" if stamp == expected else "stale",
        "total_chunks": len(selected),
        "next_offset": end if end < len(selected) else None,
        "chunks": chunks,
    }
