"""Read-only chunk preview for one immutable workspace source generation."""

from __future__ import annotations

from typing import Any

from rag.chunking import chunk_document_spans
from rag.workspaces import WorkspaceConflict, WorkspaceStore


def _utf16_positions(content: str, positions: set[int]) -> dict[int, int]:
    """Convert Python source positions to offsets used by browser string slices."""
    converted: dict[int, int] = {}
    cursor = units = 0
    for position in sorted(positions):
        units += len(content[cursor:position].encode("utf-16-le")) // 2
        converted[position] = units
        cursor = position
    return converted


def preview_workspace_chunks(
    store: WorkspaceStore,
    workspace_id: str,
    document_id: str,
    revision: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    if not isinstance(revision, str) or not revision:
        raise ValueError("A workspace revision is required.")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("A document id is required.")
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
        raise ValueError("Invalid chunk page parameters.")

    manifest = store.get(workspace_id)
    if manifest["revision"] != revision:
        raise WorkspaceConflict("Workspace changed in another client. Reload before inspecting chunks.")
    document = next((item for item in manifest["documents"] if item["id"] == document_id), None)
    if document is None:
        raise FileNotFoundError("Source not found.")
    path = store.document_path(workspace_id, revision, document)
    source_content = path.read_text(encoding="utf-8")

    # Ingestion reads this same normalized file and uses these process settings.
    # Import lazily so simple workspace metadata reads never import the pipeline.
    from rag.index_config import RAG_CONFIG_V2

    size = RAG_CONFIG_V2["chunk_size"]
    overlap = RAG_CONFIG_V2["chunk_overlap"]
    chunks = chunk_document_spans(source_content, file_path=path.name, chunk_size=size, chunk_overlap=overlap)
    end = min(len(chunks), offset + limit)
    selected: list[dict[str, Any]] = []
    for ordinal, item in enumerate(chunks[offset:end], start=offset):
        selected.append({
            "ordinal": ordinal,
            "content": item["content"],
            "heading": item["heading"],
            "start_offset": item["start_offset"],
            "end_offset": item["end_offset"],
        })
    positions = {position for item in selected for position in (item["start_offset"], item["end_offset"])
                 if position is not None}
    utf16 = _utf16_positions(source_content, positions)
    for item in selected:
        for key in ("start_offset", "end_offset"):
            if item[key] is not None:
                item[key] = utf16[item[key]]
    return {
        "schema_version": 1,
        "workspace_id": workspace_id,
        "revision": revision,
        "document_id": document_id,
        "name": document["name"],
        "source_type": document["source_type"],
        "source_content": source_content,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "total_chunks": len(chunks),
        "offset": offset,
        "next_offset": end if end < len(chunks) else None,
        "chunks": selected,
    }


def inspect_workspace_chunks(
    store: WorkspaceStore,
    workspace_id: str,
    document_id: str,
    revision: str | None = None,
    *,
    limit: int = 50,
    offset: int = 0,
    include_content: bool = False,
) -> dict[str, Any]:
    """CLI/MCP projection: metadata by default, page text only on explicit opt-in."""
    if type(include_content) is not bool:
        raise ValueError("include_content must be a boolean.")
    if revision is None:
        revision = store.get(workspace_id)["revision"]
    preview = preview_workspace_chunks(store, workspace_id, document_id, revision, limit=limit, offset=offset)
    return {
        **{key: value for key, value in preview.items() if key not in {"source_content", "chunks"}},
        "chunks": [{key: value for key, value in item.items() if include_content or key not in {"content", "heading"}}
                   for item in preview["chunks"]],
    }
