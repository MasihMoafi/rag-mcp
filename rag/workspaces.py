"""Private, versioned document workspaces shared by Studio, CLI, and MCP."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Mapping

from filelock import FileLock


_WORKSPACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_MARKDOWN_TYPES = {"markdown", "latex"}


class WorkspaceConflict(ValueError):
    """Another client changed the snapshot the caller based its edit on."""


def _default_root(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    configured = environ.get("RAG_MCP_WORKSPACE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    state_home = Path(environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return state_home / "ragmcp" / "workspaces"


class WorkspaceStore:
    @staticmethod
    def document_filename(document_id: str, source_type: str) -> str:
        suffix = ".md" if source_type in _MARKDOWN_TYPES else ".txt"
        return hashlib.sha256(document_id.encode()).hexdigest() + suffix

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        max_document_bytes: int = 16 * 1024 * 1024,
        max_total_bytes: int = 64 * 1024 * 1024,
        max_documents: int = 500,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else _default_root(environ)
        self.max_document_bytes = max_document_bytes
        self.max_total_bytes = max_total_bytes
        self.max_documents = max_documents

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> "WorkspaceStore":
        return cls(environ=environ)

    @staticmethod
    def validate_id(workspace_id: str) -> str:
        if not isinstance(workspace_id, str) or _WORKSPACE_ID.fullmatch(workspace_id) is None:
            raise ValueError("workspace id must use 1–64 letters, numbers, underscores, or hyphens")
        return workspace_id

    def _workspace(self, workspace_id: str) -> Path:
        return self.root / self.validate_id(workspace_id)

    def _lock(self) -> FileLock:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        return FileLock(self.root / ".write.lock", timeout=5)

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError, TypeError) as error:
            raise OSError("workspace metadata is unavailable") from error
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise OSError("workspace metadata is invalid")
        return value

    def replace(
        self,
        workspace_id: str,
        name: str,
        documents: list[dict[str, Any]],
        *,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        workspace_id = self.validate_id(workspace_id)
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
            raise ValueError("workspace name must contain 1–80 characters")
        if not isinstance(documents, list) or len(documents) > self.max_documents:
            raise ValueError(f"workspace exceeds the {self.max_documents}-document limit")

        normalized: list[tuple[dict[str, Any], bytes]] = []
        total_bytes = 0
        seen_ids: set[str] = set()
        for document in documents:
            if not isinstance(document, dict):
                raise ValueError("each document must be an object")
            document_id = document.get("id")
            title = document.get("name")
            source_type = document.get("source_type", document.get("sourceType"))
            content = document.get("content")
            if not isinstance(document_id, str) or not document_id or len(document_id) > 256:
                raise ValueError("document id must contain 1–256 characters")
            if document_id in seen_ids:
                raise ValueError("document ids must be unique")
            if not isinstance(title, str) or not title.strip() or len(title.strip()) > 512:
                raise ValueError("document name must contain 1–512 characters")
            if not isinstance(source_type, str) or not source_type or len(source_type) > 64:
                raise ValueError("document source type is invalid")
            if not isinstance(content, str):
                raise ValueError("document content must be text")
            encoded = content.encode("utf-8")
            if len(encoded) > self.max_document_bytes:
                raise ValueError(f"document exceeds the {self.max_document_bytes}-byte limit")
            total_bytes += len(encoded)
            if total_bytes > self.max_total_bytes:
                raise ValueError(f"workspace exceeds the {self.max_total_bytes}-byte limit")
            seen_ids.add(document_id)
            normalized.append(({
                "id": document_id,
                "name": title.strip(),
                "source_type": source_type,
                "size_bytes": len(encoded),
                "content_sha256": hashlib.sha256(encoded).hexdigest(),
            }, encoded))

        generation = f"{int(time.time() * 1000)}-{secrets.token_hex(6)}"
        workspace = self._workspace(workspace_id)
        generation_root = workspace / "generations" / generation
        document_root = generation_root / "documents"
        metadata: list[dict[str, Any]] = []

        with self._lock():
            manifest_path = workspace / "manifest.json"
            previous = self._read_manifest(manifest_path) if manifest_path.exists() else None
            if (previous["generation"] if previous else None) != expected_revision:
                raise WorkspaceConflict("Workspace changed in another client. Reload and reconcile before saving.")
            if previous and previous["name"] == name.strip() and [
                {key: value for key, value in item.items() if key != "stored_file"}
                for item in previous["documents"]
            ] == [item for item, _ in normalized]:
                return self._summary(previous)
            document_root.mkdir(parents=True, mode=0o700)
            workspace.chmod(0o700)
            (workspace / "generations").chmod(0o700)
            generation_root.chmod(0o700)
            document_root.chmod(0o700)
            try:
                for document, encoded in normalized:
                    stored_file = self.document_filename(document["id"], document["source_type"])
                    target = document_root / stored_file
                    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                    metadata.append({**document, "stored_file": stored_file})

                manifest = {
                    "schema_version": 1,
                    "id": workspace_id,
                    "name": name.strip(),
                    "generation": generation,
                    "document_count": len(metadata),
                    "total_bytes": total_bytes,
                    "updated_unix": time.time(),
                    "documents": metadata,
                }
                temporary = workspace / f"manifest-{generation}.tmp"
                with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
                    json.dump(manifest, stream, ensure_ascii=False, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(workspace / "manifest.json")
            except BaseException:
                import shutil
                shutil.rmtree(generation_root, ignore_errors=True)
                raise
        return self._summary(manifest)

    def get(self, workspace_id: str) -> dict[str, Any]:
        manifest = self._read_manifest(self._workspace(workspace_id) / "manifest.json")
        return self._summary(manifest)

    def describe(self, workspace_id: str) -> dict[str, Any]:
        """Client-facing source IDs and metadata without stored paths or text hashes."""
        value = self.get(workspace_id)
        return {
            **{key: value[key] for key in ("schema_version", "revision", "id", "name", "document_count", "total_bytes")},
            "documents": [
                {key: item[key] for key in ("id", "name", "source_type", "size_bytes")}
                for item in value["documents"]
            ],
        }

    @staticmethod
    def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "revision": manifest["generation"],
            "id": manifest["id"],
            "name": manifest["name"],
            "document_count": manifest["document_count"],
            "total_bytes": manifest["total_bytes"],
            "documents": manifest["documents"],
        }

    def snapshot(self, workspace_id: str) -> dict[str, Any]:
        """Private document endpoint for paired clients; never included in traces."""
        workspace = self._workspace(workspace_id)
        manifest = self._read_manifest(workspace / "manifest.json")
        root = workspace / "generations" / manifest["generation"] / "documents"
        return {**self._summary(manifest), "documents": [
            {"id": item["id"], "name": item["name"], "sourceType": item["source_type"],
             "content": (root / item["stored_file"]).read_text(encoding="utf-8")}
            for item in manifest["documents"]
        ]}

    def list_workspaces(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        workspaces = []
        for path in self.root.iterdir():
            if not path.is_dir() or _WORKSPACE_ID.fullmatch(path.name) is None:
                continue
            try:
                item = self.get(path.name)
            except OSError:
                continue
            workspaces.append({key: item[key] for key in ("id", "name", "document_count", "total_bytes")})
        return sorted(workspaces, key=lambda item: (item["name"].casefold(), item["id"]))

    def resolve(self, workspace_id: str) -> Path:
        workspace = self._workspace(workspace_id)
        manifest = self._read_manifest(workspace / "manifest.json")
        documents = workspace / "generations" / manifest["generation"] / "documents"
        if not documents.is_dir():
            raise OSError("workspace documents are unavailable")
        return documents
