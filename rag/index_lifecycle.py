"""Open published indexes or build complete replacement generations."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from filelock import FileLock

from utils.persistence import write_json_atomic

from .formats import INDEXABLE_EXTENSIONS


class RebuildRequired(RuntimeError):
    """Persisted storage cannot be reused; rebuild away from the active index."""


def selected_file_scope(root: str, paths: list[str]) -> str:
    """Name the exact selected-file index shared by Studio, CLI, and MCP."""
    digest = hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()[:16]
    return f"{root}:{digest}"


def source_snapshot(document_paths):
    """Cheap filesystem revision check; not an immutable content snapshot."""
    sources = []
    for document_path in document_paths:
        source = Path(document_path)
        paths = source.rglob('*') if source.is_dir() else [source]
        for path in paths:
            # Inside a folder only the files the index reads can invalidate it:
            # a hidden or unsupported file is never indexed, so touching it is
            # not a change. An explicitly named file is always read.
            if source.is_dir() and not _indexable(path.relative_to(source)):
                continue
            if path.is_file():
                stat = path.stat()
                sources.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    return sorted(sources)


def _indexable(relative_path):
    return (relative_path.name.lower().endswith(INDEXABLE_EXTENSIONS)
            and not any(part.startswith('.') for part in relative_path.parts))


def source_delta(previous_stamp, stamp):
    """Sources to re-read and to drop when nothing but the file list changed."""
    if not isinstance(previous_stamp, dict) or previous_stamp == stamp:
        return None
    settings = lambda manifest: {name: value for name, value in manifest.items() if name != 'sources'}
    if settings(previous_stamp) != settings(stamp):
        return None  # Chunking, parsing, storage or the embedding model changed.
    previous = {entry[0]: tuple(entry[1:]) for entry in previous_stamp.get('sources', [])}
    current = {entry[0]: tuple(entry[1:]) for entry in stamp['sources']}
    delta = {'changed': sorted(path for path, revision in current.items() if previous.get(path) != revision),
             'removed': sorted(set(previous) - set(current))}
    return delta if delta['changed'] or delta['removed'] else None


def active_directory(scope_root):
    root = Path(scope_root)
    try:
        generation = json.loads((root / 'active-index.json').read_text())
    except FileNotFoundError:
        return root  # Legacy indexes stored data directly in their scope directory.
    if not isinstance(generation, str) or not generation.startswith('generation-') or Path(generation).name != generation:
        raise ValueError('Invalid active index generation')
    directory = root / generation
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError('Active index generation is missing or invalid')
    return directory


def _build_generation(config, build, pending, active, delta):
    """Stage a replacement: carry the published index over when only sources moved."""
    if delta is not None:
        shutil.copytree(active, pending, dirs_exist_ok=True)
        try:
            return build({**config, 'persist_dir': str(pending), 'force_reindex': False})
        except RebuildRequired:
            shutil.rmtree(pending)
            pending.mkdir()
    return build({**config, 'persist_dir': str(pending), 'force_reindex': True, 'source_delta': None})


def open_or_build(config, build):
    """The builder must close its own resources if construction fails."""
    root = Path(config.get('index_scope', config['persist_dir']))
    active = active_directory(root)
    config = {**config, 'index_scope': str(root), 'persist_dir': str(active)}
    delta = config.get('source_delta')
    if not config.get('force_reindex', False) and delta is None and active.is_dir() and any(active.iterdir()):
        try:
            return build(config)
        except RebuildRequired:
            pass
    if active == root or not active.is_dir():
        delta = None  # A legacy index has no generation to carry forward.

    root.mkdir(parents=True, exist_ok=True)
    document_paths = config.get('document_paths', [])
    sources = config.get('source_stamp', {}).get('sources')
    if sources is None:
        sources = source_snapshot(document_paths)
    pending = Path(tempfile.mkdtemp(prefix='generation-', dir=root))
    instance = None
    published = False
    try:
        instance = _build_generation(config, build, pending, active, delta)
        if 'source_stamp' in config:
            write_json_atomic(pending / 'source-manifest.json', config['source_stamp'])
        # Storage and metadata must succeed before this generation becomes active.
        with FileLock(str(root / '.publication.lock'), timeout=30):
            if active_directory(root) != active:
                raise RuntimeError('Index changed during another refresh; retry the query.')
            if source_snapshot(document_paths) != sources:
                raise RuntimeError('Sources changed during indexing; retry the query.')
            published = True  # Once publication starts, retain data unless proven unpublished.
            try:
                write_json_atomic(root / 'active-index.json', pending.name)
            except BaseException:
                # Check under the publication lock: replacement may have succeeded.
                try:
                    published = active_directory(root) == pending
                except (OSError, ValueError):
                    pass
                raise
        return instance
    except BaseException as error:
        try:
            close = getattr(instance, 'close', None)
            if close is not None:
                close()
            if not published:
                shutil.rmtree(pending)
        except Exception as cleanup_error:
            error.add_note(f"Staging cleanup failed ({type(cleanup_error).__name__}); inspect {pending}")
        raise
