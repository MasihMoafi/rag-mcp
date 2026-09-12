"""Open published indexes or build complete replacement generations."""

import json
from pathlib import Path
import shutil
import tempfile
from filelock import FileLock

from utils.persistence import write_json_atomic


class RebuildRequired(RuntimeError):
    """Persisted storage cannot be reused; rebuild away from the active index."""


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


def open_or_build(config, build):
    """The builder must close its own resources if construction fails."""
    root = Path(config.get('index_scope', config['persist_dir']))
    active = active_directory(root)
    config = {**config, 'index_scope': str(root), 'persist_dir': str(active)}
    if not config.get('force_reindex', False) and active.is_dir() and any(active.iterdir()):
        try:
            return build(config)
        except RebuildRequired:
            pass

    root.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix='generation-', dir=root))
    instance = None
    try:
        instance = build({**config, 'persist_dir': str(pending), 'force_reindex': True})
        if 'source_stamp' in config:
            write_json_atomic(pending / 'source-manifest.json', config['source_stamp'])
        # Storage and metadata must succeed before this generation becomes active.
        with FileLock(str(root / '.publication.lock'), timeout=30):
            if active_directory(root) != active:
                raise RuntimeError('Index changed during another refresh; retry the query.')
            write_json_atomic(root / 'active-index.json', pending.name)
        return instance
    except Exception:
        if instance is not None and instance.config.get('backend', 'lancedb') == 'qdrant':
            instance.vector_db.client.close()
        shutil.rmtree(pending)
        raise
