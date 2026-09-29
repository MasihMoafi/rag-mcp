"""Paired-client run results, separate from the metadata-only trace interface."""
import json
import os
from pathlib import Path
import re
import stat
import time
import tempfile

from filelock import FileLock
from rag import tracing


class RunResultStore:
    def __init__(self, root=None):
        trace_root = tracing._directory()
        self.root = Path(root) if root is not None else trace_root / 'results' if trace_root else None

    def _path(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r'[0-9a-f]{32}', run_id):
            raise ValueError('Invalid run id')
        return self.root / f'{run_id}.json' if self.root is not None else None

    def read(self, run_id):
        path = self._path(run_id)
        if path is None:
            return None
        try:
            with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)), 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
                    raise OSError('Invalid run result')
                value = json.load(stream)
            if not isinstance(value, dict) or value.get('schema_version') != 1 or value.get('run_id') != run_id:
                raise OSError('Invalid run result')
            return value
        except FileNotFoundError:
            return None
        except (ValueError, UnicodeError) as error:
            raise OSError('Invalid run result') from error

    def save(self, result):
        path = self._path(result['run_id'])
        if path is None:
            return
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        data = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
        if len(data) > 8 * 1024 * 1024:
            raise OSError('Run result exceeds storage limit')
        with FileLock(self.root / '.write.lock', timeout=1):
            files = sorted((p for p in self.root.glob('*.json') if re.fullmatch(r'[0-9a-f]{32}\.json', p.name)), key=lambda p: p.stat().st_mtime)
            total = sum(p.stat().st_size for p in files)
            for old in files:
                if old.stat().st_mtime < time.time() - 7 * 86400 or total + len(data) > 100 * 1024 * 1024:
                    total -= old.stat().st_size
                    old.unlink()
            descriptor, name = tempfile.mkstemp(prefix='.result-', dir=self.root)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
