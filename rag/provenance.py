"""Bounded, private runtime provenance captured once per engine process."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess

from rag.studio_build_manifest import read_bundle_manifest


def _file_sha256(path: Path, *, limit: int) -> str:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError('Runtime artifact is not a bounded regular file.')
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > limit or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError('Runtime artifact changed during capture.')
        total = 0
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            total += len(block)
            if total > limit:
                raise ValueError('Runtime artifact exceeds its bound.')
            digest.update(block)
        finished = os.fstat(stream.fileno())
        if (opened.st_size, opened.st_mtime_ns) != (finished.st_size, finished.st_mtime_ns):
            raise ValueError('Runtime artifact changed during capture.')
    return digest.hexdigest()


def _code_files(root: Path) -> list[dict[str, str]]:
    paths = []
    for package in ('rag', 'utils'):
        base = root / package
        if not base.is_dir() or base.is_symlink():
            raise ValueError('Engine package is unavailable.')
        for folder, directories, files in os.walk(base, followlinks=False):
            directories[:] = [name for name in directories
                              if not name.startswith(('.', 'rag_db'))
                              and name not in {'__pycache__', 'studio_dist'}]
            paths.extend(Path(folder) / name for name in files if name.endswith('.py'))
    paths.extend(root / name for name in ('server.py', 'pyproject.toml', 'hatch_build.py')
                 if (root / name).exists())
    if len(paths) > 256:
        raise ValueError('Engine source inventory exceeds its bound.')
    return [{'path': path.relative_to(root).as_posix(), 'sha256': _file_sha256(path, limit=2 * 1024 * 1024)}
            for path in sorted(paths)]


def _dependencies() -> dict:
    packages = []
    unidentified = 0
    for distribution in metadata.distributions():
        name = distribution.metadata.get('Name')
        version = distribution.version
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name)
                or not isinstance(version, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9!+._-]*', version)):
            unidentified += 1
            continue
        packages.append({'name': re.sub(r'[-_.]+', '-', name).lower(), 'version': version})
        if len(packages) + unidentified > 5000:
            raise ValueError('Dependency inventory exceeds its bound.')
    return {'packages': sorted(packages, key=lambda item: (item['name'], item['version'])),
            'unidentified_count': unidentified}


def _git(root: Path) -> dict:
    unavailable = {'head': None, 'state': 'unavailable'}
    if not (root / '.git').exists():
        return unavailable
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                              text=True, timeout=2, check=True).stdout.strip()
        if not re.fullmatch(r'[0-9a-f]{40,64}', head):
            return unavailable
        status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=normal', '--',
                                 'rag', 'utils', 'server.py', 'pyproject.toml', 'hatch_build.py'],
                                cwd=root, capture_output=True, text=True, timeout=2, check=True)
        return {'head': head, 'state': 'dirty' if status.stdout else 'clean'}
    except (OSError, subprocess.SubprocessError):
        return unavailable


def _studio_build(bundle: Path) -> dict | None:
    if not bundle.exists():
        return None
    try:
        value = read_bundle_manifest(bundle)
    except ValueError:
        return {'attestation': 'unavailable'}
    source = value['source']
    return {'attestation': value['attestation'], 'build_id': value['build_id'],
            'server_js_sha256': value['server_js_sha256'],
            'source_sha256': source['source_sha256'] if source else None,
            'source_file_count': len(source['source_files']) if source else None,
            'installed_lock_sha256': source['installed_lock_sha256'] if source else None,
            'web_git': source['git'] if source else {'head': None, 'state': 'unavailable'},
            'local_env_files_present': bool(source['local_env_files_present']) if source else None}


def capture_manifest(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    bundle = root / 'rag' / 'studio_dist' / 'bundle.tar.gz'
    try:
        version = metadata.version('ragmcp')
    except metadata.PackageNotFoundError:
        version = None
    if version is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9!+._-]*', version):
        version = None
    value = {'schema_version': 1, 'capture_scope': 'process_first_run',
             'engine_version': version,
             'python': {'implementation': platform.python_implementation(),
                        'version': platform.python_version(),
                        'system': platform.system(), 'machine': platform.machine()},
             'code_files': _code_files(root), 'dependencies': _dependencies(),
             'studio_bundle_sha256': _file_sha256(bundle, limit=100 * 1024 * 1024) if bundle.exists() else None,
             'studio_build': _studio_build(bundle),
             'git': _git(root)}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {**value, 'sha256': digest}


@lru_cache(maxsize=1)
def runtime_manifest() -> dict:
    return capture_manifest()
