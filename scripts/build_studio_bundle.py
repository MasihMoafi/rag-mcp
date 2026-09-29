"""Build and stage the sibling Studio's traced Next server for the Python package."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT_FILES = ('package.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml', 'tsconfig.json', 'turbo.json')
_SKIP_DIRS = {'.next', '.turbo', 'node_modules', 'dist', 'coverage', '__pycache__'}


def _digest(path: Path, *, limit: int = 8 * 1024 * 1024) -> str:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise RuntimeError('Web build input is not a bounded regular file.')
    value = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise RuntimeError('Web build input changed during capture.')
        total = 0
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            total += len(block)
            if total > limit:
                raise RuntimeError('Web build input exceeds its bound.')
            value.update(block)
        finished = os.fstat(stream.fileno())
        if (opened.st_size, opened.st_mtime_ns) != (finished.st_size, finished.st_mtime_ns):
            raise RuntimeError('Web build input changed during capture.')
    return value.hexdigest()


def _git_state(root: Path) -> dict:
    unavailable = {'head': None, 'state': 'unavailable'}
    if not (root / '.git').exists():
        return unavailable
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                              text=True, timeout=2, check=True).stdout.strip()
        if not re.fullmatch(r'[0-9a-f]{40,64}', head):
            return unavailable
        changes = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=normal', '--',
                                  'apps/web', 'packages', *_SOURCE_ROOT_FILES], cwd=root,
                                 capture_output=True, text=True, timeout=2, check=True).stdout
        return {'head': head, 'state': 'dirty' if changes else 'clean'}
    except (OSError, subprocess.SubprocessError):
        return unavailable


def _snapshot(web_root: Path) -> dict:
    root = web_root.parents[1]
    paths = [root / name for name in _SOURCE_ROOT_FILES if (root / name).is_file()]
    for base in (web_root, root / 'packages'):
        if not base.is_dir() or base.is_symlink():
            raise RuntimeError('Web build source directory is unavailable.')
        for folder, directories, files in os.walk(base, followlinks=False):
            directories[:] = [name for name in directories if name not in _SKIP_DIRS and not name.startswith('.')]
            paths.extend(Path(folder) / name for name in files
                         if not name.startswith('.env') and not name.endswith('.tsbuildinfo'))
    if len(paths) > 2000:
        raise RuntimeError('Web build source inventory exceeds its bound.')
    files = [{'path': path.relative_to(root).as_posix(), 'sha256': _digest(path)} for path in sorted(paths)]
    source_sha256 = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    installed_lock = root / 'node_modules' / '.pnpm' / 'lock.yaml'
    local_env_files = [path.relative_to(root).as_posix() for base in (root, web_root)
                       for path in base.glob('.env*') if path.name != '.env.example' and path.is_file()]
    return {'source_files': files, 'source_sha256': source_sha256,
            'installed_lock_sha256': _digest(installed_lock) if installed_lock.is_file() else None,
            'local_env_files_present': sorted(local_env_files), 'git': _git_state(root)}


def stage(web_root: Path, *, build: bool = True) -> Path:
    web_root = web_root.resolve()
    before = _snapshot(web_root) if build else None
    if build:
        lock = next((item['sha256'] for item in before['source_files'] if item['path'] == 'pnpm-lock.yaml'), None)
        if lock is None or lock != before['installed_lock_sha256']:
            raise RuntimeError('Installed pnpm lock does not match the web source lock; install dependencies first.')
        env = {**os.environ, 'RAG_STUDIO_BUNDLE': '1'}
        subprocess.run(['pnpm', '--dir', str(web_root), 'build'], check=True, env=env)
        after = _snapshot(web_root)
        if (before['source_sha256'] != after['source_sha256']
                or before['installed_lock_sha256'] != after['installed_lock_sha256']
                or before['local_env_files_present'] != after['local_env_files_present']
                or before['git']['head'] != after['git']['head']):
            raise RuntimeError('Web source or installed dependencies changed during the build; retry from a stable checkout.')
    standalone = web_root / '.next' / 'standalone'
    static = web_root / '.next' / 'static'
    if not (standalone / 'apps/web/server.js').is_file() or not static.is_dir() or not (web_root / '.next/BUILD_ID').is_file():
        raise RuntimeError('Studio standalone build is incomplete. Rebuild with RAG_STUDIO_BUNDLE=1.')
    web = standalone / 'apps' / 'web'
    if not (web / '.next/BUILD_ID').is_file():
        raise RuntimeError('Standalone server lacks its build ID.')
    build_id = (web_root / '.next/BUILD_ID').read_text().strip()
    if not build_id or build_id != (web / '.next/BUILD_ID').read_text().strip():
        raise RuntimeError('Standalone server and Studio build IDs differ.')
    manifest = {'schema_version': 1, 'build_id': build_id,
                'attestation': 'measured' if before is not None else 'unattested',
                'server_js_sha256': _digest(web / 'server.js'),
                'source': before if before is not None else None}
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    if len(manifest_bytes) > 256 * 1024:
        raise RuntimeError('Studio build manifest exceeds its bound.')
    target = ROOT / 'rag' / 'studio_dist' / 'bundle.tar.gz'
    with tempfile.TemporaryDirectory(prefix='rag-studio-stage-', dir=ROOT / 'rag') as temporary:
        staged = Path(temporary) / 'bundle.tar.gz'
        with tarfile.open(staged, 'w:gz', dereference=False) as archive:
            for item in standalone.iterdir():
                archive.add(item, arcname=item.name)
            archive.add(static, arcname='apps/web/.next/static')
            public = web_root / 'public'
            if public.is_dir():
                archive.add(public, arcname='apps/web/public')
            item = tarfile.TarInfo('rag-studio-build-manifest.json')
            item.size = len(manifest_bytes)
            item.mode = 0o600
            archive.addfile(item, io.BytesIO(manifest_bytes))
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web-root', type=Path, default=ROOT.parent / 'rag-studio-next' / 'apps' / 'web')
    parser.add_argument('--skip-build', action='store_true', help='stage an existing standalone build')
    args = parser.parse_args()
    print(stage(args.web_root, build=not args.skip_build))


if __name__ == '__main__':
    main()
