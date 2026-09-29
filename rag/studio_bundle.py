"""Launch the Studio bundled in the installed ragmcp package on loopback."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
from urllib.request import ProxyHandler, Request, build_opener


class StudioBundleError(RuntimeError):
    pass


def _archive_path() -> Path:
    return Path(__file__).resolve().parent / 'studio_dist' / 'bundle.tar.gz'


def _validate_archive(archive: tarfile.TarFile) -> None:
    names = set()
    total = 0
    for item in archive.getmembers():
        path = PurePosixPath(item.name)
        if path.is_absolute() or '..' in path.parts or not item.name or not (item.isfile() or item.isdir() or item.issym()):
            raise StudioBundleError('Studio bundle contains an unsafe archive entry.')
        if item.issym():
            target = posixpath.normpath(posixpath.join(posixpath.dirname(item.name), item.linkname))
            if item.linkname.startswith('/') or target == '..' or target.startswith('../'):
                raise StudioBundleError('Studio bundle contains an unsafe link.')
        total += item.size
        names.add(item.name)
    if total > 300 * 1024 * 1024 or not {'apps/web/server.js', 'apps/web/.next/BUILD_ID'}.issubset(names) or not any(
            name.startswith('apps/web/.next/static/') for name in names):
        raise StudioBundleError('Studio bundle is incomplete or too large.')


def _check_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise StudioBundleError('Studio web port must be between 1 and 65535.')
    with socket.socket() as probe:
        try:
            if os.name == 'posix':
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
        except OSError as error:
            raise StudioBundleError(f'Local Studio web port {port} is already in use.') from error


@contextmanager
def serve_studio(port: int = 3000, *, environ: Mapping[str, str] | None = None) -> Iterator[str]:
    values = dict(os.environ if environ is None else environ)
    node = shutil.which('node', path=values.get('PATH'))
    if not node:
        raise StudioBundleError('Node.js 22 or newer is required to run the bundled Studio.')
    version = subprocess.run([node, '--version'], capture_output=True, text=True, check=False)
    match = re.fullmatch(r'v(\d+)\.\d+\.\d+\s*', version.stdout)
    if version.returncode or not match or int(match[1]) < 22:
        raise StudioBundleError('Node.js 22 or newer is required to run the bundled Studio.')
    _check_port(port)
    bundle = _archive_path()
    if not bundle.is_file():
        raise StudioBundleError('Bundled Studio is missing from this installation.')
    with tempfile.TemporaryDirectory(prefix='rag-studio-') as temporary:
        with tarfile.open(bundle, 'r:gz') as archive:
            _validate_archive(archive)
            archive.extractall(temporary)
        server = Path(temporary) / 'apps' / 'web' / 'server.js'
        url = f'http://127.0.0.1:{port}'
        process = subprocess.Popen([node, str(server)], cwd=server.parent,
            env={**values, 'HOSTNAME': '127.0.0.1', 'PORT': str(port)},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise StudioBundleError('Bundled Studio exited during startup.')
                try:
                    with opener.open(Request(f'{url}/studio'), timeout=0.5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(0.1)
            else:
                raise StudioBundleError('Bundled Studio did not become ready within 15 seconds.')
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=3)
