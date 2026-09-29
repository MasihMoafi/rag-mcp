import io
import os
import socket
import tarfile

import pytest

from rag import studio_bundle


def archive_with(*names):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        for name in names:
            item = tarfile.TarInfo(name)
            item.size = 1
            archive.addfile(item, io.BytesIO(b'x'))
    data.seek(0)
    return data


def test_bundle_requires_server_build_id_and_assets():
    with tarfile.open(fileobj=archive_with('apps/web/server.js', 'apps/web/.next/BUILD_ID',
                                           'apps/web/.next/static/test.js'), mode='r:gz') as archive:
        studio_bundle._validate_archive(archive)
    with tarfile.open(fileobj=archive_with('apps/web/server.js'), mode='r:gz') as archive:
        with pytest.raises(studio_bundle.StudioBundleError, match='incomplete'):
            studio_bundle._validate_archive(archive)


def test_bundle_rejects_unsafe_entries():
    with tarfile.open(fileobj=archive_with('../escape'), mode='r:gz') as archive:
        with pytest.raises(studio_bundle.StudioBundleError, match='unsafe'):
            studio_bundle._validate_archive(archive)
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        item = tarfile.TarInfo('apps/web/node_modules/next')
        item.type = tarfile.SYMTYPE
        item.linkname = '../../../../outside'
        archive.addfile(item)
    data.seek(0)
    with tarfile.open(fileobj=data, mode='r:gz') as archive:
        with pytest.raises(studio_bundle.StudioBundleError, match='unsafe'):
            studio_bundle._validate_archive(archive)


def test_missing_node_and_occupied_port_fail_without_launch(monkeypatch):
    monkeypatch.setattr(studio_bundle.shutil, 'which', lambda *_args, **_kwargs: None)
    with pytest.raises(studio_bundle.StudioBundleError, match='Node.js 22'):
        with studio_bundle.serve_studio(3000):
            pass
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        with pytest.raises(studio_bundle.StudioBundleError, match='already in use'):
            studio_bundle._check_port(occupied.getsockname()[1])


@pytest.mark.skipif(os.name != 'posix', reason='POSIX TIME_WAIT socket behavior')
def test_recently_closed_studio_port_can_restart():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with socket.create_connection(('127.0.0.1', port)) as client:
            peer, _ = listener.accept()
            peer.close()
            client.recv(1)
    studio_bundle._check_port(port)
