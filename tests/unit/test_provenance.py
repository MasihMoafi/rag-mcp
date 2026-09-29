import json
from pathlib import Path
from types import SimpleNamespace

from rag import provenance


def test_manifest_fingerprints_code_dependencies_bundle_and_git(tmp_path, monkeypatch):
    (tmp_path / 'rag').mkdir()
    (tmp_path / 'utils').mkdir()
    (tmp_path / 'rag' / '__init__.py').write_text('PRIVATE_CODE = 1\n')
    (tmp_path / 'utils' / '__init__.py').write_text('')
    (tmp_path / 'server.py').write_text('from rag import provenance\n')
    (tmp_path / 'rag' / 'studio_dist').mkdir()
    (tmp_path / 'rag' / 'studio_dist' / 'bundle.tar.gz').write_bytes(b'bundled Studio')
    (tmp_path / '.git').mkdir()
    monkeypatch.setattr(provenance.metadata, 'version', lambda _: '0.2.0')
    monkeypatch.setattr(provenance.metadata, 'distributions', lambda: [
        SimpleNamespace(metadata={'Name': 'Test_Package'}, version='1.2.3'),
        SimpleNamespace(metadata={'Name': 'private/path'}, version='file:///private')])
    status = {'value': ''}
    def git(command, **_kwargs):
        return SimpleNamespace(stdout='a' * 40 if command[1] == 'rev-parse' else status['value'])
    monkeypatch.setattr(provenance.subprocess, 'run', git)

    first = provenance.capture_manifest(tmp_path)
    assert first == provenance.capture_manifest(tmp_path)
    assert first['dependencies']['packages'] == [{'name': 'test-package', 'version': '1.2.3'}]
    assert first['dependencies']['unidentified_count'] == 1
    assert first['git'] == {'head': 'a' * 40, 'state': 'clean'}
    assert first['studio_bundle_sha256'] is not None
    assert str(tmp_path) not in json.dumps(first) and 'PRIVATE_CODE' not in json.dumps(first)
    assert 'private/path' not in json.dumps(first) and 'file:///private' not in json.dumps(first)
    status['value'] = ' M rag/__init__.py\n'
    dirty = provenance.capture_manifest(tmp_path)
    assert dirty['git']['state'] == 'dirty' and dirty['sha256'] != first['sha256']
    (tmp_path / 'rag' / '__init__.py').write_text('PRIVATE_CODE = 2\n')
    assert provenance.capture_manifest(tmp_path)['sha256'] != dirty['sha256']
    (tmp_path / '.git').rmdir()
    assert provenance.capture_manifest(tmp_path)['git']['state'] == 'unavailable'


def test_manifest_rejects_symlinked_source_file(tmp_path):
    (tmp_path / 'rag').mkdir()
    (tmp_path / 'utils').mkdir()
    (tmp_path / 'rag' / 'private.py').symlink_to(Path(__file__).resolve())
    try:
        provenance.capture_manifest(tmp_path)
    except ValueError as error:
        assert 'bounded regular file' in str(error)
    else:
        raise AssertionError('Symlinked source must not be read')
