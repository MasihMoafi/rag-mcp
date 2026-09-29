import io
import json
from pathlib import Path
import tarfile

import pytest

from rag.provenance import _studio_build
from rag.studio_build_manifest import read_bundle_manifest
from scripts import build_studio_bundle


@pytest.fixture
def staged_web(tmp_path, monkeypatch):
    root = tmp_path / 'web-repo'
    web = root / 'apps' / 'web'
    (web / '.next' / 'standalone' / 'apps' / 'web' / '.next').mkdir(parents=True)
    (web / '.next' / 'static').mkdir()
    (root / 'packages' / 'ui').mkdir(parents=True)
    (root / 'node_modules' / '.pnpm').mkdir(parents=True)
    (web / 'page.tsx').write_text('export default function Page() { return null }')
    (web / '.env.local').write_text('PRIVATE_BUILD_KEY=do-not-record')
    (root / 'packages' / 'ui' / 'index.ts').write_text('export const label = "Studio"')
    (root / 'package.json').write_text('{"packageManager":"pnpm@10"}')
    (root / 'pnpm-lock.yaml').write_text('lockfileVersion: 9.0')
    (root / 'node_modules' / '.pnpm' / 'lock.yaml').write_text('lockfileVersion: 9.0')
    (web / '.next' / 'BUILD_ID').write_text('build-one')
    (web / '.next' / 'standalone' / 'apps' / 'web' / '.next' / 'BUILD_ID').write_text('build-one')
    (web / '.next' / 'standalone' / 'apps' / 'web' / 'server.js').write_text('console.log("Studio")')
    (web / '.next' / 'static' / 'asset.js').write_text('asset')
    engine = tmp_path / 'engine'
    (engine / 'rag').mkdir(parents=True)
    monkeypatch.setattr(build_studio_bundle, 'ROOT', engine)
    monkeypatch.setattr(build_studio_bundle.subprocess, 'run', lambda command, **kwargs: None)
    return root, web


def test_fresh_build_records_source_and_skip_build_refuses_attestation(staged_web):
    root, web = staged_web
    bundle = build_studio_bundle.stage(web)
    measured = read_bundle_manifest(bundle)
    assert measured['attestation'] == 'measured'
    assert measured['build_id'] == 'build-one'
    assert measured['source']['installed_lock_sha256']
    assert measured['source']['local_env_files_present'] == ['apps/web/.env.local']
    assert 'PRIVATE_BUILD_KEY' not in json.dumps(measured)
    assert all(not item['path'].startswith('/') for item in measured['source']['source_files'])
    assert _studio_build(bundle)['source_sha256'] == measured['source']['source_sha256']

    (web / 'page.tsx').write_text('changed after the build')
    skipped = read_bundle_manifest(build_studio_bundle.stage(web, build=False))
    assert skipped['attestation'] == 'unattested' and skipped['source'] is None
    assert _studio_build(bundle)['source_sha256'] is None


def test_source_mutation_and_lock_mismatch_abort_before_publication(staged_web, monkeypatch):
    root, web = staged_web
    bundle = build_studio_bundle.stage(web)
    original = bundle.read_bytes()
    def edit_during_build(command, **kwargs):
        (web / 'page.tsx').write_text('changed while building')
    monkeypatch.setattr(build_studio_bundle.subprocess, 'run', edit_during_build)
    with pytest.raises(RuntimeError, match='changed during the build'):
        build_studio_bundle.stage(web)
    assert bundle.read_bytes() == original

    (root / 'node_modules' / '.pnpm' / 'lock.yaml').write_text('stale lock')
    with pytest.raises(RuntimeError, match='Installed pnpm lock'):
        build_studio_bundle.stage(web)
    assert bundle.read_bytes() == original


def test_mismatched_build_ids_and_tampered_manifest_are_rejected(staged_web, tmp_path):
    _, web = staged_web
    bundle = build_studio_bundle.stage(web, build=False)
    (web / '.next' / 'standalone' / 'apps' / 'web' / '.next' / 'BUILD_ID').write_text('older-build')
    with pytest.raises(RuntimeError, match='build IDs differ'):
        build_studio_bundle.stage(web, build=False)

    tampered = tmp_path / 'tampered.tar.gz'
    with tarfile.open(bundle, 'r:gz') as source, tarfile.open(tampered, 'w:gz') as target:
        for item in source.getmembers():
            payload = source.extractfile(item).read() if item.isfile() else None
            if item.name == 'rag-studio-build-manifest.json':
                data = json.loads(payload)
                data['build_id'] = 'wrong-build'
                payload = json.dumps(data).encode()
                item.size = len(payload)
            target.addfile(item, io.BytesIO(payload) if payload is not None else None)
    with pytest.raises(ValueError, match='identity'):
        read_bundle_manifest(tampered)
    assert _studio_build(tampered) == {'attestation': 'unavailable'}
