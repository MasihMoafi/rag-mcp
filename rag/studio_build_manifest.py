"""Validate the Studio artifact identity embedded in a package bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tarfile


def read_bundle_manifest(bundle: Path) -> dict:
    try:
        with tarfile.open(bundle, 'r:gz') as archive:
            names = {item.name for item in archive.getmembers()}
            if not any(name.startswith('apps/web/.next/static/') for name in names):
                raise ValueError('Studio static assets are missing.')
            manifest_member = archive.getmember('rag-studio-build-manifest.json')
            build_id_member = archive.getmember('apps/web/.next/BUILD_ID')
            server_member = archive.getmember('apps/web/server.js')
            if (not manifest_member.isfile() or manifest_member.size > 256 * 1024
                    or not build_id_member.isfile() or build_id_member.size > 128
                    or not server_member.isfile() or server_member.size > 2 * 1024 * 1024):
                raise ValueError('Studio bundle members are invalid.')
            value = json.load(archive.extractfile(manifest_member))
            build_id = archive.extractfile(build_id_member).read().decode().strip()
            server_sha256 = hashlib.sha256(archive.extractfile(server_member).read()).hexdigest()
    except (OSError, KeyError, UnicodeError, tarfile.TarError) as error:
        raise ValueError('Studio bundle is incomplete or invalid.') from error
    if (not isinstance(value, dict) or value.get('schema_version') != 1
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', build_id)
            or value.get('build_id') != build_id or value.get('server_js_sha256') != server_sha256
            or value.get('attestation') not in {'measured', 'unattested'}):
        raise ValueError('Studio build identity is invalid.')
    source = value.get('source')
    if value['attestation'] == 'unattested':
        if source is not None:
            raise ValueError('Unattested build claims a source snapshot.')
        return value
    if not isinstance(source, dict) or not isinstance(source.get('source_files'), list):
        raise ValueError('Measured build lacks a source snapshot.')
    files = source['source_files']
    if not files or len(files) > 2000 or any(
            not isinstance(item, dict) or not isinstance(item.get('path'), str)
            or item['path'].startswith('/') or '..' in Path(item['path']).parts
            or any(part.startswith('.env') for part in Path(item['path']).parts)
            or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', '')))
            for item in files):
        raise ValueError('Measured source inventory is invalid.')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    git = source.get('git')
    env_files = source.get('local_env_files_present')
    if (source.get('source_sha256') != digest or not isinstance(git, dict)
            or git.get('state') not in {'clean', 'dirty', 'unavailable'}
            or (git.get('head') is not None and not re.fullmatch(r'[0-9a-f]{40,64}', str(git['head'])))
            or not re.fullmatch(r'[0-9a-f]{64}', str(source.get('installed_lock_sha256', '')))
            or not isinstance(env_files, list)
            or any(not isinstance(name, str) or name.startswith('/') or '..' in Path(name).parts
                   for name in env_files)):
        raise ValueError('Measured source identity is invalid.')
    return value
