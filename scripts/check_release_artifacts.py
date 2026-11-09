"""Check the exact release packages and their version before publishing."""
import email
import json
from pathlib import Path
import sys
import tarfile
import tomllib
import zipfile


root = Path(__file__).resolve().parents[1]
dist = Path(sys.argv[1])
tag = sys.argv[2]
with (root / 'pyproject.toml').open('rb') as stream:
    version = tomllib.load(stream)['project']['version']
if tag != f'v{version}':
    raise SystemExit(f'Tag {tag!r} does not match v{version}')
expected = {f'ragmcp_cli-{version}-py3-none-any.whl', f'ragmcp_cli-{version}.tar.gz', f'ragmcp-{version}.tgz'}
actual = {path.name for path in dist.iterdir() if path.is_file()}
if actual != expected:
    raise SystemExit(f'Unexpected artifacts: {sorted(actual)}; expected {sorted(expected)}')

with zipfile.ZipFile(dist / f'ragmcp_cli-{version}-py3-none-any.whl') as wheel:
    names = wheel.namelist()
    metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
    if len(metadata_names) != 1 or 'rag/server.py' not in names or 'rag/cli.py' not in names:
        raise SystemExit('Wheel is missing metadata, server or CLI')
    metadata = email.message_from_bytes(wheel.read(metadata_names[0]))
    if metadata['Name'] != 'ragmcp-cli' or metadata['Version'] != version:
        raise SystemExit('Wheel metadata does not match the release')
    if any('studio_dist/' in name for name in names):
        raise SystemExit('Studio bundle must not be released')

with tarfile.open(dist / f'ragmcp_cli-{version}.tar.gz') as sdist:
    names = sdist.getnames()
    if not any(name.endswith('/rag/server.py') for name in names) or any('/studio_dist/' in name for name in names):
        raise SystemExit('Source distribution is missing server or contains Studio')

with tarfile.open(dist / f'ragmcp-{version}.tgz') as npm:
    manifest = npm.extractfile('package/package.json')
    if manifest is None:
        raise SystemExit('npm manifest missing')
    package = json.load(manifest)
    if package['name'] != 'ragmcp' or package['version'] != version or 'package/bin/rag.js' not in npm.getnames():
        raise SystemExit('npm package metadata or launcher does not match the release')
print(f'Validated wheel, sdist and npm launcher for {tag}')
