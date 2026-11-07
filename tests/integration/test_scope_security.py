import json
import os
import subprocess
import sys

import pytest

from rag import server


def test_explicit_files_obey_scan_filters(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('ignored.md\n')
    for name in ['allowed.md', '.hidden.md', 'ignored.md', 'test_secret.py', 'large.md']:
        (tmp_path / name).write_text('searchable content')
    (tmp_path / '.private').mkdir()
    (tmp_path / '.private' / 'secret.md').write_text('private')
    monkeypatch.setattr(server, 'workspace_root', tmp_path)
    assert server._validate_rag_scope(tmp_path / 'allowed.md') is None
    for name in ['.hidden.md', 'ignored.md', 'test_secret.py']:
        with pytest.raises(ValueError):
            server._validate_rag_scope(tmp_path / name)
    with pytest.raises(ValueError):
        server._validate_rag_scope(tmp_path / '.private' / 'secret.md')
    with pytest.raises(ValueError):
        server._validate_rag_scope(tmp_path / '.private')
    monkeypatch.setenv('RAG_MCP_MAX_TOKENS', '1')
    with pytest.raises(ValueError, match='token limit'):
        server._validate_rag_scope(tmp_path / 'large.md')


def test_allowed_roots_confines_absolute_relative_and_symlink_paths(tmp_path, monkeypatch):
    root = tmp_path / 'allowed'
    root.mkdir()
    (root / 'ok.md').write_text('ok')
    (tmp_path / 'outside.md').write_text('private')
    (root / 'link.md').symlink_to(tmp_path / 'outside.md')
    (root / 'folder').mkdir()
    (root / 'folder' / 'inside.md').write_text('inside')
    (root / 'folder-link').symlink_to(root / 'folder', target_is_directory=True)
    monkeypatch.setattr(server, 'workspace_root', root)
    monkeypatch.setenv('RAG_MCP_ALLOWED_ROOTS', str(root))
    assert server._normalize_path('ok.md') == str(root / 'ok.md')
    assert server._normalize_path(str(root / 'ok.md')) == str(root / 'ok.md')
    for name in ['../outside.md', str(tmp_path / 'outside.md'), 'link.md']:
        with pytest.raises(ValueError, match='allowed roots'):
            server._normalize_path(name)
    with pytest.raises(ValueError, match='Symbolic links'):
        server._normalize_path('folder-link/inside.md')
    monkeypatch.delenv('RAG_MCP_ALLOWED_ROOTS')
    assert server._normalize_path('../outside.md') == str(tmp_path / 'outside.md')


def test_mcp_import_enforces_allowed_roots(tmp_path):
    root = tmp_path / 'allowed'
    root.mkdir()
    (root / 'ok.md').write_text('A safe document.')
    (tmp_path / 'outside.md').write_text('Private.')
    requests = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
        *[{'jsonrpc': '2.0', 'id': number, 'method': 'tools/call', 'params': {
            'name': 'add_workspace_sources', 'arguments': {'workspace_id': f'book{number}',
            'path': path, 'revision': None}}} for number, path in
          [(2, 'ok.md'), (3, '../outside.md'), (4, str(tmp_path / 'outside.md'))]],
    ]
    env = {**os.environ, 'RAG_MCP_ALLOWED_ROOTS': str(root), 'RAG_MCP_WORKSPACE_ROOT': str(root),
           'RAG_MCP_CONFIG': str(tmp_path / 'config.json'), 'RAG_MCP_DATA_DIR': str(tmp_path / 'index'),
           'RAG_MCP_WORKSPACE_DIR': str(tmp_path / 'workspaces'), 'RAG_MCP_TRACE_DIR': str(tmp_path / 'traces')}
    result = subprocess.run([sys.executable, str(server.repo_root / 'server.py')],
        input=''.join(json.dumps(request) + '\n' for request in requests),
        text=True, capture_output=True, cwd=root, env=env, timeout=15, check=True)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert not replies[1]['result'].get('isError', False)
    assert all(reply['result']['isError'] for reply in replies[2:])
    assert not (tmp_path / 'workspaces' / 'book3').exists()
