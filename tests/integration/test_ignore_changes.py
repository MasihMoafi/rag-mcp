import subprocess

import pytest

from rag import server, workspace_import
from rag.workspaces import WorkspaceStore


def test_ignore_changes_take_effect_in_a_running_server(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    document = tmp_path / 'document.md'
    document.write_text('A searchable document.\n')
    rules = tmp_path / '.gitignore'
    rules.write_text('other.md\n')
    assert str(document) in server._validate_rag_scope(tmp_path).files
    rules.write_text('document.md\n')
    assert str(document) not in server._validate_rag_scope(tmp_path).files
    rules.write_text('other.md\n')
    assert str(document) in server._validate_rag_scope(tmp_path).files


def test_warm_scan_reuses_ignore_decisions(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'document.md').write_text('content')
    real_run = server.subprocess.run
    calls = []

    def run(*args, **kwargs):
        if 'check-ignore' in args[0]:
            calls.append(args[0])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(server.subprocess, 'run', run)
    server._validate_rag_scope(tmp_path)
    first = len(calls)
    server._validate_rag_scope(tmp_path)
    assert first > 0
    assert len(calls) == first
    (tmp_path / '.gitignore').write_text('document.md\n')
    assert str(tmp_path / 'document.md') not in server._validate_rag_scope(tmp_path).files


def test_git_info_exclude_changes_are_seen(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    document = tmp_path / 'document.md'
    document.write_text('content')
    exclude = tmp_path / '.git' / 'info' / 'exclude'
    assert str(document) in server._validate_rag_scope(tmp_path).files
    exclude.write_text('document.md\n')
    assert str(document) not in server._validate_rag_scope(tmp_path).files
    exclude.write_text('other.md\n')
    assert str(document) in server._validate_rag_scope(tmp_path).files


def test_configured_global_exclude_changes_are_seen(tmp_path):
    folder = tmp_path / 'repo'
    folder.mkdir()
    subprocess.run(['git', 'init', '-q', str(folder)], check=True)
    document = folder / 'document.md'
    document.write_text('content')
    exclude = tmp_path / 'global-ignore'
    exclude.write_text('other.md\n')
    subprocess.run(['git', '-C', str(folder), 'config', 'core.excludesFile', str(exclude)], check=True)
    assert str(document) in server._validate_rag_scope(folder).files
    exclude.write_text('document.md\n')
    assert str(document) not in server._validate_rag_scope(folder).files
    exclude.write_text('other.md\n')
    assert str(document) in server._validate_rag_scope(folder).files


def test_git_worktree_uses_common_info_exclude(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'test@example.test'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Test'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-q', '--allow-empty', '-m', 'initial'], check=True)
    folder = tmp_path / 'worktree'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '-q', '-b', 'test', str(folder)], check=True)
    document = folder / 'document.md'
    document.write_text('content')
    exclude = repo / '.git' / 'info' / 'exclude'
    assert str(document) in server._validate_rag_scope(folder).files
    exclude.write_text('document.md\n')
    assert str(document) not in server._validate_rag_scope(folder).files
    exclude.write_text('other.md\n')
    assert str(document) in server._validate_rag_scope(folder).files


def test_import_rechecks_ignore_rules_before_publish(tmp_path, monkeypatch):
    folder = tmp_path / 'corpus'
    folder.mkdir()
    subprocess.run(['git', 'init', '-q', str(folder)], check=True)
    (folder / 'document.md').write_text('content')
    (folder / '.gitignore').write_text('other.md\n')
    store = WorkspaceStore(tmp_path / 'store')
    extract = workspace_import.extract_content

    def change_rules(path):
        content = extract(path)
        (folder / '.gitignore').write_text('document.md\n')
        return content

    monkeypatch.setattr(workspace_import, 'extract_content', change_rules)
    with pytest.raises(ValueError, match='changed during import'):
        workspace_import.import_workspace_sources(store, 'book', folder)
    assert not (store.root / 'book' / 'manifest.json').exists()


def test_index_changes_invalidate_cached_ignore_decision(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    doc = tmp_path / 'document.md'
    doc.write_text('content')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'document.md'], check=True)
    (tmp_path / '.gitignore').write_text('document.md\n')
    assert str(doc) in server._validate_rag_scope(tmp_path).files
    subprocess.run(['git', '-C', str(tmp_path), 'rm', '--cached', '-q', 'document.md'], check=True)
    assert str(doc) not in server._validate_rag_scope(tmp_path).files


def test_failed_git_check_is_not_cached_or_treated_as_unignored(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'document.md').write_text('content')
    (tmp_path / '.gitignore').write_text('document.md\n')
    real_run = server.subprocess.run
    failed = False
    def run(*args, **kwargs):
        nonlocal failed
        if 'check-ignore' in args[0] and not failed:
            failed = True
            raise OSError('git unavailable')
        return real_run(*args, **kwargs)
    monkeypatch.setattr(server.subprocess, 'run', run)
    with pytest.raises(ValueError, match='ignore'):
        server._validate_rag_scope(tmp_path)
    assert str(tmp_path / 'document.md') not in server._validate_rag_scope(tmp_path).files


def test_scope_without_git_and_outside_repository(tmp_path, monkeypatch):
    document = tmp_path / 'document.md'
    document.write_text('content')
    assert str(document) in server._validate_rag_scope(tmp_path).files
    monkeypatch.setenv('PATH', str(tmp_path / 'empty-bin'))
    assert str(document) in server._validate_rag_scope(tmp_path).files
    assert server._validate_rag_scope(document) is None


def test_scope_without_git_inside_repository(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    document = tmp_path / 'document.md'
    document.write_text('content')
    (tmp_path / '.gitignore').write_text('document.md\n')
    assert str(document) not in server._validate_rag_scope(tmp_path).files
    monkeypatch.setenv('PATH', str(tmp_path / 'empty-bin'))
    assert str(document) in server._validate_rag_scope(tmp_path).files
