from rag.server import _is_git_repository_marker


def test_git_repository_markers(tmp_path):
    marker = tmp_path / '.git'
    assert not _is_git_repository_marker(marker)
    marker.mkdir()
    assert not _is_git_repository_marker(marker)
    (marker / 'HEAD').write_text('ref: refs/heads/main\n')
    assert _is_git_repository_marker(marker)
    (marker / 'HEAD').unlink()
    marker.rmdir()
    marker.write_text('gitdir: /elsewhere/worktree\n')
    assert _is_git_repository_marker(marker)
    marker.write_text('other content\n')
    assert not _is_git_repository_marker(marker)
