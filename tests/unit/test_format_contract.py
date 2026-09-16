import server
import pytest
from rag import fetch


SUPPORTED = tuple('.rs .toml .yaml .yml .py .md .txt .json .js .ts .tsx .jsx .c .h .cpp .hpp .go .sh .bash .zsh .css .html .sql .java .kt .proto .pdf .ipynb .rst .ini .cfg .conf .docx .pptx .xlsx .csv'.split())


@pytest.mark.parametrize('entrypoint', ['mcp', 'discovery'])
def test_traversal_failure_cannot_return_partial_scope(tmp_path, monkeypatch, entrypoint):
    from pathlib import Path
    from rag import discovery

    (tmp_path / 'good.txt').write_text('evidence')
    unreadable = tmp_path / 'restricted'
    unreadable.mkdir()
    failure = PermissionError('injected traversal failure')
    monkeypatch.setattr(server, '_validated_scope_cache', {})
    operation = 'scandir' if entrypoint == 'mcp' else 'listdir'
    original = getattr(discovery.os, operation)

    def fail_selected(path):
        if Path(path) == unreadable:
            raise failure
        return original(path)

    monkeypatch.setattr(discovery.os, operation, fail_selected)
    with pytest.raises(PermissionError) as caught:
        if entrypoint == 'mcp':
            server._validate_rag_scope(tmp_path)
        else:
            discovery.find_all_indexable_files(str(tmp_path))
    assert caught.value is failure
    assert not server._validated_scope_cache


def test_mcp_stat_failure_cannot_cache_partial_scope(tmp_path, monkeypatch):
    from pathlib import Path

    document = tmp_path / 'source.txt'
    document.write_text('evidence')
    original = Path.stat
    failure = OSError('injected stat failure')
    monkeypatch.setattr(server, '_validated_scope_cache', {})

    def fail_selected(path, *args, **kwargs):
        if path == document and kwargs.get('follow_symlinks', True):
            raise failure
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', fail_selected)
    with pytest.raises(OSError) as caught:
        server._validate_rag_scope(tmp_path)
    assert caught.value is failure
    assert not server._validated_scope_cache


@pytest.mark.parametrize('count', [500, 501])
def test_mcp_rejects_overflow_without_caching_partial_manifest(tmp_path, monkeypatch, count):
    # Pin the cap so this asserts the overflow mechanism rather than whatever the
    # shipped default happens to be.
    monkeypatch.setenv('RAG_MCP_MAX_FILES', '500')
    monkeypatch.setattr(server, '_validated_scope_cache', {})
    for index in range(count):
        (tmp_path / f'doc-{index}.txt').touch()
    (tmp_path / 'test_excluded.py').touch()
    if count == 501:
        with pytest.raises(ValueError, match='500-document limit'):
            server._validate_rag_scope(tmp_path)
        assert not server._validated_scope_cache
    else:
        assert len(server._validate_rag_scope(tmp_path).files) == 500


def test_server_and_fetch_preserve_existing_format_contract(tmp_path):
    assert fetch._INDEXABLE_EXTENSIONS == SUPPORTED
    assert server._INDEXABLE_EXTENSIONS == set(SUPPORTED)
    for extension in SUPPORTED:
        (tmp_path / ('source' + extension.upper())).write_text('evidence')
    (tmp_path / 'source.tex').write_text('unsupported in directory discovery')
    server._validated_scope_cache.clear()
    manifest = server._validate_rag_scope(tmp_path)
    assert set(manifest.files) == set(fetch.find_all_indexable_files(str(tmp_path)))
    assert len(manifest.files) == len(SUPPORTED)


def test_mcp_startup_does_not_load_retrieval_dependencies():
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, '-c',
         'import sys, server; assert not {"torch", "rag.core", "rag.fetch", "lancedb"}.intersection(sys.modules)'],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_images_are_never_indexed(tmp_path):
    from pathlib import Path
    from rag.discovery import find_all_indexable_files
    for extension in ('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp'):
        (tmp_path / f'screenshot{extension}').touch()
    (tmp_path / 'notes.md').write_text('text')
    assert [Path(path).name for path in find_all_indexable_files(str(tmp_path))] == ['notes.md']
