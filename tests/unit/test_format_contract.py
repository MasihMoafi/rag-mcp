import server
import pytest
from rag import fetch


SUPPORTED = tuple('.rs .toml .yaml .yml .py .md .txt .json .js .ts .tsx .jsx .c .h .cpp .hpp .go .sh .bash .zsh .css .html .sql .java .kt .proto .pdf .ipynb .rst .ini .cfg .conf .png .jpg .jpeg .webp .tiff .bmp .docx .pptx .xlsx .csv'.split())


@pytest.mark.parametrize('count', [500, 501])
def test_mcp_rejects_overflow_without_caching_partial_manifest(tmp_path, monkeypatch, count):
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
