from pathlib import Path


def test_release_quick_start_and_security_disclosures():
    readme = (Path(__file__).resolve().parents[2] / 'README.md').read_text()
    assert 'uvx --from ragmcp rag setup' in readme
    assert 'Bare `rag` prints help' in readme
    assert 'RAG_MCP_ALLOWED_ROOTS' in readme
    assert 'trusted like a shell' in readme
    assert 'host is responsible' in readme
    assert 'rag studio                    # start the bundled Studio' not in readme
    assert '[Live Demo]' not in readme
