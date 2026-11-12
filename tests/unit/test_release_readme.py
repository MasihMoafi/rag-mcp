import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_release_quick_start_and_security_disclosures():
    readme = (ROOT / "README.md").read_text()
    assert "Bare `rag` opens the menu at a terminal and prints help otherwise" in readme
    assert "RAG_MCP_ALLOWED_ROOTS" in readme
    assert "trusted like a shell" in readme
    assert "host is responsible" in readme
    assert "rag studio                    # start the bundled Studio" not in readme
    assert "[Live Demo]" not in readme


def test_readme_install_line_points_at_the_installer_for_this_package():
    readme = (ROOT / "README.md").read_text()
    match = re.search(
        r"curl -LsSf https://raw\.githubusercontent\.com/MasihMoafi/rag-mcp/main/(\S+) \| sh",
        readme,
    )
    assert match, "README has no curl install line"
    installer = (ROOT / match.group(1)).read_text()
    name = re.search(r'(?m)^name = "([^"]+)"', (ROOT / "pyproject.toml").read_text())
    assert name, "pyproject.toml has no project name"
    assert f'package="{name.group(1)}"' in installer
    # Registries the package is not published to must not be offered as install paths.
    assert "uvx --from ragmcp" not in readme
    assert "npx ragmcp" not in readme
