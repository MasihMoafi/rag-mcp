"""The three independently shipped version declarations must agree."""

import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest depends on tomli there.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_package_versions_match_mcp_initialize():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        python_version = tomllib.load(stream)["project"]["version"]
    npm_version = json.loads((ROOT / "npm/package.json").read_text())["version"]
    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    process = subprocess.run(
        [sys.executable, str(ROOT / "server.py")],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    mcp_version = json.loads(process.stdout)["result"]["serverInfo"]["version"]
    assert python_version == npm_version == mcp_version


def test_registry_server_json_matches_the_package():
    """The MCP registry entry names the released PyPI version and the README carries its ownership marker."""
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    server = json.loads((ROOT / "server.json").read_text())
    package = server["packages"][0]
    assert server["version"] == package["version"] == project["version"]
    assert package["identifier"] == project["name"]
    assert package["identifier"] in project["scripts"]
    assert f"<!-- mcp-name: {server['name']} -->" in (ROOT / "README.md").read_text()
