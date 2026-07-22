from pathlib import Path
import json
import sys
import tempfile
import types


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


def test_default_workspace_call_uses_workspace_root() -> None:
    original_workspace = server.workspace_root
    original_import = server.importlib.import_module
    captured = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            server.workspace_root = Path(directory)
            server.importlib.import_module = lambda _name: types.SimpleNamespace(
                fetchExternalKnowledge=lambda **kwargs: captured.update(kwargs) or "ok"
            )
            assert json.loads(server.query_knowledge_base("find context")) == {"result": "ok"}
    finally:
        server.workspace_root = original_workspace
        server.importlib.import_module = original_import

    assert captured["doc_path"] == str(Path(directory).resolve())


def test_explicit_path_keeps_its_scope() -> None:
    original_import = server.importlib.import_module
    captured = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            server.importlib.import_module = lambda _name: types.SimpleNamespace(
                fetchExternalKnowledge=lambda **kwargs: captured.update(kwargs) or "ok"
            )
            server.query_knowledge_base("find context", directory)
            assert captured["doc_path"] == str(Path(directory).resolve())
    finally:
        server.importlib.import_module = original_import


def test_node_modules_scope_has_clear_recovery() -> None:
    with tempfile.TemporaryDirectory() as directory:
        unsafe = Path(directory) / "node_modules"
        unsafe.mkdir()
        output = server.query_knowledge_base("find context", str(unsafe))

    assert "will not scan dependency or build folders" in output
    assert "Choose the project source folder instead" in output


def test_depth_limit_has_clear_recovery() -> None:
    original = server.os.environ.get("RAG_MCP_MAX_DEPTH")
    server.os.environ["RAG_MCP_MAX_DEPTH"] = "1"
    try:
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "one" / "two"
            nested.mkdir(parents=True)
            output = server.query_knowledge_base("find context", directory)
    finally:
        if original is None:
            server.os.environ.pop("RAG_MCP_MAX_DEPTH", None)
        else:
            server.os.environ["RAG_MCP_MAX_DEPTH"] = original

    assert "configured 1-folder depth limit" in output
    assert "RAG_MCP_MAX_DEPTH" in output
