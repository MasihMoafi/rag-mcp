from pathlib import Path
import json
import sys
import tempfile
import types


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


def test_folder_manifest_excludes_linked_files_and_directories(tmp_path):
    scope = tmp_path / 'scope'
    scope.mkdir()
    local = scope / 'allowed.md'
    local.write_text('allowed')
    outside = tmp_path / 'outside.md'
    outside.write_text('private')
    (scope / 'linked.md').symlink_to(outside)
    (scope / 'cycle').symlink_to(scope, target_is_directory=True)
    manifest = server._validate_rag_scope(scope)
    assert manifest.files == (str(local),)


def test_query_requires_an_explicit_document_scope() -> None:
    original_workspace = server.workspace_root
    original_import = server.importlib.import_module
    captured = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            server.workspace_root = Path(directory)
            server.importlib.import_module = lambda _name: types.SimpleNamespace(
                fetchExternalKnowledge=lambda **kwargs: captured.update(kwargs) or "ok"
            )
            assert "doc_path must name an explicit supported file" in json.loads(
                server.query_knowledge_base("find context")
            )["error"]
    finally:
        server.workspace_root = original_workspace
        server.importlib.import_module = original_import

    assert captured == {}


def test_explicit_path_keeps_its_scope() -> None:
    original_import = server.importlib.import_module
    captured = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "facts.md").write_text("Scoped evidence.")
            server.importlib.import_module = lambda _name: types.SimpleNamespace(
                fetchExternalKnowledge=lambda **kwargs: captured.update(kwargs) or "ok"
            )
            server.query_knowledge_base("find context", directory)
            assert captured["doc_path"] == [str(Path(directory).resolve() / "facts.md")]
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


def test_token_limit_prefers_narrowing_to_raising_the_limit() -> None:
    original = server.os.environ.get("RAG_MCP_MAX_TOKENS")
    server.os.environ["RAG_MCP_MAX_TOKENS"] = "1"
    try:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "canonical.md").write_text("enough text", encoding="utf-8")
            output = server.query_knowledge_base("find context", directory)
    finally:
        if original is None:
            server.os.environ.pop("RAG_MCP_MAX_TOKENS", None)
        else:
            server.os.environ["RAG_MCP_MAX_TOKENS"] = original

    assert "Retry with doc_path set to the canonical file or a smaller folder" in output
    assert "only when the broader corpus is deliberate" in output


def test_repeated_scope_queries_reuse_manifest_and_invalidate_on_change() -> None:
    original_import = server.importlib.import_module
    original_walk = server.os.walk
    captured = []
    walk_count = 0

    def counting_walk(top, *args, **kwargs):
        nonlocal walk_count
        # The runtime manifest also walks the engine's own code once. Only scans of the queried folder count.
        if Path(top) == scope:
            walk_count += 1
        return original_walk(top, *args, **kwargs)

    try:
        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory)
            first_file = scope / "first.md"
            first_file.write_text("first document\n", encoding="utf-8")
            server._validated_scope_cache.clear()
            server.os.walk = counting_walk
            server.importlib.import_module = lambda _name: types.SimpleNamespace(
                fetchExternalKnowledge=lambda **kwargs: captured.append(kwargs) or "ok"
            )

            assert json.loads(server.query_knowledge_base("find context", directory))["result"] == "ok"
            assert json.loads(server.query_knowledge_base("find context", directory))["result"] == "ok"
            assert walk_count == 1
            assert captured[0]["doc_path"] == captured[1]["doc_path"]
            assert captured[0]["doc_path"] == [str(first_file)]

            second_file = scope / "second.md"
            second_file.write_text("second document\n", encoding="utf-8")
            assert json.loads(server.query_knowledge_base("find context", directory))["result"] == "ok"
            assert walk_count == 2
            assert captured[2]["doc_path"] == [str(first_file), str(second_file)]

            first_file.write_text("updated document\n", encoding="utf-8")
            assert json.loads(server.query_knowledge_base("find context", directory))["result"] == "ok"
            assert walk_count == 3
            assert captured[3]["doc_path"] == [str(first_file), str(second_file)]
    finally:
        server.importlib.import_module = original_import
        server.os.walk = original_walk
        server._validated_scope_cache.clear()
