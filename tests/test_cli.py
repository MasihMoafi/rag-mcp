import io
import json
from pathlib import Path

from rag.cli import main


class TTYStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_bare_cli_prints_help_and_menu_is_explicit():
    output = io.StringIO()
    assert main([], stdin=TTYStringIO(), stdout=output) == 0
    assert "usage: rag" in output.getvalue()


def test_query_json_format_is_stable_in_a_tty(tmp_path: Path):
    document = tmp_path / "facts.md"
    document.write_text("The launch codename is Cedar Lantern.")
    output = TTYStringIO()

    assert main(
        ["query", "codename?", "--path", str(document), "--format", "json"],
        stdin=TTYStringIO(),
        stdout=output,
        query_fn=lambda query, path: json.dumps({"result": "Cedar Lantern."}),
    ) == 0
    assert json.loads(output.getvalue()) == {"result": "Cedar Lantern."}


def test_query_text_format_is_stable_without_a_tty(tmp_path: Path):
    document = tmp_path / "facts.md"
    document.write_text("The launch codename is Cedar Lantern.")
    output = io.StringIO()

    assert main(
        ["query", "codename?", "--path", str(document), "--format", "text"],
        stdout=output,
        query_fn=lambda query, path: json.dumps({"result": "The launch codename is Cedar Lantern."}),
    ) == 0
    assert "The launch codename is Cedar Lantern." in output.getvalue()
    assert '"result"' not in output.getvalue()


def test_setup_does_not_connect_an_agent_as_a_side_effect(tmp_path: Path, monkeypatch):
    called = []
    monkeypatch.setattr(
        "rag.cli.user_settings.run_setup",
        lambda **kwargs: 0,
    )
    monkeypatch.setattr("rag.cli._offer_agent_connection", lambda *args: called.append(args))

    assert main(
        ["setup"],
        stdin=TTYStringIO(),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        environ={"RAG_MCP_CONFIG": str(tmp_path / "config.json")},
    ) == 0
    assert called == []


def test_doctor_reports_codex_login_as_json(tmp_path: Path):
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nprintf 'Logged in using ChatGPT\\n'\n")
    codex.chmod(0o755)
    output = io.StringIO()
    config_path = tmp_path / "config.json"

    code = main(
        ["doctor", "--json"],
        stdout=output,
        environ={"PATH": str(tmp_path), "RAG_MCP_CONFIG": str(config_path)},
    )

    report = json.loads(output.getvalue())
    assert code == 0
    assert report["codex"] == {
        "available": True,
        "authenticated": True,
        "method": "ChatGPT",
    }
    assert report["embedding"]["provider"] == "none"
    assert report["embedding"]["configured"] is False
    assert report["embedding"]["config_path"] == str(config_path)
    assert isinstance(report["embedding"]["full_extra"], bool)


def test_setup_flags_save_every_embedding_field_without_asking(tmp_path: Path):
    config_path = tmp_path / "config.json"
    probes = []
    output = io.StringIO()

    code = main(
        [
            "setup",
            "--provider", "openai_compatible",
            "--model", "text-embedding-3-large",
            "--base-url", "https://example.test/v1",
            "--api-key", "sk-test",
        ],
        stdout=output,
        environ={"RAG_MCP_CONFIG": str(config_path)},
        embedding_probe=lambda config: probes.append(dict(config)) or 3072,
    )

    assert code == 0
    assert "Works: 3072 dimensions." in output.getvalue()
    assert json.loads(config_path.read_text()) == {
        "embed_provider": "openai_compatible",
        "embed_model": "text-embedding-3-large",
        "embed_base_url": "https://example.test/v1",
        "embed_api_key": "sk-test",
        "ollama_url": "http://localhost:11434",
    }
    assert probes == [json.loads(config_path.read_text())]


def test_setup_ollama_url_and_default_model_and_no_verify(tmp_path: Path):
    config_path = tmp_path / "config.json"

    code = main(
        ["setup", "--provider", "ollama", "--ollama-url", "http://gpu-box:11434", "--no-verify"],
        stdout=io.StringIO(),
        environ={"RAG_MCP_CONFIG": str(config_path)},
        embedding_probe=lambda _config: (_ for _ in ()).throw(AssertionError("must not verify")),
    )

    assert code == 0
    saved = json.loads(config_path.read_text())
    assert saved["embed_provider"] == "ollama"
    assert saved["embed_model"] == "nomic-embed-text"
    assert saved["ollama_url"] == "http://gpu-box:11434"


def test_contextual_retrieval_can_be_toggled_from_config_command(tmp_path: Path):
    config_path = tmp_path / "config.json"
    environ = {"RAG_MCP_CONFIG": str(config_path)}

    assert main(
        ["config", "contextual", "on"],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        environ=environ,
    ) == 0
    assert json.loads(config_path.read_text())["contextual_retrieval"] == "on"

    assert main(
        ["config", "contextual", "off"],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        environ=environ,
    ) == 0
    assert json.loads(config_path.read_text())["contextual_retrieval"] == "off"


def test_contextual_retrieval_config_rejects_unknown_mode(tmp_path: Path):
    stderr = io.StringIO()
    assert main(
        ["config", "contextual", "maybe"],
        stdout=io.StringIO(),
        stderr=stderr,
        environ={"RAG_MCP_CONFIG": str(tmp_path / "config.json")},
    ) == 2
    assert "on or off" in stderr.getvalue()


def test_setup_api_option_requires_a_key(tmp_path: Path):
    stderr = io.StringIO()
    code = main(
        ["setup", "--provider", "openai_compatible", "--no-verify"],
        stdout=io.StringIO(),
        stderr=stderr,
        environ={"RAG_MCP_CONFIG": str(tmp_path / "config.json")},
    )
    assert code == 2
    assert "--api-key is required" in stderr.getvalue()
    assert not (tmp_path / "config.json").exists()


def test_query_warns_when_only_exact_word_search_is_configured(tmp_path: Path):
    document = tmp_path / "facts.md"
    document.write_text("The launch codename is Cedar Lantern.")
    stderr = io.StringIO()

    code = main(
        ["query", "codename?", "--path", str(document)],
        stdout=io.StringIO(),
        stderr=stderr,
        environ={"RAG_MCP_CONFIG": str(tmp_path / "config.json")},
        query_fn=lambda query, path: json.dumps({"result": "No matching information."}),
    )

    assert code == 0
    assert "Exact-word search (BM25) only" in stderr.getvalue()
    assert "rag setup" in stderr.getvalue()


def test_query_prints_the_real_query_result_contract(tmp_path: Path):
    document = tmp_path / "facts.md"
    document.write_text("The launch codename is Cedar Lantern.")
    output = io.StringIO()

    code = main(
        ["query", "What is the launch codename?", "--path", str(document)],
        stdout=output,
        query_fn=lambda query, path: json.dumps({"result": document.read_text()}),
    )

    assert code == 0
    assert output.getvalue().strip() == "The launch codename is Cedar Lantern."


def test_doctor_reports_missing_and_signed_out_codex(tmp_path: Path):
    output = io.StringIO()
    assert main(["doctor", "--json"], stdout=output, environ={"PATH": str(tmp_path)}) == 0
    assert json.loads(output.getvalue())["codex"] == {
        "available": False,
        "authenticated": False,
        "method": None,
    }

    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nprintf 'Not logged in\n'\nexit 1\n")
    codex.chmod(0o755)
    output = io.StringIO()
    assert main(["doctor", "--json"], stdout=output, environ={"PATH": str(tmp_path)}) == 0
    assert json.loads(output.getvalue())["codex"] == {
        "available": True,
        "authenticated": False,
        "method": None,
    }


def test_connect_codex_registers_the_current_server(tmp_path: Path):
    calls = tmp_path / "calls"
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/sh\n"
        "if [ \"$1 $2\" = \"mcp get\" ]; then exit 1; fi\n"
        f"printf '%s\\n' \"$@\" > '{calls}'\n"
    )
    codex.chmod(0o755)
    output = io.StringIO()

    code = main(
        ["connect-codex"],
        stdout=output,
        environ={"PATH": str(tmp_path), "HOME": str(tmp_path)},
    )

    assert code == 0
    args = calls.read_text().splitlines()
    assert args[:4] == ["mcp", "add", "rag", "--"]
    assert Path(args[4]).resolve() == Path(__import__("sys").executable).resolve()
    assert Path(args[5]).resolve().name == "server.py"
    assert not (tmp_path / ".codex" / "prompts" / "rag.md").exists()


def test_connect_codex_does_not_replace_a_matching_registration(tmp_path: Path):
    import server
    import sys

    marker = tmp_path / "unexpected-mutation"
    codex = tmp_path / "codex"
    transport = json.dumps(
        {
            "transport": {
                "command": sys.executable,
                "args": [str(Path(server.__file__).resolve())],
            }
        }
    )
    codex.write_text(
        "#!/bin/sh\n"
        "if [ \"$1 $2\" = \"mcp get\" ]; then\n"
        f"  printf '%s\\n' '{transport}'\n"
        "  exit 0\n"
        "fi\n"
        f"printf 'unexpected mutation\\n' > '{marker}'\n"
        "exit 99\n"
    )
    codex.chmod(0o755)
    output = io.StringIO()

    assert main(
        ["connect-codex"],
        stdout=output,
        stderr=io.StringIO(),
        environ={"PATH": str(tmp_path), "HOME": str(tmp_path)},
    ) == 0
    assert "already connected" in output.getvalue()
    assert not marker.exists()


def test_studio_opens_a_session_token_and_serves_on_loopback():
    events = []

    class Server:
        server_port = 47831

        def serve_forever(self):
            events.append("served")

        def server_close(self):
            events.append("closed")

    output = io.StringIO()
    code = main(
        ["studio"],
        stdout=output,
        token_factory=lambda: "paired-token",
        server_factory=lambda token, port: events.append((token, port)) or Server(),
        open_url=lambda url: events.append(url) or True,
    )

    assert code == 0
    assert events == [
        ("paired-token", 47831),
        "https://rag.masihmoafi.com/studio#local=paired-token",
        "served",
        "closed",
    ]


def test_query_top_shorthand_sets_the_result_count(tmp_path: Path, monkeypatch):
    document = tmp_path / "facts.md"
    document.write_text("The launch codename is Cedar Lantern.")
    seen = {}
    monkeypatch.delenv("RAG_MCP_TOP_K", raising=False)

    code = main(
        ["query", "codename?", "--path", str(document), "--top3"],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        environ={"RAG_MCP_CONFIG": str(tmp_path / "config.json")},
        query_fn=lambda query, path: seen.update(
            {"final": __import__("os").environ["RAG_MCP_RERANK_TOP_K"], "candidates": __import__("os").environ["RAG_MCP_TOP_K"]}
        ) or json.dumps({"result": "Cedar Lantern."}),
    )

    assert code == 0
    assert seen == {"final": "3", "candidates": "5"}


def test_connect_claude_registers_the_server_in_user_scope(tmp_path: Path):
    calls = tmp_path / "calls"
    claude = tmp_path / "claude"
    claude.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{calls}'\n")
    claude.chmod(0o755)
    output = io.StringIO()

    code = main(
        ["connect", "claude", "--install-prompt"],
        stdout=output,
        environ={"PATH": str(tmp_path), "HOME": str(tmp_path)},
    )

    assert code == 0
    assert "description:" in (tmp_path / ".claude" / "commands" / "rag.md").read_text()
    lines = calls.read_text().splitlines()
    assert lines[:5] == ["mcp", "remove", "rag", "-s", "user"]
    assert lines[5:11] == ["mcp", "add", "rag", "-s", "user", "--"]
    assert Path(lines[12]).name == "server.py"
    assert "Connected Claude Code" in output.getvalue()


def test_studio_can_open_a_local_studio():
    events = []

    class Server:
        server_port = 47831

        def serve_forever(self):
            pass

        def server_close(self):
            pass

    code = main(
        ["studio", "--studio-url", "http://localhost:3000/"],
        stdout=io.StringIO(),
        token_factory=lambda: "paired-token",
        server_factory=lambda token, port: Server(),
        open_url=lambda url: events.append(url) or True,
    )

    assert code == 0
    assert events == ["http://localhost:3000/studio#local=paired-token"]
