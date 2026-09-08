import io
import json
from pathlib import Path

from rag.cli import main


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
        query_fn=lambda query, path: json.dumps({"result": []}),
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
        query_fn=lambda query, path: json.dumps(
            {"result": [{"content": document.read_text(), "source": path}]}
        ),
    )

    assert code == 0
    assert json.loads(output.getvalue()) == {
        "result": [
            {
                "content": "The launch codename is Cedar Lantern.",
                "source": str(document),
            }
        ]
    }


def test_doctor_reports_missing_and_signed_out_codex(tmp_path: Path):
    output = io.StringIO()
    assert main(["doctor", "--json"], stdout=output, environ={"PATH": str(tmp_path)}) == 1
    assert json.loads(output.getvalue())["codex"] == {
        "available": False,
        "authenticated": False,
        "method": None,
    }

    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nprintf 'Not logged in\n'\nexit 1\n")
    codex.chmod(0o755)
    output = io.StringIO()
    assert main(["doctor", "--json"], stdout=output, environ={"PATH": str(tmp_path)}) == 1
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
        ) or "{}",
    )

    assert code == 0
    assert seen == {"final": "3", "candidates": "5"}


def test_connect_claude_registers_the_server_in_user_scope(tmp_path: Path):
    calls = tmp_path / "calls"
    claude = tmp_path / "claude"
    claude.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{calls}'\n")
    claude.chmod(0o755)
    output = io.StringIO()

    code = main(["connect", "claude"], stdout=output, environ={"PATH": str(tmp_path), "HOME": str(tmp_path)})

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
