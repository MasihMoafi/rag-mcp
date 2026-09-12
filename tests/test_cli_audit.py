import io
import json
from pathlib import Path

import pytest

from rag import cli, detect, settings, model_catalog


def test_terminal_query_shows_passages_without_transport_json(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.prompts.interactive", lambda *_: True)
    document = tmp_path / "notes.txt"
    document.write_text("The deadline is Friday.")
    result = json.dumps({"result": [{"content": "The deadline is Friday.", "source": str(document), "score": 0.8}]})
    output = io.StringIO()
    assert cli.main(["query", "deadline", "--path", str(document)], stdin=io.StringIO(), stdout=output,
                    stderr=io.StringIO(), environ={"RAG_MCP_CONFIG": str(tmp_path / "config.json")},
                    query_fn=lambda *_: result) == 0
    assert "1. " + str(document) in output.getvalue()
    assert "The deadline is Friday." in output.getvalue()
    assert '"result"' not in output.getvalue()
    assert "score" not in output.getvalue()


def test_cli_query_errors_have_a_failure_exit_code(tmp_path):
    doc = tmp_path / "facts.md"
    doc.write_text("Cedar Lantern")
    assert cli.main(["query", "fact?", "--path", str(doc)], stdout=io.StringIO(), stderr=io.StringIO(), query_fn=lambda *_: '{"error":"Embedding failed"}') == 1


def test_mcp_tool_errors_are_reported_as_errors(monkeypatch):
    import json
    import server

    replies = []
    monkeypatch.setattr(server, "_initialized", True)
    monkeypatch.setattr(server, "_result", lambda request_id, result: replies.append(result))
    server._handle_request({"id": 1, "method": "tools/call", "params": {"name": "query_knowledge_base", "arguments": {"query": ""}}})
    assert replies[0]["isError"] is True
    assert "query must be" in json.loads(replies[0]["content"][0]["text"])["error"]


@pytest.mark.parametrize("answer", ["0", "-1", "7", "nonsense"])
def test_setup_rejects_invalid_numbers_without_saving(tmp_path, monkeypatch, answer):
    monkeypatch.setattr(detect, "detect_all", lambda *_: {})
    env = {"RAG_MCP_CONFIG": str(tmp_path / "config.json")}
    assert settings.run_setup(stdin=io.StringIO(answer + "\n"), stdout=io.StringIO(), stderr=io.StringIO(), environ=env) == 2
    assert not settings.config_path(env).exists()


def test_installed_models_are_selectable_and_keep_the_detected_server(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, "detect_all", lambda *_: {
        "ollama": {"running": True, "url": "http://gpu-box:11434", "embedding_models": ["qwen3-embedding:0.6b", "bge-m3:latest"]},
    })
    env = {"RAG_MCP_CONFIG": str(tmp_path / "config.json")}
    settings.save_settings({"top_k": "3", "reranker_type": "off"}, env)
    out = io.StringIO()
    assert settings.run_setup(stdin=io.StringIO("2\n"), stdout=out, stderr=io.StringIO(), environ=env, verify=lambda _: 1024) == 0
    assert "qwen3-embedding:0.6b · Ollama" in out.getvalue()
    saved = settings.load_settings(env)
    assert saved["embed_model"] == "bge-m3:latest"
    assert saved["ollama_url"] == "http://gpu-box:11434"
    assert saved["top_k"] == "3"
    assert saved["reranker_type"] == "off"


def test_detection_respects_configured_ollama_server(tmp_path, monkeypatch):
    env = {"RAG_MCP_CONFIG": str(tmp_path / "config.json"), "RAG_MCP_OLLAMA_URL": "http://gpu-box:11434"}
    calls = []
    monkeypatch.setattr(detect, "detect_ollama", lambda url: calls.append(url) or {"running": True, "embedding_models": ["bge-m3"]})
    monkeypatch.setattr(detect, "detect_local_runtimes", lambda: [])
    recommended = detect.recommend(detect.detect_all(env))
    assert calls == ["http://gpu-box:11434"]
    assert recommended["ollama_url"] == "http://gpu-box:11434"


def test_api_key_is_selected_independently_of_ollama_recommendation(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, "detect_all", lambda *_: {"api_keys": ["openrouter"]})
    monkeypatch.setattr(model_catalog, "api_models", lambda *args: ["qwen/qwen3-embedding-4b"])
    env = {"RAG_MCP_CONFIG": str(tmp_path / "config.json"), "OPENROUTER_API_KEY": "test-key"}
    probes, output = [], io.StringIO()
    assert settings.run_setup(stdin=io.StringIO("1\n3\n1\n"), stdout=output, stderr=io.StringIO(), environ=env, verify=lambda config: probes.append(config) or 2560) == 0
    assert probes[0]["embed_base_url"] == "https://openrouter.ai/api/v1"
    assert probes[0]["embed_model"] == "qwen/qwen3-embedding-4b"
    assert probes[0]["embed_api_key"] == ""
    assert "OpenRouter · key available" in output.getvalue()


@pytest.mark.parametrize("command", ["serve", "studio"])
@pytest.mark.parametrize("port", [47831, 47839])
def test_companion_startup_links_the_actual_port(command, port):
    class Server:
        server_port = port
        closed = False

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            self.closed = True

    server, out, opened = Server(), io.StringIO(), []
    assert cli.main([command, "--port", str(port)], stdout=out, stderr=io.StringIO(), token_factory=lambda: "test-token", server_factory=lambda **_: server, open_url=opened.append) == 0
    suffix = f"&port={port}" if port != 47831 else ""
    url = f"https://rag.masihmoafi.com/studio#local=test-token{suffix}"
    assert url in out.getvalue()
    assert server.closed
    assert opened == ([url] if command == "studio" else [])


def test_menu_rejects_zero():
    assert cli._menu(stdin=io.StringIO("0\n"), stdout=io.StringIO(), stderr=io.StringIO(), environ={}) == 2


def test_menu_search_uses_saved_result_count(tmp_path, monkeypatch):
    env = {"RAG_MCP_CONFIG": str(tmp_path / "config.json")}
    settings.save_settings({"top_k": "3"}, env)
    calls = []
    monkeypatch.setattr(cli, "main", lambda argv, **_: calls.append(argv) or 0)
    assert cli._menu(stdin=io.StringIO("2\n.\nfact?\n\n"), stdout=io.StringIO(), stderr=io.StringIO(), environ=env) == 0
    assert calls[0][-4:] == ["--top", "3", "--format", "text"]


def test_menu_command_dispatches_to_the_human_ui(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "_menu", lambda **kwargs: called.append(kwargs) or 0)
    assert cli.main(["menu"], stdout=io.StringIO(), stderr=io.StringIO()) == 0
    assert called


def test_cli_keeps_human_ui_and_agent_integrations_out_of_the_dispatcher():
    assert cli._menu.__module__ == "rag.cli_menu"
    assert cli._connect_agent.__module__ == "rag.cli_agents"
