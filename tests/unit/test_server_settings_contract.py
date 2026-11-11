"""MCP setting delegation preserves order, validation and cache invalidation."""

from unittest.mock import Mock

import pytest
import server
from rag import cli, fetch
from rag.workspaces import WorkspaceStore


def test_settings_report_uses_captured_server_identity(monkeypatch):
    from rag import detect

    assert (
        server._SERVER_SOURCE_SHA256
        == server.hashlib.sha256(server.Path(server.__file__).read_bytes()).hexdigest()
    )
    monkeypatch.setattr(server, "_SERVER_SOURCE_SHA256", "startup-fingerprint")
    monkeypatch.setattr(
        server.Path,
        "read_bytes",
        lambda *args: pytest.fail("Do not rehash changed disk contents"),
    )
    monkeypatch.setattr(server.user_settings, "describe_settings", list)
    monkeypatch.setattr(
        server,
        "embedding_config",
        lambda: {"embed_provider": "none", "embed_model": "", "ollama_url": "unused"},
    )
    monkeypatch.setattr(server.user_settings, "retrieval_config", dict)
    monkeypatch.setattr(detect, "detect_all", lambda *args: {})
    assert server._settings_report()["runtime"] == {
        "pid": server.os.getpid(),
        "server_source_sha256_at_startup": "startup-fingerprint",
    }


def test_mcp_retrieval_error_does_not_echo_provider_payload(monkeypatch, tmp_path):
    import json

    failure = RuntimeError(
        "provider body contains sk-private-example and private document text"
    )
    monkeypatch.setattr(
        server, "_validate_rag_scope", lambda path, extensions=None: None
    )
    monkeypatch.setattr(fetch, "fetchExternalKnowledge", Mock(side_effect=failure))
    output = json.loads(server.query_knowledge_base("question", str(tmp_path)))
    assert "error" in output
    assert "RuntimeError" in output["error"]
    assert "sk-private-example" not in output["error"]
    assert "private document text" not in output["error"]


def test_mcp_embedding_probe_never_returns_provider_error(monkeypatch, tmp_path):
    import json

    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setattr(
        cli,
        "_probe_embedding",
        Mock(
            side_effect=ValueError(
                "Authorization: Bearer sk-secret-example; body: private provider response"
            )
        ),
    )
    responses = []
    monkeypatch.setattr(server, "_initialized", True)
    monkeypatch.setattr(
        server, "_result", lambda request_id, result: responses.append(result)
    )
    server._handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "set_rag_settings",
                "arguments": {
                    "embedding_provider": "ollama",
                    "embedding_model": "fixture",
                },
            },
        }
    )
    output = json.dumps(responses)
    assert "verification failed" in output.lower()
    assert (
        "sk-secret-example" not in output
        and "Authorization" not in output
        and "private provider response" not in output
    )


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("get_rag_settings", {}),
        (
            "set_rag_settings",
            {"embedding_provider": "ollama", "embedding_model": "fixture"},
        ),
    ],
)
def test_mcp_detection_failure_does_not_echo_provider_error(
    monkeypatch, tmp_path, name, arguments
):
    import json

    from rag import detect

    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setattr(
        detect,
        "detect_all",
        Mock(side_effect=ValueError("Authorization: Bearer sk-detection-secret")),
    )
    responses = []
    monkeypatch.setattr(server, "_initialized", True)
    monkeypatch.setattr(
        server, "_result", lambda request_id, result: responses.append(result)
    )
    server._handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    output = json.dumps(responses)
    assert "detection" in output.lower()
    assert "Authorization" not in output and "sk-detection-secret" not in output


def test_mcp_workspace_id_resolves_the_shared_engine_workspace(monkeypatch, tmp_path):
    import json
    import types

    root = tmp_path / "workspaces"
    store = WorkspaceStore(root)
    store.replace(
        "book",
        "RAG book",
        [
            {
                "id": "chapter",
                "name": "chapter.md",
                "source_type": "markdown",
                "content": "Cedar Lantern",
            }
        ],
    )
    monkeypatch.setenv("RAG_MCP_WORKSPACE_DIR", str(root))
    paths = []
    validate = server._validate_rag_scope

    def record(path, extensions=None, **kwargs):
        # Record the scope and run the real check: workspace queries use its validated file list.
        paths.append(path)
        return validate(path, extensions, **kwargs)

    monkeypatch.setattr(server, "_validate_rag_scope", record)
    monkeypatch.setattr(
        server.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(
            fetchExternalKnowledge=lambda **_kwargs: "Shared answer"
        ),
    )

    assert json.loads(
        server.query_knowledge_base("codename?", workspace_id="book")
    ) == {"result": "Shared answer"}
    assert paths == [store.resolve("book")]
    assert "error" in json.loads(
        server.query_knowledge_base(
            "codename?", doc_path=str(tmp_path), workspace_id="book"
        )
    )


def test_multi_part_question_retrieves_facets_but_synthesizes_once(
    monkeypatch, tmp_path
):
    import json
    import types

    from rag import openrouter_pipeline

    # Never read the developer's own settings (a saved context_radius changes the code path).
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))

    document = tmp_path / "chapter.md"
    document.write_text("Evaluation evidence.")
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs["query"])
        marker = len(calls)
        return [
            {"content": f"facet {marker} passage {rank}", "file_path": str(document)}
            for rank in range(1, 4)
        ]

    captured = {}

    def synthesize(query, passages, config):
        captured.update(query=query, passages=passages)
        return "Combined answer."

    monkeypatch.setattr(
        server.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(
            fetchExternalKnowledge=fetch,
            expand_answer_context=lambda passages, *_args: passages,
        ),
    )
    monkeypatch.setattr(openrouter_pipeline, "synthesize_answer", synthesize)
    query = (
        "Why can retrieval metrics look good while the answer is poor? "
        "Explain the failure points, and give the practical evaluation approach."
    )

    output = json.loads(server.query_knowledge_base(query, str(document)))

    assert output == {"result": "Combined answer."}
    # One retrieval per user facet (each keeps the user's own words), then one synthesis.
    from rag.query_decomposition import decompose_query

    assert calls == decompose_query(query)
    assert len(calls) == 3
    assert calls[0].startswith("Why can retrieval metrics look good")
    assert calls[1].startswith("Explain the failure points")
    assert calls[2].startswith("give the practical evaluation approach")
    assert captured["query"] == query
    assert len(captured["passages"]) == 5
    assert {passage["content"].split()[1] for passage in captured["passages"]} == {
        "1",
        "2",
        "3",
    }


def test_single_content_facet_retrieves_without_answer_instructions(
    monkeypatch, tmp_path
):
    import json

    from rag import openrouter_pipeline

    document = tmp_path / "chapter.md"
    document.write_text("Book evidence.")
    calls = []
    captured = {}
    question = "Which passages discuss ambition?"
    query = question + " Return two passages from this book."

    def retrieve(**kwargs):
        calls.append(kwargs["query"])
        return [{"content": "Book evidence."}]

    def synthesize(original, passages, config):
        captured["query"] = original
        return "Book answer."

    monkeypatch.setattr(fetch, "fetchExternalKnowledge", retrieve)
    monkeypatch.setattr(openrouter_pipeline, "synthesize_answer", synthesize)

    assert json.loads(server.query_knowledge_base(query, str(document))) == {
        "result": "Book answer."
    }
    assert calls == [question]
    assert captured["query"] == query


@pytest.mark.parametrize("read_fails", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_server_shutdown_releases_pipelines(
    monkeypatch, read_fails, cleanup_fails, capsys
):
    failure = OSError("input failed")
    read = Mock(side_effect=failure) if read_fails else Mock(return_value=None)
    reset = Mock(side_effect=RuntimeError("cleanup failed") if cleanup_fails else None)
    monkeypatch.setattr(server, "_read_payload", read)
    monkeypatch.setattr(fetch, "reset_pipelines", reset)
    if read_fails:
        with pytest.raises(OSError) as caught:
            server.main()
        assert caught.value is failure
    else:
        server.main()
    reset.assert_called_once_with()
    output = capsys.readouterr()
    assert not output.out
    if cleanup_fails:
        assert "cleanup" in output.err.lower()


@pytest.fixture
def delegation(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    calls = []

    def configure(args, **kwargs):
        calls.append(
            (args.setting, args.value, args.provider, args.no_verify, kwargs["verify"])
        )
        return 0

    monkeypatch.setattr(cli, "_config_command", configure)
    monkeypatch.setattr(
        server.user_settings, "describe_settings", lambda: ["configured"]
    )
    reset = Mock()
    monkeypatch.setattr(fetch, "reset_pipelines", reset)
    return calls, reset


@pytest.mark.parametrize("failure", ["validation", "probe"])
def test_failed_request_preserves_live_settings(monkeypatch, tmp_path, failure):
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    path = server.user_settings.save_settings({"top_k": "5", "custom": "preserved"})
    original = path.read_bytes()
    reset = Mock()
    monkeypatch.setattr(fetch, "reset_pipelines", reset)
    monkeypatch.setattr(
        cli, "_probe_embedding", Mock(side_effect=RuntimeError("probe failed"))
    )
    arguments = (
        {"top": 9, "contextual_retrieval": "invalid"}
        if failure == "validation"
        else {"top": 9, "embedding_provider": "ollama", "embedding_model": "fixture"}
    )
    assert "error" in server._apply_settings(arguments)
    assert path.read_bytes() == original
    reset.assert_not_called()


def test_successful_request_publishes_all_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    server.user_settings.save_settings({"top_k": "5", "custom": "preserved"})
    reset = Mock()
    monkeypatch.setattr(fetch, "reset_pipelines", reset)
    result = server._apply_settings({"top": 9, "reranker": "off"})
    assert result["changed"] == ["top", "reranker"]
    assert server.user_settings.load_settings() == {
        "top_k": "9",
        "custom": "preserved",
        "reranker_type": "off",
        "reranker_model": "",
    }
    reset.assert_called_once_with()


def test_staged_mcp_edit_does_not_overwrite_interleaved_cli_change(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    server.user_settings.save_settings({"top_k": "5", "answer_model": "old/model"})
    configure = cli._config_command

    def interleaved(args, **kwargs):
        result = configure(args, **kwargs)
        server.user_settings.update_settings({"answer_model": "new/cli-model"})
        return result

    monkeypatch.setattr(cli, "_config_command", interleaved)
    with pytest.raises(server.user_settings.SettingsConflict):
        server._apply_settings({"top": 9})
    assert server.user_settings.load_settings() == {
        "top_k": "5",
        "answer_model": "new/cli-model",
    }


def test_setting_order_and_alias_are_preserved(delegation):
    calls, reset = delegation
    result = server._apply_settings(
        {"contextual_retrieval": "on", "reranker": "off", "top": 3}
    )
    assert result == {
        "changed": ["top", "reranker", "contextual_retrieval"],
        "settings": ["configured"],
    }
    assert calls == [
        ("top", "3", None, True, None),
        ("reranker", "off", None, True, None),
        ("contextual", "on", None, True, None),
    ]
    reset.assert_called_once_with()


@pytest.mark.parametrize("failed", ["top", "reranker", "contextual"])
def test_validation_failure_stops_later_changes(monkeypatch, delegation, failed):
    calls, reset = delegation

    def configure(args, **kwargs):
        calls.append(args.setting)
        if args.setting == failed:
            kwargs["stderr"].write("invalid setting\n")
            return 2
        return 0

    monkeypatch.setattr(cli, "_config_command", configure)
    result = server._apply_settings(
        {"top": 3, "reranker": "off", "contextual_retrieval": "on"}
    )
    assert result == {"error": "invalid setting"}
    order = ["top", "reranker", "contextual"]
    assert calls == order[: order.index(failed) + 1]
    reset.assert_not_called()


def test_empty_change_does_not_reset(delegation):
    calls, reset = delegation
    assert "error" in server._apply_settings({})
    assert calls == []
    reset.assert_not_called()


def test_embedding_keeps_probe_and_runs_after_retrieval(delegation):
    calls, reset = delegation
    result = server._apply_settings(
        {"embedding_model": "test", "embedding_provider": "ollama", "top": 2}
    )
    assert result["changed"] == ["top", "embedding"]
    assert calls == [
        ("top", "2", None, True, None),
        ("embedding", "test", "ollama", False, cli._probe_embedding),
    ]
    reset.assert_called_once_with()


def test_mcp_local_runtime_setting_uses_detected_endpoint(monkeypatch, tmp_path):
    from rag import detect, settings

    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    settings.update_settings({"embed_api_key": "private-hosted-key"})
    monkeypatch.setattr(
        detect,
        "detect_all",
        lambda *_: {
            "ollama": {"embedding_models": []},
            "local_runtimes": [
                {
                    "label": "vLLM",
                    "base_url": "http://localhost:8000/v1",
                    "embedding_models": ["local-embedding"],
                }
            ],
            "api_keys": [],
        },
    )
    monkeypatch.setattr(cli, "_probe_embedding", lambda _: 384)
    result = server._apply_settings(
        {
            "embedding_model": "local-embedding",
            "embedding_provider": "openai_compatible",
            "embedding_runtime": "vLLM",
        }
    )
    assert result["changed"] == ["embedding"]
    assert settings.load_settings()["embed_base_url"] == "http://localhost:8000/v1"
    assert settings.load_settings()["embed_api_key"] == ""


def test_answer_length_and_timeouts_can_be_set_from_chat(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    for name in (
        "RAG_MCP_ANSWER_MAX_TOKENS",
        "RAG_MCP_ANSWER_TIMEOUT",
        "RAG_MCP_RERANKER_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    server.user_settings.save_settings({"top_k": "5"})
    monkeypatch.setattr(fetch, "reset_pipelines", Mock())
    assert {"answer_max_tokens", "answer_timeout", "reranker_timeout"} <= set(
        server._SETTINGS_SET_TOOL["inputSchema"]["properties"]
    )
    result = server._apply_settings(
        {"answer_max_tokens": 1200, "answer_timeout": 120, "reranker_timeout": 30}
    )
    assert result["changed"] == [
        "answer_max_tokens",
        "answer_timeout",
        "reranker_timeout",
    ]
    retrieval = server.user_settings.retrieval_config()
    assert (
        retrieval["answer_max_tokens"],
        retrieval["answer_timeout"],
        retrieval["reranker_timeout"],
    ) == ("1200", "120", "30")


@pytest.mark.parametrize(
    "arguments",
    [{"answer_max_tokens": 10}, {"answer_timeout": 0}, {"reranker_timeout": 9999}],
)
def test_out_of_range_answer_length_or_timeout_changes_nothing(
    monkeypatch, tmp_path, arguments
):
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    path = server.user_settings.save_settings({"top_k": "5"})
    original = path.read_bytes()
    assert "error" in server._apply_settings(arguments)
    assert path.read_bytes() == original
