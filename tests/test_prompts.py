import io

import pytest
from rag import model_catalog
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from rag import cli, prompts, settings


@pytest.mark.parametrize("keys, expected", [("\x1b[B\r", 1), ("\x1b[A\r", 2), ("2\x1b[B\r", 1)])
def test_selector_uses_arrow_keys_and_ignores_typed_menu_numbers(monkeypatch, keys, expected):
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        monkeypatch.setattr("prompt_toolkit.input.create_input", lambda **_: pipe)
        monkeypatch.setattr("prompt_toolkit.output.create_output", lambda **_: DummyOutput())
        assert prompts.select("Model", ["One", "Two", "Other"], stdin=io.StringIO(), stdout=io.StringIO()) == expected


@pytest.mark.parametrize("key", ["\x1b", "\x03"])
def test_selector_escape_and_ctrl_c_cancel_without_a_typed_answer(monkeypatch, key):
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    with create_pipe_input() as pipe:
        pipe.send_text(key)
        monkeypatch.setattr("prompt_toolkit.input.create_input", lambda **_: pipe)
        monkeypatch.setattr("prompt_toolkit.output.create_output", lambda **_: DummyOutput())
        with pytest.raises(prompts.Cancelled):
            prompts.select("Model", ["One", "Two"], stdin=io.StringIO(), stdout=io.StringIO())


def test_presets_only_allow_typing_after_the_final_option(monkeypatch):
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    typed = []
    monkeypatch.setattr(prompts, "text", lambda *a, **k: typed.append(True) or "my-model")
    options = []
    monkeypatch.setattr(prompts, "select", lambda _, choices, **k: options.extend(choices) or 0)
    assert prompts.preset("Model", ["Small", "Large"], ["small", "large"], stdin=io.StringIO(), stdout=io.StringIO()) == "small"
    assert typed == []
    assert options[-1] == "Type your own…"
    monkeypatch.setattr(prompts, "select", lambda _, choices, **k: len(choices) - 1)
    assert prompts.preset("Model", ["Small", "Large"], ["small", "large"], stdin=io.StringIO(), stdout=io.StringIO()) == "my-model"
    assert typed == [True]


def test_path_picker_navigates_to_a_file_without_typing(tmp_path, monkeypatch):
    folder = tmp_path / "notes"
    folder.mkdir()
    doc = folder / "facts.md"
    doc.write_text("Cedar Lantern")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    choices_seen = []
    def choose(_, choices, **kwargs):
        choices_seen.append(choices)
        return 2
    monkeypatch.setattr(prompts, "select", choose)
    assert prompts.path(stdin=io.StringIO(), stdout=io.StringIO()) == str(doc)
    assert all(choices[-1] == "Type a path…" for choices in choices_seen)


def test_cancelled_custom_entry_returns_to_its_choices(monkeypatch):
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    selections = iter([2, 1])
    monkeypatch.setattr(prompts, "select", lambda *a, **k: next(selections))
    def cancelled(*args, **kwargs):
        raise prompts.Cancelled
    monkeypatch.setattr(prompts, "text", cancelled)
    assert prompts.preset("Model", ["Small", "Large"], ["small", "large"], stdin=io.StringIO(), stdout=io.StringIO()) == "large"


def test_custom_text_escape_cancels(monkeypatch):
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b")
        monkeypatch.setattr("prompt_toolkit.input.create_input", lambda **_: pipe)
        monkeypatch.setattr("prompt_toolkit.output.create_output", lambda **_: DummyOutput())
        with pytest.raises(prompts.Cancelled):
            prompts.text("Custom model", stdin=io.StringIO(), stdout=io.StringIO())


def test_menu_returns_after_saving_a_setting(tmp_path, monkeypatch):
    env = {"RAG_MCP_CONFIG": str(tmp_path / "config.json")}
    selections = iter([3, 0, 0, 6])
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    monkeypatch.setattr(prompts, "select", lambda *a, **k: next(selections))
    assert cli._menu(stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO(), environ=env) == 0
    assert settings.retrieval_config(env)["top_k"] == "1"


def test_cancelling_model_setup_returns_to_the_menu_without_saving(tmp_path, monkeypatch):
    env = {"RAG_MCP_CONFIG": str(tmp_path / "config.json")}
    monkeypatch.setattr("rag.detect.detect_all", lambda *_: {})
    selections = iter([0, prompts.Cancelled(), 6])
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    def choose(*args, **kwargs):
        value = next(selections)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(prompts, "select", choose)
    assert cli._menu(stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO(), environ=env) == 0
    assert not settings.config_path(env).exists()


def test_api_model_is_selected_from_catalog_without_text_entry(tmp_path, monkeypatch):
    env = {"RAG_MCP_CONFIG": str(tmp_path / "config.json")}
    monkeypatch.setattr("rag.detect.detect_all", lambda *_: {"api_keys": ["openrouter"]})
    monkeypatch.setattr(prompts, "interactive", lambda *_: True)
    env["OPENROUTER_API_KEY"] = "test-key"
    monkeypatch.setattr(model_catalog, "api_models", lambda *args: ["new/vendor-model-a", "new/vendor-model-b"])
    selections = iter([0, 2, 1])
    monkeypatch.setattr(prompts, "select", lambda *a, **k: next(selections))
    monkeypatch.setattr(prompts, "text", lambda *a, **k: pytest.fail("preset flow asked for typed input"))
    assert settings.run_setup(stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO(), environ=env, verify=lambda _: 1024) == 0
    assert settings.embedding_config(env)["embed_model"] == "new/vendor-model-b"
