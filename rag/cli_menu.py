"""Human-facing terminal UI for the :mod:`rag.cli` command line."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TextIO

from rag import settings as user_settings


MENU = (
    ("Ask about a file or folder", None),
    ("Connect your coding agent (Claude Code, Codex, ...)", None),
    ("Add or change the embedding model (meaning search)", ["setup"]),
    ("Add or change the answer key (written answers)", ["key", "set"]),
    ("Change settings", None),
    ("Check setup", ["doctor"]),
    ("Inspect recent runs", ["runs"]),
    ("Quit", None),
)


def status_line(environ: Mapping[str, str]) -> str:
    """What works right now, in one line: search always, meaning search and answers when set up."""
    config = user_settings.embedding_config(environ)
    embedding = (user_settings.describe_embedding(config) if user_settings.has_embedding(config)
                 else "off")
    answers = ("on (OpenRouter key saved)" if user_settings.resolve_openrouter_key(config, environ)
               else "off: questions return the matching passages")
    return f"Exact-word search: on · Meaning search: {embedding} · Written answers: {answers}"


def menu(
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    environ: Mapping[str, str],
    **passthrough: object,
) -> int:
    """Run the deliberate interactive UI; never use this path for agents."""
    from rag import prompts

    stdout.write("rag · search your documents from this terminal or from your coding agent\n")
    stdout.write(status_line(environ) + "\n\n")
    selected = 0
    while True:
        try:
            selected = prompts.select(
                "What would you like to do?",
                [label for label, _ in MENU],
                default=selected,
                stdin=stdin,
                stdout=stdout,
            )
        except prompts.Cancelled:
            stdout.write("\n")
            return 0
        except ValueError as error:
            stderr.write(f"{error}\n")
            return 2
        if MENU[selected][0] == "Quit":
            return 0
        try:
            code = _menu_action(
                selected,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                environ=environ,
                **passthrough,
            )
        except prompts.Cancelled:
            code = 130
        if not prompts.interactive(stdin, stdout):
            return code
        stdout.write("\n")


def _search_questions(*, stdin: TextIO, stdout: TextIO, environ: Mapping[str, str]) -> list[str]:
    from rag import prompts

    step, path, question = 0, "", ""
    while True:
        try:
            if step == 0:
                path = prompts.path(stdin=stdin, stdout=stdout, start=path)
                step = 1
            if step == 1:
                question = prompts.preset(
                    "What would you like to find?",
                    ["Main topics", "Dates and deadlines", "Requirements and obligations"],
                    [
                        "What are the main topics?",
                        "What dates and deadlines are mentioned?",
                        "What are the requirements and obligations?",
                    ],
                    default=question,
                    stdin=stdin,
                    stdout=stdout,
                )
                if not question.strip():
                    stdout.write("Enter a question or choose one from the list.\n")
                    continue
                step = 2
            top = prompts.result_count(
                user_settings.retrieval_config(environ)["top_k"],
                stdin=stdin,
                stdout=stdout,
            )
            return ["query", question, "--path", path, "--top", top, "--format", "text"]
        except prompts.Cancelled:
            if step == 0 or not prompts.interactive(stdin, stdout):
                raise
            step -= 1


def _settings_questions(
    *,
    stdin: TextIO,
    stdout: TextIO,
    environ: Mapping[str, str],
    selected: int = 0,
) -> list[str]:
    from rag import prompts

    while True:
        config = user_settings.retrieval_config(environ)
        selected = prompts.select(
            "Which setting would you like to change?",
            [
                f"Passages for answer · {config['top_k']}",
                f"Reranker · {config['reranker_type']}",
                "Embedding model",
            ],
            default=selected,
            stdin=stdin,
            stdout=stdout,
        )
        try:
            if selected == 2:
                return ["setup"]
            if selected == 0:
                value = prompts.result_count(config["top_k"], stdin=stdin, stdout=stdout)
                if not value.isdigit() or int(value) < 1:
                    stdout.write("Choose a result count or enter a positive whole number.\n")
                    continue
                return ["config", "top", value]
            current = config["reranker_type"]
            if current == "cross-encoder" or (
                current == "fastembed"
                and config["reranker_model"] not in {"", user_settings.FASTEMBED_RERANKER}
            ):
                current = config["reranker_model"]
            value = prompts.preset(
                "Which reranker would you like to use?",
                ["Off", "Automatic", "MiniLM cross-encoder · local CPU"],
                ["off", "auto", "fastembed"],
                default=current,
                stdin=stdin,
                stdout=stdout,
                validate=lambda answer: bool(answer.strip())
                or "Enter a model ID, or press Esc to choose from the list.",
            )
            return ["config", "reranker", value]
        except prompts.Cancelled:
            if not prompts.interactive(stdin, stdout):
                raise


def _menu_action(
    selected: int,
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    environ: Mapping[str, str],
    **passthrough: object,
) -> int:
    from rag import cli, prompts

    label, command = MENU[selected]
    if label == "Quit":
        return 0
    if label.startswith("Ask"):
        command = _search_questions(stdin=stdin, stdout=stdout, environ=environ)
    elif label.startswith("Change"):
        setting = 0
        while True:
            command = _settings_questions(
                stdin=stdin,
                stdout=stdout,
                environ=environ,
                selected=setting,
            )
            code = cli.main(
                command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                environ=environ,
                **passthrough,
            )
            if command != ["setup"] or code != 130 or not prompts.interactive(stdin, stdout):
                return code
            setting = 2
    elif label.startswith("Connect"):
        available = cli._available_agents(environ)
        choices = [label for _, label in available] + ["Another MCP app — show config"]
        answer = prompts.select(
            "Which app should use RAG MCP?",
            choices,
            stdin=stdin,
            stdout=stdout,
        )
        if answer == len(available):
            return cli._show_generic_mcp_config(environ, stdout)
        command = ["connect", available[answer][0]]
    return cli.main(
        command,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ=environ,
        **passthrough,
    )  # type: ignore[arg-type]
