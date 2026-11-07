"""Agent integrations for the local RAG MCP server."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO


def server_command(environ: Mapping[str, str]) -> list[str]:
    """Return a durable command an MCP host can use after this CLI exits."""
    spec = environ.get("RAGMCP_UVX_FROM", "").strip()
    if spec:
        return ["uvx", "--from", spec, "rag", "mcp"]
    if "/environments-v2/" in sys.executable or "/.cache/uv/" in sys.executable:
        from importlib.metadata import version

        return ["uvx", "--from", f"ragmcp=={version('ragmcp')}", "rag", "mcp"]
    installed = shutil.which("rag", path=environ.get("PATH"))
    if installed and ".venv" not in installed:
        return [installed, "mcp"]
    from rag import server

    return [sys.executable, str(Path(server.__file__).resolve())]


def _run(
    argv: list[str], environ: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, capture_output=True, check=False, text=True, env=dict(environ)
    )


def _replace_registration(
    add: list[str],
    remove: list[str] | None,
    restore: list[str] | None,
    environ: Mapping[str, str],
    stderr: TextIO,
) -> int:
    """Register `rag`, replacing an existing entry without ever losing it.

    The new entry is added first. Only when that fails (an entry named `rag` already exists) and the old entry can be
    put back (`restore` is known) is the old one removed and the add retried; if the retry fails, the old entry is
    restored. Clients whose current entry cannot be read keep it untouched and get the error instead.
    """
    added = _run(add, environ)
    if added.returncode == 0:
        return 0
    if remove is None or restore is None:
        print((added.stderr or added.stdout).strip(), file=stderr)
        return added.returncode
    _run(remove, environ)
    retried = _run(add, environ)
    if retried.returncode == 0:
        return 0
    restored = _run(restore, environ)
    print((retried.stderr or retried.stdout).strip(), file=stderr)
    if restored.returncode != 0:
        print(
            "The previous rag registration could not be restored: " + " ".join(restore),
            file=stderr,
        )
    return retried.returncode


def _claude_entry(
    executable: str, environ: Mapping[str, str]
) -> dict[str, object] | None:
    """The current `rag` stdio entry in Claude Code, from `claude mcp get rag`, or None."""
    current = _run([executable, "mcp", "get", "rag"], environ)
    if current.returncode != 0:
        return None
    fields = {}
    for line in current.stdout.splitlines():
        key, _, value = line.strip().partition(":")
        fields[key.strip().lower()] = value.strip()
    if fields.get("type", "stdio") != "stdio" or not fields.get("command"):
        return None
    return {
        "type": "stdio",
        "command": fields["command"],
        "args": shlex.split(fields.get("args", "")),
    }


def connect_codex(
    environ: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
    *,
    install_prompt: bool = False,
) -> int:
    executable = shutil.which("codex", path=environ.get("PATH"))
    if executable is None:
        print("Codex is not installed or is not on PATH.", file=stderr)
        return 1
    command = server_command(environ)
    current = subprocess.run(
        [executable, "mcp", "get", "rag", "--json"],
        capture_output=True,
        check=False,
        text=True,
        env=dict(environ),
    )
    restore = None
    if current.returncode == 0:
        try:
            transport = json.loads(current.stdout)["transport"]
        except (json.JSONDecodeError, KeyError, TypeError):
            transport = {}
        if (
            transport.get("command") == command[0]
            and transport.get("args") == command[1:]
        ):
            print("Codex is already connected to this RAG server.", file=stdout)
            return 0
        if transport.get("command"):
            old = [transport["command"], *transport.get("args", [])]
            restore = [executable, "mcp", "add", "rag", "--", *old]
    code = _replace_registration(
        [executable, "mcp", "add", "rag", "--", *command],
        [executable, "mcp", "remove", "rag"],
        restore,
        environ,
        stderr,
    )
    if code != 0:
        return code
    print("Connected Codex to the local RAG server.", file=stdout)
    if install_prompt:
        install_slash_commands(stdout, "codex", environ)
    return 0


def connect_claude(
    environ: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
    scope: str = "user",
    *,
    install_prompt: bool = False,
) -> int:
    executable = shutil.which("claude", path=environ.get("PATH"))
    command = [
        "claude",
        "mcp",
        "add",
        "rag",
        "-s",
        scope,
        "--",
        *server_command(environ),
    ]
    if executable is None:
        print(
            "Claude Code is not installed or is not on PATH. Register manually with:",
            file=stderr,
        )
        print("  " + " ".join(command), file=stderr)
        return 1
    entry = _claude_entry(executable, environ)
    restore = (
        [executable, "mcp", "add-json", "rag", json.dumps(entry), "-s", scope]
        if entry
        else None
    )
    code = _replace_registration(
        [executable, *command[1:]],
        [executable, "mcp", "remove", "rag", "-s", scope],
        restore,
        environ,
        stderr,
    )
    if code != 0:
        return code
    where = "all your projects" if scope == "user" else "this project (.mcp.json)"
    print(
        f"Connected Claude Code to the local RAG server for {where}: {' '.join(server_command(environ))}",
        file=stdout,
    )
    if install_prompt:
        install_slash_commands(stdout, "claude", environ)
    return 0


def connect_copilot(environ: Mapping[str, str], stdout: TextIO, stderr: TextIO) -> int:
    executable = shutil.which("copilot", path=environ.get("PATH"))
    if executable is None:
        print("GitHub Copilot CLI is not installed or is not on PATH.", file=stderr)
        return 1
    command = server_command(environ)
    added = subprocess.run(
        [executable, "mcp", "add", "rag", "--", *command],
        capture_output=True,
        check=False,
        text=True,
        env=dict(environ),
    )
    if added.returncode != 0:
        print((added.stderr or added.stdout).strip(), file=stderr)
        return added.returncode
    print(
        f"Connected GitHub Copilot CLI to the local RAG server: {' '.join(shlex.quote(part) for part in command)}",
        file=stdout,
    )
    return 0


def connect_gemini(
    environ: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
    scope: str = "user",
) -> int:
    executable = shutil.which("gemini", path=environ.get("PATH"))
    if executable is None:
        print("Gemini CLI is not installed or is not on PATH.", file=stderr)
        return 1
    command = server_command(environ)
    added = subprocess.run(
        [
            executable,
            "mcp",
            "add",
            "--scope",
            scope,
            "--transport",
            "stdio",
            "rag",
            *command,
        ],
        capture_output=True,
        check=False,
        text=True,
        env=dict(environ),
    )
    if added.returncode != 0:
        print((added.stderr or added.stdout).strip(), file=stderr)
        return added.returncode
    where = "your user settings" if scope == "user" else "this project's settings"
    print(
        f"Connected Gemini CLI to {where}: {' '.join(shlex.quote(part) for part in command)}",
        file=stdout,
    )
    return 0


def connect_agy(environ: Mapping[str, str], stdout: TextIO, stderr: TextIO) -> int:
    executable = shutil.which("agy", path=environ.get("PATH"))
    if executable is None:
        print("Agy is not installed or is not on PATH.", file=stderr)
        return 1
    command = server_command(environ)
    added = subprocess.run(
        [executable, "mcp", "add", "rag", command[0], *command[1:]],
        capture_output=True,
        check=False,
        text=True,
        env=dict(environ),
    )
    if added.returncode != 0:
        print((added.stderr or added.stdout).strip(), file=stderr)
        return added.returncode
    print(
        f"Connected Agy to the local RAG server: {' '.join(shlex.quote(part) for part in command)}",
        file=stdout,
    )
    return 0


AGENT_LABELS = (
    ("codex", "Codex", "codex"),
    ("claude", "Claude Code", "claude"),
    ("copilot", "GitHub Copilot CLI", "copilot"),
    ("gemini", "Gemini CLI", "gemini"),
    ("agy", "Agy", "agy"),
)


def available_agents(environ: Mapping[str, str]) -> list[tuple[str, str]]:
    return [
        (key, label)
        for key, label, executable in AGENT_LABELS
        if shutil.which(executable, path=environ.get("PATH"))
    ]


def show_generic_mcp_config(environ: Mapping[str, str], stdout: TextIO) -> int:
    command = server_command(environ)
    print("For another MCP-compatible app, add this local stdio server:", file=stdout)
    print(
        json.dumps(
            {"mcpServers": {"rag": {"command": command[0], "args": command[1:]}}},
            indent=2,
        ),
        file=stdout,
    )
    print(
        "The app must support local/stdio MCP servers. This command only shows the config; it does not edit unknown app files.",
        file=stdout,
    )
    return 0


def connect_agent(
    agent: str,
    environ: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
    scope: str = "user",
    *,
    install_prompt: bool = False,
) -> int:
    if agent == "codex":
        return connect_codex(environ, stdout, stderr, install_prompt=install_prompt)
    if agent == "claude":
        return connect_claude(
            environ, stdout, stderr, scope=scope, install_prompt=install_prompt
        )
    if agent == "copilot":
        return connect_copilot(environ, stdout, stderr)
    if agent == "gemini":
        return connect_gemini(environ, stdout, stderr, scope=scope)
    if agent == "agy":
        return connect_agy(environ, stdout, stderr)
    return show_generic_mcp_config(environ, stdout)


def offer_agent_connection(
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    environ: Mapping[str, str],
) -> None:
    """Compatibility helper for callers that explicitly request this flow."""
    if not getattr(stdin, "isatty", lambda: False)():
        return
    from rag.prompts import select

    available = available_agents(environ)
    choices = [label for _, label in available] + [
        "Another MCP app — show config",
        "Skip",
    ]
    answer = select(
        "Connect an app to RAG MCP?",
        choices,
        default=len(choices) - 1,
        stdin=stdin,
        stdout=stdout,
    )
    if answer < len(available):
        connect_agent(available[answer][0], environ, stdout, stderr)
    elif answer == len(available):
        show_generic_mcp_config(environ, stdout)


SLASH_COMMAND = """\
---
description: Change or inspect the RAG answer pipeline in plain words (answer passages, reranker, embedding model)
---
The user wants to change or inspect their RAG answer pipeline: $ARGUMENTS

Use the `rag` command line tool through the shell (if `rag` is not on PATH, use `uvx ragmcp` with the same arguments). Map the request to exactly one of these and run it:
- `rag config show`
- `rag config top <N>`                       (also for "top5", "top 5", "use 5 passages for the answer")
- `rag config reranker off|auto|fastembed|openrouter|<model>`
- `rag config embedding <model> [--provider fastembed|ollama|openai_compatible|sentencetransformer]`
- `rag setup --auto`                          (start over with what this machine already has)
- `rag doctor`                                (what is installed and running)
Never ask for or paste API keys in this chat; for keys, tell the user to run `rag setup` in a terminal.
Finish by running `rag config show` and report its lines to the user, nothing more.
"""


def install_slash_commands(
    stdout: TextIO, agent: str, environ: Mapping[str, str]
) -> None:
    """Install the optional agent prompt only when the caller opted in."""
    home = Path(environ.get("HOME") or Path.home())
    targets = {
        "claude": home / ".claude" / "commands" / "rag.md",
        "codex": home / ".codex" / "prompts" / "rag.md",
    }
    target = targets[agent]
    target.parent.mkdir(parents=True, exist_ok=True)
    body = SLASH_COMMAND if agent == "claude" else SLASH_COMMAND.split("---\n", 2)[-1]
    target.write_text(body, encoding="utf-8")
    hint = (
        "/rag <what you want>" if agent == "claude" else "/prompts:rag <what you want>"
    )
    print(f"Installed the {hint} command ({target}).", file=stdout)
