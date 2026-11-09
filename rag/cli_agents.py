"""Agent integrations for the local RAG MCP server."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO, cast
from utils.persistence import write_json_atomic

_SAFE_ENV = {'RAG_MCP_ALLOWED_ROOTS', 'RAG_MCP_WORKSPACE_ROOT'}


def _safe_old_args(args: object) -> bool:
    if not isinstance(args, list):
        return False
    values = cast(list[object], args)
    return all(isinstance(value, str) and (
        value in {'mcp', '-m', 'rag.cli', 'rag', '--from'}
        or value.endswith('/server.py')
        or re.fullmatch(r'ragmcp-cli==[0-9A-Za-z.+_-]+', value) is not None
    ) for value in values)


def _safe_entry_env(entry: dict[str, Any] | None) -> dict[str, str] | None:
    raw: Any = (entry or {}).get('env', {}) or {}
    if not isinstance(raw, dict):
        return None
    values = cast(dict[object, object], raw)
    if any(not isinstance(k, str) or not isinstance(v, str) or k not in _SAFE_ENV
           for k, v in values.items()):
        return None
    return cast(dict[str, str], values)


def server_command(environ: Mapping[str, str]) -> list[str]:
    """Return a durable command an MCP host can use after this CLI exits."""
    spec = environ.get("RAGMCP_UVX_FROM", "").strip()
    if spec:
        return ["uvx", "--from", spec, "rag", "mcp"]
    if "/environments-v2/" in sys.executable or "/.cache/uv/" in sys.executable:
        from importlib.metadata import version

        return ["uvx", "--from", f"ragmcp-cli=={version('ragmcp-cli')}", "rag", "mcp"]
    installed = shutil.which("rag", path=environ.get("PATH"))
    if installed and ".venv" not in installed:
        return [installed, "mcp"]
    from rag import server

    return [sys.executable, str(Path(server.__file__).resolve())]


def _run(
    argv: list[str], environ: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, capture_output=True, check=False, text=True, env=dict(environ)
        )
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess(argv, 1, '', '')


def _replace_registration(
    add: list[str],
    remove: list[str] | None,
    restore: list[str] | None,
    environ: Mapping[str, str],
    stderr: TextIO,
    old_entry: dict[str, Any] | None = None,
    prepare_recovery: Callable[[], tuple[Callable[[], bool], Callable[[], None]] | None] | None = None,
    recovery_location: str = 'mcpServers.rag',
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
        print('Could not register the RAG server; the existing entry was left unchanged.', file=stderr)
        return added.returncode
    recovery = prepare_recovery() if prepare_recovery else None
    if recovery is None:
        print('Could not safely back up the existing RAG registration; it was left unchanged.', file=stderr)
        return added.returncode
    recover, cleanup = recovery
    try:
        removed = _run(remove, environ)
        if removed.returncode:
            print('Could not remove the existing RAG registration.', file=stderr)
            return removed.returncode
        retried = _run(add, environ)
        if retried.returncode == 0:
            return 0
        restored = _run(restore, environ)
        print('Could not register the replacement RAG server.', file=stderr)
        if restored.returncode != 0:
            if recover():
                print('The previous rag registration was restored from its backup.', file=stderr)
            else:
                print(f'The previous rag registration could not be restored. Restore {recovery_location} manually.', file=stderr)
        return retried.returncode
    finally:
        cleanup()


def _claude_config_path(environ: Mapping[str, str], scope: str) -> Path:
    home = Path(environ.get('HOME') or Path.home())
    config_dir = Path(environ.get('CLAUDE_CONFIG_DIR') or home)
    return Path.cwd() / '.mcp.json' if scope == 'project' else config_dir / '.claude.json'


def _claude_entry(environ: Mapping[str, str], scope: str) -> dict[str, Any] | None:
    """Read the original entry at its actual scope; CLI display omits fields."""
    path = _claude_config_path(environ, scope)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if scope == 'local':
            data = data['projects'][str(Path.cwd())]
        entry = data['mcpServers']['rag']
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return cast(dict[str, Any], entry) if isinstance(entry, dict) else None


def _prepare_claude_recovery(environ: Mapping[str, str], scope: str, entry: dict[str, Any]) -> tuple[Callable[[], bool], Callable[[], None]] | None:
    path = _claude_config_path(environ, scope)
    backup_path: Path | None = None
    try:
        if path.is_symlink():
            return None
        original = path.read_bytes()
        decoded = json.loads(original)
        container = decoded['projects'][str(Path.cwd())] if scope == 'local' else decoded
        if container['mcpServers']['rag'] != entry:
            return None
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.rag-backup-', delete=False) as backup:
            backup_path = Path(backup.name)
            backup.write(json.dumps(entry).encode('utf-8'))
            backup.flush()
            os.fsync(backup.fileno())
    except (OSError, ValueError, KeyError, TypeError):
        if backup_path is not None:
            backup_path.unlink(missing_ok=True)
        return None

    assert backup_path is not None

    def recover() -> bool:
        try:
            if path.is_symlink():
                return False
            data = json.loads(path.read_text(encoding='utf-8'))
            container = data['projects'][str(Path.cwd())] if scope == 'local' else data
            servers = container.setdefault('mcpServers', {})
            if 'rag' in servers and servers['rag'] != entry:
                return False
            servers['rag'] = entry
            write_json_atomic(path, data)
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False

    def cleanup() -> None:
        backup_path.unlink(missing_ok=True)

    return recover, cleanup


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
    restore: list[str] | None = None
    entry: dict[str, Any] | None = None
    flags: list[str] = []
    if current.returncode == 0:
        transport: dict[str, Any] = {}
        try:
            decoded: Any = json.loads(current.stdout)
            if isinstance(decoded, dict):
                entry = cast(dict[str, Any], decoded)
                raw_transport: Any = entry.get('transport')
                if isinstance(raw_transport, dict):
                    transport = cast(dict[str, Any], raw_transport)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        if (
            transport.get("command") == command[0]
            and transport.get("args") == command[1:]
        ):
            print("Codex is already connected to this RAG server.", file=stdout)
            return 0
        extras = {k: v for k, v in transport.items() if k not in {'type', 'command', 'args', 'env'} and v}
        extras.update({k: v for k, v in (entry or {}).items()
                       if k not in {'name', 'transport', 'disabled_reason', 'enabled'} and v is not None})
        if entry and entry.get('enabled') is False:
            extras['enabled'] = False
        raw_env: Any = transport.get('env') or {}
        safe_env = _safe_entry_env(transport)
        if safe_env is None:
            print('The existing registration contains environment values that cannot be passed safely. Update the client configuration manually.', file=stderr)
            return 1
        env_vars = cast(dict[object, object], raw_env) if isinstance(raw_env, dict) else {}
        old_command: Any = transport.get('command')
        old_args: Any = transport.get('args') or []
        if (isinstance(old_command, str) and isinstance(old_args, list) and
                _safe_old_args(cast(object, old_args)) and
                not extras and isinstance(raw_env, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in env_vars.items())):
            old = [old_command, *cast(list[str], old_args)]
            flags = [item for key, value in cast(dict[str, str], env_vars).items()
                     for item in ('--env', f'{key}={value}')]
            restore = [executable, "mcp", "add", "rag", *flags, "--", *old]
        if restore is None:
            print('The existing Codex registration cannot be preserved safely. Update it manually.', file=stderr)
            return 1
    code = _replace_registration(
        [executable, "mcp", "add", "rag", *flags, "--", *command],
        None,
        restore,
        environ,
        stderr,
        entry,
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
    entry = _claude_entry(environ, scope)
    if entry and (_safe_entry_env(entry) is None or not _safe_old_args(entry.get('args', []))
                  or set(entry) - {'type', 'command', 'args', 'env', 'cwd'}):
        print('The existing registration cannot be passed safely. Update the Claude configuration manually.', file=stderr)
        return 1
    restore = (
        [executable, "mcp", "add-json", "rag", json.dumps(entry), "-s", scope]
        if entry
        else None
    )
    replacement = ({**entry, 'command': server_command(environ)[0], 'args': server_command(environ)[1:]}
                   if entry else None)
    add = ([executable, 'mcp', 'add-json', 'rag', json.dumps(replacement), '-s', scope]
           if replacement else [executable, *command[1:]])
    code = _replace_registration(
        add,
        [executable, "mcp", "remove", "rag", "-s", scope],
        restore,
        environ,
        stderr,
        entry,
        (lambda: _prepare_claude_recovery(environ, scope, entry)) if entry else None,
        f'projects[{json.dumps(str(Path.cwd()))}].mcpServers.rag' if scope == 'local' else 'mcpServers.rag',
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

Use the `rag` command line tool through the shell (if `rag` is not on PATH, use `uvx --from ragmcp-cli rag` with the same arguments). Map the request to exactly one of these and run it:
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
