from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from rag import settings as user_settings


def _codex_status(environ: Mapping[str, str]) -> dict[str, object]:
    executable = shutil.which("codex", path=environ.get("PATH"))
    if executable is None:
        return {"available": False, "authenticated": False, "method": None}

    completed = subprocess.run(
        [executable, "login", "status"],
        capture_output=True,
        check=False,
        text=True,
        env=dict(environ),
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    authenticated = completed.returncode == 0 and "Logged in" in output
    method = "ChatGPT" if "ChatGPT" in output else ("API key" if authenticated else None)
    return {"available": True, "authenticated": authenticated, "method": method}


def _full_extra_installed() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _probe_embedding(config: Mapping[str, str]) -> int:
    from rag.embeddings import build_embedder, probe_dimensions

    return probe_dimensions(build_embedder(config))


def _should_offer_setup(environ: Mapping[str, str], stdin: TextIO) -> bool:
    """First interactive query with no saved choice: ask before searching."""
    if environ.get("RAG_MCP_EMBED_PROVIDER", "").strip():
        return False
    if user_settings.load_settings(environ):
        return False
    return bool(getattr(stdin, "isatty", lambda: False)())


_TOP_SHORTHAND = re.compile(r"^--?top-?(\d+)$")


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Accept `--top5` / `--top-5` as shorthand for `--top 5`."""
    normalized: list[str] = []
    for item in argv:
        match = _TOP_SHORTHAND.match(item)
        if match:
            normalized.extend(["--top", match.group(1)])
        else:
            normalized.append(item)
    return normalized


def _server_command(environ: Mapping[str, str]) -> list[str]:
    """How a coding agent should start the MCP server so it keeps working after this process exits."""
    spec = environ.get("RAGMCP_UVX_FROM", "").strip()
    if spec:  # launched through the npm launcher or uvx: re-run the same package spec
        return ["uvx", "--from", spec, "rag", "mcp"]
    if "/environments-v2/" in sys.executable or "/.cache/uv/" in sys.executable:
        from importlib.metadata import version

        return ["uvx", "--from", f"ragmcp=={version('ragmcp')}", "rag", "mcp"]
    installed = shutil.which("rag", path=environ.get("PATH"))
    if installed and ".venv" not in installed:  # uv tool install / pipx / npm -g
        return [installed, "mcp"]
    import server

    return [sys.executable, str(Path(server.__file__).resolve())]


def _connect_codex(environ: Mapping[str, str], stdout: TextIO, stderr: TextIO) -> int:
    executable = shutil.which("codex", path=environ.get("PATH"))
    if executable is None:
        print("Codex is not installed or is not on PATH.", file=stderr)
        return 1
    command = _server_command(environ)
    current = subprocess.run(
        [executable, "mcp", "get", "rag", "--json"],
        capture_output=True,
        check=False,
        text=True,
        env=dict(environ),
    )
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
        subprocess.run(
            [executable, "mcp", "remove", "rag"],
            capture_output=True,
            check=False,
            text=True,
            env=dict(environ),
        )
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
    print("Connected Codex to the local RAG server.", file=stdout)
    _install_slash_commands(stdout, "codex", environ)
    return 0


def _connect_claude(environ: Mapping[str, str], stdout: TextIO, stderr: TextIO, scope: str = "user") -> int:
    executable = shutil.which("claude", path=environ.get("PATH"))
    command = ["claude", "mcp", "add", "rag", "-s", scope, "--", *_server_command(environ)]
    if executable is None:
        print("Claude Code is not installed or is not on PATH. Register manually with:", file=stderr)
        print("  " + " ".join(command), file=stderr)
        return 1
    subprocess.run([executable, "mcp", "remove", "rag", "-s", scope], capture_output=True, check=False, text=True, env=dict(environ))
    added = subprocess.run([executable, *command[1:]], capture_output=True, check=False, text=True, env=dict(environ))
    if added.returncode != 0:
        print((added.stderr or added.stdout).strip(), file=stderr)
        return added.returncode
    where = "all your projects" if scope == "user" else "this project (.mcp.json)"
    print(f"Connected Claude Code to the local RAG server for {where}: {' '.join(_server_command(environ))}", file=stdout)
    _install_slash_commands(stdout, "claude", environ)
    return 0


def _offer_agent_connection(stdin: TextIO, stdout: TextIO, stderr: TextIO, environ: Mapping[str, str]) -> None:
    """After an interactive setup: hook the MCP server to a coding agent in one step."""
    if not getattr(stdin, "isatty", lambda: False)():
        return
    stdout.write("Connect the MCP server to a coding agent now? [codex/claude/skip] [skip]: ")
    stdout.flush()
    answer = (stdin.readline() or "").strip().lower()
    if answer == "codex":
        _connect_codex(environ, stdout, stderr)
    elif answer == "claude":
        _connect_claude(environ, stdout, stderr)


MENU = (
    ("Set up meaning-based search (choose an embedding model)", ["setup"]),
    ("Search a file or folder", None),
    ("Connect a coding agent (Claude Code or Codex)", None),
    ("Change settings (passages per query, reranker, embedding model)", None),
    ("Open the Studio paired with this machine", ["studio"]),
    ("Check this machine (doctor)", ["doctor"]),
    ("Quit", None),
)


def _menu(*, stdin: TextIO, stdout: TextIO, stderr: TextIO, environ: Mapping[str, str], **passthrough: object) -> int:
    """`rag` with no arguments in a terminal: a numbered menu instead of usage text."""
    from rag.settings import _ask

    stdout.write("rag · local search for your documents and code\n\n")
    for index, (label, _) in enumerate(MENU, 1):
        stdout.write(f"  {index}) {label}\n")
    stdout.write("\n")
    answer = _ask(stdin, stdout, "Choice", "1")
    try:
        label, command = MENU[int(answer) - 1]
    except (ValueError, IndexError):
        stderr.write(f"Not a listed choice: {answer}\n")
        return 2
    if label == "Quit":
        return 0
    if label.startswith("Search"):
        path = _ask(stdin, stdout, "File or folder to search", ".")
        question = _ask(stdin, stdout, "Question")
        if not question:
            stderr.write("A question is required.\n")
            return 2
        top = _ask(stdin, stdout, "How many passages", "5")
        command = ["query", question, "--path", path, "--top", top]
    elif label.startswith("Change settings"):
        for line in user_settings.describe_settings(environ)[:3]:
            stdout.write(f"  {line}\n")
        setting = _ask(stdin, stdout, "Change which [top/reranker/embedding]", "top")
        if setting not in {"top", "reranker", "embedding"}:
            stderr.write(f"Not a listed setting: {setting}\n")
            return 2
        prompts = {"top": "Passages per query", "reranker": "Reranker [off/auto/fastembed/model]", "embedding": "Embedding model"}
        value = _ask(stdin, stdout, prompts[setting])
        command = ["config", setting, value]
    elif label.startswith("Connect"):
        agent = _ask(stdin, stdout, "Agent [claude/codex]", "claude").lower()
        if agent not in {"claude", "codex"}:
            stderr.write(f"Not a listed agent: {agent}\n")
            return 2
        command = ["connect", agent]
    return main(command, stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, **passthrough)  # type: ignore[arg-type]


SLASH_COMMAND = """\
---
description: Change or inspect the local rag search setup in plain words (top N, reranker, embedding model)
---
The user wants to change or inspect their local rag search setup: $ARGUMENTS

Use the `rag` command line tool through the shell (if `rag` is not on PATH, use `uvx ragmcp` with the same arguments). Map the request to exactly one of these and run it:
- `rag config show`
- `rag config top <N>`                       (also for "top5", "top 5", "return 5 passages")
- `rag config reranker off|auto|fastembed|<model>`
- `rag config embedding <model> [--provider fastembed|ollama|openai_compatible|sentencetransformer]`
- `rag setup --auto`                          (start over with what this machine already has)
- `rag doctor`                                (what is installed and running)
Never ask for or paste API keys in this chat; for keys, tell the user to run `rag setup` in a terminal.
Finish by running `rag config show` and report its lines to the user, nothing more.
"""


def _resolve_embedding_choice(model: str, provider: str | None, environ: Mapping[str, str]) -> dict[str, str]:
    """Turn `rag config embedding <model>` into a full embedding config, guessing the provider from what runs here."""
    from rag import detect

    config = dict(user_settings.DEFAULTS)
    detection = detect.detect_all(environ)
    ollama_models = (detection.get("ollama") or {}).get("embedding_models") or []
    if provider is None:
        match = next((name for name in ollama_models if name == model or name.startswith(model + ":") or name.split(":")[0] == model), None)
        if match:
            provider, model = "ollama", match
        elif any(model.lower().startswith(prefix) for prefix in ("text-embedding-", "gemini-embedding", "qwen/", "voyage")):
            provider = "openai_compatible"
        elif "/" in model or model.lower() in {"all-minilm-l6-v2"}:
            provider = "fastembed"
        else:
            raise ValueError(
                f"Cannot tell where '{model}' runs. Add --provider ollama|fastembed|openai_compatible|sentencetransformer."
                + (f" Ollama has: {', '.join(ollama_models)}." if ollama_models else "")
            )
    config["embed_provider"] = provider
    config["embed_model"] = model
    if provider == "openai_compatible":
        family = "gemini" if "gemini" in model.lower() else "openrouter" if "/" in model else "openai"
        base_url, _ = detect.API_ENDPOINTS[family]
        config["embed_base_url"] = base_url
        saved = user_settings.load_settings(environ)
        if saved.get("embed_base_url") == base_url and saved.get("embed_api_key"):
            config["embed_api_key"] = str(saved["embed_api_key"])
        elif family not in detection.get("api_keys", []):
            raise ValueError(f"No API key for {family} in the environment. Run `rag setup` to enter one.")
    return config


def _config_command(args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO, environ: Mapping[str, str], verify) -> int:
    setting, value = args.setting, args.value.strip()
    if setting == "show":
        for line in user_settings.describe_settings(environ):
            print(line, file=stdout)
        return 0
    if setting == "top":
        if not value.isdigit() or int(value) < 1:
            print("Give a whole number, e.g. `rag config top 5`.", file=stderr)
            return 2
        user_settings.update_settings({"top_k": value}, environ)
    elif setting == "reranker":
        if not value:
            print("Give off, auto, fastembed or a model name.", file=stderr)
            return 2
        if value in {"off", "auto"}:
            user_settings.update_settings({"reranker_type": value, "reranker_model": ""}, environ)
        elif value == "fastembed" or value.startswith(("Xenova/", "jinaai/", "BAAI/bge-reranker")):
            model = user_settings.FASTEMBED_RERANKER if value == "fastembed" else value
            if verify is not None:
                from rag.embeddings import FastEmbedReranker

                print("Fetching the reranker model...", file=stdout)
                try:
                    scores = FastEmbedReranker(model).predict([("probe", "a passage")])
                except Exception as error:
                    print(f"Could not load the reranker: {error}\nNothing was saved.", file=stderr)
                    return 1
                print(f"Works: reranker answered {len(scores)} score(s).", file=stdout)
            user_settings.update_settings({"reranker_type": "fastembed", "reranker_model": model}, environ)
        else:
            user_settings.update_settings({"reranker_type": "cross-encoder", "reranker_model": value}, environ)
    elif setting == "embedding":
        if not value:
            print("Give a model name, e.g. `rag config embedding qwen3-embedding:0.6b`.", file=stderr)
            return 2
        try:
            config = _resolve_embedding_choice(value, args.provider, environ)
        except ValueError as error:
            print(str(error), file=stderr)
            return 2
        code = user_settings.finish_setup(config, stdout=stdout, stderr=stderr, environ=environ, verify=verify)
        if code != 0:
            return code
    for line in user_settings.describe_settings(environ)[:3]:
        print(line, file=stdout)
    return 0


def _install_slash_commands(stdout: TextIO, agent: str, environ: Mapping[str, str]) -> None:
    """Give the coding agent a /rag command that drives `rag config` from plain words."""
    home = Path(environ.get("HOME") or Path.home())
    targets = {"claude": home / ".claude" / "commands" / "rag.md", "codex": home / ".codex" / "prompts" / "rag.md"}
    target = targets[agent]
    target.parent.mkdir(parents=True, exist_ok=True)
    body = SLASH_COMMAND if agent == "claude" else SLASH_COMMAND.split("---\n", 2)[-1]
    target.write_text(body, encoding="utf-8")
    hint = "/rag <what you want>" if agent == "claude" else "/prompts:rag <what you want>"
    print(f"Installed the {hint} command ({target}).", file=stdout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Local hybrid search: exact words (BM25) always on, plus the embedding model you choose.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="Choose how meaning-based search should work")
    setup.add_argument("--provider", choices=user_settings.PROVIDERS, help="Skip the questions and set the provider directly")
    setup.add_argument("--model", default="", help="Embedding model name for the provider")
    setup.add_argument("--base-url", default="", dest="base_url", help="OpenAI-compatible endpoint URL")
    setup.add_argument("--api-key", default="", dest="api_key", help="API key for openai_compatible")
    setup.add_argument("--ollama-url", default="", dest="ollama_url", help="Ollama URL (default http://localhost:11434)")
    setup.add_argument("--no-verify", action="store_true", dest="no_verify", help="Save without embedding a test sentence")
    setup.add_argument("--auto", action="store_true", help="Pick the best option this machine already has, no questions")

    config = commands.add_parser("config", help="Show or change settings: top N, reranker, embedding model")
    config.add_argument("setting", nargs="?", choices=["show", "top", "reranker", "embedding"], default="show")
    config.add_argument("value", nargs="?", default="")
    config.add_argument("--provider", choices=user_settings.PROVIDERS, help="Where the embedding model runs (guessed when omitted)")
    config.add_argument("--no-verify", action="store_true", dest="no_verify")

    doctor = commands.add_parser("doctor", help="Check the local RAG, embedding and Codex setup")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    query = commands.add_parser("query", help="Query a local file or folder")
    query.add_argument("question")
    query.add_argument("--path", required=True)
    query.add_argument("--top", type=int, default=None, metavar="N", help="How many passages to return (also --top5, --top-3); default from `rag config top`")
    connect = commands.add_parser("connect", help="Register the MCP server with a coding agent")
    connect.add_argument("agent", choices=["codex", "claude"])
    connect.add_argument("--scope", choices=["user", "project"], default="user", help="Claude Code scope: user (all projects) or project (.mcp.json here)")
    commands.add_parser("connect-codex", help="Same as `rag connect codex`")
    serve = commands.add_parser("serve", help="Run the local Studio companion")
    serve.add_argument("--port", type=int, default=47831)
    studio = commands.add_parser("studio", help="Open Studio with the local companion")
    studio.add_argument("--port", type=int, default=47831)
    studio.add_argument("--no-open", action="store_true")
    studio.add_argument("--studio-url", default="https://rag.masihmoafi.com", dest="studio_url", help="Studio to open, e.g. http://localhost:3000 for a local checkout")
    commands.add_parser("mcp", help="Run the stdio MCP server")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    query_fn: Callable[[str, str], str] | None = None,
    embedding_probe: Callable[[Mapping[str, str]], int] | None = None,
    token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    server_factory: Callable[..., object] | None = None,
    open_url: Callable[[str], object] = webbrowser.open,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    environ = environ or os.environ
    raw = list(argv) if argv is not None else sys.argv[1:]
    if not raw:
        if getattr(stdin, "isatty", lambda: False)():
            return _menu(stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, embedding_probe=embedding_probe,
                         token_factory=token_factory, server_factory=server_factory, open_url=open_url)
        _parser().print_help(stdout)
        return 0
    args = _parser().parse_args(_normalize_argv(raw))

    if args.command == "setup":
        verify = None if args.no_verify else (embedding_probe or _probe_embedding)
        if args.auto:
            return user_settings.auto_setup(stdout=stdout, stderr=stderr, environ=environ, verify=verify)
        if args.provider is None:
            code = user_settings.run_setup(stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, verify=verify)
            if code == 0:
                _offer_agent_connection(stdin, stdout, stderr, environ)
            return code
        if args.provider == "openai_compatible" and not args.api_key:
            print("--api-key is required for openai_compatible.", file=stderr)
            return 2
        config = dict(user_settings.DEFAULTS)
        config.update(
            {
                "embed_provider": args.provider,
                "embed_model": args.model or user_settings.default_model(args.provider),
                "embed_base_url": args.base_url,
                "embed_api_key": args.api_key,
            }
        )
        if args.ollama_url:
            config["ollama_url"] = args.ollama_url
        return user_settings.finish_setup(config, stdout=stdout, stderr=stderr, environ=environ, verify=verify)

    if args.command == "config":
        verify = None if args.no_verify else (embedding_probe or _probe_embedding)
        return _config_command(args, stdout=stdout, stderr=stderr, environ=environ, verify=verify)

    if args.command == "doctor":
        from rag import detect

        embedding = user_settings.embedding_config(environ)
        detection = detect.detect_all(environ, embedding["ollama_url"])
        report = {
            "detected": detection,
            "python": {"available": True, "version": sys.version.split()[0]},
            "codex": _codex_status(environ),
            "embedding": {
                "provider": embedding["embed_provider"],
                "model": embedding["embed_model"],
                "configured": user_settings.has_embedding(embedding),
                "config_path": str(user_settings.config_path(environ)),
                "full_extra": _full_extra_installed(),
            },
        }
        if args.as_json:
            print(json.dumps(report), file=stdout)
        else:
            codex = report["codex"]
            status = "signed in" if codex["authenticated"] else "not signed in"
            print(
                f"Python {report['python']['version']} · Codex {status} · "
                f"Embedding: {user_settings.describe_embedding(embedding)}",
                file=stdout,
            )
            for line in detect.describe(detection):
                print(f"Detected: {line}", file=stdout)
            if not report["embedding"]["configured"]:
                print("Run `rag setup` (or `rag setup --auto`) to add meaning-based search.", file=stdout)
        return 0 if report["codex"]["authenticated"] else 1

    if args.command == "connect-codex" or (args.command == "connect" and args.agent == "codex"):
        return _connect_codex(environ, stdout, stderr)
    if args.command == "connect":
        return _connect_claude(environ, stdout, stderr, scope=args.scope)

    if args.command in {"serve", "studio"}:
        if server_factory is None:
            from rag.companion import create_companion_server

            server_factory = create_companion_server
        token = token_factory()
        try:
            server = server_factory(token=token, port=args.port)
        except OSError as error:
            print(f"Could not start the local companion: {error}", file=stderr)
            return 1
        url = f"{args.studio_url.rstrip('/')}/studio#local={token}"
        print(f"Local companion: http://127.0.0.1:{server.server_port}", file=stdout)
        print(f"Embedding: {user_settings.describe_embedding(user_settings.embedding_config(environ))}", file=stdout)
        print(f"Studio: {url}", file=stdout)
        if args.command == "studio" and not args.no_open:
            open_url(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0

    if args.command == "mcp":
        from server import main as mcp_main

        mcp_main()
        return 0

    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"Path does not exist: {path}", file=stderr)
        return 2
    if query_fn is None:
        if _should_offer_setup(environ, stdin):
            code = user_settings.run_setup(
                stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, verify=embedding_probe or _probe_embedding
            )
            if code != 0:
                return code
        from server import query_knowledge_base

        query_fn = query_knowledge_base
    if args.top is not None:
        # The pipeline reads these when it starts: N final passages, at least N candidates per method.
        os.environ["RAG_MCP_RERANK_TOP_K"] = str(max(1, args.top))
        os.environ["RAG_MCP_TOP_K"] = str(max(args.top, int(os.environ.get("RAG_MCP_TOP_K") or 5)))
    print(query_fn(args.question, str(path)), file=stdout)
    if not user_settings.has_embedding(user_settings.embedding_config(environ)):
        print(user_settings.BM25_ONLY_NOTICE, file=stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
