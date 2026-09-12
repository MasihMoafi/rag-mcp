from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from rag import settings as user_settings
from rag.cli_agents import (
    AGENT_LABELS,
    available_agents as _available_agents,
    connect_agent as _connect_agent,
    connect_agy as _connect_agy,
    connect_claude as _connect_claude,
    connect_codex as _connect_codex,
    connect_copilot as _connect_copilot,
    connect_gemini as _connect_gemini,
    install_slash_commands as _install_slash_commands,
    offer_agent_connection as _offer_agent_connection,
    server_command as _server_command,
    show_generic_mcp_config as _show_generic_mcp_config,
)
from rag.cli_menu import MENU, menu as _menu


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
    elif setting == "contextual":
        mode = value.lower()
        if mode not in {"on", "off"}:
            print("Give on or off for contextual retrieval.", file=stderr)
            return 2
        user_settings.update_settings({"contextual_retrieval": mode}, environ)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Local hybrid search: exact words (BM25) always on, plus the embedding source you choose. Studio or an MCP app supplies answers.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="Choose where retrieval embeddings run, then choose a model")
    setup.add_argument("--provider", choices=user_settings.PROVIDERS, help="Skip the questions and set the provider directly")
    setup.add_argument("--model", default="", help="Embedding model name for the provider")
    setup.add_argument("--base-url", default="", dest="base_url", help="OpenAI-compatible endpoint URL")
    setup.add_argument("--api-key", default="", dest="api_key", help="API key for openai_compatible")
    setup.add_argument("--ollama-url", default="", dest="ollama_url", help="Ollama URL (default http://localhost:11434)")
    setup.add_argument("--no-verify", action="store_true", dest="no_verify", help="Save without embedding a test sentence")
    setup.add_argument("--auto", action="store_true", help="Pick the best option this machine already has, no questions")
    commands.add_parser("menu", help="open the interactive human terminal UI")

    config = commands.add_parser("config", help="Show or change settings: top N, reranker, contextual retrieval, embedding model")
    config.add_argument("setting", nargs="?", choices=["show", "top", "reranker", "contextual", "embedding"], default="show")
    config.add_argument("value", nargs="?", default="")
    config.add_argument("--provider", choices=user_settings.PROVIDERS, help="Where the embedding model runs (guessed when omitted)")
    config.add_argument("--no-verify", action="store_true", dest="no_verify")

    doctor = commands.add_parser("doctor", help="Check local RAG, embedding and MCP-app setup")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    query = commands.add_parser("query", help="Query a local file or folder")
    query.add_argument("question")
    query.add_argument("--path", required=True)
    query.add_argument("--top", type=int, default=None, metavar="N", help="How many passages to return (also --top5, --top-3); default from `rag config top`")
    query.add_argument(
        "--format",
        dest="output_format",
        choices=("auto", "text", "json"),
        default="auto",
        help="output format (default: auto-detect terminal)",
    )
    connect = commands.add_parser("connect", help="Register the MCP server with an installed MCP app")
    connect.add_argument("agent", choices=["codex", "claude", "copilot", "gemini", "agy"])
    connect.add_argument("--scope", choices=["user", "project"], default="user", help="Configuration scope where the app supports it")
    connect.add_argument(
        "--install-prompt",
        action="store_true",
        help="also install the optional slash prompt for supported agents",
    )
    connect_codex = commands.add_parser("connect-codex", help="Same as `rag connect codex`")
    connect_codex.add_argument("--install-prompt", action="store_true")
    serve = commands.add_parser("serve", help="Run the local Studio companion")
    serve.add_argument("--port", type=int, default=47831)
    serve.add_argument("--studio-url", default="https://rag.masihmoafi.com", dest="studio_url")
    studio = commands.add_parser("studio", help="Open Studio with the local companion")
    studio.add_argument("--port", type=int, default=47831)
    studio.add_argument("--no-open", action="store_true")
    studio.add_argument("--studio-url", default="https://rag.masihmoafi.com", dest="studio_url", help="Studio to open, e.g. http://localhost:3000 for a local checkout")
    commands.add_parser("mcp", help="Run the stdio MCP server")
    return parser


def _main(
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
        _parser().print_help(stdout)
        return 0
    args = _parser().parse_args(_normalize_argv(raw))

    if args.command == "menu":
        return _menu(stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, embedding_probe=embedding_probe,
                     token_factory=token_factory, server_factory=server_factory, open_url=open_url)

    if args.command == "setup":
        verify = None if args.no_verify else (embedding_probe or _probe_embedding)
        if args.auto:
            return user_settings.auto_setup(stdout=stdout, stderr=stderr, environ=environ, verify=verify)
        if args.provider is None:
            code = user_settings.run_setup(stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, verify=verify)
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
            "mcp_apps": {key: bool(shutil.which(executable, path=environ.get("PATH"))) for key, _, executable in AGENT_LABELS},
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
            apps = [label for key, label, _ in AGENT_LABELS if report["mcp_apps"][key]]
            if apps:
                print("MCP apps available: " + ", ".join(apps), file=stdout)
            if not report["embedding"]["configured"]:
                print("Run `rag setup` (or `rag setup --auto`) to add meaning-based search.", file=stdout)
        # MCP hosts are optional integrations, not prerequisites for local RAG.
        return 0

    if args.command == "connect-codex":
        return _connect_codex(environ, stdout, stderr, install_prompt=args.install_prompt)
    if args.command == "connect":
        return _connect_agent(
            args.agent,
            environ,
            stdout,
            stderr,
            scope=args.scope,
            install_prompt=args.install_prompt,
        )

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
        if server.server_port != 47831:
            url += f"&port={server.server_port}"
        print("Starting the local companion for Studio.", file=stdout)
        print("Keep this terminal open while Studio uses your local documents and models.", file=stdout)
        print(f"Local companion: http://127.0.0.1:{server.server_port}", file=stdout)
        print(f"Embedding: {user_settings.describe_embedding(user_settings.embedding_config(environ))}", file=stdout)
        if args.command == "studio":
            print(f"Opening the live Studio in your browser: {url}", file=stdout)
            print("Press Ctrl-C in this terminal to stop the companion.", file=stdout)
        else:
            print(f"Studio pairing URL: {url}", file=stdout)
            print("Open that URL in a browser. Press Ctrl-C here to stop the companion.", file=stdout)
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
    from rag.prompts import interactive

    human_output = args.output_format == "text" or (
        args.output_format == "auto" and interactive(stdin, stdout)
    )
    previous = {name: os.environ.get(name) for name in ("RAG_MCP_RERANK_TOP_K", "RAG_MCP_TOP_K")}
    try:
        if args.top is not None:
            os.environ["RAG_MCP_RERANK_TOP_K"] = str(max(1, args.top))
            os.environ["RAG_MCP_TOP_K"] = str(max(args.top, int(os.environ.get("RAG_MCP_TOP_K") or 5)))
        if human_output:
            print(f"Searching {path.name}…", file=stdout, flush=True)
            # Native database diagnostics also write directly to fd 2.
            with tempfile.TemporaryFile(mode="w+") as diagnostics:
                original_stderr = os.dup(2)
                try:
                    os.dup2(diagnostics.fileno(), 2)
                    with contextlib.redirect_stderr(diagnostics):
                        result = query_fn(args.question, str(path))
                finally:
                    os.dup2(original_stderr, 2)
                    os.close(original_stderr)
        else:
            result = query_fn(args.question, str(path))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        payload = None
    display = result
    if human_output and isinstance(payload, dict):
        display = payload.get("error") or payload.get("result", result)
        if isinstance(display, list):
            passages = []
            for index, passage in enumerate(display, 1):
                if isinstance(passage, dict):
                    source = passage.get("source") or passage.get("file_path") or path.name
                    heading = f" · {passage['heading']}" if passage.get("heading") else ""
                    passages.append(f"\n{index}. {source}{heading}\n{passage.get('content', '')}")
                else:
                    passages.append(str(passage))
            display = "\n".join(passages) or "No matching passages found."
    print(display, file=stdout)
    if isinstance(payload, dict) and "error" in payload:
        return 1
    if not user_settings.has_embedding(user_settings.embedding_config(environ)):
        print(user_settings.BM25_ONLY_NOTICE, file=stderr)
    return 0


def main(argv: Sequence[str] | None = None, **kwargs: Any) -> int:
    from rag.prompts import Cancelled

    try:
        return _main(argv, **kwargs)
    except Cancelled:
        return 130
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=kwargs.get("stdout") or sys.stdout)
        return 130
    except ValueError as error:
        print(str(error), file=kwargs.get("stderr") or sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
