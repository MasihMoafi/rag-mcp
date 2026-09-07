from __future__ import annotations

import argparse
import json
import os
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

    doctor = commands.add_parser("doctor", help="Check the local RAG, embedding and Codex setup")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    query = commands.add_parser("query", help="Query a local file or folder")
    query.add_argument("question")
    query.add_argument("--path", required=True)
    commands.add_parser("connect-codex", help="Register this RAG server with Codex")
    serve = commands.add_parser("serve", help="Run the local Studio companion")
    serve.add_argument("--port", type=int, default=47831)
    studio = commands.add_parser("studio", help="Open Studio with the local companion")
    studio.add_argument("--port", type=int, default=47831)
    studio.add_argument("--no-open", action="store_true")
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
    args = _parser().parse_args(argv)

    if args.command == "setup":
        verify = None if args.no_verify else (embedding_probe or _probe_embedding)
        if args.provider is None:
            return user_settings.run_setup(stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, verify=verify)
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

    if args.command == "doctor":
        embedding = user_settings.embedding_config(environ)
        report = {
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
            if not report["embedding"]["configured"]:
                print("Run `rag setup` to add meaning-based search.", file=stdout)
        return 0 if report["codex"]["authenticated"] else 1

    if args.command == "connect-codex":
        executable = shutil.which("codex", path=environ.get("PATH"))
        if executable is None:
            print("Codex is not installed or is not on PATH.", file=stderr)
            return 1
        import server

        server_path = str(Path(server.__file__).resolve())
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
                transport.get("command") == sys.executable
                and transport.get("args") == [server_path]
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
            [executable, "mcp", "add", "rag", "--", sys.executable, server_path],
            capture_output=True,
            check=False,
            text=True,
            env=dict(environ),
        )
        if added.returncode != 0:
            print((added.stderr or added.stdout).strip(), file=stderr)
            return added.returncode
        print("Connected Codex to the local RAG server.", file=stdout)
        return 0

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
        url = f"https://rag.masihmoafi.com/studio#local={token}"
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
    print(query_fn(args.question, str(path)), file=stdout)
    if not user_settings.has_embedding(user_settings.embedding_config(environ)):
        print(user_settings.BM25_ONLY_NOTICE, file=stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
