from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TextIO


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Check the local RAG and Codex setup")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    query = commands.add_parser("query", help="Query a local file or folder")
    query.add_argument("question")
    query.add_argument("--path", required=True)
    commands.add_parser("connect-codex", help="Register this RAG server with Codex")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    query_fn: Callable[[str, str], str] | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    environ = environ or os.environ
    args = _parser().parse_args(argv)

    if args.command == "doctor":
        report = {
            "python": {"available": True, "version": sys.version.split()[0]},
            "codex": _codex_status(environ),
        }
        if args.as_json:
            print(json.dumps(report), file=stdout)
        else:
            codex = report["codex"]
            status = "signed in" if codex["authenticated"] else "not signed in"
            print(f"Python {report['python']['version']} · Codex {status}", file=stdout)
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

    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"Path does not exist: {path}", file=stderr)
        return 2
    if query_fn is None:
        from server import query_knowledge_base

        query_fn = query_knowledge_base
    print(query_fn(args.question, str(path)), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
