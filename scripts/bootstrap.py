#!/usr/bin/env python3
"""One-command bootstrap for fresh rag-mcp environments."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


MIN_PYTHON = (3, 10)


def _venv_python(repo_root: Path) -> Path:
    if os.name == "nt":
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python"


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        required = ".".join(str(part) for part in MIN_PYTHON)
        current = ".".join(str(part) for part in sys.version_info[:3])
        raise RuntimeError(
            f"Python {required}+ is required (found {current}). "
            "Install a newer Python and retry."
        )


def _check_uv() -> None:
    if shutil.which("uv") is None:
        raise RuntimeError(
            "Missing required tool: uv.\n"
            "Install guide: https://docs.astral.sh/uv/getting-started/installation/"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap rag-mcp on a fresh system and verify core tests."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate prerequisites; do not install dependencies or run tests.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    try:
        _check_python()
        _check_uv()
        if args.check_only:
            print("OK: prerequisites satisfied (Python and uv).")
            return 0

        print("Installing dependencies with uv...")
        _run(["uv", "sync", "--group", "dev"], cwd=repo_root)

        venv_python = _venv_python(repo_root)
        if not venv_python.exists():
            raise RuntimeError(f"Expected virtualenv interpreter not found: {venv_python}")

        print("Running test suite...")
        _run([str(venv_python), "-m", "pytest", "tests/", "-v"], cwd=repo_root)

        print("Bootstrap complete: rag-mcp is ready.")
        return 0
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
