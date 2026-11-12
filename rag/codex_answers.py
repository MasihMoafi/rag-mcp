"""Written answers through the user's Codex sign-in (a ChatGPT plan), with no API key.

rag's answer request (prompt, evidence and, for passage selection, its JSON schema) is sent as one
read-only `codex exec` turn in an empty folder; the reply comes back in the chat-completions shape
the OpenRouter path returns, so the rest of the answer step is unchanged.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PREFERRED_MODEL = (
    "gpt-6.1-sol"  # the answer model behind the EnterpriseRAG-Bench results
)
EFFORT = "low"  # one grounded answer from supplied passages; low keeps it to seconds


@functools.lru_cache(maxsize=1)
def signed_in() -> bool:
    """Whether Codex is installed and signed in (checked once per process)."""
    executable = shutil.which("codex")
    if executable is None:
        return False
    try:
        completed = subprocess.run(
            [executable, "login", "status"],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return (
        completed.returncode == 0
        and "Logged in" in f"{completed.stdout}\n{completed.stderr}"
    )


def model(configured: str = "") -> str:
    """The configured model, else the preferred one when the account lists it, else Codex's first."""
    if configured.strip():
        return configured.strip()
    from rag.companion import codex_cached_models

    listed = [entry["id"] for entry in codex_cached_models()]
    return PREFERRED_MODEL if PREFERRED_MODEL in listed or not listed else listed[0]


def complete(request: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one answer request through Codex; return it as a chat-completions response."""
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("Codex is not installed or is not on PATH.")
    prompt = "\n\n".join(str(message["content"]) for message in request["messages"])
    prompt += "\n\nDo not use tools. Return only the answer."
    schema = (request.get("response_format") or {}).get("json_schema", {}).get("schema")
    with tempfile.TemporaryDirectory(prefix="rag-codex-answer-") as directory:
        folder = Path(directory)
        command = [
            executable,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model(str(config.get("codex_model") or "")),
            "-c",
            f"model_reasoning_effort={config.get('codex_effort') or EFFORT}",
            "--cd",
            directory,
            "--output-last-message",
            str(folder / "answer.txt"),
        ]
        if schema:
            (folder / "schema.json").write_text(json.dumps(schema))
            command += ["--output-schema", str(folder / "schema.json")]
        completed = subprocess.run(
            [*command, prompt.replace("\0", "")],
            capture_output=True,
            check=False,
            text=True,
            timeout=float(config.get("answer_timeout", 180)),
            env=dict(os.environ),
        )
        answer_file = folder / "answer.txt"
        if completed.returncode != 0 or not answer_file.exists():
            raise RuntimeError("Codex could not generate an answer.")
        text = answer_file.read_text().strip()
    if not text:
        raise RuntimeError("Codex returned an empty answer.")
    return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
