"""`rag key`: show, set or remove the saved OpenRouter key without ever printing it."""

from __future__ import annotations

import getpass
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import TextIO
from urllib.parse import urlsplit

from rag import settings as user_settings

OPENROUTER_URL = "https://openrouter.ai/api/v1"


def mask(key: str) -> str:
    return f"…{key[-4:]}" if len(key) >= 8 else "…"


def status_lines(environ: Mapping[str, str]) -> list[str]:
    data = user_settings.load_settings(environ)
    saved = str(data.get("openrouter_api_key") or "").strip()
    env_key = environ.get("OPENROUTER_API_KEY", "").strip()
    embedding = user_settings.embedding_config(environ)
    legacy = (embedding['embed_api_key'] if urlsplit(embedding['embed_base_url']).hostname == 'openrouter.ai' else '')
    lines = [
        f"Saved key: {mask(saved) if saved else 'none'} ({user_settings.config_path(environ)})",
        f"OPENROUTER_API_KEY in environment: {mask(env_key) if env_key else 'none'}",
    ]
    in_use = saved or legacy or env_key
    lines.append(f"In use: {mask(in_use) if in_use else 'none; run `rag key set`'}")
    return lines


def check_openrouter(key: str) -> tuple[bool, str]:
    """Ask OpenRouter whether the key is valid (GET /key costs nothing)."""
    request = urllib.request.Request(
        f"{OPENROUTER_URL}/key", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8")).get("data") or {}
    except urllib.error.HTTPError as error:
        return False, f"OpenRouter refused it (HTTP {error.code})"
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        return False, f"could not reach OpenRouter ({error.__class__.__name__})"
    left = data.get("limit_remaining")
    return True, "OpenRouter accepts it" + (
        f"; limit left {left}" if left is not None else ""
    )


def read_key(stdin: TextIO, stdout: TextIO) -> str:
    if stdin.isatty():
        return getpass.getpass(
            "Paste the OpenRouter key (hidden), then Enter: ", stream=stdout
        )
    return stdin.readline()


def key_command(
    action: str,
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    environ: Mapping[str, str],
    check: Callable[[str], tuple[bool, str]] = check_openrouter,
) -> int:
    if action == "show":
        for line in status_lines(environ):
            print(line, file=stdout)
        return 0
    if action == "remove":
        user_settings.update_settings({"openrouter_api_key": ""}, environ)
        print("Saved key removed.", file=stdout)
        for line in status_lines(environ)[1:]:
            print(line, file=stdout)
        return 0
    key = read_key(stdin, stdout).strip()
    if not key or any(c.isspace() or ord(c) < 32 for c in key):
        print("No key saved: the input was empty or had spaces.", file=stderr)
        return 2
    ok, message = check(key)
    print(("Works: " if ok else "Warning: ") + message + ".",
          file=stdout if ok else stderr)
    if not ok:
        return 1
    user_settings.update_settings({"openrouter_api_key": key}, environ)
    print(f"Saved {mask(key)}.", file=stdout)
    return 0


def main_env() -> Mapping[str, str]:
    return os.environ
