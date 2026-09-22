"""Fail-closed decoding for the public answer-only boundary."""

from __future__ import annotations

import json


INVALID_PUBLIC_ANSWER = "RAG answer pipeline returned an invalid public result."


def decode_public_answer(response: object) -> tuple[str, bool]:
    """Return answer/error text and reject every other response shape."""
    if not isinstance(response, str):
        return INVALID_PUBLIC_ANSWER, True
    try:
        payload = json.loads(response)
    except (TypeError, ValueError):
        return INVALID_PUBLIC_ANSWER, True
    if not isinstance(payload, dict):
        return INVALID_PUBLIC_ANSWER, True
    if set(payload) == {"result"} and isinstance(payload["result"], str):
        answer = payload["result"].strip()
        return (answer, False) if answer else (INVALID_PUBLIC_ANSWER, True)
    if set(payload) == {"error"} and isinstance(payload["error"], str):
        error = payload["error"].strip()
        return (error, True) if error else (INVALID_PUBLIC_ANSWER, True)
    return INVALID_PUBLIC_ANSWER, True
