"""Tiny stand-in for an OpenAI-compatible embeddings API and for Ollama's /api/embed.

Vectors are deterministic word hashes (8 dimensions) so texts that share words
land close together; enough to prove that the CLI sends requests to the
configured URL with the configured key and model, and ranks with the answer.
Every request is appended to the log file as one JSON line.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

DIMENSIONS = 8


def embed(text: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for word in text.lower().split():
        digest = hashlib.md5(word.encode("utf-8")).digest()
        vector[digest[0] % DIMENSIONS] += 1.0
        vector[digest[1] % DIMENSIONS] += 0.5
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def main(port: int, log_path: str) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _log(self, body: dict) -> None:
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(json.dumps({
                    "path": self.path,
                    "authorization": self.headers.get("Authorization", ""),
                    "model": body.get("model"),
                    "count": len(body.get("input", [])) if isinstance(body.get("input"), list) else 1,
                }) + "\n")

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._send(200, {"data": [{"id": "mock-embed-interactive"}]})
                return
            if self.path == "/api/ps":
                self._send(200, {"models": []})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            self._log(body)
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            vectors = [embed(text) for text in inputs]
            if self.path.rstrip("/").endswith("/embeddings"):
                self._send(200, {
                    "object": "list",
                    "model": body.get("model"),
                    "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                })
            elif self.path == "/api/embed":
                self._send(200, {"model": body.get("model"), "embeddings": vectors})
            else:
                self._send(404, {"error": "not found"})

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
