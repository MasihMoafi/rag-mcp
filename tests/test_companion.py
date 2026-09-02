import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rag.companion import create_companion_server


def request(url, token="secret", origin="https://rag.masihmoafi.com", body=None):
    headers = {"Authorization": f"Bearer {token}", "Origin": origin}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    return urlopen(Request(url, data=data, headers=headers), timeout=2)


def test_companion_rejects_unpaired_requests_and_streams_grounded_answers():
    calls = []
    server = create_companion_server(
        token="secret",
        run_codex=lambda payload: calls.append(payload) or "Cedar Lantern [1]",
        codex_ready=lambda: True,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with request(f"{base}/health") as response:
            assert json.load(response) == {"ready": True, "model": "gpt-5.6-luna"}
            assert response.headers["Access-Control-Allow-Origin"] == "https://rag.masihmoafi.com"

        for token, origin in (("wrong", "https://rag.masihmoafi.com"), ("secret", "https://evil.example")):
            try:
                request(f"{base}/generate", token=token, origin=origin, body={"question": "Fact?", "chunks": []})
            except HTTPError as error:
                assert error.code == 403
            else:
                raise AssertionError("unpaired companion request was accepted")

        payload = {
            "question": "What is the codename?",
            "history": [],
            "chunks": [{
                "id": "1",
                "documentId": "facts",
                "documentTitle": "facts.md",
                "content": "The codename is Cedar Lantern.",
                "rank": 1,
            }],
        }
        with request(f"{base}/generate", body=payload) as response:
            events = [json.loads(line) for line in response.read().decode().splitlines()]
        assert events == [
            {"type": "meta", "mode": "provider", "provider": "codex", "model": "gpt-5.6-luna"},
            {"type": "delta", "text": "Cedar Lantern [1]"},
            {"type": "done"},
        ]
        assert calls == [payload]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
