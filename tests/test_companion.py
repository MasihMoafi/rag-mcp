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


class FakeEmbedder:
    model_name = "fake-embedder"

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 2.0]


def test_companion_rejects_unpaired_requests_and_streams_grounded_answers(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.delenv("RAG_MCP_EMBED_PROVIDER", raising=False)
    calls = []
    server = create_companion_server(
        token="secret",
        run_codex=lambda payload: calls.append(payload) or "Cedar Lantern [1]",
        codex_ready=lambda: True,
        embedder_factory=lambda: None,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with request(f"{base}/health") as response:
            assert json.load(response) == {
                "ready": True,
                "model": "gpt-5.6-luna",
                "embedding": {"provider": "none", "model": ""},
            }
            assert response.headers["Access-Control-Allow-Origin"] == "https://rag.masihmoafi.com"

        try:
            request(f"{base}/embed", body={"texts": ["hello"]})
        except HTTPError as error:
            assert error.code == 409
            assert "rag setup" in json.load(error)["error"]
        else:
            raise AssertionError("embedding without a model must be refused")

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
            {"type": "meta", "mode": "provider", "provider": "Codex · Local", "model": "gpt-5.6-luna"},
            {"type": "delta", "text": "Cedar Lantern [1]"},
            {"type": "done"},
        ]
        assert calls == [payload]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_companion_embeds_with_the_configured_model():
    built = []

    def factory():
        built.append(True)
        return FakeEmbedder()

    server = create_companion_server(token="secret", codex_ready=lambda: False, embedder_factory=factory)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with request(f"{base}/health") as response:
            assert response.status == 200
            assert json.load(response)["ready"] is False
        with request(f"{base}/embed", body={"texts": ["ab", "abcd"]}) as response:
            assert json.load(response) == {
                "vectors": [[2.0, 1.0], [4.0, 1.0]],
                "model": "fake-embedder",
                "dimensions": 2,
            }
        with request(f"{base}/embed", body={"texts": ["x"]}) as response:
            assert json.load(response)["dimensions"] == 2
        assert built == [True], "the embedder is built once and reused"

        with request(f"{base}/embed", body={"texts": ["ab"], "kind": "query"}) as response:
            assert json.load(response)["vectors"] == [[2.0, 2.0]]

        for bad in ({"texts": []}, {"texts": "not a list"}, {"texts": ["", "ok"]}, {"texts": ["ok"], "kind": "invalid"}, {}):
            try:
                request(f"{base}/embed", body=bad)
            except HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError(f"invalid embed payload accepted: {bad}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
