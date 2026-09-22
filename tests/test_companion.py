import json
from types import SimpleNamespace
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rag.companion import GenerationResult, create_companion_server, run_codex_generation


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


def test_codex_generation_reads_reported_usage(monkeypatch):
    monkeypatch.setattr("rag.companion.shutil.which", lambda _name: "/usr/bin/codex")

    def run(command, **_kwargs):
        assert "--json" in command
        output = command[command.index("--output-last-message") + 1]
        with open(output, "w") as stream:
            stream.write("Grounded answer [1]\n")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"turn.completed","usage":{"input_tokens":41,"output_tokens":9}}\n',
        )

    monkeypatch.setattr("rag.companion.subprocess.run", run)
    result = run_codex_generation({"question": "What?", "history": [], "chunks": []})
    assert result == GenerationResult("Grounded answer [1]", 41, 9)


def test_companion_rejects_unpaired_requests_and_streams_grounded_answers(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.delenv("RAG_MCP_EMBED_PROVIDER", raising=False)
    calls = []
    server = create_companion_server(
        token="secret",
        run_codex=lambda payload: calls.append(payload) or GenerationResult(
            answer="Cedar Lantern [1]", input_tokens=123, output_tokens=17
        ),
        codex_ready=lambda: True,
        embedder_factory=lambda: None,
        metrics_path=tmp_path / "metrics.json",
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
        with request(f"{base}/v1/dashboard") as response:
            dashboard = json.load(response)
        assert dashboard["operations"]["generation"]["requests"] == 1
        assert dashboard["operations"]["generation"]["successes"] == 1
        assert dashboard["operations"]["generation"]["failures"] == 0
        assert dashboard["operations"]["generation"]["average_latency_ms"] >= 0
        assert dashboard["models"]["generation"] == "gpt-5.6-luna"
        assert dashboard["usage"] == {"input_tokens": 123, "output_tokens": 17, "cost_usd": None}
        assert dashboard["gates"]["native_operation_metrics"] is True
        assert dashboard["gates"]["durable_operation_metrics"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    restarted = create_companion_server(
        token="secret",
        codex_ready=lambda: True,
        embedder_factory=lambda: None,
        metrics_path=tmp_path / "metrics.json",
    )
    thread = Thread(target=restarted.serve_forever, daemon=True)
    thread.start()
    try:
        with request(f"http://127.0.0.1:{restarted.server_port}/v1/dashboard") as response:
            dashboard = json.load(response)
        assert dashboard["usage"]["input_tokens"] == 123
        assert dashboard["usage"]["output_tokens"] == 17
    finally:
        restarted.shutdown()
        restarted.server_close()
        thread.join(timeout=2)


def test_companion_embeds_with_the_configured_model(tmp_path):
    built = []

    def factory():
        built.append(True)
        return FakeEmbedder()

    metrics_path = tmp_path / "metrics.json"
    server = create_companion_server(token="secret", codex_ready=lambda: False, embedder_factory=factory, metrics_path=metrics_path)
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

        with request(f"{base}/v1/dashboard") as response:
            dashboard = json.load(response)
        assert dashboard["operations"]["embedding"]["requests"] == 3
        assert dashboard["operations"]["embedding"]["successes"] == 3
        assert dashboard["models"]["embedding"] == "fake-embedder"

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

    restarted = create_companion_server(token="secret", codex_ready=lambda: False, embedder_factory=factory, metrics_path=metrics_path)
    thread = Thread(target=restarted.serve_forever, daemon=True)
    thread.start()
    try:
        with request(f"http://127.0.0.1:{restarted.server_port}/v1/dashboard") as response:
            dashboard = json.load(response)
        assert dashboard["operations"]["embedding"]["requests"] == 3
        assert dashboard["operations"]["embedding"]["successes"] == 3
    finally:
        restarted.shutdown()
        restarted.server_close()
        thread.join(timeout=2)
