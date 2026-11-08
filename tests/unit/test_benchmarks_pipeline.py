"""Observable benchmark contracts; all hosted calls and downloads are offline fakes."""

import json
import math
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from evals.benchmarks.answers import accounting, run_answers, usage_cost
from evals.benchmarks.build import build_answer_task, build_notebook
from evals.benchmarks.common import (
    Ledger,
    file_hash,
    read_ledger,
    retrieval_metrics,
    write_json,
)
from evals.benchmarks.datasets import normalize, prepare, wikipedia_text
from evals.benchmarks.retrieve import (
    RERANKER,
    RagRetriever,
    check_bundle,
    run_retrieval,
)

FIXTURES = Path(__file__).resolve().parents[2] / "evals/benchmarks/fixtures"


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    shutil.copytree(FIXTURES, root)
    write_json(
        root / "source.json", {"dataset": "enterprise", "revision": "offline-fixture"}
    )
    schema = {
        "version": 1,
        "dataset": "enterprise",
        "readme_sha256": file_hash(root / "README.md"),
        "questions": {
            "file": "questions.jsonl",
            "id": "key",
            "text": "prompt",
            "answer": "answer",
            "gold": "evidence[].key",
            "category": "group",
        },
        "documents": {
            "format": "records",
            "globs": ["documents.jsonl"],
            "id": "key",
            "text": ["body"],
        },
    }
    path = tmp_path / "schema.json"
    write_json(path, schema)
    return root, path, schema


@pytest.fixture
def bundle(source, tmp_path):
    root, schema_path, _ = source
    output = tmp_path / "bundle"
    normalize(root, schema_path, output)
    return output


def test_known_document_metrics_and_chunk_duplicates():
    ranked = ["noise", "a", "a", "noise2", "noise3", "noise4", "b"]
    scores = retrieval_metrics(ranked, ["a", "b"])
    assert scores == pytest.approx(
        {
            "recall@5": 0.5,
            "recall@10": 1.0,
            "MRR": 0.5,
            "nDCG@10": (1 / math.log2(3) + 1 / math.log2(7)) / (1 + 1 / math.log2(3)),
        }
    )
    assert retrieval_metrics([], ["a"]) == {
        "recall@5": 0,
        "recall@10": 0,
        "nDCG@10": 0,
        "MRR": 0,
    }
    assert retrieval_metrics(["a"], ["a"])["nDCG@10"] == 1
    assert retrieval_metrics(["x"] * 12 + ["a"], ["a"])["recall@10"] == 1
    with pytest.raises(ValueError, match="nonempty gold"):
        retrieval_metrics(["a"], [])


@pytest.mark.parametrize(
    "break_schema", ["field", "type", "readme", "duplicates", "no_coverage"]
)
def test_schema_rejects_before_corpus_output(source, tmp_path, break_schema):
    root, path, schema = source
    if break_schema == "field":
        schema["questions"]["text"] = "guessed_field"
    elif break_schema == "type":
        schema["questions"]["gold"] = "answer"
    elif break_schema == "readme":
        schema["readme_sha256"] = "stale"
    elif break_schema == "duplicates":
        with (root / "documents.jsonl").open("a") as stream:
            stream.write((root / "documents.jsonl").read_text().splitlines()[0] + "\n")
    else:
        schema["documents"]["id"] = "body"
    write_json(path, schema)
    output = tmp_path / "bad-bundle"
    with pytest.raises(ValueError, match="Schema|schema"):
        normalize(root, path, output)
    assert not output.exists()


def test_slice_filter_reports_qualification_and_preserves_all_corpus_docs(
    source, tmp_path
):
    root, path, _ = source
    rows = [
        json.loads(line) for line in (root / "questions.jsonl").read_text().splitlines()
    ]
    rows[0]["evidence"].append({"key": "unloaded"})
    rows[1]["evidence"] = []
    (root / "questions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    output = tmp_path / "filtered"
    manifest = normalize(root, path, output)
    assert (
        manifest["questions_total"],
        manifest["questions_qualified"],
        manifest["questions_selected"],
        manifest["questions_without_gold"],
    ) == (3, 1, 1, 1)
    assert check_bundle(output)[2][0]["id"] == "q3"
    assert len(list((output / "corpus").glob("*.txt"))) == 3
    next((output / "corpus").glob("*.txt")).write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        check_bundle(output)


def test_runtime_repository_snapshot_slice_download_and_schema_binding(
    tmp_path, monkeypatch
):
    def metadata(url):
        if "/commits/" in url:
            return {"sha": "pinned-commit"}
        assert url.endswith("/releases/latest")
        return {
            "tag_name": "v1.0.0",
            "assets": [
                {"name": name, "browser_download_url": f"https://example.test/{name}"}
                for name in ("questions.jsonl", "github_documents.zip", "other.zip")
            ],
        }

    downloaded = []

    def fetch(url, target):
        downloaded.append(url)
        target = Path(target)
        if target.suffix == ".zip":
            import zipfile

            rows = (FIXTURES / "documents.jsonl").read_text().splitlines()[:2]
            with zipfile.ZipFile(target, "w") as archive:
                archive.writestr("docs.jsonl", "\n".join(rows) + "\n")
        else:
            shutil.copyfile(FIXTURES / target.name, target)

    monkeypatch.setattr("evals.benchmarks.datasets.get_json", metadata)
    monkeypatch.setattr("evals.benchmarks.datasets.download", fetch)
    root = tmp_path / "downloaded"
    metadata = prepare("enterprise", root, ["github"])
    assert metadata["revision"] == "pinned-commit"
    assert len(downloaded) == 3 and not any("other.zip" in url for url in downloaded)
    inspection = json.loads((root / "inspection.json").read_text())
    assert (
        next(r for r in inspection if r["path"] == "questions.jsonl")["sample_shape"][
            "prompt"
        ]
        == "str"
    )
    schema = {
        "version": 1,
        "dataset": "enterprise",
        "readme_sha256": metadata["readme_sha256"],
        "questions": {
            "file": "questions.jsonl",
            "id": "key",
            "text": "prompt",
            "answer": "answer",
            "gold": "evidence[].key",
        },
        "documents": {
            "format": "records",
            "globs": ["extracted/**/*.jsonl"],
            "id": "key",
            "text": ["body"],
        },
    }
    schema_path = tmp_path / "download-schema.json"
    write_json(schema_path, schema)
    result = normalize(root, schema_path, tmp_path / "download-bundle")
    assert result["questions_qualified"] == 2 and result["documents"] == 2


def retrieval_fixture(bundle, tmp_path):
    dataset, mapping, questions = check_bundle(bundle)
    manifest = {
        "dataset": dataset,
        "rag_mcp_commit": "offline",
        "models": {},
        "questions": 3,
    }
    output = tmp_path / "retrieval"
    return questions, output, manifest


def test_retrieval_graceful_stop_resume_keeps_partial_and_skips_done(bundle, tmp_path):
    questions, output, manifest = retrieval_fixture(bundle, tmp_path)
    calls = []

    class Engine:
        indexed_count = 3

        def __call__(self, q):
            calls.append(q["id"])
            if q["id"] == "q2" and len(calls) == 2:
                raise KeyboardInterrupt("pilot interrupted")
            return {"ranked_doc_ids": q["gold_doc_ids"], "scores": [1], "passages": []}

        def close(self):
            pass

    result = run_retrieval(questions, ["bm25"], lambda arm: Engine(), output, manifest)
    assert result["arms"]["bm25"]["completed"] == 1
    assert "pilot interrupted" in result["stopped_early"]
    prefix = (output / "retrieval.jsonl").read_bytes()
    assert (
        json.loads((output / "metrics.json").read_text())["arms"]["bm25"]["completed"]
        == 1
    )
    result = run_retrieval(questions, ["bm25"], lambda arm: Engine(), output, manifest)
    assert calls == ["q1", "q2", "q2", "q3"]
    assert result["arms"]["bm25"]["completed"] == 3
    assert (output / "retrieval.jsonl").read_bytes().startswith(prefix)
    assert any(
        e["event"] == "error" for e in read_ledger(output / "retrieval.jsonl")[1]
    )
    with pytest.raises(ValueError, match="identity"):
        run_retrieval(
            questions,
            ["bm25"],
            lambda arm: Engine(),
            output,
            dict(manifest, models={"changed": 1}),
        )


@pytest.mark.parametrize(
    "arm,missing_document", [("bm25", False), ("bm25", True), ("hybrid_qwen", False)]
)
def test_rag_adapter_explicit_files_index_audit_and_passage_scores(
    bundle, tmp_path, monkeypatch, arm, missing_document
):
    monkeypatch.setattr("evals.benchmarks.retrieve.tune_embeddings", lambda: None)
    questions, output, manifest = retrieval_fixture(bundle, tmp_path)
    mapping = json.loads((bundle / "documents.json").read_text())
    files = [str((bundle / "corpus" / name).resolve()) for name in sorted(mapping)]
    # Include >501 explicit files to guard against reintroducing a bounded folder scan.
    for i in range(502):
        name = f"extra-{i}.txt"
        mapping[name] = f"extra-{i}"
        files.append(str((bundle / "corpus" / name).resolve()))
    settings, scopes = [], []
    observer = {}

    def search(query):
        doc = {"id": "chunk1", "file_path": files[0], "content": "source text"}
        observer["callback"](
            "candidates", [dict(doc, score=0.25), dict(doc, id="chunk2", score=0.2)]
        )
        return [doc]

    scan = SimpleNamespace()
    scan.select = lambda columns: scan
    scan.limit = lambda count: scan
    scan.to_arrow = lambda: SimpleNamespace(
        to_pylist=lambda: [
            {"file_path": path} for path in (files[:-1] if missing_document else files)
        ]
    )
    pipeline = SimpleNamespace(
        vector_db=SimpleNamespace(table=SimpleNamespace(search=lambda: scan)),
        search=search,
        device="cuda",
        reranker_type="qwen3",
        reranker_model=object(),
        embedding_fn=object(),
    )

    def get_pipeline(doc_path, scope):
        scopes.append((doc_path, scope))
        return pipeline

    def context(callback):
        observer["callback"] = callback
        return nullcontext()

    monkeypatch.setitem(
        sys.modules,
        "rag.fetch",
        SimpleNamespace(get_rag_pipeline_v2=get_pipeline, reset_pipelines=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules, "rag.settings", SimpleNamespace(save_settings=settings.append)
    )
    monkeypatch.setitem(
        sys.modules, "rag.index_config", SimpleNamespace(RAG_CONFIG_V2={})
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda model, **kwargs: model),
    )
    import rag

    monkeypatch.setattr(
        rag, "tracing", SimpleNamespace(execution_context=context), raising=False
    )
    original_config = __import__("os").environ.get("RAG_MCP_CONFIG")
    result = run_retrieval(
        questions,
        [arm],
        lambda selected_arm: RagRetriever(
            selected_arm,
            bundle,
            output,
            mapping,
            "embedding",
            50,
            700,
            {"embedding": "embedding-revision", RERANKER: "reranker-revision"},
        ),
        output,
        manifest,
    )
    assert __import__("os").environ.get("RAG_MCP_CONFIG") == original_config
    if missing_document:
        assert "Indexed-document audit failed" in result["stopped_early"]
        assert result["arms"]["bm25"]["completed"] == 0
        return
    if arm == "hybrid_qwen":
        assert "Reranker fell back" in result["stopped_early"]
        assert result["arms"][arm]["completed"] == 0
        return
    assert result["stopped_early"] is None, result
    assert sorted(scopes[0][0]) == sorted(files)
    assert (
        settings[0]["embed_provider"] == "none"
        and settings[0]["reranker_type"] == "off"
    )
    events = read_ledger(output / "retrieval.jsonl")[1]
    assert next(e for e in events if e["event"] == "index_verified")["documents"] == 505
    done = [e for e in events if e["event"] == "complete"]
    assert len(done) == 3
    assert all(e["passages"][0]["text"] and e["scores"][0] > 0 for e in done)
    assert __import__("os").environ.get("RAG_MCP_CONFIG") == original_config


class FakeLLM:
    def __init__(self, model, fail_at=None, cost=0.002):
        self.model, self.calls, self.fail_at, self.cost = model, [], fail_at, cost
        self.client = SimpleNamespace(with_options=lambda **kwargs: self.client)

    def respond(self, schema, **kwargs):
        self.calls.append(schema.__name__)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("provider rejected")
        # Each generated job has its own valid document id.
        if schema.__name__ == "Answer":
            prompt = self.kbench.prompts[-1]
            doc = json.loads(prompt.split("\n", 1)[1])["evidence"][0]["doc_id"]
            content = schema(text=f"Fixture answer [{doc}]", citations=[doc])
        else:
            content = schema(label="correct", reason="Matches gold")
        usage = SimpleNamespace(
            input_tokens_cost_nanodollars=self.cost * 1e9,
            output_tokens_cost_nanodollars=0,
        )
        return SimpleNamespace(content=content, usage=usage)


def fake_kbench(model="answer-model", judge="judge-model", fail_at=None):
    module = ModuleType("kaggle_benchmarks")
    module.prompts = []
    module.user = SimpleNamespace(send=module.prompts.append)
    module.chats = SimpleNamespace(new=lambda *args, **kwargs: nullcontext())
    module.llms = {model: FakeLLM(model), judge: FakeLLM(judge, fail_at)}
    for llm in module.llms.values():
        llm.kbench = module

    def task(**kwargs):
        def decorate(fn):
            fn.run = fn
            return fn

        return decorate

    module.task = task
    return module


@pytest.fixture
def retrieval_evidence(bundle, tmp_path):
    questions, output, manifest = retrieval_fixture(bundle, tmp_path)

    class Engine:
        indexed_count = 3

        def __call__(self, q):
            return {
                "ranked_doc_ids": q["gold_doc_ids"],
                "scores": [1],
                "passages": [
                    {"doc_id": q["gold_doc_ids"][0], "text": q["gold_answer"]}
                ],
            }

        def close(self):
            pass

    run_retrieval(questions, ["bm25"], lambda arm: Engine(), output, manifest)
    return output / "retrieval.jsonl", output / "manifest.json"


def test_generated_three_question_task_executes_offline(
    retrieval_evidence, tmp_path, monkeypatch
):
    retrieval, manifest = retrieval_evidence
    code = build_answer_task(
        retrieval,
        manifest,
        arm="bm25",
        model="answer-model",
        judge="judge-model",
        task="fixture",
        ceiling=0.5,
        limit=3,
    )
    bench = fake_kbench()
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", bench)
    monkeypatch.setenv("RAG_BENCH_WORKING_DIR", str(tmp_path))
    namespace = {"__name__": "generated_fixture"}
    exec(compile(code, "<fixture-task>", "exec"), namespace)
    output = tmp_path / "answer_results"
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["completed"] == 3 and metrics["labels"] == {"correct": 3}
    assert bench.llms["answer-model"].calls == ["Answer"] * 3
    assert bench.llms["judge-model"].calls == ["Grade"] * 3
    assert all("gold_answer" not in p for p in bench.prompts[::2])


def test_hosted_resume_reuses_answer_and_reserves_rejected_call(
    retrieval_evidence, tmp_path
):
    code = build_answer_task(
        *retrieval_evidence,
        arm="bm25",
        model="answer-model",
        judge="judge-model",
        task="fixture",
        ceiling=0.5,
        limit=3,
    )
    # Read literal config through AST, without invoking the generated task.
    import ast

    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(code).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("MANIFEST", "JOBS")
    }
    manifest, jobs = values["MANIFEST"], values["JOBS"]
    output = tmp_path / "answers"
    bench = fake_kbench(fail_at=1)
    summary = run_answers(bench, *bench.llms.values(), jobs, output, manifest)
    assert "provider rejected" in summary["stopped_early"]
    assert summary["accounting"]["reserved_unknown_usd"] == 0.01
    seed = (output / "answers.jsonl").read_bytes()
    resumed_code = build_answer_task(
        *retrieval_evidence,
        arm="bm25",
        model="answer-model",
        judge="judge-model",
        task="fixture-resume",
        ceiling=0.2,
        limit=3,
        resume_from=output / "answers.jsonl",
    )
    embedded_seed = next(
        ast.literal_eval(n.value)
        for n in ast.parse(resumed_code).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "RESUME_LEDGER"
    )
    assert embedded_seed == seed
    bench = fake_kbench()
    summary = run_answers(bench, *bench.llms.values(), jobs, output, manifest, seed)
    assert summary["completed"] == 3
    assert bench.llms["answer-model"].calls == ["Answer", "Answer"]
    assert (output / "answers.jsonl").read_bytes().startswith(seed)
    latest = (output / "answers.jsonl").read_bytes()
    before = len(bench.prompts)
    summary = run_answers(bench, *bench.llms.values(), jobs, output, manifest, latest)
    assert len(bench.prompts) == before and summary["completed"] == 3
    # A zero remaining allocation stops before making a call and still writes metrics.
    bench = fake_kbench()
    summary = run_answers(
        bench,
        *bench.llms.values(),
        jobs,
        tmp_path / "no-budget",
        dict(manifest, incremental_ceiling_usd=0.001),
    )
    assert summary["completed"] == 0 and not bench.prompts
    assert "Spend ceiling" in summary["stopped_early"]
    assert accounting([{"event": "call_started", "call_id": "lost"}], 0.05)[
        "unresolved_call"
    ]


def test_hosted_preflight_rejects_invalid_cost_and_oversized_task(
    retrieval_evidence, tmp_path
):
    kwargs = dict(
        arm="bm25",
        model="answer-model",
        judge="judge-model",
        task="fixture",
        ceiling=0.5,
    )
    with pytest.raises(ValueError, match="1..30"):
        build_answer_task(*retrieval_evidence, limit=31, **kwargs)
    code = build_answer_task(*retrieval_evidence, **kwargs)
    import ast

    config = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(code).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("MANIFEST", "JOBS")
    }
    manifest = config["MANIFEST"]
    events = [
        {
            "identity": manifest["identity"],
            "event": "call_started",
            "call_id": "bad",
            "arm": "bm25",
            "question_id": "q1",
            "step": "answer",
            "model": "answer-model",
        },
        {
            "identity": manifest["identity"],
            "event": "call_complete",
            "call_id": "bad",
            "arm": "bm25",
            "question_id": "q1",
            "step": "answer",
            "cost_usd": -1,
        },
    ]
    seed = tmp_path / "bad.jsonl"
    seed.write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(ValueError, match="invalid call cost"):
        build_answer_task(*retrieval_evidence, resume_from=seed, **kwargs)
    assert (
        usage_cost(
            {
                "usage": {
                    "cost": {
                        "input_tokens_cost_nanodollars": 1000000,
                        "output_tokens_cost_nanodollars": 2000000,
                    }
                }
            }
        )
        == 0.003
    )
    assert usage_cost({"usage": {}}) is None


def test_notebook_artifact_has_executable_cells_and_gpu_metadata(tmp_path):
    output = tmp_path / "notebook"
    build_notebook(output, "owner", "fixture", slices=["github"])
    notebook = json.loads((output / "benchmark.ipynb").read_text())
    for cell in notebook["cells"]:
        compile("".join(cell["source"]), "<notebook>", "exec")
    meta = json.loads((output / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]


def test_wikipedia_fetch_is_cached_and_redirect_response_retained(
    tmp_path, monkeypatch
):
    calls = []

    def response(url):
        calls.append(url)
        return {
            "query": {
                "redirects": [{"from": "Cedar", "to": "Cedar tree"}],
                "pages": {
                    "1": {"title": "Cedar tree", "extract": "An evergreen tree."}
                },
            }
        }

    monkeypatch.setattr("evals.benchmarks.datasets.get_json", response)
    for _ in range(2):
        assert (
            wikipedia_text("https://en.wikipedia.org/wiki/Cedar", tmp_path)
            == "An evergreen tree."
        )
    assert len(calls) == 1
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["query"]["redirects"]


def test_ledger_refuses_torn_or_foreign_parent(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"event":"partial"}')
    with pytest.raises(ValueError, match="newline"):
        Ledger(path, "id")
    path.write_text('{"event":"partial","identity":"other"}\n')
    with pytest.raises(ValueError, match="identity"):
        Ledger(path, "id")


def test_paid_provider_keys_are_removed_and_unknown_keys_refuse_to_run():
    import pytest
    from evals.benchmarks.retrieve import forbid_paid_calls

    environ = {"OPENROUTER_API_KEY": "x", "RAG_MCP_EMBED_API_KEY": "y", "PATH": "/bin"}
    assert sorted(forbid_paid_calls(environ)) == [
        "OPENROUTER_API_KEY",
        "RAG_MCP_EMBED_API_KEY",
    ]
    assert environ == {"PATH": "/bin"}
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        forbid_paid_calls({"MISTRAL_API_KEY": "z"})
