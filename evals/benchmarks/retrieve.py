"""In-process rag-mcp retrieval; no benchmark-owned search implementation."""

import argparse
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

from .common import (
    Ledger,
    digest,
    file_hash,
    finish_safely,
    graceful_signals,
    read_ledger,
    retrieval_summary,
    run_lock,
    write_json,
)

EMBEDDINGS = {size: f"Qwen/Qwen3-Embedding-{size}" for size in ("0.6B", "4B", "8B")}
RERANKER = "Qwen/Qwen3-Reranker-4B"
ARMS = ("bm25", "hybrid_qwen")
# Only local.py asks for this arm; it uses Masih's key on this machine (8 Nov), never on Kaggle.
API_ARM = "hybrid_api"
API_EMBEDDING = "qwen/qwen3-embedding-8b"
API_RERANKER = "cohere/rerank-4-fast"


PAID_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "COHERE_API_KEY",
    "VOYAGE_API_KEY",
    "RAG_MCP_EMBED_API_KEY",
)


def forbid_paid_calls(environ=os.environ):
    """Masih (8 Nov): no paid calls. Remove every provider key so rag cannot reach a hosted model."""
    removed = [key for key in PAID_KEYS if environ.pop(key, None)]
    leftover = [key for key in environ if key.endswith("_API_KEY") and environ[key]]
    if leftover:
        raise ValueError(
            f"Unexpected API keys present; refusing to run: {sorted(leftover)}"
        )
    return removed


def tune_embeddings(batch_size=128):
    """rag calls encode() with library defaults and is silent until indexing ends: use
    half precision and larger batches on the GPU, and print progress for every batch rag sends."""
    from sentence_transformers import SentenceTransformer

    if getattr(SentenceTransformer, "_rag_bench_tuned", False):
        return
    plain_init, plain_encode = SentenceTransformer.__init__, SentenceTransformer.encode
    embedded, started = [0], time.monotonic()

    def init(self, *args, **kwargs):
        kwargs.setdefault("model_kwargs", {"torch_dtype": "float16"})
        plain_init(self, *args, **kwargs)

    def encode(self, texts, *args, **kwargs):
        kwargs.setdefault("batch_size", batch_size)
        vectors = plain_encode(self, texts, *args, **kwargs)
        if isinstance(texts, list) and len(texts) > 1:
            embedded[0] += len(texts)
            print(
                f"[bench] embedded {embedded[0]} chunks in {time.monotonic() - started:.0f}s",
                flush=True,
            )
        return vectors

    SentenceTransformer.__init__, SentenceTransformer.encode = init, encode
    SentenceTransformer._rag_bench_tuned = True


def check_bundle(bundle):
    bundle = Path(bundle)
    dataset = json.loads((bundle / "dataset.json").read_text())
    mapping = json.loads((bundle / "documents.json").read_text())
    _, questions = read_ledger(bundle / "questions.jsonl")
    documents = {
        key: (bundle / "corpus" / name).read_text() for name, key in mapping.items()
    }
    if (
        digest(documents) != dataset["corpus_sha256"]
        or digest(questions) != dataset["questions_sha256"]
    ):
        raise ValueError("Normalized corpus/questions changed since schema validation")
    if (
        not documents
        or not questions
        or any(
            not q["gold_doc_ids"] or not set(q["gold_doc_ids"]).issubset(documents)
            for q in questions
        )
    ):
        raise ValueError("Invalid normalized corpus or gold scope")
    return dataset, mapping, questions


def preflight_models(embedding):
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise ValueError("A CUDA GPU is required; enable Kaggle GPU before running")
    revisions = {}
    for model in (embedding, RERANKER):
        path = Path(snapshot_download(model))
        revisions[model] = path.name
    return {
        "revisions": revisions,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


class RagRetriever:
    """One arm owns one isolated store and one pipeline; arms run sequentially."""

    def __init__(
        self,
        arm,
        bundle,
        output,
        mapping,
        embedding,
        candidates,
        chunk_size,
        revisions=None,
    ):
        from evals.paidrun import isolated_env

        self.old_env = dict(os.environ)
        self.pipeline = None
        self.old_config = None
        self.mapping = mapping
        self.arm = arm
        try:
            # Indexes go to scratch when set, so saved notebook output stays small.
            stores = Path(os.environ.get("RAG_BENCH_SCRATCH") or output) / "stores"
            env = isolated_env(stores / arm, os.environ)
            # Do not inherit retrieval/model/credential choices from a notebook.
            for key in list(os.environ):
                if key.startswith("RAG_MCP_"):
                    del os.environ[key]
            os.environ.update(
                {
                    key: value
                    for key, value in env.items()
                    if key
                    in (
                        "RAG_MCP_CONFIG",
                        "RAG_MCP_DATA_DIR",
                        "RAG_MCP_WORKSPACE_DIR",
                        "RAG_MCP_TRACE_DIR",
                    )
                }
            )
            Path(os.environ["RAG_MCP_CONFIG"]).write_text("{}\n")
            embed_path, reranker_path = "", ""
            if arm == "hybrid_qwen":
                from huggingface_hub import snapshot_download

                embed_path = snapshot_download(
                    embedding, revision=revisions[embedding], local_files_only=True
                )
                reranker_path = snapshot_download(
                    RERANKER, revision=revisions[RERANKER], local_files_only=True
                )
                tune_embeddings()
            from rag.fetch import get_rag_pipeline_v2
            from rag.settings import save_settings

            hosted = (
                {
                    "embed_provider": "openai_compatible",
                    "embed_model": API_EMBEDDING,
                    "embed_base_url": "https://openrouter.ai/api/v1",
                    "reranker_type": "openrouter",
                    "reranker_model": API_RERANKER,
                }
                if arm == API_ARM
                else {
                    "embed_provider": "none"
                    if arm == "bm25"
                    else "sentencetransformer",
                    "embed_model": embed_path,
                    "reranker_type": "off" if arm == "bm25" else "cross-encoder",
                    "reranker_model": reranker_path,
                }
            )
            save_settings(
                {
                    **hosted,
                    "top_k": str(candidates),
                    "candidate_k": str(candidates),
                    "contextual_retrieval": "off",
                }
            )
            # These defaults are imported once; set the actual config explicitly.
            from rag.index_config import RAG_CONFIG_V2

            self.old_config = dict(RAG_CONFIG_V2)
            RAG_CONFIG_V2.update(
                chunk_size=chunk_size,
                chunk_overlap=100,
                top_k=candidates,
                backend="lancedb",
            )
            files = [
                str((Path(bundle) / "corpus" / name).resolve())
                for name in sorted(mapping)
            ]
            self.pipeline = get_rag_pipeline_v2(
                files, scope=str(Path(output).resolve() / arm)
            )
            # Table projection avoids materializing every embedding vector.
            stored = (
                self.pipeline.vector_db.table.search()
                .select(["file_path"])
                .limit(None)
                .to_arrow()
                .to_pylist()
            )
            indexed = {str(Path(row["file_path"]).resolve()) for row in stored}
            if indexed != set(files):
                raise ValueError(
                    f"Indexed-document audit failed: expected {len(files)}, got "
                    f"{len(indexed)}; missing={sorted(set(files) - indexed)[:10]}"
                )
            if arm == "hybrid_qwen" and (
                self.pipeline.device != "cuda"
                or self.pipeline.reranker_type != "qwen3"
                or self.pipeline.reranker_model is None
                or self.pipeline.embedding_fn is None
            ):
                raise ValueError("Requested GPU dense/reranker models were not loaded")
            self.indexed_count = len(indexed)
        except BaseException:
            self.close()
            raise

    def __call__(self, question):
        from rag import tracing

        reranked, fused = {}, {}

        def observe(name, documents):
            target = (
                reranked
                if name == "ranked"
                else fused
                if name == "candidates"
                else None
            )
            if target is not None:
                target.clear()
                target.update({d["id"]: d.get("score") for d in documents})

        with tracing.execution_context(observe):
            results = self.pipeline.search(question["question"])
        if isinstance(results, str):
            if results != "No relevant information found.":
                raise ValueError(f"Unexpected search response: {results}")
            results = []
        if self.arm in ("hybrid_qwen", API_ARM) and len(fused) >= 2 and not reranked:
            raise ValueError(
                "Reranker fell back without producing ranked scores; arm invalid"
            )
        scores = reranked or fused
        passages = []
        for result in results:
            key = self.mapping[Path(result["file_path"]).name]
            score = scores.get(result["id"])
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError(
                    "Library did not expose a finite score for a returned passage"
                )
            passages.append(
                {
                    "doc_id": key,
                    "chunk_id": result["id"],
                    "text": result.get("original_content", result["content"]),
                    "location": result.get("location"),
                    "score": score,
                    "score_kind": "reranker" if reranked else "rrf",
                }
            )
        unique = {}
        for passage in passages:
            unique.setdefault(passage["doc_id"], passage["score"])
        return {
            "ranked_doc_ids": list(unique),
            "scores": list(unique.values()),
            "passages": passages[:5],
        }

    def close(self):
        try:
            if self.pipeline is not None:
                from rag.fetch import reset_pipelines

                reset_pipelines()
        finally:
            if self.old_config is not None:
                from rag.index_config import RAG_CONFIG_V2

                RAG_CONFIG_V2.clear()
                RAG_CONFIG_V2.update(self.old_config)
            os.environ.clear()
            os.environ.update(self.old_env)


def run_retrieval(questions, arms, factory, output, manifest):
    """Preflight identity/ledger before work, then return normally on any stop."""
    output = Path(output)
    identity = digest(manifest)
    ledger = Ledger(output / "retrieval.jsonl", identity)
    write_json(
        output / "manifest.json",
        dict(manifest, identity=identity, parent_ledger_sha256=ledger.parent_hash),
    )
    done, stopped = ledger.completed(), None
    current = None
    start = None
    try:
        with graceful_signals():
            for arm in arms:
                pending = [q for q in questions if (arm, q["id"]) not in done]
                if not pending:
                    continue
                retriever = None
                try:
                    ledger.append({"event": "index_started", "arm": arm})
                    retriever = factory(arm)
                    ledger.append(
                        {
                            "event": "index_verified",
                            "arm": arm,
                            "documents": retriever.indexed_count,
                        }
                    )
                    for question in pending:
                        current = (arm, question["id"])
                        ledger.append(
                            {
                                "event": "started",
                                "arm": arm,
                                "question_id": question["id"],
                            }
                        )
                        start = time.monotonic()
                        result = retriever(question)
                        ledger.append(
                            dict(
                                result,
                                event="complete",
                                arm=arm,
                                question_id=question["id"],
                                question=question["question"],
                                gold_answer=question["gold_answer"],
                                gold_doc_ids=question["gold_doc_ids"],
                                caveat=manifest.get("dataset", {}).get("caveat"),
                                latency_seconds=time.monotonic() - start,
                                errors=[],
                            )
                        )
                        current = None
                finally:
                    if retriever is not None:
                        retriever.close()
    except BaseException as error:  # noqa: BLE001 - runbook: record and stop, never lose finished results
        stopped = f"{type(error).__name__}: {error}"
        try:
            ledger.append(
                {
                    "event": "error",
                    "arm": current[0] if current else arm,
                    "question_id": current[1] if current else None,
                    "errors": [stopped],
                    "ranked_doc_ids": [],
                    "scores": [],
                    "latency_seconds": time.monotonic() - start
                    if current and start
                    else None,
                    "caveat": manifest.get("dataset", {}).get("caveat"),
                }
            )
        except BaseException as logging_error:  # noqa: BLE001 - runbook: record and stop, never lose finished results
            print("STOPPED_EARLY error write failed:", repr(logging_error), flush=True)
    try:
        summary = retrieval_summary(ledger.events, questions, arms, stopped)
        summary["caveat"] = manifest.get("dataset", {}).get("caveat")
    except BaseException as error:  # noqa: BLE001 - runbook: record and stop, never lose finished results
        summary = {"stopped_early": stopped or repr(error)}
    return finish_safely(output, ledger, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-size", choices=EMBEDDINGS, default="0.6B")
    parser.add_argument("--arm", choices=(*ARMS, API_ARM), action="append")
    parser.add_argument(
        "--rag-commit", help="commit of a rag-mcp not installed from GitHub"
    )
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=700)
    args = parser.parse_args()
    if args.candidates < 10 or args.chunk_size < 100:
        parser.error("--candidates must be >=10; --chunk-size must be >=100")
    if API_ARM not in (args.arm or ()):
        forbid_paid_calls()
    dataset, mapping, questions = check_bundle(args.bundle)
    embedding = EMBEDDINGS[args.embedding_size]
    arms = list(dict.fromkeys(args.arm or ARMS))
    models = preflight_models(embedding) if "hybrid_qwen" in arms else {}
    dist = importlib.metadata.distribution("ragmcp")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = args.rag_commit or direct.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise ValueError(
            "Install ragmcp from GitHub main; pip must record its commit in direct_url.json"
        )
    manifest = {
        "schema": "rag-bench.retrieval.v1",
        "dataset": dataset,
        "rag_mcp_commit": commit,
        "rag_mcp_version": dist.version,
        "models": models,
        "embedding": API_EMBEDDING
        if API_ARM in arms
        else embedding
        if "hybrid_qwen" in arms
        else None,
        "reranker": API_RERANKER
        if API_ARM in arms
        else RERANKER
        if "hybrid_qwen" in arms
        else None,
        "flags": {
            "arms": arms,
            "candidates": args.candidates,
            "chunk_size": args.chunk_size,
            "chunk_overlap": 100,
        },
        "runner_sha256": {
            name: file_hash(Path(__file__).with_name(name + ".py"))
            for name in ("retrieve", "common", "datasets")
        },
        "questions": len(questions),
        "documents": len(mapping),
        "mrr_cutoff_passages": args.candidates,
    }
    with run_lock(args.output):
        run_retrieval(
            questions,
            arms,
            lambda arm: RagRetriever(
                arm,
                args.bundle,
                args.output,
                mapping,
                embedding,
                args.candidates,
                args.chunk_size,
                models.get("revisions"),
            ),
            args.output,
            manifest,
        )


if __name__ == "__main__":
    main()
