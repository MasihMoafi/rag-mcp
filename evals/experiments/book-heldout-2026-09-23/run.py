"""Paired, bounded book retrieval evaluation with private ranked evidence."""

import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import lancedb

from rag import tracing
from rag.embeddings import build_embedder
from rag.lancedb_backend import LanceDBBackend
from rag.reranking import fuse_rankings
from rag.settings import embedding_config


HERE = Path(__file__).parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())


def verify_corpus(index_root):
    source_root = Path(MANIFEST["source_directory"])
    paths = sorted(source_root.glob("*.md"))
    digest = hashlib.sha256()
    if len(paths) != MANIFEST["source_count"]:
        raise RuntimeError("Source count changed")
    for path, source in zip(paths, MANIFEST["sources"]):
        content = path.read_bytes()
        content_sha = hashlib.sha256(content).digest()
        if (path.name != source["name"] or len(content) != source["bytes"]
                or content_sha.hex() != source["sha256"]):
            raise RuntimeError("Book source changed")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(content_sha)
    if digest.hexdigest() != MANIFEST["aggregate_source_sha256"]:
        raise RuntimeError("Aggregate book fingerprint changed")
    docs_paths = list(index_root.glob("*/generation-*/docs.json"))
    if len(docs_paths) != 1:
        raise RuntimeError("Expected exactly one published index")
    docs = json.loads(docs_paths[0].read_text())
    if len(docs) != MANIFEST["expected_chunk_count"]:
        raise RuntimeError("Published chunk count changed")
    anchors = {}
    for question in MANIFEST["questions"]:
        source = next(item for item in MANIFEST["sources"] if item["chapter"] == question["chapter"])
        text = (source_root / source["name"]).read_text()
        phrase = question["anchor"]
        if text.count(phrase) != 1 or text.index(phrase) != question["anchor_offset"]:
            raise RuntimeError("Question anchor changed")
        matches = {doc["id"] for doc in docs if phrase in doc["content"]}
        if not matches:
            raise RuntimeError("Question anchor absent from published chunks")
        anchors[question["id"]] = matches
    return docs_paths[0].parent / "lancedb_storage", anchors


def anchor_rank(rows, anchor_ids):
    return next((rank for rank, doc in enumerate(rows, 1) if doc["id"] in anchor_ids), None)


def main():
    index_path, anchors = verify_corpus(Path(os.environ["RAG_MCP_DATA_DIR"]))
    config = embedding_config()
    if (config["embed_provider"] != "openai_compatible"
            or config["embed_model"] != MANIFEST["embedding_model"]
            or config["embed_base_url"] != MANIFEST["embedding_endpoint"]):
        raise RuntimeError("Embedding settings differ from preregistration")
    if os.environ.get("RAG_MCP_TRACE_CONTENT") == "1":
        raise RuntimeError("Content capture must be disabled")
    tables = lancedb.connect(index_path).table_names()
    if len(tables) != 1:
        raise RuntimeError("Expected one LanceDB table")
    backend = LanceDBBackend(tables[0], build_embedder(config, device="cpu"),
                             persist_path=str(index_path))
    private = Path(os.environ["EVAL_PRIVATE_DIR"])
    private.mkdir(mode=0o700, parents=True, exist_ok=False)
    fusion = SimpleNamespace(config={"ensemble_weights": MANIFEST["ensemble_weights"]})
    report = {"manifest_sha256": hashlib.sha256((HERE / "manifest.json").read_bytes()).hexdigest(),
              "index_chunks": MANIFEST["expected_chunk_count"], "runs": []}
    started = time.monotonic()
    for question in MANIFEST["questions"]:
        began = time.monotonic()
        trace_id = uuid.uuid4().hex
        with tracing.request_trace(trace_id):
            with tracing.span("retrieval.lexical"):
                lexical = backend.search_fts(question["query"], MANIFEST["candidate_k"])
                tracing.passages("candidates", lexical)
            with tracing.span("retrieval.dense"):
                dense = backend.search_vector(question["query"], MANIFEST["candidate_k"])
                tracing.passages("candidates", dense)
            with tracing.span("retrieval.rrf", score_kind="rrf"):
                ranked_hybrid = []

                def record_fusion(ranked):
                    ranked_hybrid.extend(ranked)
                    tracing.passages("candidates", [dict(doc, score=score)
                                                     for doc, score in ranked[:MANIFEST["candidate_k"]]])

                hybrid = fuse_rankings(
                    fusion,
                    [(doc, doc["score"]) for doc in dense],
                    [(doc, doc["score"]) for doc in lexical],
                    k=MANIFEST["rrf_k"],
                    on_ranked=record_fusion,
                )
        anchor_ids = anchors[question["id"]]
        raw = {"question_id": question["id"], "trace_id": trace_id,
               "lexical": [{"id": doc["id"], "score": doc["score"]} for doc in lexical],
               "dense": [{"id": doc["id"], "score": doc["score"]} for doc in dense],
               "hybrid": [{"id": doc["id"], "score": score} for doc, score in ranked_hybrid],
               "anchor_ids": sorted(anchor_ids)}
        (private / f"{question['id']}.json").write_text(json.dumps(raw) + "\n")
        trace_path = Path(os.environ["RAG_MCP_TRACE_DIR"]) / f"trace-{trace_id}.json"
        trace = json.loads(trace_path.read_text())
        report["runs"].append({"id": question["id"], "chapter": question["chapter"],
                               "trace_id": trace_id, "status": "success",
                               "lexical_rank_at_5": anchor_rank(lexical[:MANIFEST["final_k"]], anchor_ids),
                               "hybrid_rank_at_5": anchor_rank(hybrid[:MANIFEST["final_k"]], anchor_ids),
                               "dense_rank_at_50": anchor_rank(dense, anchor_ids),
                               "lexical_rank_at_50": anchor_rank(lexical, anchor_ids),
                               "hybrid_rank_at_50": anchor_rank(hybrid[:MANIFEST["candidate_k"]], anchor_ids),
                               "duration_seconds": round(time.monotonic() - began, 3),
                               "reported_cost_usd": trace["accounting"]["reported_cost_subtotal_usd"]})
        (private / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = report["runs"]
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["lexical_hits_at_5"] = sum(row["lexical_rank_at_5"] is not None for row in rows)
    report["hybrid_hits_at_5"] = sum(row["hybrid_rank_at_5"] is not None for row in rows)
    report["paired_wins"] = sum(row["lexical_rank_at_5"] is None and row["hybrid_rank_at_5"] is not None for row in rows)
    report["paired_losses"] = sum(row["lexical_rank_at_5"] is not None and row["hybrid_rank_at_5"] is None for row in rows)
    report["reported_cost_usd"] = round(sum(row["reported_cost_usd"] or 0 for row in rows), 8)
    report["quality_threshold_met"] = (len(rows) == len(MANIFEST["questions"])
                                       and report["hybrid_hits_at_5"] - report["lexical_hits_at_5"] >= 4
                                       and report["paired_losses"] <= 2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
