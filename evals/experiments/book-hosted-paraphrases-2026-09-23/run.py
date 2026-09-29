"""Bounded retrieval-only, redacted report for the nine-chapter book."""

import hashlib
import json
import os
from pathlib import Path
import time

from rag.execution import execute_run
from rag.settings import embedding_config, retrieval_config
from rag.workspaces import WorkspaceStore


MANIFEST = json.loads((Path(__file__).parent / "manifest.json").read_text())


def source_fingerprint(root):
    files = sorted(root.glob("*.md"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return len(files), sum(path.stat().st_size for path in files), digest.hexdigest()


def main():
    root = Path(MANIFEST["source_directory"])
    if source_fingerprint(root) != (MANIFEST["source_count"], MANIFEST["source_bytes"], MANIFEST["aggregate_source_sha256"]):
        raise RuntimeError("Book source changed after preregistration.")
    embedding = embedding_config()
    retrieval = retrieval_config()
    if (embedding["embed_provider"] != "openai_compatible"
            or embedding["embed_model"] != MANIFEST["embedding_model"]
            or embedding["embed_base_url"] != MANIFEST["embedding_endpoint"]
            or retrieval["reranker_type"] != "off"
            or retrieval["contextual_retrieval"] != "off"
            or int(retrieval["candidate_k"]) != 50
            or int(retrieval["top_k"]) != 5):
        raise RuntimeError("Retrieval settings differ from the manifest.")
    store = WorkspaceStore()
    snapshot = store.snapshot("book-hosted")
    if len(snapshot["documents"]) != MANIFEST["source_count"]:
        raise RuntimeError("Imported workspace has the wrong source count.")
    report = {"workspace_revision": snapshot["revision"], "runs": [], "index_chunks": None}
    started = time.monotonic()
    for question in MANIFEST["questions"]:
        began = time.monotonic()
        result = execute_run(question["query"], workspace_id="book-hosted", client="cli",
                             workspace_revision=snapshot["revision"], mode="retrieve")
        passages = result.get("retrieved_passages") or []
        anchor_rank = next((i for i, row in enumerate(passages, 1)
                            if question["anchor"] in row["content"].lower()), None)
        chapter_rank = next((i for i, row in enumerate(passages, 1)
                             if row["name"].startswith(question["chapter"])), None)
        trace_path = Path(os.environ["RAG_MCP_TRACE_DIR"]) / f"trace-{result['run_id']}.json"
        trace = json.loads(trace_path.read_text())
        report["runs"].append({"run_id": result["run_id"], "status": result["status"],
                               "anchor_rank": anchor_rank, "chapter_rank": chapter_rank,
                               "passage_count": len(passages),
                               "reported_cost_usd": trace["accounting"]["reported_cost_subtotal_usd"],
                               "duration_seconds": round(time.monotonic() - began, 3)})
        if len(report["runs"]) == 1:
            files = list(Path(os.environ["RAG_MCP_DATA_DIR"]).glob("*/generation-*/docs.json"))
            report["index_chunks"] = len(json.loads(files[0].read_text())) if len(files) == 1 else None
            if report["index_chunks"] != MANIFEST["expected_chunk_count"]:
                break
        if result["status"] != "success":
            break
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["anchors_in_top_five"] = sum(row["anchor_rank"] is not None and row["anchor_rank"] <= 5
                                         for row in report["runs"])
    report["accepted"] = (len(report["runs"]) == 4 and all(row["status"] == "success" for row in report["runs"])
                          and report["anchors_in_top_five"] >= 3
                          and report["index_chunks"] == MANIFEST["expected_chunk_count"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
