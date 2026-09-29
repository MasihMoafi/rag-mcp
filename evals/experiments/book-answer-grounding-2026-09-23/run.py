"""Three bounded live answers; keep private text out of the redacted report."""

import hashlib
import json
import os
from pathlib import Path
import re
import time

from rag.execution import execute_run
from rag.settings import embedding_config, retrieval_config
from rag.workspaces import WorkspaceStore


HERE = Path(__file__).parent
MANIFEST_PATH = Path(os.environ.get("EVAL_MANIFEST", HERE / "manifest.json"))
MANIFEST = json.loads(MANIFEST_PATH.read_text())


def validate():
    root = Path(MANIFEST["source_directory"])
    files = sorted(root.glob("*.md"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    if (len(files) != MANIFEST["source_count"]
            or sum(path.stat().st_size for path in files) != MANIFEST["source_bytes"]
            or digest.hexdigest() != MANIFEST["aggregate_source_sha256"]):
        raise RuntimeError("Book sources changed")
    index_files = list(Path(os.environ["RAG_MCP_DATA_DIR"]).glob("*/generation-*/docs.json"))
    if len(index_files) != 1 or len(json.loads(index_files[0].read_text())) != MANIFEST["expected_chunk_count"]:
        raise RuntimeError("Published book index changed")
    embedding = embedding_config()
    retrieval = retrieval_config()
    if (embedding["embed_provider"] != "openai_compatible"
            or embedding["embed_model"] != MANIFEST["embedding_model"]
            or embedding["embed_base_url"] != MANIFEST["embedding_endpoint"]
            or retrieval["answer_model"] != MANIFEST["answer_model"]
            or int(retrieval["answer_max_tokens"]) != 240
            or retrieval["reranker_type"] != "off"
            or retrieval["contextual_retrieval"] != "off"
            or int(retrieval["candidate_k"]) != 50
            or int(retrieval["top_k"]) != 5):
        raise RuntimeError("Run settings differ from preregistration")
    if os.environ.get("RAG_MCP_TRACE_CONTENT") == "1":
        raise RuntimeError("Content capture must be disabled")
    snapshot = WorkspaceStore().snapshot(MANIFEST["workspace_id"])
    if len(snapshot["documents"]) != MANIFEST["source_count"]:
        raise RuntimeError("Imported workspace changed")
    return snapshot["revision"]


def main():
    revision = validate()
    private = Path(os.environ["EVAL_PRIVATE_DIR"])
    private.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {"manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
              "workspace_revision": revision, "runs": []}
    started = time.monotonic()
    for question in MANIFEST["questions"]:
        began = time.monotonic()
        result = execute_run(question["query"], workspace_id=MANIFEST["workspace_id"],
                             workspace_revision=revision, client="studio", mode="answer")
        (private / f"{question['id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        trace_path = Path(os.environ["RAG_MCP_TRACE_DIR"]) / f"trace-{result['run_id']}.json"
        trace = json.loads(trace_path.read_text())
        spans = {span["operation"]: span for span in trace["spans"]}
        answer = result.get("answer") or ""
        passages = result.get("retrieved_passages") or []
        anchor = question["anchor"].lower()
        report["runs"].append({"id": question["id"], "run_id": result["run_id"],
                               "status": result["status"], "error_code": result.get("code"),
                               "answer_chars": len(answer),
                               "citation_numbers": [int(x) for x in re.findall(r"\[(\d+)\]", answer)],
                               "passage_count": len(passages),
                               "anchor_rank": next((i for i, row in enumerate(passages, 1)
                                                    if anchor in row["content"].lower()), None),
                               "synthesis_status": spans.get("synthesis", {}).get("status"),
                               "reported_cost_usd": trace["accounting"]["reported_cost_subtotal_usd"],
                               "duration_seconds": round(time.monotonic() - began, 3)})
        (private / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        if result["status"] != "success":
            break
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["reported_cost_usd"] = round(sum(row["reported_cost_usd"] or 0 for row in report["runs"]), 8)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
