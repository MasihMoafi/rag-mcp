"""Inspect exact-anchor ranks in a published private index; print no passages."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import lancedb

from rag.embeddings import build_embedder
from rag.lancedb_backend import LanceDBBackend
from rag.reranking import fuse_rankings
from rag.settings import embedding_config


HERE = Path(__file__).parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())


def main():
    root = Path(os.environ["RAG_MCP_DATA_DIR"])
    docs_paths = list(root.glob("*/generation-*/docs.json"))
    if len(docs_paths) != 1:
        raise RuntimeError("Expected exactly one published index")
    docs = json.loads(docs_paths[0].read_text())
    if len(docs) != MANIFEST["expected_chunk_count"]:
        raise RuntimeError("Published chunk count changed")
    config = embedding_config()
    if (config["embed_provider"] != "openai_compatible"
            or config["embed_model"] != MANIFEST["embedding_model"]
            or config["embed_base_url"] != MANIFEST["embedding_endpoint"]):
        raise RuntimeError("Embedding settings changed")
    path = docs_paths[0].parent / "lancedb_storage"
    names = lancedb.connect(path).table_names()
    if len(names) != 1:
        raise RuntimeError("Expected exactly one LanceDB table")
    backend = LanceDBBackend(names[0], build_embedder(config, device="cpu"), persist_path=str(path))
    result = []
    for question in MANIFEST["questions"]:
        anchors = {doc["id"] for doc in docs if question["anchor"] in doc["content"].lower()}
        if len(anchors) != 1:
            raise RuntimeError("Expected exactly one exact anchor chunk")
        rows = {}
        candidates_by_stage = {}
        for stage, search in (("dense", backend.search_vector), ("lexical", backend.search_fts)):
            candidates = search(question["query"], top_k=200)
            candidates_by_stage[stage] = candidates
            rows[stage] = next((rank for rank, doc in enumerate(candidates, 1)
                                if doc["id"] in anchors), None)
        fusion = SimpleNamespace(config={"ensemble_weights": [1.0, 1.0]})
        for limit in (50, 200):
            fused = fuse_rankings(
                fusion,
                [(doc, doc["score"]) for doc in candidates_by_stage["dense"][:limit]],
                [(doc, doc["score"]) for doc in candidates_by_stage["lexical"][:limit]],
                k=60,
            )
            rows[f"fused_{limit}"] = next((rank for rank, doc in enumerate(fused, 1)
                                          if doc["id"] in anchors), None)
        result.append({"chapter": question["chapter"], "anchor_rank_within_dense_200": rows["dense"],
                       "anchor_rank_within_lexical_200": rows["lexical"],
                       "anchor_rank_fused_50": rows["fused_50"],
                       "anchor_rank_fused_200": rows["fused_200"]})
    print(json.dumps({"candidate_limit": 200, "results": result}, indent=2))


if __name__ == "__main__":
    main()
