"""Compare raw and instructed dense queries against an existing book index."""

import json
from pathlib import Path
from types import SimpleNamespace

from fastembed import TextEmbedding

from rag.embeddings import _fastembed_cache_dir
from rag.lancedb_backend import LanceDBBackend
from rag.reranking import fuse_rankings


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "query-prefix-followup.manifest.json").read_text())
INDEX = Path("/tmp/rag-semantic-smoke-38iZ18Ur/indexes/documents:c12f995c59794226_d745c2c6/generation-zflmcrar/lancedb_storage")
ANCHOR = MANIFEST["anchor"].lower()


class Embedder:
    def __init__(self):
        self.model_name = MANIFEST["model"]
        self.model = TextEmbedding(model_name=self.model_name, cache_dir=_fastembed_cache_dir())

    def embed_query(self, text):
        return next(iter(self.model.query_embed(text))).tolist()


def rank_of(rows):
    return next((rank for rank, row in enumerate(rows, 1) if ANCHOR in row["content"].lower()), None)


def main():
    backend = LanceDBBackend(
        table_name="rag_v2_generation_zflmcrar",
        embedding_fn=Embedder(),
        vector_size=384,
        persist_path=str(INDEX),
    )
    query = MANIFEST["query"]
    lexical = backend.search_fts(query, top_k=50)
    output = {"lexical_anchor_rank": rank_of(lexical), "variants": {}}
    fuser = SimpleNamespace(config={"ensemble_weights": [1.0, 1.0]})
    for label, dense_query in (("raw", query), ("prefixed", MANIFEST["prefix"] + query)):
        dense = backend.search_vector(dense_query, top_k=50)
        fused = fuse_rankings(fuser, [(row, row["score"]) for row in dense],
                              [(row, row["score"]) for row in lexical], k=60)
        output["variants"][label] = {
            "dense_anchor_rank": rank_of(dense),
            "fused_anchor_rank": rank_of(fused),
            "top_five_offsets": [[row.get("start_offset"), row.get("end_offset")] for row in fused[:5]],
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
