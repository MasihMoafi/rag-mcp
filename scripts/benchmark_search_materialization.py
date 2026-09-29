"""Local, deterministic search benchmark; no models, credentials or downloads.

Run with an output JSON path. Compare result digests before/after code changes.
"""
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from rag.bm25 import BM25Index
from rag.lancedb_backend import LanceDBBackend


def measure(operation):
    expected = operation()
    samples = []
    for _ in range(15):
        start = time.perf_counter()
        result = operation()
        samples.append((time.perf_counter() - start) * 1000)
        assert result == expected
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "result_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()}


index = BM25Index()
for i in range(20000):
    index.add_document({"id": i, "content": f"common document group{i % 1000} " * (1 + i % 5)})
report = {"bm25_sparse": measure(lambda: index.search("group17 group29", k=20)),
          "bm25_common": measure(lambda: index.search("common", k=20))}
with tempfile.TemporaryDirectory() as temporary:
    vector = np.random.default_rng(42).normal(size=4096).astype('float32').tolist()
    class Embedder:
        def embed_query(self, text):
            return vector
    backend = LanceDBBackend('benchmark', Embedder(), vector_size=4096, persist_path=temporary)
    backend.table = backend.db.create_table('benchmark', [
        {"id": i, "content": f"common document group{i % 10}", "vector": vector}
        for i in range(1000)
    ])
    backend.table.create_fts_index('content', replace=True)
    report['vector_4096_top100'] = measure(lambda: backend.search_vector('common', top_k=100))
    report['fts_4096_top100'] = measure(lambda: backend.search_fts('common', top_k=100))
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value['median_ms'] for key, value in report.items()}))
