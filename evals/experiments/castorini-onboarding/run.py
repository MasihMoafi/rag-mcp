"""Reproduce official retrieval lessons on a disposable GitHub Actions runner."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile


ROOT = Path(os.environ.get("RUNNER_TEMP", "/nonexistent")) / "castorini-course"
EVIDENCE = ROOT / "evidence"
RUNS = ROOT / "runs"
JAR = ROOT / "anserini-2.3.0-fatjar.jar"
PYTHON = sys.executable
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
commands = []
metrics = {}


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def run(label, argv):
    argv = [str(value) for value in argv]
    commands.append({"label": label, "argv": argv})
    (EVIDENCE / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print(label, flush=True)
    with (EVIDENCE / f"{label}.log").open("w") as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((EVIDENCE / f"{label}.log").read_text()[-12000:], flush=True)
        raise subprocess.CalledProcessError(result.returncode, argv)


def java(main, *args):
    return ["java", "--add-modules", "jdk.incubator.vector", "-cp", JAR, main, *args]


def module(name, *args):
    return [PYTHON, "-m", name, *args]


def prebuilt_index(name, metadata_file):
    # Read the jar's pinned registry from raw content: its unauthenticated
    # GitHub directory-list API is rate-limited on shared Actions addresses.
    url = "https://raw.githubusercontent.com/castorini/prebuilt-indexes/367560b4de7d9c3486f666dcc2df7783ca7758f2/lucene/" + metadata_file
    with urllib.request.urlopen(url, timeout=60) as response:
        entry, = [item for item in json.load(response) if item["name"] == name]
    (EVIDENCE / f"{name}-input.json").write_text(json.dumps(entry, indent=2) + "\n")

    class CheckedStream:
        def __init__(self, source):
            self.source = source
            self.digest = hashlib.md5()
            self.bytes_read = 0

        def read(self, size=-1):
            chunk = self.source.read(size)
            self.digest.update(chunk)
            self.bytes_read += len(chunk)
            return chunk

    # Stream the exact published archive into extraction, retaining its hash
    # without needing two additional 26 GB temporary copies on the runner.
    with urllib.request.urlopen(entry["urls"][0], timeout=120) as source:
        checked = CheckedStream(source)
        with tarfile.open(fileobj=checked, mode="r|gz") as archive:
            archive.extractall(ROOT / "indexes", filter="data")
        while checked.read(1024 * 1024):
            pass
    assert checked.bytes_read == entry["size"], checked.bytes_read
    assert checked.digest.hexdigest() == entry["md5"], checked.digest.hexdigest()
    index = ROOT / "indexes" / entry["filename"].removesuffix(".tar.gz")
    assert index.is_dir(), index
    return index


def evaluate(label, qrels, ranking, expected, *options):
    run(label, java("trec_eval", "-c", *options, qrels, ranking))
    found = {}
    for line in (EVIDENCE / f"{label}.log").read_text().splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] in expected and fields[1] == "all":
            found[fields[0]] = float(fields[2])
    assert found.keys() == expected.keys(), (label, found, expected)
    for metric, value in expected.items():
        assert f"{found[metric]:.4f}" == f"{value:.4f}", (label, metric, found, expected)
    metrics[label] = {"expected": expected, "observed": found}
    print(json.dumps(metrics[label]), flush=True)


def prepare():
    EVIDENCE.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    (ROOT / "indexes").mkdir(exist_ok=True)
    os.environ["PYSERINI_CACHE"] = str(ROOT / "cache")
    download("https://repo.maven.apache.org/maven2/io/anserini/anserini/2.3.0/anserini-2.3.0-fatjar.jar", JAR)
    expected = "35ea338af9d483823e80a8fa888af22be42f22a7c095ef384e4581abacd11a0382ed68583eeef3e46a75eec773a6fd3d5a3358756be31016dcfffa77d392c363"
    assert hashlib.file_digest(JAR.open("rb"), "sha512").hexdigest() == expected
    os.environ["ANSERINI_CLASSPATH"] = str(ROOT)
    (ROOT / "pyserini/pyserini/resources/jars" / JAR.name).symlink_to(JAR)
    metadata = {"anserini": "7b08f2954d8390a85370e68ab3a3ff8164f7dd0b", "jar_version": "2.3.0", "jar_sha512": expected,
                "pyserini": "33712371e8964c42fe83faaac4b8f36ebb077839", "python": sys.version,
                "execution": "GitHub Actions CPU runner; no laptop compute"}
    (EVIDENCE / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


def msmarco_data(full):
    data = ROOT / "data/msmarco-passage"
    data.mkdir(parents=True)
    archive = ROOT / "collectionandqueries.tar.gz"
    download("https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz", archive)
    assert hashlib.file_digest(archive.open("rb"), "md5").hexdigest() == "31644046b18952c1386cd4564ba2ae69"
    with tarfile.open(archive) as tar:
        members = None if full else [member for member in tar if Path(member.name).name in {"queries.dev.tsv", "qrels.dev.small.tsv"}]
        tar.extractall(data, members=members, filter="data")
    scripts = ROOT / "anserini/src/main/python/msmarco"
    run("filter-queries", [PYTHON, scripts / "filter_queries.py", "--qrels", data / "qrels.dev.small.tsv", "--queries", data / "queries.dev.tsv", "--output", data / "queries.dev.small.tsv"])
    assert sum(1 for _ in (data / "queries.dev.small.tsv").open()) == 6980
    run("convert-qrels", [PYTHON, scripts / "convert_msmarco_to_trec_qrels.py", "--input", data / "qrels.dev.small.tsv", "--output", data / "qrels.trec"])
    if full:
        run("convert-passages", [PYTHON, scripts / "convert_collection_to_jsonl.py", "--collection-path", data / "collection.tsv", "--output-folder", data / "collection_jsonl"])
        counts = [sum(1 for _ in file.open()) for file in sorted((data / "collection_jsonl").glob("*.json"))]
        assert counts == [1000000] * 8 + [841823], counts
    return data


def msmarco_bm25():
    data = msmarco_data(full=True)
    index = ROOT / "indexes/msmarco"
    run("anserini-index", java("io.anserini.index.IndexCollection", "-collection", "JsonCollection", "-input", data / "collection_jsonl", "-index", index, "-generator", "DefaultLuceneDocumentGenerator", "-threads", "4", "-storePositions", "-storeDocvectors", "-storeRaw"))
    queries = data / "queries.dev.small.tsv"
    qrels = data / "qrels.trec"
    msmarco = RUNS / "anserini-bm25.tsv"
    trec = RUNS / "anserini-bm25.trec"
    run("anserini-search", java("io.anserini.search.SearchCollection", "-index", index, "-topics", queries, "-topicReader", "TsvInt", "-output", msmarco, "-format", "msmarco", "-parallelism", "4", "-bm25", "-bm25.k1", "0.82", "-bm25.b", "0.68", "-hits", "1000"))
    scripts = ROOT / "anserini/src/main/python/msmarco"
    run("convert-run", [PYTHON, scripts / "convert_msmarco_to_trec_run.py", "--input", msmarco, "--output", trec])
    evaluate("anserini-bm25-eval", qrels, trec, {"map": 0.1957, "recall_1000": 0.8573}, "-m", "map", "-m", "recall.1000")
    evaluate("anserini-mrr", qrels, trec, {"recip_rank": 0.1874}, "-M", "10", "-m", "recip_rank")
    # Build with Pyserini too: the exercise covers the Python indexing entry point.
    run("pyserini-index", module("pyserini.index.lucene", "--collection", "JsonCollection", "--input", data / "collection_jsonl", "--index", ROOT / "indexes/pyserini-msmarco", "--generator", "DefaultLuceneDocumentGenerator", "--threads", "4", "--memoryBuffer", "1024", "--storePositions", "--storeDocvectors", "--storeRaw"))
    run("pyserini-search", module("pyserini.search.lucene", "--index", ROOT / "indexes/pyserini-msmarco", "--topics", queries, "--output", RUNS / "pyserini-bm25.trec", "--hits", "1000", "--bm25", "--k1", "0.82", "--b", "0.68", "--threads", "4", "--batch-size", "16"))
    evaluate("pyserini-bm25-eval", qrels, RUNS / "pyserini-bm25.trec", {"map": 0.1958, "recall_1000": 0.8573}, "-m", "map", "-m", "recall.1000")
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    from pyserini.index.lucene import LuceneIndexReader
    from pyserini.search.lucene import LuceneSearcher
    reader = LuceneIndexReader(str(index))
    analyzer = Analyzer(get_lucene_analyzer())
    searcher = LuceneSearcher(str(index))
    assert searcher.num_docs == 8841823
    query = "what is paula deen's brother"
    terms = set(analyzer.analyze(query))
    weights = {term: reader.compute_bm25_term_weight("7187158", term, analyzer=None) for term in reader.get_document_vector("7187158") if term in terms}
    score = sum(weights.values())
    assert math.isclose(score, 17.949487686157227, abs_tol=2e-5)
    assert math.isclose(score, reader.compute_query_document_score("7187158", query), abs_tol=2e-5)
    assert searcher.search(query)[0].docid == "7187158"
    assert not searcher.search("qzxwxyzabcinvented")
    (EVIDENCE / "bm25-vector-proof.json").write_text(json.dumps({"query": query, "weights": weights, "dot_product": score}, indent=2) + "\n")
    searcher.close()


def msmarco_dense():
    data = msmarco_data(full=False)
    queries, qrels = data / "queries.dev.small.tsv", data / "qrels.trec"
    inverted = prebuilt_index("msmarco-v1-passage", "msmarco-v1-passage-inverted.json")
    run("prebuilt-bm25-search", java("io.anserini.search.SearchCollection", "-index", inverted, "-topics", queries, "-topicReader", "TsvInt", "-output", RUNS / "prebuilt-bm25.trec", "-parallelism", "4", "-bm25", "-bm25.k1", "0.82", "-bm25.b", "0.68", "-hits", "1000"))
    evaluate("prebuilt-bm25-mrr", qrels, RUNS / "prebuilt-bm25.trec", {"recip_rank": 0.1875}, "-M", "10", "-m", "recip_rank")
    dense = prebuilt_index("msmarco-v1-passage.bge-base-en-v1.5.hnsw", "msmarco-v1-passage-hnsw.json")
    run("prebuilt-bge-search", java("io.anserini.search.SearchHnswDenseVectors", "-index", dense, "-topics", queries, "-topicReader", "TsvInt", "-output", RUNS / "prebuilt-bge.trec", "-encoder", "BgeBaseEn15", "-hits", "1000", "-threads", "4"))
    evaluate("prebuilt-bge-mrr", qrels, RUNS / "prebuilt-bge.trec", {"recip_rank": 0.3521}, "-M", "10", "-m", "recip_rank")


def nfcorpus():
    from huggingface_hub import snapshot_download
    import faiss
    import numpy as np
    archive = ROOT / "nfcorpus.zip"
    download("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip", archive)
    assert hashlib.file_digest(archive.open("rb"), "sha256").hexdigest() == "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b"
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(ROOT / "data")
    data = ROOT / "data/nfcorpus"
    model = snapshot_download("BAAI/bge-base-en-v1.5", revision=MODEL_REVISION, allow_patterns=["config.json", "model.safetensors", "tokenizer*", "special_tokens_map.json", "vocab.txt"])
    docs = [json.loads(line) for line in (data / "corpus.jsonl").open()]
    assert len(docs) == 3633
    with (data / "queries.tsv").open("w") as out:
        for line in (data / "queries.jsonl").open():
            query = json.loads(line)
            out.write(query["_id"] + "\t" + query["text"] + "\n")
    with (data / "qrels/test.tsv").open() as source, (data / "qrels/test.qrels").open("w") as out:
        next(source)
        for line in source:
            qid, docid, grade = line.split()
            out.write(f"{qid}\tQ0\t{docid}\t{grade}\n")
    dense = ROOT / "indexes/nfcorpus.bge"
    run("bge-encode", module("pyserini.encode", "input", "--corpus", data / "corpus.jsonl", "--fields", "title", "text", "output", "--embeddings", dense, "--to-faiss", "encoder", "--encoder", model, "--l2-norm", "--device", "cpu", "--pooling", "mean", "--fields", "title", "text", "--batch", "32"))
    run("bge-search", module("pyserini.search.faiss", "--encoder-class", "auto", "--encoder", model, "--l2-norm", "--pooling", "mean", "--index", dense, "--topics", data / "queries.tsv", "--output", RUNS / "nfcorpus-bge.trec", "--batch", "128", "--threads", "4", "--hits", "1000"))
    evaluate("nfcorpus-bge-eval", data / "qrels/test.qrels", RUNS / "nfcorpus-bge.trec", {"ndcg_cut_10": 0.3808}, "-m", "ndcg_cut.10")
    corpus = data / "pyserini-corpus"
    corpus.mkdir()
    with (corpus / "corpus.jsonl").open("w") as out:
        for doc in docs:
            out.write(json.dumps({"id": doc["_id"], "contents": doc["title"] + " " + doc["text"]}) + "\n")
    sparse = ROOT / "indexes/nfcorpus.bm25"
    run("nfcorpus-bm25-index", module("pyserini.index.lucene", "--collection", "JsonCollection", "--input", corpus, "--index", sparse, "--generator", "DefaultLuceneDocumentGenerator", "--threads", "4", "--storePositions", "--storeDocvectors", "--storeRaw"))
    run("nfcorpus-bm25-search", module("pyserini.search.lucene", "--index", sparse, "--topics", data / "queries.tsv", "--output", RUNS / "nfcorpus-bm25.trec", "--hits", "1000", "--bm25", "--threads", "4", "--batch-size", "16"))
    evaluate("nfcorpus-bm25-eval", data / "qrels/test.qrels", RUNS / "nfcorpus-bm25.trec", {"ndcg_cut_10": 0.3218}, "-m", "ndcg_cut.10")
    from pyserini.encode import AutoQueryEncoder
    from pyserini.search.faiss import FaissSearcher
    vectors = faiss.read_index(str(dense / "index"))
    assert vectors.ntotal == 3633 and vectors.d == 768
    embeddings = vectors.reconstruct_n(0, vectors.ntotal)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
    docids = (dense / "docid").read_text().splitlines()
    query = "How to Help Prevent Abdominal Aortic Aneurysms"
    encoder = AutoQueryEncoder(model, device="cpu", pooling="mean", l2_norm=True)
    query_vector = encoder.encode(query)
    scores = embeddings @ query_vector
    top = np.argsort(-scores)[:10]
    hits = FaissSearcher(str(dense), encoder).search(query, k=10)
    assert [docids[i] for i in top] == [hit.docid for hit in hits]
    score = float(scores[docids.index("MED-4555")])
    assert math.isclose(score, 0.7913785, abs_tol=2e-5), score
    (EVIDENCE / "dense-vector-proof.json").write_text(json.dumps({"query": query, "documents": 3633, "dimensions": 768, "model_revision": MODEL_REVISION, "MED-4555_dot_product": score, "top_ids": [docids[i] for i in top]}, indent=2) + "\n")


if __name__ == "__main__":
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Course computation is allowed only on GitHub Actions; do not run on the laptop CPU.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=["nfcorpus", "msmarco-bm25", "msmarco-dense"])
    part = parser.parse_args().part
    prepare()
    try:
        {"nfcorpus": nfcorpus, "msmarco-bm25": msmarco_bm25, "msmarco-dense": msmarco_dense}[part]()
    finally:
        (EVIDENCE / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
