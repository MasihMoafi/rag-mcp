"""Run EnterpriseRAG-Bench search on this machine with hosted embedding and reranking.

Downloads only the named slices; the key comes from rag's saved settings into this process and
is never printed or written to the run's files.

    python -m evals.benchmarks.local --work DIR --scratch DIR --commit SHA
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import datasets
from .retrieve import main as retrieve_main

REPOSITORY = "onyx-dot-app/EnterpriseRAG-Bench"
SCHEMA = Path(__file__).with_name("schemas") / "enterprise-v1.0.0.json"


def release_download(url, target):
    """GitHub's direct release links fail on this network; `gh` works, without the proxy."""
    if "/releases/download/" not in url:
        return plain_download(url, target)
    tag, name = url.rsplit("/", 2)[-2:]
    environ = {
        k: v
        for k, v in os.environ.items()
        if k.lower() not in ("https_proxy", "http_proxy", "all_proxy")
    }
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            tag,
            "-R",
            REPOSITORY,
            "-p",
            name,
            "-O",
            str(target),
            "--clobber",
        ],
        check=True,
        env=environ,
        timeout=600,
    )
    return None


plain_download = datasets.download


def load_key(path):
    """The OpenRouter key from rag's saved settings; only this process's environment gets it."""
    from rag.settings import resolve_openrouter_key

    key = resolve_openrouter_key(json.loads(Path(path).read_text()))
    if not key:
        raise ValueError("No OpenRouter key saved; run `rag key set`")
    os.environ["OPENROUTER_API_KEY"] = key


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True, help="results folder")
    parser.add_argument(
        "--scratch", type=Path, required=True, help="dataset and index folder"
    )
    parser.add_argument(
        "--commit", required=True, help="the rag-mcp commit being measured"
    )
    parser.add_argument("--limit", type=int, default=104)
    parser.add_argument("--slice", action="append", default=["github", "confluence"])
    parser.add_argument(
        "--config", type=Path, default=Path.home() / ".config/rag/config.json"
    )
    args = parser.parse_args()
    load_key(args.config)
    os.environ["RAG_BENCH_SCRATCH"] = str(args.scratch)
    source, bundle = args.scratch / "dataset_source", args.scratch / "bundle"
    if not bundle.exists():
        if not source.exists():
            datasets.download = release_download
            datasets.prepare("enterprise", source, args.slice, "v1.0.0", None, [])
        datasets.normalize(source, SCHEMA, bundle, args.limit)
    args.work.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "retrieve",
        "--bundle",
        str(bundle),
        "--output",
        str(args.work),
        "--arm",
        "bm25",
        "--arm",
        "hybrid_api",
        "--rag-commit",
        args.commit,
    ]
    retrieve_main()


if __name__ == "__main__":
    main()
