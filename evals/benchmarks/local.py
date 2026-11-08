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


def gh(*args, **options):
    """Plain GitHub URLs return 503 on this network; `gh` without the proxy works."""
    environ = {
        k: v
        for k, v in os.environ.items()
        if k.lower() not in ("https_proxy", "http_proxy", "all_proxy")
    }
    return subprocess.run(
        ["gh", *args],
        check=True,
        env=environ,
        timeout=600,
        capture_output=True,
        **options,
    ).stdout


def gh_json(url):
    return json.loads(gh("api", url.removeprefix("https://api.github.com/")))


def gh_download(url, target):
    if "/releases/download/" in url:
        tag, name = url.rsplit("/", 2)[-2:]
        gh(
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
        )
    else:
        commit, name = url.rsplit("/", 2)[-2:]
        Path(target).write_bytes(
            gh(
                "api",
                "-H",
                "Accept: application/vnd.github.raw",
                f"repos/{REPOSITORY}/contents/{name}?ref={commit}",
            )
        )


def pin_embedding_provider(provider="DeepInfra"):
    """OpenRouter sends Qwen3 8B to the first provider it picks, and Nebius's quota runs out at
    once. Same price at DeepInfra; no fallback, because SiliconFlow charges four times as much."""
    from openai.resources.embeddings import Embeddings

    plain_create = Embeddings.create

    def create(self, **kwargs):
        kwargs.setdefault(
            "extra_body", {"provider": {"order": [provider], "allow_fallbacks": False}}
        )
        return plain_create(self, **kwargs)

    Embeddings.create = create


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
    parser.add_argument(
        "--run", default="full", help="subfolder of --scratch for this run"
    )
    parser.add_argument(
        "--distractors",
        type=int,
        help="pipeline check: gold documents plus this many others",
    )
    parser.add_argument("--slice", action="append", default=["github", "confluence"])
    parser.add_argument(
        "--config", type=Path, default=Path.home() / ".config/rag/config.json"
    )
    args = parser.parse_args()
    load_key(args.config)
    # Hosted embedding providers limit tokens per minute; a long index waits instead of failing.
    from rag.embeddings import OpenAICompatibleWrapper

    OpenAICompatibleWrapper._create_with_retry.__defaults__ = ((10,) * 360,)
    pin_embedding_provider()
    # This machine sets ALL_PROXY to a socks:// URL, which httpx refuses; HTTPS_PROXY still applies.
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)
    source = args.scratch / "dataset_source"
    bundle = args.scratch / args.run / "bundle"
    os.environ["RAG_BENCH_SCRATCH"] = str(args.scratch / args.run)
    if not bundle.exists():
        if not source.exists():
            datasets.download, datasets.get_json = gh_download, gh_json
            datasets.prepare("enterprise", source, args.slice, "v1.0.0", None, [])
        bundle.parent.mkdir(parents=True, exist_ok=True)
        datasets.normalize(source, SCHEMA, bundle, args.limit, args.distractors)
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
