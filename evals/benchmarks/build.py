"""Build self-contained Kaggle GPU notebooks and small hosted-answer tasks offline."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .answers import validate_answer_ledger
from .common import digest, read_ledger, write_json

HERE = Path(__file__).resolve().parent


def bootstrap(names, paidrun=False):
    sources = {"evals/__init__.py": "", "evals/benchmarks/__init__.py": ""}
    sources.update(
        {
            f"evals/benchmarks/{name}.py": (HERE / f"{name}.py").read_text()
            for name in names
        }
    )
    if paidrun:
        sources["evals/paidrun.py"] = (HERE.parent / "paidrun.py").read_text()
    return f"""import sys
import os
from pathlib import Path
SOURCES = {sources!r}
working = Path(os.environ.get('RAG_BENCH_WORKING_DIR', '/kaggle/working'))
source_root = working / 'rag_bench_source'
for name, content in SOURCES.items():
    target = source_root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
sys.path.insert(0, str(source_root))
"""


def build_answer_task(
    retrieval,
    retrieval_manifest,
    *,
    arm,
    model,
    judge,
    task,
    ceiling,
    call_budget=0.05,
    limit=30,
    offset=0,
    resume_from=None,
):
    if not 1 <= limit <= 30 or offset < 0:
        raise ValueError("Hosted tasks require 1..30 questions and nonnegative offset")
    if model == judge:
        raise ValueError("Select a distinct second model for grading")
    if (
        not all(math.isfinite(v) and v > 0 for v in (ceiling, call_budget))
        or call_budget < 0.01
    ):
        raise ValueError(
            "Positive finite spend ceiling and call budget >= $0.01 required"
        )
    raw, events = read_ledger(retrieval)
    source_manifest = json.loads(Path(retrieval_manifest).read_text())
    expected_identity = source_manifest.pop("identity")
    source_manifest.pop("parent_ledger_sha256", None)
    if digest(source_manifest) != expected_identity:
        raise ValueError("Retrieval manifest identity mismatch")
    if any(e.get("identity") != expected_identity for e in events):
        raise ValueError("Retrieval evidence belongs to another manifest")
    results = {}
    for event in events:
        if event.get("event") == "complete" and event["arm"] == arm:
            if event["question_id"] in results:
                raise ValueError("Duplicate completed retrieval question")
            if not isinstance(event.get("passages"), list) or any(
                not isinstance(p.get("text"), str) or not p.get("doc_id")
                for p in event["passages"]
            ):
                raise ValueError("Retrieval passages require document ids and text")
            results[event["question_id"]] = event
    jobs = [
        {
            "id": e["question_id"],
            "arm": arm,
            "question": e["question"],
            "gold_answer": e["gold_answer"],
            "passages": e["passages"][:5],
            "caveat": e.get("caveat"),
        }
        for e in list(results.values())[offset : offset + limit]
    ]
    if not jobs:
        raise ValueError("No completed questions in the selected arm/shard")
    protocol = {
        "schema": "rag-bench.answers.v1",
        "answer_model": model,
        "judge_model": judge,
        "retrieval_manifest": source_manifest,
        "retrieval_identity": expected_identity,
        "jobs": jobs,
        "call_budget_usd": call_budget,
        "call_settings": {
            "temperature": 1,
            "max_completion_tokens": 1024,
            "timeout": 60,
        },
        "source_sha256": {
            name: hashlib.sha256((HERE / f"{name}.py").read_bytes()).hexdigest()
            for name in ("common", "answers")
        },
    }
    identity = digest(protocol)
    seed, prior = read_ledger(resume_from) if resume_from else (b"", [])
    allowed = {(job["arm"], job["id"]) for job in jobs}
    if any(e.get("identity") != identity for e in prior):
        raise ValueError("Resume parent differs from the exact answer protocol/shard")
    if any(
        (e["arm"], e["question_id"]) not in allowed
        for e in prior
        if e.get("question_id") is not None
    ):
        raise ValueError("Resume parent contains questions outside this task")
    manifest = dict(
        protocol,
        identity=identity,
        incremental_ceiling_usd=ceiling,
        retrieval_jsonl_sha256=hashlib.sha256(raw).hexdigest(),
        parent_sha256=hashlib.sha256(seed).hexdigest() if seed else None,
        caveat=source_manifest.get("dataset", {}).get("caveat"),
    )
    validate_answer_ledger(prior, jobs, manifest)
    code = (
        bootstrap(("common", "answers"))
        + f"""
import kaggle_benchmarks as kbench
from evals.benchmarks.answers import run_answers
from evals.benchmarks.common import run_lock
MANIFEST = {manifest!r}
JOBS = {jobs!r}
RESUME_LEDGER = {seed!r}

@kbench.task(name={task!r})
def answer_task(llm):
    output = working / 'answer_results'
    with run_lock(output):
        run_answers(kbench, llm, kbench.llms[{judge!r}], JOBS, output, MANIFEST, RESUME_LEDGER)

answer_task.run(kbench.llms[{model!r}])
"""
    )
    compile(code, "<hosted-task>", "exec")
    return code


def build_notebook(
    output,
    owner,
    slug,
    dataset="enterprise",
    slices=(),
    revision="main",
    hf_id=None,
    files=(),
    schema=None,
    limit=30,
    embedding_size="0.6B",
    candidates=50,
    resume_from=None,
    distractors=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    schema_value = json.loads(Path(schema).read_text()) if schema else None
    seed, _ = read_ledger(resume_from) if resume_from else (b"", [])
    if seed and schema is None:
        raise ValueError(
            "Retrieval resume requires a schema and the same dataset/model flags"
        )
    code = (
        bootstrap(("common", "datasets", "retrieve"), paidrun=True)
        + f"""
import json
import os
from evals.benchmarks.datasets import prepare, normalize
from evals.benchmarks.common import write_json
import shutil
# Large data (archives, corpus, indexes) stays outside /kaggle/working: everything there is
# saved as notebook output, and `kaggle kernels output` would download it to Masih's machine.
scratch = Path('/kaggle/temp' if Path('/kaggle/temp').is_dir() else '/tmp') / 'rag-bench'
os.environ['RAG_BENCH_SCRATCH'] = str(scratch)
source = scratch / 'dataset_source'
prepare({dataset!r}, source, {list(slices)!r}, {revision!r}, {hf_id!r}, {list(files)!r})
info = working / 'dataset_info'
info.mkdir(exist_ok=True)
for name in ('README.md', 'inspection.json', 'source.json'):
    shutil.copyfile(source / name, info / name)
print((source / 'README.md').read_text())
print((source / 'inspection.json').read_text())
SCHEMA = {schema_value!r}
if SCHEMA is not None:
    schema_path = working / 'schema.json'
    write_json(schema_path, SCHEMA)
    bundle = scratch / 'bundle'
    normalize(source, schema_path, bundle, {limit!r}, {distractors!r})
    result = working / 'retrieval_results'
    result.mkdir(exist_ok=True)
    seed = {seed!r}
    if seed:
        with (result / 'retrieval.jsonl').open('xb') as stream:
            stream.write(seed)
    # main() performs preflight before entering the durable retrieval loop.
    from evals.benchmarks.retrieve import main
    sys.argv = ['retrieve', '--bundle', str(bundle), '--output', str(result),
                '--embedding-size', {embedding_size!r}, '--candidates', {str(candidates)!r}]
    main()
"""
    )
    install = """import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install',
    'ragmcp[full] @ git+https://github.com/MasihMoafi/rag-mcp',
    'huggingface_hub', 'pyarrow'], check=True)
"""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "cell_type": "code",
                "id": name,
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": content.splitlines(keepends=True),
            }
            for name, content in (("install", install), ("benchmark", code))
        ],
    }
    for cell in notebook["cells"]:
        compile("".join(cell["source"]), "<notebook-cell>", "exec")
    write_json(output / "benchmark.ipynb", notebook)
    write_json(
        output / "kernel-metadata.json",
        {
            "id": f"{owner}/{slug}",
            "title": slug,
            "code_file": "benchmark.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    notebook = sub.add_parser("notebook")
    notebook.add_argument("--output", type=Path, required=True)
    notebook.add_argument("--owner", required=True)
    notebook.add_argument("--slug", required=True)
    notebook.add_argument(
        "--dataset",
        choices=("enterprise", "freshstack", "frames"),
        default="enterprise",
    )
    notebook.add_argument("--slice", action="append", default=[])
    notebook.add_argument("--revision", default="main")
    notebook.add_argument("--hf-id")
    notebook.add_argument("--file", action="append", default=[])
    notebook.add_argument("--schema", type=Path)
    notebook.add_argument("--limit", type=int, default=30)
    notebook.add_argument(
        "--distractors",
        type=int,
        help="pipeline check: gold documents plus this many others",
    )
    notebook.add_argument(
        "--embedding-size", choices=("0.6B", "4B", "8B"), default="0.6B"
    )
    notebook.add_argument("--candidates", type=int, default=50)
    notebook.add_argument("--resume-from", type=Path)
    answers = sub.add_parser("answers")
    answers.add_argument("--retrieval", type=Path, required=True)
    answers.add_argument("--retrieval-manifest", type=Path, required=True)
    answers.add_argument("--arm", choices=("bm25", "hybrid_qwen"), required=True)
    answers.add_argument("--model", required=True)
    answers.add_argument("--judge", required=True)
    answers.add_argument("--task", required=True)
    answers.add_argument("--ceiling", type=float, required=True)
    answers.add_argument("--call-budget", type=float, default=0.05)
    answers.add_argument("--limit", type=int, default=30)
    answers.add_argument("--offset", type=int, default=0)
    answers.add_argument("--resume-from", type=Path)
    answers.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command, output = args.pop("command"), args.pop("output")
    if command == "notebook":
        args["slices"] = args.pop("slice")
        args["files"] = args.pop("file")
        build_notebook(output, **args)
    else:
        code = build_answer_task(**args)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(code)


if __name__ == "__main__":
    main()
