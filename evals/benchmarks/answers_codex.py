"""Answer and grade saved Kaggle search results locally through a signed-in Codex CLI.

Calls use the ChatGPT subscription: every *_API_KEY is removed from the child environment, so
Codex cannot bill an API account. Results append to answers.jsonl and a rerun skips finished
questions. Run a few first:

    python -m evals.benchmarks.answers_codex --retrieval retrieval.jsonl --arm hybrid_qwen \
        --output answers-pilot --limit 5
"""

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from .answers import answer_prompt, grade_prompt, validate_answer
from .common import ANSWER_PASSAGES, read_ledger, write_json
from .quality import QUALITY_SCHEMA, mean_scores, quality_prompt, quality_scores

# Masih, 8 Nov: Sol writes and judges, each call in its own session.
DEFAULT_JUDGE = "gpt-6.1-sol"
# Masih, 8 Nov: a second, blind Sol grades a sample to check the judge.
BLIND_JUDGE = "gpt-6.1-sol"
# Scores from different judge wording must not be compared; every result records this.
PROMPTS_SHA256 = hashlib.sha256(
    "".join(
        inspect.getsource(f) for f in (answer_prompt, grade_prompt, quality_prompt)
    ).encode()
).hexdigest()

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "citations"],
    "additionalProperties": False,
}
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["correct", "partial", "wrong"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}
MAX_CONSECUTIVE_FAILURES = 3
PROXY_VARIABLES = {"http_proxy", "https_proxy", "all_proxy"}


def subscription_env(environ):
    """The child environment without API keys or proxies: only the sign-in, never the paid VPN."""
    return {
        key: value
        for key, value in environ.items()
        if not key.endswith("_API_KEY") and key.lower() not in PROXY_VARIABLES
    }


def run_codex(prompt, schema, model, effort):
    """One read-only Codex turn in an empty folder; returns the parsed JSON reply."""
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("Codex is not installed or is not on PATH")
    with tempfile.TemporaryDirectory(prefix="rag-bench-codex-") as directory:
        folder = Path(directory)
        (folder / "schema.json").write_text(json.dumps(schema))
        completed = subprocess.run(
            [
                executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                model,
                "-c",
                f"model_reasoning_effort={effort}",
                "--cd",
                directory,
                "--output-schema",
                str(folder / "schema.json"),
                "--output-last-message",
                str(folder / "reply.json"),
                prompt,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
            env=subscription_env(os.environ),
        )
        if completed.returncode != 0 or not (folder / "reply.json").exists():
            raise RuntimeError(f"Codex failed with exit code {completed.returncode}")
        return json.loads((folder / "reply.json").read_text())


def bundle_documents(bundle):
    """Document id -> full text, from a normalized bundle (documents.json and corpus/)."""
    bundle = Path(bundle)
    mapping = json.loads((bundle / "documents.json").read_text())
    return {
        key: (bundle / "corpus" / name).read_text() for name, key in mapping.items()
    }


def whole_documents(passages, documents, max_chars):
    """A short document replaces its passages, once, at its first rank; long ones keep passages.

    The search finds the right document but often not the part that answers (9 Nov analysis).
    """
    whole, seen = [], set()
    for passage in passages:
        text = documents.get(passage["doc_id"])
        if text is None or len(text) > max_chars:
            whole.append(passage)
        elif passage["doc_id"] not in seen:
            seen.add(passage["doc_id"])
            whole.append(dict(passage, text=text, location="whole document"))
    return whole


def load_jobs(retrieval, arm, limit, questions=None, documents=None, max_chars=8000):
    """Completed searches for one arm; `questions` is the dataset's file, for facts and gold documents.

    With `documents` (id -> text), short documents are read whole (see whole_documents).
    """
    _, events = read_ledger(retrieval)
    labels = {}
    if questions:
        for line in Path(questions).read_text().splitlines():
            row = json.loads(line)
            labels[row["question_id"]] = row
    jobs = {}
    for event in events:
        if event.get("event") == "complete" and event.get("arm") == arm:
            label = labels.get(event["question_id"], {})
            jobs[event["question_id"]] = {
                "id": event["question_id"],
                "arm": arm,
                "question": event["question"],
                "gold_answer": event["gold_answer"],
                "passages": whole_documents(
                    event["passages"][:ANSWER_PASSAGES], documents, max_chars
                )
                if documents
                else event["passages"][:ANSWER_PASSAGES],
                "answer_facts": label.get("answer_facts", []),
                "gold_doc_ids": label.get("expected_doc_ids", []),
            }
    if not jobs:
        raise ValueError(f"No completed questions for arm {arm!r}")
    return list(jobs.values())[:limit]


def agreement(rows):
    """How often a blind second grader matches the judge: raw share and Cohen's kappa."""
    pairs = [
        (row["label"], row["blind_label"]) for row in rows if row.get("blind_label")
    ]
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    chance = sum(first[k] * second[k] for k in first) / len(pairs) ** 2
    kappa = 1.0 if chance == 1 else (observed - chance) / (1 - chance)
    return {
        "graded_twice": len(pairs),
        "agreement": round(observed, 4),
        "cohen_kappa": round(kappa, 4),
    }


def blind_sample_ids(jobs, size):
    """A fixed sample that does not depend on question order."""
    return set(
        sorted(
            (job["id"] for job in jobs),
            key=lambda q: hashlib.sha256(q.encode()).hexdigest(),
        )[:size]
    )


def answer_one(job, call, model, effort, judge, blind_judge=None):
    """Answer with `model`, then grade and score with `judge`; an invalid answer gets one retry.

    `blind_judge` grades again in a separate session that sees only the question, the reference
    answer and the candidate: not the first grade, the arm or which model wrote the answer."""
    started = time.monotonic()
    for attempt in (1, 2):
        answer = call(answer_prompt(job), ANSWER_SCHEMA, model, effort)
        try:
            validate_answer(answer, job)
            break
        except ValueError:
            if attempt == 2:
                raise
    if judge is None:  # answer only; grading runs separately (regrade_onyx)
        grade, verdict = {"label": None, "reason": None}, None
    else:
        grade = call(grade_prompt(job, answer), GRADE_SCHEMA, judge, effort)
        verdict = call(quality_prompt(job, answer), QUALITY_SCHEMA, judge, effort)
    blind = (
        call(grade_prompt(job, answer), GRADE_SCHEMA, blind_judge, effort)
        if blind_judge
        else None
    )
    return {
        "blind_label": blind["label"] if blind else None,
        "blind_judge": blind_judge if blind else None,
        "prompts_sha256": PROMPTS_SHA256,
        "question_id": job["id"],
        "arm": job["arm"],
        "status": "ok",
        "answer": answer,
        "label": grade["label"],
        "reason": grade["reason"],
        "quality": quality_scores(verdict, job, answer) if verdict else {},
        "verdict": verdict,
        "model": model,
        "judge": judge,
        "effort": effort,
        "seconds": round(time.monotonic() - started, 1),
    }


def summarize(rows, jobs):
    finished = [row for row in rows if row["status"] == "ok"]
    labels = Counter(row["label"] for row in finished)
    return {
        "questions": len(jobs),
        "answered": len(finished),
        "failed": len(
            {row["question_id"] for row in rows if row["status"] == "error"}
            - {row["question_id"] for row in finished}
        ),
        "correct": labels["correct"],
        "partial": labels["partial"],
        "wrong": labels["wrong"],
        "accuracy": round(labels["correct"] / len(finished), 4) if finished else None,
        "quality": mean_scores(finished),
        "judge_agreement": agreement(finished),
        "prompts_sha256": PROMPTS_SHA256,
    }


def run(
    jobs,
    output,
    model,
    effort,
    workers,
    call=run_codex,
    judge=DEFAULT_JUDGE,
    blind_judge=BLIND_JUDGE,
    blind_sample=0,
):
    """Never raises once calls have started: failures are recorded and the summary still written."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "answers.jsonl"
    rows = (
        [json.loads(line) for line in ledger.read_text().splitlines()]
        if ledger.exists()
        else []
    )
    done = {(row["arm"], row["question_id"]) for row in rows if row["status"] == "ok"}
    pending = [job for job in jobs if (job["arm"], job["id"]) not in done]
    lock = Lock()
    failures = [0]
    sampled = blind_sample_ids(jobs, blind_sample)

    def work(job):
        if failures[0] >= MAX_CONSECUTIVE_FAILURES:
            return
        try:
            row = answer_one(
                job,
                call,
                model,
                effort,
                judge,
                blind_judge if job["id"] in sampled else None,
            )
        except Exception as error:  # noqa: BLE001 - runbook: record and stop, never lose finished results
            row = {
                "question_id": job["id"],
                "arm": job["arm"],
                "status": "error",
                "error": f"{type(error).__name__}: {str(error)[:300]}",
            }
        with lock:
            failures[0] = 0 if row["status"] == "ok" else failures[0] + 1
            rows.append(row)
            with ledger.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, pending))
    summary = summarize(rows, jobs)
    summary["stopped_early"] = failures[0] >= MAX_CONSECUTIVE_FAILURES
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--arm", default="hybrid_qwen")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--blind-judge", default=BLIND_JUDGE)
    parser.add_argument("--blind-sample", type=int, default=20)
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="write answers without our judge; grade them with regrade_onyx",
    )
    parser.add_argument(
        "--questions",
        help="the dataset's questions.jsonl, for facts and gold documents",
    )
    parser.add_argument(
        "--whole-documents",
        metavar="BUNDLE",
        help="normalized bundle of the searched corpus: read short documents whole",
    )
    parser.add_argument("--whole-document-chars", type=int, default=8000)
    args = parser.parse_args()
    jobs = load_jobs(
        args.retrieval,
        args.arm,
        args.limit,
        args.questions,
        bundle_documents(args.whole_documents) if args.whole_documents else None,
        args.whole_document_chars,
    )
    print(
        json.dumps(
            run(
                jobs,
                args.output,
                args.model,
                args.effort,
                args.workers,
                judge=None if args.answer_only else args.judge,
                blind_judge=args.blind_judge,
                blind_sample=0 if args.answer_only else args.blind_sample,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
