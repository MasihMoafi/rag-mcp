"""Regrade saved answers for correctness with EnterpriseRAG-Bench's own judge prompt.

Calls go through the signed-in Codex CLI. Results append to grades.jsonl and a rerun skips
finished questions. Answer-fact completeness and Onyx's gold-document correction are not run.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import TimeoutExpired
from threading import Lock

from .answers_codex import DEFAULT_JUDGE, run_codex
from .common import write_json

# Verbatim from onyx-dot-app/EnterpriseRAG-Bench (MIT), src/prompts/answer_evaluation.py.
ONYX_CORRECTNESS_PROMPT = """
You are a wholistic and detail-oriented answer evaluator. Given a query, a gold answer, and a candidate answer, evaluate if the candidate answer aligned with the gold answer.
Use the following metrics for evaluating the answer:
- The candidate answer must provide loosely the same information as the gold answer. The core aspects directly asked by the query must be addressed in the candidate answer and they must not conflict with the gold answer.
- If there are any specific quantities mentioned in both answers, they must match.
- The candidate answer is not required to contain all of the same details as the gold answer.
- The candidate answer must address the key parts of the query, if it is missing anything critical to the question, it is misaligned.
- The candidate answer may contain more details, richer information, or other helpful relevant information than the gold answer, this is ok.
- The candidate answer may offer up additional loosely related information that adds to the context of the answer, this is ok as long as it does not lead the user to an incorrect conclusion (compared to the gold answer).
- Do not penalize the candidate answer for stylistic differences. If the candidate answer offers follow up questions, asks additional clarifications to the user, or offers additional context, \
this is ok as long as it contains the necessary information to answer the question.

There is a separate check for answer completeness, this is not in scope for this evaluation. However, if there are core parts of the question being left out, this is misaligned.

## Query
```
{query}
```

## Gold Answer
```
{gold_answer}
```

## Candidate Answer
```
{candidate_answer}
```

## Output Format
Output a JSON with "reason" and "aligned" fields. The "reason" field should be a as concise as possible (max 1 sentence) explanation of why the candidate answer is aligned or misaligned with the gold answer. \
The "aligned" field should be a simple "yes" or "no", use only those two strings literally and nothing else.

CRITICAL: Output only a JSON object with the following fields in the order shown below (with no additional text or formatting):
{{
  "reason": "reason for the classification",
  "aligned": "yes or no"
}}
""".strip()

SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "aligned": {"type": "string"}},
    "required": ["reason", "aligned"],
    "additionalProperties": False,
}
CITATIONS = re.compile(r"\[(?:\s*dsid_[0-9a-f]+\s*[,;]?\s*)+\]|dsid_[0-9a-f]+")


def strip_citations(text):
    return re.sub(r" {2,}", " ", CITATIONS.sub("", text)).replace(" .", ".").strip()


def grade(row, question, effort):
    prompt = ONYX_CORRECTNESS_PROMPT.format(
        query=question["question"],
        gold_answer=question["gold_answer"],
        candidate_answer=strip_citations(row["answer"]["text"]),
    )
    reply = run_codex(prompt, SCHEMA, DEFAULT_JUDGE, effort)
    return {
        "question_id": row["question_id"],
        "question_type": question["question_type"],
        "old_label": row["label"],
        "aligned": re.search(r"\byes\b", reply["aligned"], re.IGNORECASE) is not None,
        "reason": reply["reason"],
    }


def summarize(rows):
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row["aligned"])
    percent = {k: 100 * sum(v) / len(v) for k, v in sorted(by_type.items())}
    return {
        "questions": len(rows),
        "correct_percent": round(100 * sum(r["aligned"] for r in rows) / len(rows), 1),
        "category_average_percent": round(sum(percent.values()) / len(percent), 1),
        "by_category": {k: round(v, 1) for k, v in percent.items()},
        "correct_by_old_label": dict(
            Counter(r["old_label"] for r in rows if r["aligned"])
        ),
        "old_label_counts": dict(Counter(r["old_label"] for r in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10**9)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--effort", default="medium")
    args = parser.parse_args()

    questions = {
        q["question_id"]: q
        for q in map(json.loads, args.questions.read_text().splitlines())
    }
    answers = [json.loads(l) for l in args.answers.read_text().splitlines()]
    answers = [a for a in answers if a["status"] == "ok"][: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "grades.jsonl"
    done = (
        [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    )
    seen = {row["question_id"] for row in done}
    lock = Lock()

    def work(row):
        result = None
        for _ in range(3):
            try:
                result = grade(row, questions[row["question_id"]], args.effort)
                break
            except (RuntimeError, ValueError, KeyError, OSError, TimeoutExpired) as error:
                print(row["question_id"], type(error).__name__, flush=True)
        if result is None:
            return
        with lock:
            done.append(result)
            with path.open("a") as stream:
                stream.write(json.dumps(result) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, [a for a in answers if a["question_id"] not in seen]))
    summary = summarize(done)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
