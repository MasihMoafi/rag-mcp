"""Search report from saved results: ranking metrics, latency, and a paired test between two arms.

python -m evals.benchmarks.report --retrieval retrieval.jsonl --questions questions.jsonl
"""

import argparse
import json
import math
from pathlib import Path

from .common import read_ledger, retrieval_metrics

KEYS = ("recall@5", "recall@10", "precision@5", "MAP", "nDCG@10", "MRR")


def ranking_metrics(ranked, gold):
    """retrieval_metrics plus precision@5 and average precision over the whole ranking."""
    scores = dict(retrieval_metrics(ranked, gold))
    ranked = list(dict.fromkeys(ranked))
    wanted = set(gold)
    hits = [doc in wanted for doc in ranked]
    scores["precision@5"] = sum(hits[:5]) / 5
    found, total = 0, 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:
            found += 1
            total += found / rank
    scores["MAP"] = total / len(wanted)
    return scores


def percentile(values, share):
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def sign_test(first, second):
    """Exact two-sided sign test on paired scores; ties are dropped."""
    wins = sum(a > b for a, b in zip(first, second, strict=True))
    losses = sum(a < b for a, b in zip(first, second, strict=True))
    trials = wins + losses
    if not trials:
        return {"first_better": 0, "second_better": 0, "p": 1.0}
    tail = sum(math.comb(trials, k) for k in range(min(wins, losses) + 1)) / 2**trials
    return {"first_better": wins, "second_better": losses, "p": min(1.0, 2 * tail)}


def build_report(events, gold):
    arms = {}
    for event in events:
        if event.get("event") == "complete" and gold.get(event["question_id"]):
            arms.setdefault(event["arm"], {})[event["question_id"]] = event
    report = {"arms": {}}
    for arm, done in arms.items():
        rows = [ranking_metrics(e["ranked_doc_ids"], gold[q]) for q, e in done.items()]
        latency = [e["latency_seconds"] for e in done.values()]
        report["arms"][arm] = {
            "questions": len(rows),
            **{key: round(sum(r[key] for r in rows) / len(rows), 4) for key in KEYS},
            "latency_p50_s": round(percentile(latency, 0.5), 3),
            "latency_p95_s": round(percentile(latency, 0.95), 3),
        }
    names = sorted(arms)
    if len(names) == 2:
        shared = sorted(set(arms[names[0]]) & set(arms[names[1]]))
        first, second = (
            [
                ranking_metrics(arms[arm][q]["ranked_doc_ids"], gold[q])["nDCG@10"]
                for q in shared
            ]
            for arm in names
        )
        report["paired_nDCG@10"] = {
            "first": names[0],
            "second": names[1],
            **sign_test(first, second),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--questions", required=True)
    args = parser.parse_args()
    _, events = read_ledger(args.retrieval)
    gold = {}
    for line in Path(args.questions).read_text().splitlines():
        row = json.loads(line)
        gold[row["question_id"]] = row.get("expected_doc_ids") or []
    print(json.dumps(build_report(events, gold), indent=2))


if __name__ == "__main__":
    main()
