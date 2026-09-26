"""Retrieval-only eval on a code corpus: does search return the right passages?

    python evals/code-retrieval/run.py --corpus ~/src/project/core/src --type rs --label baseline

Imports the corpus into a workspace (once), runs every question through `rag search`
(no answer synthesis, so no generation cost), and scores where the expected file and
the answer's identifier first appear among the returned passages.
"""
import argparse
import json
import os
import statistics
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--corpus", required=True, help="folder to index; question files are relative to it")
parser.add_argument("--type", default="rs", help="file extension to import")
parser.add_argument("--workspace", default="code-retrieval-eval")
parser.add_argument("--questions", default=os.path.join(HERE, "questions.json"))
parser.add_argument("--label", default="run")
parser.add_argument("--rag", default="rag", help="the rag executable to test")
parser.add_argument("--out", default="/tmp/code-retrieval-eval")
args = parser.parse_args()


def rag(*argv):
    return subprocess.run([args.rag, *argv], capture_output=True, text=True)


if args.workspace not in rag("workspaces").stdout.split():
    added = rag("sources", "add", args.workspace, "--path", os.path.expanduser(args.corpus), "--type", args.type, "--json")
    if added.returncode:
        raise SystemExit(f"could not import the corpus: {added.stderr[-300:]}")

questions = json.load(open(args.questions))
out = os.path.join(args.out, args.label)
os.makedirs(out, exist_ok=True)
rows = []
for q in questions:
    started = time.time()
    run = rag("search", "--workspace", args.workspace, "--json", q["q"])
    seconds = time.time() - started
    try:
        passages = json.loads(run.stdout)["passages"]
    except (ValueError, KeyError):
        passages = []
        print(f"FAILED {q['id']}: {run.stderr[-160:]}")
    json.dump(passages, open(os.path.join(out, f"{q['id']}.json"), "w"))
    names = [p["name"] for p in passages]
    rows.append({
        "id": q["id"], "secs": round(seconds, 1), "top": names[:5],
        "file_rank": next((i + 1 for i, n in enumerate(names) if n in q["files"]), None),
        "text_rank": next((i + 1 for i, p in enumerate(passages) if q["kw"] in p.get("content", "")), None),
    })

count = len(rows)
hits = lambda key, k: sum(1 for r in rows if r[key] and r[key] <= k)
mrr = sum(1 / r["file_rank"] for r in rows if r["file_rank"]) / count
print(f"[{args.label}] expected file: top-1 {hits('file_rank', 1)}/{count}, top-5 {hits('file_rank', 5)}/{count}, "
      f"MRR {mrr:.2f} | answer identifier in a passage: top-1 {hits('text_rank', 1)}/{count}, top-5 {hits('text_rank', 5)}/{count} "
      f"| median {statistics.median(r['secs'] for r in rows)}s")
for r in rows:
    if r["file_rank"] != 1 or r["text_rank"] is None:
        print(f"  {r['id']}: file rank {r['file_rank']}, identifier rank {r['text_rank']}, top {r['top']}")
json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=1)
