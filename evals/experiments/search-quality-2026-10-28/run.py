"""Search quality today: 153 questions, today's search, top 5 per question (see experiment.md).
Runs the rag CLI from the worktree at main with its own settings and indexes. Every OpenRouter call uses the
OpenRouter key, and the run stops before a collection if the key's spend since the start passes CAP."""
import csv, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
STORE = Path.home() / ".cache/rag-experiments" / HERE.name  # outside the repo: rag skips git-ignored files
DATA = HERE.parent / "experiment-isolated-scaling/data"
CODE = Path.home() / "Desktop/p/.worktrees/rag-mcp-eval-main"
PY = Path.home() / "Desktop/p/rag-mcp/.venv/bin/python"
KEY = os.environ["OPENROUTER_API_KEY"]
CAP = 1.00  # dollars, by the key's reported usage (it lags)
RERANK = 0.002  # dollars per search; the run also stops when its own count of reranks passes CAP
COLLECTIONS = {  # workspace: (paths, question column, answer column)
    "attention": (["attention/att.pdf"], "Q: The Question", "Exact Answer"),
    "brain-and-behavior": ([str(STORE / "inputs/brain_and_behavior.txt")], "Q: The Question", "Exact Answer"),  # the 141 MB PDF passes the 16 MB import limit only as the text rag itself extracts from it
    "napoleon": (["napoleon/Napoleon.pdf"], "Question", "Answer"),
    "fire-and-blood": (["fire-and-blood/Fire And Blood_George.R.R.Martin_furfalling.ir.pdf"], "Q: The Question", "Answer"),
    "mixed-codebase": (["mixed-codebase/python", "mixed-codebase/rust", "mixed-codebase/notebook"], "Question", "Expected_Answer"),
}
ENV = {k: v for k, v in os.environ.items() if not k.startswith("OPENROUTER_API_KEY")}
ENV.update(OPENROUTER_API_KEY=KEY, RAG_MCP_CONFIG=str(HERE / "config.json"), RAG_MCP_DATA_DIR=str(HERE / "index"),
           RAG_MCP_WORKSPACE_DIR=str(STORE / "workspaces"), RAG_MCP_TRACE_DIR=str(STORE / "traces"))  # own engine lock, workspaces, traces
CONFIG = {"embed_provider": "openai_compatible", "embed_model": "qwen/qwen3-embedding-8b",
          "embed_base_url": "https://openrouter.ai/api/v1", "embed_api_key": "", "top_k": "5",
          "reranker_type": "openrouter", "reranker_model": "cohere/rerank-4-fast"}


def usage():
    request = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + KEY})
    return json.load(urllib.request.urlopen(request, timeout=30))["data"]["usage"]


def rag(*args, timeout=1800):
    run = subprocess.run([str(PY), "-m", "rag.cli", *args], cwd=CODE, env=ENV, capture_output=True, text=True, timeout=timeout)
    return run.returncode, run.stdout, run.stderr


def search(workspace, question):
    """rag runs one engine query at a time across the machine; wait out another query instead of recording a failure."""
    for _ in range(120):
        code, so, se = rag("search", "--workspace", workspace, "--json", question)
        try:
            result = json.loads(so)
        except ValueError:
            return code, None, (se or so)[-1500:]
        if result.get("code") != "engine_busy":
            return code, result, result.get("error")
        time.sleep(5)
    return code, None, "engine busy for 10 minutes"


def main(only=None, limit=None):
    done = set()
    if (HERE / "results.jsonl").exists():
        done = {(r["collection"], r["n"]) for r in map(json.loads, (HERE / "results.jsonl").open()) if r.get("result") and "passages" in r["result"]}
    if not (HERE / "config.json").exists():
        (HERE / "config.json").write_text(json.dumps(CONFIG, indent=2))
    start = usage()
    log = open(HERE / "run.log", "a")
    out = open(HERE / "results.jsonl", "a")
    for workspace, (paths, qcol, acol) in COLLECTIONS.items():
        if only and workspace != only:
            continue
        spent = usage() - start
        print(f"{workspace}: spent so far ${spent:.4f}", file=log, flush=True)
        if spent > CAP:
            sys.exit(f"stopped before {workspace}: ${spent:.2f} spent, cap ${CAP}")
        for path in paths if not (STORE / "workspaces" / workspace).exists() else []:
            code, so, se = rag("sources", "add", workspace, "--path", str(DATA / path), "--json")
            print(workspace, path, code, se[-300:], file=log, flush=True)
        rows = list(csv.DictReader(open(DATA / workspace / "queries.csv", newline="")))[:limit]
        for i, row in enumerate(rows, 1):
            if (workspace, i) in done:
                continue
            began = time.monotonic()
            code, result, error = search(workspace, row[qcol])
            record = {"collection": workspace, "n": i, "question": row[qcol], "expected": row.get(acol, ""), "row": row,
                      "seconds": round(time.monotonic() - began, 2), "code": code}
            if result and "passages" in result:
                record["result"] = result
            else:
                record["error"] = error or json.dumps(result)[-1500:]
            out.write(json.dumps(record, ensure_ascii=False) + "\n"); out.flush()
            searches = globals().setdefault("SEARCHES", [0]); searches[0] += 1
            if "error" in record:  # stop at the first failure instead of paying for more of them
                sys.exit(f"stopped at the first failure ({workspace} #{i}): {record['error'][:300]}")
            if searches[0] * RERANK > CAP:
                sys.exit(f"stopped: {searches[0]} searches reach the ${CAP} cap")
    print(f"done: spent ${usage() - start:.4f}", file=log, flush=True)
    print(f"spent ${usage() - start:.4f}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or [None]), *(int(a) for a in sys.argv[2:3]))
