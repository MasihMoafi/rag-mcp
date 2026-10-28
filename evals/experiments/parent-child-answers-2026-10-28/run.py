"""Parent-child answers: the 22 anchored book questions, answered with parent_context off and on (see experiment.md).
Both modes run the same code (the parent-child worktree), each with its own settings and index. Every OpenRouter call
uses the OpenRouter key; the run stops if the key's spend since the start passes CAP. Grading happens afterwards, blind,
from grading-packet.jsonl; the mode behind each A/B label is in key.json."""
import json, os, random, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
EXPERIMENTS = HERE.parent
BOOK = Path.home() / "Desktop/books/rag"
CODE = Path.home() / "Desktop/p/.worktrees/rag-mcp-parent-child"
PY = Path.home() / "Desktop/p/rag-mcp/.venv/bin/python"
KEY = os.environ["OPENROUTER_API_KEY"]
CAP = 0.50  # dollars
BASE = {"embed_provider": "openai_compatible", "embed_model": "qwen/qwen3-embedding-8b",
        "embed_base_url": "https://openrouter.ai/api/v1", "embed_api_key": "", "top_k": "5",
        "reranker_type": "openrouter", "reranker_model": "cohere/rerank-4-fast"}

questions = json.loads((EXPERIMENTS / "book-heldout-2026-09-23/manifest.json").read_text())["questions"]
questions += [dict(q, id=f"p{i}") for i, q in enumerate(
    json.loads((EXPERIMENTS / "book-hosted-paraphrases-2026-09-23/manifest.json").read_text())["questions"], 1)]
chapters = {path.name.split(".")[0]: path.read_text() for path in BOOK.glob("*.md")}


def excerpt(q):
    text = chapters[q["chapter"].rstrip(".")]
    at = text.find(q["anchor"])
    return text[max(0, at - 900): at + len(q["anchor"]) + 900]


def usage():
    request = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + KEY})
    return json.load(urllib.request.urlopen(request, timeout=30))["data"]["usage"]


def answer(mode, q):
    config = HERE / f"config-{mode}.json"
    config.write_text(json.dumps({**BASE, "parent_context": mode}, indent=2))
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENROUTER_API_KEY")}
    env.update(OPENROUTER_API_KEY=KEY, RAG_MCP_CONFIG=str(config), RAG_MCP_DATA_DIR=str(HERE / f"index-{mode}"),
               RAG_MCP_WORKSPACE_DIR=str(Path.home() / ".cache/rag-experiments" / HERE.name / "workspaces"), RAG_MCP_TRACE_DIR=str(Path.home() / ".cache/rag-experiments" / HERE.name / "traces"))  # own engine lock and traces
    run = subprocess.run([str(PY), "-m", "rag.cli", "query", "--path", str(BOOK), "--type", "md", "--format", "json", q["query"]],
                         cwd=CODE, env=env, capture_output=True, text=True, timeout=1800)
    try:
        return json.loads(run.stdout)
    except ValueError:
        return {"error": (run.stderr or run.stdout)[-1500:], "code": run.returncode}


def main(limit=None):
    assert len(questions) == 22, len(questions)
    start, done = usage(), set()
    path = HERE / "answers.jsonl"
    if path.exists():
        done = {(r["mode"], r["id"]) for r in map(json.loads, path.open()) if "error" not in r["result"]}
    with path.open("a") as out, (HERE / "run.log").open("a") as log:
        for mode in ("off", "on"):
            for q in questions[:limit]:
                if (mode, q["id"]) in done:
                    continue
                spent = usage() - start
                if spent > CAP:
                    sys.exit(f"stopped: ${spent:.2f} spent, cap ${CAP}")
                result = answer(mode, q)
                out.write(json.dumps({"mode": mode, "id": q["id"], "query": q["query"], "result": result}, ensure_ascii=False) + "\n")
                out.flush()
                if "error" in result or not result.get("result"):  # stop at the first failure instead of paying for more of them
                    sys.exit(f"stopped at the first failure ({mode} {q['id']}): {json.dumps(result)[:300]}")
                print(mode, q["id"], f"spent ${usage() - start:.4f}", file=log, flush=True)
    if limit:
        print(f"smoke test of {limit} questions per mode passed; spent ${usage() - start:.4f}")
        return
    rows = [json.loads(l) for l in path.open()]
    by = {(r["mode"], r["id"]): r["result"] for r in rows if "error" not in r["result"]}  # the last success per question
    rng, packet, key = random.Random(20261028), [], {}
    for q in questions:
        modes = ["off", "on"]; rng.shuffle(modes)
        text = lambda m: (by[(m, q["id"])] or {}).get("result") or json.dumps(by[(m, q["id"])])[:3000]
        packet.append({"id": q["id"], "question": q["query"], "anchor": q["anchor"], "passage": excerpt(q),
                       "answer_A": text(modes[0]), "answer_B": text(modes[1])})
        key[q["id"]] = {"A": modes[0], "B": modes[1]}
    rng.shuffle(packet)
    (HERE / "grading-packet.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in packet))
    (HERE / "key.json").write_text(json.dumps(key, indent=1))
    print(f"spent ${usage() - start:.4f}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
