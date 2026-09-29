"""End-to-end answer quality: the 22 existing source-anchored book questions,
answered by `rag query` exactly as a user runs it, graded by a different model
against the source passage around each anchor."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import urllib.request

HERE = Path(__file__).parent
EXPERIMENTS = HERE.parent
BOOK = Path.home() / "Desktop/books/rag"
RAG = Path(__file__).resolve().parents[3] / ".venv/bin/rag"
JUDGE = "anthropic/claude-sonnet-5.5"

questions = json.loads((EXPERIMENTS / "book-heldout-2026-09-23/manifest.json").read_text())["questions"]
questions += [dict(q, id=f"p{i}") for i, q in enumerate(
    json.loads((EXPERIMENTS / "book-hosted-paraphrases-2026-09-23/manifest.json").read_text())["questions"], 1)]
chapters = {path.name.split(".")[0]: path.read_text() for path in BOOK.glob("*.md")}


def excerpt(question):
    text = chapters[question["chapter"].rstrip(".")]
    at = text.find(question["anchor"])
    if at < 0:
        raise RuntimeError(f"anchor not found for {question['id']}")
    return text[max(0, at - 900): at + len(question["anchor"]) + 900]


def answer(question):
    run = subprocess.run([str(RAG), "query", "--path", str(BOOK), "--type", "md", "--format", "json", question["query"]],
                         capture_output=True, text=True, timeout=600)
    return json.loads(run.stdout)["result"] if run.returncode == 0 else f"<error {run.returncode}>"


GRADING = """You grade a RAG system's answer against the book passage that contains the answer.
Question: {query}
Book passage (the key sentence contains: "{anchor}"):
<<<{excerpt}>>>
System answer:
<<<{answer}>>>
Grade:
- correct: states the key point the passage makes for this question; nothing contradicts the passage.
- partial: on topic but misses or blurs the key point.
- wrong: contradicts the passage, or answers something else.
- abstained: says the sources do not contain the answer.
Reply with JSON only: {{"grade": "...", "why": "<one short sentence>"}}"""


def grade(question, reply, passage):
    body = json.dumps({"model": JUDGE, "temperature": 0, "max_tokens": 1500, "messages": [{"role": "user", "content": GRADING.format(
        query=question["query"], anchor=question["anchor"], excerpt=passage, answer=reply)}]}).encode()
    request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body, {
        "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY_COMPANY"], "Content-Type": "application/json"})
    content = json.load(urllib.request.urlopen(request, timeout=120))["choices"][0]["message"]["content"]
    found = re.search(r"\{.*\}", content or "", re.S)
    if not found:
        raise RuntimeError(f"judge gave no JSON for {question['id']}: {content!r}")
    return json.loads(found.group(0))


def evaluate(question):
    passage = excerpt(question)
    reply = answer(question)
    return {"id": question["id"], "query": question["query"], **grade(question, reply, passage)}  # book-derived answer text stays out of the repo


with ThreadPoolExecutor(1) as pool:  # the engine runs one query at a time
    rows = list(pool.map(evaluate, questions))
counts = {g: sum(r["grade"] == g for r in rows) for g in ("correct", "partial", "wrong", "abstained")}
(HERE / "result.json").write_text(json.dumps({"judge": JUDGE, "counts": counts, "rows": rows}, ensure_ascii=False, indent=1))
print(counts)
for r in rows:
    print(f"{r['id']:>3} {r['grade']:<9} {r['why']}")
