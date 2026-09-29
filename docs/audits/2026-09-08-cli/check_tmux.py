"""Live tmux UX walkthrough. Requires running Ollama and an OpenRouter key.

Run with the project's Python. Uses temporary documents/settings and leaves the
session open at the main menu for inspection: tmux attach -t rag-ux-check
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

from rag import detect

session = "rag-ux-check"
evidence = Path(__file__).parent
temporary = Path(tempfile.mkdtemp(prefix="rag-tmux-"))
documents = temporary / "documents"
documents.mkdir()
(documents / "project.txt").write_text("The Cedar Lantern project launches on Friday.\nAll reports must be submitted by Thursday.\n")
config = temporary / "config.json"
env = dict(os.environ, RAG_MCP_CONFIG=str(config), RAG_MCP_DATA_DIR=str(temporary / "index"),
           RAG_MCP_RERANKER_TYPE="off", PROMPT_TOOLKIT_NO_CPR="1", TERM="xterm-256color")
for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    env.pop(name, None)


def tmux(*args):
    return subprocess.check_output(["tmux", *args], text=True, env=env)


def screen():
    return tmux("capture-pane", "-p", "-t", session)


def expect(question, timeout=45):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        pane = screen()
        questions = [line for line in pane.splitlines() if line.startswith("◆ ")]
        if questions and questions[-1] == f"◆ {question}" and "❯" in pane.rsplit(f"◆ {question}\n", 1)[-1]:
            time.sleep(0.15)
            return pane
        time.sleep(0.1)
    raise AssertionError(f"Did not reach {question}:\n{screen()}")


def keys(*values):
    tmux("send-keys", "-t", session, *values)


def save(name):
    (evidence / f"tmux-{name}.txt").write_text(screen())


launch = ["env"]
for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    launch += ["-u", name]
launch += [f"{name}={env[name]}" for name in ("RAG_MCP_CONFIG", "RAG_MCP_DATA_DIR", "RAG_MCP_RERANKER_TYPE", "PROMPT_TOOLKIT_NO_CPR", "TERM")]
launch += ["rag"]
tmux("new-session", "-d", "-s", session, "-x", "110", "-y", "32", "-c", str(documents), shlex.join(launch))
detected = detect.detect_all(env)
local_count = len(detected["ollama"]["embedding_models"]) + len(detected["cached_models"]) + sum(len(item["embedding_models"]) for item in detected["local_runtimes"])
main = "What would you like to do?"
model_question = "Which embedding model would you like to use?"
expect(main)
keys("Enter")
expect(model_question)
save("local-models")
assert "BGE small · English" not in screen()
keys("Enter")
expect("Connect a coding agent?")
assert json.loads(config.read_text())["embed_provider"] == "ollama"
save("verified-model")
keys("Enter")
expect(main)

# Live provider catalog, bounded viewport and wrap to the final Back action.
keys("Enter")
expect(model_question)
keys(*(["Down"] * local_count), "Enter")
expect("Which API provider do you use?")
keys("Down", "Down", "Enter")
expect(model_question)
pane = screen()
assert "Loading OpenRouter models" in pane
assert pane.count("embedding") < 15, "Catalog should scroll inside an eight-row viewport"
save("api-models")
keys("Up")
time.sleep(0.2)
assert "❯ Back to providers" in screen()
save("api-scroll")
keys("Enter")
expect("Which API provider do you use?")
keys("Escape")
expect(model_question)

# Explicit download catalog, with no download until a model is selected.
keys(*(["Down"] * (local_count + 1)), "Enter")
expect("Which model would you like to download?", timeout=60)
assert "download ~" in screen()
save("downloads")
keys("Escape")
expect(model_question)
keys("Escape")
expect(main)

# Arrow-only file choice, question choice and real retrieval.
keys("Down", "Enter")
expect(str(documents))
save("files")
keys("Down", "Down", "Enter")
expect("What would you like to find?")
keys("Down", "Enter")
expect("Results per query")
keys("Enter")
expect(main, timeout=60)
history = tmux("capture-pane", "-p", "-S", "-", "-t", session)
assert "Cedar Lantern" in history
assert '"result":' not in history
assert "WARN  lance::" not in history
assert "[RAG V2]" not in history
assert "1. " + str(documents / "project.txt") in history
assert "Traceback" not in history
save("search")
(evidence / "tmux-walkthrough.txt").write_text(history)

# Custom source questions explain exact URL syntax and recover from bad input.
keys("Enter")
expect(model_question)
keys("Up", "Enter")
expect("Where is your model?")
keys("Enter")
time.sleep(0.3)
assert "http://192.168.1.10:11434" in screen()
keys("localhost", "Enter")
time.sleep(0.3)
assert "Enter a full URL beginning with http:// or https://" in screen()
keys("http://127.0.0.1:1", "Enter")
expect(model_question)
save("unavailable-server")
assert "No embedding models found" in screen()
keys("Escape")
expect(main)
assert json.loads(config.read_text())["embed_provider"] == "ollama"
print(f"PASS: live model verification, live API catalog and scrolling, download catalog, cancellation, file selection, readable search, custom URL guidance and unavailable-server recovery. Inspect with: tmux attach -t {session}")
