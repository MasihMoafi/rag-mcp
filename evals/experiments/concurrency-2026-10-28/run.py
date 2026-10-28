"""Concurrent queries: N rag search processes started together, before and after d2e428e (see experiment.md).
Keyword search only, no API key in the environment, every store isolated under ~/.cache/rag-experiments/."""
import csv, json, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).parent
PY = Path.home() / 'Desktop/p/rag-mcp/.venv/bin/python'
DOC = HERE.parent / 'experiment-isolated-scaling/data/attention/att.pdf'
QUESTIONS = [r['Q: The Question'] for r in csv.DictReader(open(HERE.parent / 'experiment-isolated-scaling/data/attention/queries.csv', newline=''))]
VARIANTS = {'before': Path.home() / 'Desktop/p/.worktrees/rag-mcp-before-queue', 'after': Path.home() / 'Desktop/p/.worktrees/rag-mcp-after-queue'}
LEVELS = (1, 2, 5, 10, 20)


def env(variant):
    store = Path.home() / '.cache/rag-experiments/concurrency-2026-10-28' / variant
    store.mkdir(parents=True, exist_ok=True)
    e = {k: v for k, v in os.environ.items() if 'API_KEY' not in k}
    e.update(RAG_MCP_CONFIG=str(store / 'config.json'), RAG_MCP_DATA_DIR=str(store / 'index'),
             RAG_MCP_WORKSPACE_DIR=str(store / 'workspaces'), RAG_MCP_TRACE_DIR=str(store / 'traces'))
    return e


def outcome(stdout):
    try:
        result = json.loads(stdout)
    except ValueError:
        return 'other'
    return 'answered' if result.get('passages') else result.get('code') or 'other'


results = {}
for variant, code in VARIANTS.items():
    e = env(variant)
    subprocess.run([str(PY), '-m', 'rag.cli', 'sources', 'add', 'paper', '--path', str(DOC), '--json'], cwd=code, env=e, capture_output=True, timeout=600)
    warm = subprocess.run([str(PY), '-m', 'rag.cli', 'search', '--workspace', 'paper', '--json', QUESTIONS[0]], cwd=code, env=e, capture_output=True, text=True, timeout=900)
    assert outcome(warm.stdout) == 'answered', warm.stdout[-500:] + warm.stderr[-500:]  # smoke test: one query works before the load levels
    for n in LEVELS:
        began = time.monotonic()
        procs = [subprocess.Popen([str(PY), '-m', 'rag.cli', 'search', '--workspace', 'paper', '--json', QUESTIONS[i % len(QUESTIONS)]],
                                  cwd=code, env=e, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for i in range(n)]
        outcomes = [outcome(p.communicate(timeout=1200)[0]) for p in procs]
        results.setdefault(variant, {})[n] = {'answered': outcomes.count('answered'), 'engine_busy': outcomes.count('engine_busy'),
                                              'other': len(outcomes) - outcomes.count('answered') - outcomes.count('engine_busy'),
                                              'seconds': round(time.monotonic() - began, 2)}
        print(variant, n, results[variant][n], flush=True)
(HERE / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
