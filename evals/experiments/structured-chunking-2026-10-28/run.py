"""Structured parsing and book windows: today's search on the branch for the 60 Napoleon and Fire and Blood
questions (see experiment.md). Isolated settings, index, workspaces, traces and lock; OpenRouter key only;
stops at the first failure and at its own count of paid searches.
Usage: run.py [N]  smoke test of N questions per collection, or the whole set."""
import csv, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
STORE = Path.home() / '.cache/rag-experiments' / HERE.name  # outside the repo: rag skips git-ignored files
DATA = HERE.parent / 'experiment-isolated-scaling/data'
CODE = Path.home() / 'Desktop/p/.worktrees/rag-mcp-structured-chunking'
PY = Path.home() / 'Desktop/p/rag-mcp/.venv/bin/python'
KEY = os.environ['OPENROUTER_API_KEY']
CAP, RERANK = 0.50, 0.002
COLLECTIONS = {'napoleon': ('napoleon/Napoleon.pdf', 'Question', 'Answer'),
               'fire-and-blood': ('fire-and-blood/Fire And Blood_George.R.R.Martin_furfalling.ir.pdf', 'Q: The Question', 'Answer')}
CONFIG = {'embed_provider': 'openai_compatible', 'embed_model': 'qwen/qwen3-embedding-8b', 'embed_base_url': 'https://openrouter.ai/api/v1',
          'embed_api_key': '', 'top_k': '5', 'reranker_type': 'openrouter', 'reranker_model': 'cohere/rerank-4-fast'}
ENV = {k: v for k, v in os.environ.items() if not k.startswith('OPENROUTER_API_KEY')}
ENV.update(OPENROUTER_API_KEY=KEY, RAG_MCP_CONFIG=str(HERE / 'config.json'), RAG_MCP_DATA_DIR=str(HERE / 'index'),
           RAG_MCP_WORKSPACE_DIR=str(STORE / 'workspaces'), RAG_MCP_TRACE_DIR=str(STORE / 'traces'))


def rag(*args):
    run = subprocess.run([str(PY), '-m', 'rag.cli', *args], cwd=CODE, env=ENV, capture_output=True, text=True, timeout=3600)
    return run.returncode, run.stdout, run.stderr


def search(workspace, question):
    for _ in range(120):
        code, out, err = rag('search', '--workspace', workspace, '--json', question)
        try:
            result = json.loads(out)
        except ValueError:
            return None, (err or out)[-1500:]
        if result.get('code') != 'engine_busy':
            return (result, None) if result.get('passages') else (None, json.dumps(result)[-1500:])
        time.sleep(5)
    return None, 'engine busy for 10 minutes'


def main(limit=None):
    (HERE / 'config.json').write_text(json.dumps(CONFIG, indent=2))
    path = HERE / 'results.jsonl'
    done = {(r['collection'], r['n']) for r in map(json.loads, path.open())} if path.exists() else set()
    searches = 0
    with path.open('a') as out, (HERE / 'run.log').open('a') as log:
        for workspace, (doc, qcol, acol) in COLLECTIONS.items():
            if not (STORE / 'workspaces' / workspace).exists():
                code, so, se = rag('sources', 'add', workspace, '--path', str(DATA / doc), '--json')
                print(workspace, 'import', code, se[-300:], file=log, flush=True)
                if code:
                    sys.exit(f'import failed for {workspace}: {(se or so)[-300:]}')
            rows = list(csv.DictReader(open(DATA / workspace / 'queries.csv', newline='')))
            for i, row in enumerate(rows[:limit], 1):
                if (workspace, i) in done:
                    continue
                began = time.monotonic()
                result, error = search(workspace, row[qcol])
                searches += 1
                if error:
                    sys.exit(f'stopped at the first failure ({workspace} #{i}): {error[:300]}')
                out.write(json.dumps({'collection': workspace, 'n': i, 'question': row[qcol], 'expected': row.get(acol, ''), 'row': row,
                                      'seconds': round(time.monotonic() - began, 2), 'result': result}, ensure_ascii=False) + '\n')
                out.flush()
                if searches * RERANK > CAP:
                    sys.exit(f'stopped: {searches} searches reach the ${CAP} cap')
    print(f'{searches} searches this run, about ${searches * RERANK:.2f} in reranking')


if __name__ == '__main__':
    main(*(int(a) for a in sys.argv[1:2]))
