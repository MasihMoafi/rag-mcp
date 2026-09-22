"""Run the frozen three-case diagnostic with an explicitly capped, unused key.

Defaults to an offline preview. --execute additionally requires a dedicated
OPENROUTER_API_KEY whose provider limit is no greater than --budget-usd, has no
reset and includes BYOK spending. This never creates keys or replenishes credit.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/audits/2026-09-22-readiness'


def verify_budget(data, budget, estimate):
    """Refuse unknown accounting, reused keys, automatic refills or excess limits."""
    def amount(value):
        if type(value) not in (str, int, float, Decimal):
            raise ValueError('Invalid spending amount')
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError('Invalid spending amount')
        return result
    budget, estimate = amount(budget), amount(estimate)
    limit, remaining, usage = (amount(data.get(k)) for k in ('limit', 'limit_remaining', 'usage'))
    if (not Decimal(0) < estimate <= remaining <= limit <= budget or usage != 0
            or data.get('limit_reset', 'unknown') is not None
            or data.get('include_byok_in_limit') is not True):
        raise ValueError('Require a fresh non-resetting capped key covering the estimate, including BYOK')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--budget-usd', type=Decimal, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    preflight = json.loads(subprocess.check_output([
        sys.executable, str(AUDIT / 'diagnostic_preflight.py'), str(args.source.resolve())], text=True))
    estimate = Decimal(str(preflight['planning_estimate_usd']))
    if not args.budget_usd.is_finite() or args.budget_usd < estimate:
        raise ValueError(f'Budget is insufficient: planning estimate ${estimate}; no inference submitted')
    if not args.execute:
        print(json.dumps({'mode': 'offline-preview', 'budget_usd': str(args.budget_usd),
                          'preflight': preflight}, indent=2))
        return 0
    if args.output is None or args.output.exists():
        raise ValueError('Execution requires a new --output directory')
    if datetime.now(timezone.utc).date().isoformat() != '2026-09-22':
        raise ValueError('Refresh the dated price snapshot and preflight before execution')
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not key:
        raise ValueError('Set a dedicated capped OPENROUTER_API_KEY; do not reuse the normal key')
    request = urllib.request.Request('https://openrouter.ai/api/v1/key',
                                     headers={'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(request, timeout=20) as response:
        verify_budget(json.load(response)['data'], args.budget_usd, estimate)
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700)
    # The dedicated directory owns all index/config/trace state for this run.
    environment = {k: v for k, v in os.environ.items() if not k.startswith('RAG_MCP_')}
    environment.update(
        RAG_MCP_CONFIG=str(output / 'config.json'), RAG_MCP_DATA_DIR=str(output / 'index'),
        RAG_MCP_TRACE_DIR=str(output / 'traces'), RAG_MCP_TRACE_CONTENT='1',
        RAG_MCP_EMBED_PROVIDER='openai_compatible', RAG_MCP_EMBED_MODEL='qwen/qwen3-embedding-4b',
        RAG_MCP_EMBED_API_KEY=key, RAG_MCP_EMBED_BASE_URL='https://openrouter.ai/api/v1',
        RAG_MCP_CONTEXTUAL_RETRIEVAL='off', RAG_MCP_CANDIDATE_K='50',
        RAG_MCP_RERANKER_TYPE='openrouter', RAG_MCP_RERANKER_MODEL='cohere/rerank-4-fast',
        RAG_MCP_RERANK_TOP_K='5', RAG_MCP_ANSWER_MODEL='deepseek/deepseek-v4.1-flash',
        RAG_MCP_ANSWER_MAX_TOKENS='700', RAG_MCP_BACKEND='lancedb',
        RAG_MCP_CHUNK_SIZE='700', RAG_MCP_CHUNK_OVERLAP='100')
    manifest = json.loads((AUDIT / 'diagnostic-inputs.json').read_text())
    def save(name, value):
        with os.fdopen(os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
            json.dump(value, stream, indent=2)
    configuration = {k: v for k, v in environment.items() if k.startswith('RAG_MCP_') and not k.endswith('API_KEY')}
    save('runtime.json', {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT, text=True).encode()).hexdigest(),
        'config_sha256': hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
        'configuration': configuration, 'python': sys.version,
        'input_manifest_sha256': hashlib.sha256((AUDIT / 'diagnostic-inputs.json').read_bytes()).hexdigest(),
        'budget_usd': str(args.budget_usd), 'split': 'development',
    })
    sys.path.insert(0, str(ROOT))
    records = []
    with patch.dict(os.environ, environment, clear=True):
        import server
        from rag import fetch
        try:
            for case in manifest['cases']:
                answer = json.loads(server.query_knowledge_base(case['question'], str(args.source.resolve())))
                records.append({'id': case['id'], 'response': answer})
                # Preserve failed attempts; do not retry or spend on more cases.
                if 'error' in answer:
                    break
        finally:
            try:
                fetch.reset_pipelines()
            finally:
                save('answers.json', {'preflight': preflight, 'cases': records})
    print(json.dumps({'completed_cases': len(records), 'quality': 'unreviewed', 'output': str(output)}))
    return 0 if len(records) == 3 and all('result' in r['response'] for r in records) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, InvalidOperation, KeyError) as error:
        print(f'Diagnostic refused or failed ({type(error).__name__}); no automatic retry.', file=sys.stderr)
        raise SystemExit(1)
