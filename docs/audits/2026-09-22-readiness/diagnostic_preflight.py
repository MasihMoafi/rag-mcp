"""Validate frozen development inputs and estimate cost without network or keys."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rag.chunking import chunk_text
from rag.query_decomposition import decompose_query

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
args = parser.parse_args()
manifest = json.loads(Path(__file__).with_name('diagnostic-inputs.json').read_text())
raw = args.source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == manifest['source_sha256'], 'Source changed; do not run frozen diagnostic'
lines = raw.decode('utf-8').splitlines(keepends=True)
for claim in manifest['claims']:
    start, end = claim['lines_inclusive']
    evidence = ''.join(lines[start - 1:end]).encode()
    assert hashlib.sha256(evidence).hexdigest() == claim['sha256'], claim['id']
chunks = chunk_text(raw.decode('utf-8'), 700, 100)
cases = [{'id': case['id'], 'facets': decompose_query(case['question'])} for case in manifest['cases']]
facets = [facet for case in cases for facet in case['facets']]
# Planning allowances, not a provider-enforced token/billing ceiling. Include one
# embedding retry, no rerank/synthesis retry, one fresh LanceDB corpus build.
embedding_token_allowance = 2 * (sum(len(c.encode()) for c in chunks) + sum(len(q.encode()) for q in facets))
evidence_bytes = sum(sorted((len(c.encode()) for c in chunks), reverse=True)[:5])
input_allowance = sum(evidence_bytes + len(c['question'].encode()) + 2048 for c in manifest['cases'])
output_allowance = 700 * len(cases)
components = {
    'embedding_usd': embedding_token_allowance * 0.00000002,
    'rerank_usd': len(facets) * 0.002,
    'synthesis_usd': input_allowance * 0.000000375 + output_allowance * 0.0000015,
}
print(json.dumps({
    'status': 'offline_preflight_only', 'split': 'development',
    'source_sha256': manifest['source_sha256'], 'chunk_count': len(chunks),
    'cases': cases, 'embedding_token_allowance': embedding_token_allowance,
    'synthesis_input_token_allowance': input_allowance,
    'synthesis_output_token_cap_total': output_allowance,
    'cost_components': components, 'planning_estimate_usd': sum(components.values()),
    'hard_spend_cap': False, 'paid_calls': 0,
    'limitations': 'Byte-based token allowances; provider framing, rerank billing units and price drift need verification before execution. No quality score or causal finding.'
}, indent=2))
