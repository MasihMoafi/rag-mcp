"""Blind side-by-side packet: for each question, today's top 5 (baseline) and the branch's top 5, as A and B in a
seeded random order. The mapping goes to ~/.cache so the grader cannot read it; it returns after grading."""
import json, random
from pathlib import Path

HERE = Path(__file__).parent
KEY = Path.home() / '.cache/rag-experiments' / HERE.name / 'key.json'
base = {(r['collection'], r['n']): r for r in map(json.loads, open(HERE.parent / 'search-quality-2026-10-28/results.jsonl'))
        if r['collection'] in ('napoleon', 'fire-and-blood')}
new = {(r['collection'], r['n']): r for r in map(json.loads, open(HERE / 'results.jsonl'))}
assert set(base) == set(new) and len(new) == 60, (len(base), len(new))
rng, packet, key = random.Random(20261028), [], {}
passages = lambda r: [{'rank': p['rank'], 'text': p['content']} for p in r['result']['passages']]
for k in sorted(new):
    order = ['baseline', 'branch']; rng.shuffle(order)
    sets = {'baseline': passages(base[k]), 'branch': passages(new[k])}
    packet.append({'collection': k[0], 'n': k[1], 'question': new[k]['question'], 'expected_answer': new[k]['expected'], 'csv_row': new[k]['row'],
                   'passages_A': sets[order[0]], 'passages_B': sets[order[1]]})
    key[f'{k[0]}#{k[1]}'] = {'A': order[0], 'B': order[1]}
rng.shuffle(packet)
(HERE / 'grading-packet.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in packet))
KEY.parent.mkdir(parents=True, exist_ok=True); KEY.write_text(json.dumps(key, indent=1))
print(len(packet), 'questions;', sum(len(json.dumps(p)) for p in packet) // 1000, 'k chars; key at', KEY)
