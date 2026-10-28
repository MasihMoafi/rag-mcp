"""Every number in report.html comes from this script, read from the raw per-question files of each experiment.
Run: python3 build_data.py  -> data.json"""
import json, math
from collections import Counter
from pathlib import Path

E = Path(__file__).parent.parent
SQ = E / 'search-quality-2026-10-28'
rank = {'exact': 2, 'partial': 1, 'miss': 0, 'correct': 2, 'abstained': 0, 'wrong': 0}


def wilson(k, n, z=1.96):
    if not n:
        return [0, 0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def sign_test(b, c):
    """Exact two-sided binomial (McNemar) p-value on the discordant pairs."""
    n, k = b + c, min(b, c)
    if not n:
        return 1.0
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def grades(files, key=('collection', 'n')):
    out = {}
    for f in files:
        for line in open(f):
            r = json.loads(line); out[tuple(r[k] for k in key)] = r['grade']
    return out


runs = {'August, isolated': grades([SQ / 'regrade-august/isolated-grades-1.jsonl', SQ / 'regrade-august/isolated-grades-2.jsonl']),
        'August, merged': grades([SQ / 'regrade-august/unified-grades-1.jsonl', SQ / 'regrade-august/unified-grades-2.jsonl']),
        'Today': grades([SQ / 'grades-1.jsonl', SQ / 'grades-2.jsonl'])}
assert all(len(g) == 153 for g in runs.values())
cols = ['attention', 'brain-and-behavior', 'napoleon', 'fire-and-blood', 'mixed-codebase']
names = {'attention': 'Attention paper', 'brain-and-behavior': 'Brain and Behavior', 'napoleon': 'Napoleon',
         'fire-and-blood': 'Fire and Blood', 'mixed-codebase': 'Mixed codebase'}


def summary(g, keys=None):
    v = [g[k] for k in (keys or g)]
    c = Counter(v); n = len(v)
    return {'n': n, 'exact': c['exact'], 'partial': c['partial'], 'miss': c['miss'],
            'exact_ci': wilson(c['exact'], n), 'found_ci': wilson(c['exact'] + c['partial'], n)}


def transitions(a, b, keys, labels=('exact', 'partial', 'miss')):
    m = {x: {y: 0 for y in labels} for x in labels}
    for k in keys:
        m[a[k]][b[k]] += 1
    return m


search = {'runs': {name: summary(g) for name, g in runs.items()},
          'by_collection': {names[c]: {name: summary(g, [k for k in g if k[0] == c]) for name, g in runs.items()} for c in cols}}
a, t = runs['August, isolated'], runs['Today']
b = sum(t[k] == 'exact' and a[k] != 'exact' for k in t); c = sum(a[k] == 'exact' and t[k] != 'exact' for k in t)
search['paired_exact'] = {'today_only': b, 'august_only': c, 'p': sign_test(b, c)}
search['better_worse'] = {'better': sum(rank[t[k]] > rank[a[k]] for k in t), 'worse': sum(rank[t[k]] < rank[a[k]] for k in t)}
search['transitions'] = transitions(a, t, list(t))

# Structured parsing and book windows: blind side by side, unblinded with key.json
S = E / 'structured-chunking-2026-10-28'
key = json.load(open(S / 'key.json'))
base, branch = {}, {}
for f in ('grades-1.jsonl', 'grades-2.jsonl'):
    for line in open(S / f):
        r = json.loads(line); m = key[f"{r['collection']}#{r['n']}"]; got = {m['A']: r['A']['grade'], m['B']: r['B']['grade']}
        base[(r['collection'], r['n'])] = got['baseline']; branch[(r['collection'], r['n'])] = got['branch']
assert len(base) == 60
b2 = sum(branch[k] == 'exact' and base[k] != 'exact' for k in base); c2 = sum(base[k] == 'exact' and branch[k] != 'exact' for k in base)
structured = {'baseline': summary(base), 'branch': summary(branch),
              'by_book': {names[x]: {'baseline': summary(base, [k for k in base if k[0] == x]), 'branch': summary(branch, [k for k in base if k[0] == x])}
                          for x in ('napoleon', 'fire-and-blood')},
              'paired_exact': {'branch_only': b2, 'baseline_only': c2, 'p': sign_test(b2, c2)},
              'transitions': transitions(base, branch, list(base))}

# Parent-child answers: blind A/B, unblinded with key.json
P = E / 'parent-child-answers-2026-10-28'
pkey = json.load(open(P / 'key.json')); pc = {'off': Counter(), 'on': Counter()}; poff, pon = {}, {}
for line in open(P / 'grades.jsonl'):
    r = json.loads(line); m = pkey[r['id']]
    for label in ('A', 'B'):
        pc[m[label]][r[label]['grade']] += 1
        (poff if m[label] == 'off' else pon)[r['id']] = r[label]['grade']
parent = {mode: dict(v) for mode, v in pc.items()}
parent['changed'] = [[i, poff[i], pon[i]] for i in sorted(poff) if poff[i] != pon[i]]

conc = json.load(open(E / 'concurrency-2026-10-28/results.json'))
spend = [['Search quality, 153 questions', 0.29, 'OpenRouter usage, failed first attempts included'],
         ['Structured parsing, 60 questions', 0.12, 'own count: 60 reranks'],
         ['Parent–child, 44 answers', 0.13, 'OpenRouter usage readings over all five attempts (upper end; they lag)'],
         ['Concurrency, keyword only', 0.0, 'no key in the processes']]
json.dump({'search': search, 'structured': structured, 'parent_child': parent, 'concurrency': conc, 'spend': spend},
          open(Path(__file__).parent / 'data.json', 'w'), indent=1)
print(json.dumps({'today': search['runs']['Today'], 'paired': search['paired_exact'], 'structured_paired': structured['paired_exact'], 'parent': parent}, indent=0)[:900])
