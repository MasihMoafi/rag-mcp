"""Builds report.html (one view: the story, the charts, links to the evidence) and evidence.html (every
question with its grades, the graders' reasons and a link to its run trace). Run build_data.py first.

The pages follow an existing approved layout (Masih's rule 7): the chart primitives and the CSS of the Elpis
eval dashboard, read from that folder at build time, not copied. Titles are yes-or-no answers (rule 23)."""
import html, json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
E = HERE.parent
DASH = Path.home() / 'Desktop/p/Elpis-wt-install/docs/evals/dashboard'
sys.path.insert(0, str(DASH))
import charts  # noqa: E402  (the dashboard's SVG primitives)

CSS = re.search(r'CSS = """(.*?)"""', (DASH / 'build.py').read_text(), re.S).group(1)
D = json.loads((HERE / 'data.json').read_text())
TEAL, GREY = '#17a398', '#b8b0a4'
esc = html.escape


def page(title, body):
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>{CSS}</style></head><body>\n<div class="wrap">\n{body}\n</div></body></html>\n')


def kpi(question, answer, line):
    return f'<div class="kpi"><div class="k">{question}</div><div class="n">{answer}</div><div class="l">{line}</div></div>'


def card(title, svg, caption, key=()):
    keys = ''.join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in key)
    return (f'<section class="card"><h3>{title}</h3>' + (f'<div class="key">{keys}</div>' if keys else '')
            + f'{svg}<p class="cap">{caption}</p></section>')


def link(path, text):
    return f'<a href="{path}">{text}</a>'


sq, st, pc, cc = D['search'], D['structured'], D['parent_child'], D['concurrency']
runs = sq['runs']; today, aug = runs['Today'], runs['August, isolated']
total = sum(r[1] for r in D['spend'])
cols = list(sq['by_collection']); books = list(st['by_book'])

# ---------- report.html: one view ----------
view = [
    '<header><p class="eyebrow">rag-mcp · evaluation · 28 October 2026</p>'
    f'<h1>Does rag-mcp find answers better than in August? Yes: {today["exact"]} of 153, up from {aug["exact"]}.</h1>'
    f'<p class="sub">Same 153 questions, same grading model and rubric. Structured parsing adds {st["branch"]["exact"] - st["baseline"]["exact"]} more exact answers on the two books. '
    f'Every number links to the files below. Paid calls cost ${total:.2f}.</p></header>',
    '<div class="kpis">'
    + kpi('Search better than in August?', 'Yes', f'{today["exact"]} vs {aug["exact"]} exact · not chance (p = {sq["paired_exact"]["p"]})')
    + kpi('Structured parsing finds more?', 'Yes', f'{st["branch"]["exact"]} vs {st["baseline"]["exact"]} of 60 · not chance (p = {st["paired_exact"]["p"]})')
    + kpi('Parent–child helps?', 'No', f'{pc["on"]["correct"]} vs {pc["off"]["correct"]} of 22 correct · removed')
    + kpi('20 searches at once answered?', 'Yes', f'{cc["after"]["20"]["answered"]} vs {cc["before"]["20"]["answered"]} before the fix')
    + '</div>',
    '<div class="kpis">'
    + card('Exact answers per collection',
           charts.grouped([c.replace(' and ', ' & ') for c in cols],
                          [('August', [sq['by_collection'][c]['August, isolated']['exact'] for c in cols], GREY),
                           ('Today', [sq['by_collection'][c]['Today']['exact'] for c in cols], TEAL)], w=620, h=300),
           'The books and the code gained most.', [('August', GREY), ('Today', TEAL)])
    + card('Books: old parsing vs structured parsing',
           charts.grouped(books, [('Old', [st['by_book'][b]['baseline']['exact'] for b in books], GREY),
                                  ('Structured', [st['by_book'][b]['branch']['exact'] for b in books], TEAL)], w=620, h=300),
           f'{st["paired_exact"]["branch_only"]} questions became exact, {st["paired_exact"]["baseline_only"]} became partial. Now on main.',
           [('Old', GREY), ('Structured', TEAL)])
    + '</div>',
    '<section class="card"><h3>Evidence: open the files</h3><div class="scroll"><table>'
    '<tr><th>Test</th><th>Per question</th><th>Grades</th><th>Runs and traces</th><th>Plan and result</th></tr>'
    f'<tr><td>Search, 153 questions</td><td>{link("evidence.html#search", "evidence table")}</td>'
    f'<td>{link("../search-quality-2026-10-28/grades-1.jsonl", "today 1")} · {link("../search-quality-2026-10-28/grades-2.jsonl", "today 2")} · {link("../search-quality-2026-10-28/regrade-august/", "August")}</td>'
    f'<td>{link("../search-quality-2026-10-28/results.jsonl", "results")} · {link("../search-quality-2026-10-28/traces/", "153 traces")}</td>'
    f'<td>{link("../search-quality-2026-10-28/experiment.md", "experiment.md")}</td></tr>'
    f'<tr><td>Structured parsing, 60</td><td>{link("evidence.html#parsing", "evidence table")}</td>'
    f'<td>{link("../structured-chunking-2026-10-28/grades-1.jsonl", "blind 1")} · {link("../structured-chunking-2026-10-28/grades-2.jsonl", "blind 2")} · {link("../structured-chunking-2026-10-28/key.json", "key")}</td>'
    f'<td>{link("../structured-chunking-2026-10-28/results.jsonl", "results")} · {link("../structured-chunking-2026-10-28/traces/", "60 traces")}</td>'
    f'<td>{link("../structured-chunking-2026-10-28/experiment.md", "experiment.md")}</td></tr>'
    f'<tr><td>Parent–child, 22</td><td>{link("evidence.html#parent", "evidence table")}</td>'
    f'<td>{link("../parent-child-answers-2026-10-28/grades.jsonl", "blind")} · {link("../parent-child-answers-2026-10-28/key.json", "key")}</td>'
    f'<td>{link("../parent-child-answers-2026-10-28/answers.jsonl", "answers")} · {link("../parent-child-answers-2026-10-28/traces/", "traces")}</td>'
    f'<td>{link("../parent-child-answers-2026-10-28/experiment.md", "experiment.md")}</td></tr>'
    f'<tr><td>20 searches at once</td><td>—</td><td>—</td><td>{link("../concurrency-2026-10-28/results.json", "results.json")}</td>'
    f'<td>{link("../concurrency-2026-10-28/experiment.md", "experiment.md")}</td></tr>'
    '</table></div><p class="cap">Limits: each test ran once; 16 test items look faulty; Part 1 changed four things at once. '
    f'Numbers: {link("data.json", "data.json")} from {link("build_data.py", "build_data.py")}.</p></section>']
(HERE / 'report.html').write_text(page('rag-mcp evaluation', '\n'.join(view)))

# ---------- evidence.html: every question ----------
def rows_of(path):
    return [json.loads(l) for l in open(path)]


def cls(g):
    return {'exact': 'win', 'correct': 'win', 'miss': 'lose', 'abstained': 'lose', 'wrong': 'lose'}.get(g, 'neu')


SQ = E / 'search-quality-2026-10-28'
res = {(r['collection'], r['n']): r for r in rows_of(SQ / 'results.jsonl')}
grade = lambda files: {(r['collection'], r['n']): r for f in files for r in rows_of(f)}
gt = grade([SQ / 'grades-1.jsonl', SQ / 'grades-2.jsonl'])
ga = grade([SQ / 'regrade-august/isolated-grades-1.jsonl', SQ / 'regrade-august/isolated-grades-2.jsonl'])
ev = ['<header><p class="eyebrow">rag-mcp · evidence · 28 October 2026</p><h1>Every question, its grades and its trace</h1>'
      f'<p class="sub">{link("report.html", "Back to the report")}. A trace is the engine\'s own record of one search: stages, models, candidates and scores.</p></header>',
      '<section class="card" id="search"><h3>Search: August vs today, 153 questions</h3><div class="scroll"><table>'
      '<tr><th>Question</th><th>August</th><th>Today</th><th>Grader on today</th><th>Trace</th></tr>']
for k in sorted(gt):
    q = res[k]; rid = q['result']['run_id']
    ev.append(f'<tr><td>{esc(k[0])} #{k[1]}: {esc(q["question"][:110])}</td><td class="{cls(ga[k]["grade"])}">{ga[k]["grade"]}</td>'
              f'<td class="{cls(gt[k]["grade"])}">{gt[k]["grade"]}</td><td class="neu">{esc(gt[k]["why"][:160])}</td>'
              f'<td>{link(f"../search-quality-2026-10-28/traces/trace-{rid}.json", "open")}</td></tr>')
ev.append('</table></div></section>')

S2 = E / 'structured-chunking-2026-10-28'
key = json.load(open(S2 / 'key.json')); new = {(r['collection'], r['n']): r for r in rows_of(S2 / 'results.jsonl')}
ev.append('<section class="card" id="parsing"><h3>Books: old parsing vs structured parsing, 60 questions (graded blind)</h3><div class="scroll"><table>'
          '<tr><th>Question</th><th>Old</th><th>Structured</th><th>Grader on structured</th><th>Traces</th></tr>')
for f in ('grades-1.jsonl', 'grades-2.jsonl'):
    pass
sg = {}
for f in ('grades-1.jsonl', 'grades-2.jsonl'):
    for r in rows_of(S2 / f):
        m = key[f"{r['collection']}#{r['n']}"]; sg[(r['collection'], r['n'])] = {m['A']: r['A'], m['B']: r['B']}
for k in sorted(sg):
    old, nw = sg[k]['baseline'], sg[k]['branch']
    ev.append(f'<tr><td>{esc(k[0])} #{k[1]}: {esc(new[k]["question"][:110])}</td><td class="{cls(old["grade"])}">{old["grade"]}</td>'
              f'<td class="{cls(nw["grade"])}">{nw["grade"]}</td><td class="neu">{esc(nw["why"][:160])}</td>'
              f'<td>{link(f"../search-quality-2026-10-28/traces/trace-{res[k]["result"]["run_id"]}.json", "old")} · '
              f'{link(f"../structured-chunking-2026-10-28/traces/trace-{new[k]["result"]["run_id"]}.json", "new")}</td></tr>')
ev.append('</table></div></section>')

P = E / 'parent-child-answers-2026-10-28'
pkey = json.load(open(P / 'key.json')); queries = {r['id']: r['query'] for r in rows_of(P / 'answers.jsonl')}
ev.append('<section class="card" id="parent"><h3>Parent–child: passage only vs whole section, 22 answers (graded blind)</h3><div class="scroll"><table>'
          '<tr><th>Question</th><th>Passage only</th><th>Whole section</th><th>Grader on whole section</th></tr>')
for r in rows_of(P / 'grades.jsonl'):
    m = pkey[r['id']]; off = r['A'] if m['A'] == 'off' else r['B']; on = r['B'] if m['A'] == 'off' else r['A']
    ev.append(f'<tr><td>{r["id"]}: {esc(queries.get(r["id"], "")[:110])}</td><td class="{cls(off["grade"])}">{off["grade"]}</td>'
              f'<td class="{cls(on["grade"])}">{on["grade"]}</td><td class="neu">{esc(on["why"][:160])}</td></tr>')
ev.append('</table></div></section>')
(HERE / 'evidence.html').write_text(page('rag-mcp evidence', '\n'.join(ev)))
print('report.html and evidence.html written')
