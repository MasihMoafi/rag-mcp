"""Standalone chart images for the README and the website, from data.json, in the Elpis eval dashboard's
chart style on a dark background (the website is dark-only; the old images had a dark background too)."""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
DASH = Path.home() / 'Desktop/p/Elpis-wt-install/docs/evals/dashboard'
sys.path.insert(0, str(DASH))
import charts  # noqa: E402

D = json.loads((HERE / 'data.json').read_text())
TEAL, GREY = '#28c2b5', '#6f675d'
STYLE = ('<style>.grid{stroke:#2c2924;stroke-width:1}.ax{fill:#9a9188;font-size:11px;font-family:ui-sans-serif,system-ui,sans-serif}'
         '.blab{fill:#efeae2;font-size:12.5px;font-family:ui-sans-serif,system-ui,sans-serif}'
         '.bval{fill:#9a9188;font-size:12px;font-family:ui-sans-serif,system-ui,sans-serif}</style>')


def image(title, subtitle, legend, inner, w=1020, h=380):
    body = re.sub(r'^<svg[^>]*>|</svg>$', '', inner)
    keys = ''.join(f'<rect x="{w - 300 + i * 150}" y="30" width="11" height="11" rx="3" fill="{c}"/>'
                   f'<text x="{w - 284 + i * 150}" y="40" class="bval">{n}</text>' for i, (n, c) in enumerate(legend))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h + 70}" viewBox="0 0 {w} {h + 70}" role="img" aria-label="{title}">'
            f'<title>{title}</title>{STYLE}<rect width="{w}" height="{h + 70}" fill="#100f0e"/>'
            f'<text x="30" y="34" fill="#efeae2" font-family="ui-serif,Georgia,serif" font-size="22" font-weight="600">{title}</text>'
            f'<text x="30" y="56" class="bval">{subtitle}</text>{keys}<g transform="translate(0,70)">{body}</g></svg>\n')


sq, st = D['search'], D['structured']
cols = list(sq['by_collection'])
a = image('Exact answers per collection: August vs October 2026',
          f'153 questions, top 5 passages, one grading model and rubric for both runs. Total {sq["runs"]["August, isolated"]["exact"]} → {sq["runs"]["Today"]["exact"]}.',
          [('August', GREY), ('October', TEAL)],
          charts.grouped([c.replace(' and ', ' &amp; ') for c in cols],
                         [('August', [sq['by_collection'][c]['August, isolated']['exact'] for c in cols], GREY),
                          ('October', [sq['by_collection'][c]['Today']['exact'] for c in cols], TEAL)], h=380))
books = list(st['by_book'])
b = image('Books: flat text vs structured parsing',
          f'60 questions on two books, graded blind. Exact answers {st["baseline"]["exact"]} → {st["branch"]["exact"]}; 10 questions gained, 1 lost.',
          [('Flat text', GREY), ('Structured', TEAL)],
          charts.grouped(books, [('Flat text', [st['by_book'][x]['baseline']['exact'] for x in books], GREY),
                                 ('Structured', [st['by_book'][x]['branch']['exact'] for x in books], TEAL)], h=380))
runs = sq['runs']
hero = image('rag-mcp retrieval, October 2026', '153 questions with known answers, top 5 passages. Same grading model and rubric for both runs.', [],
             charts.hbars([('Exact, August', runs['August, isolated']['exact'], GREY), ('Exact, October', runs['Today']['exact'], TEAL),
                           ('Exact or partial, October', runs['Today']['exact'] + runs['Today']['partial'], TEAL)],
                          w=720, pl=210, vmax=153, fmtf=lambda v: f'{v:.0f} of 153'), w=720, h=150)
REPO = HERE.parent.parent.parent
(REPO / 'assets/retrieval-comparison.svg').write_text(a)
(REPO / 'assets/retrieval-hero.svg').write_text(hero)
(REPO / 'assets/retrieval-structured-parsing.svg').write_text(b)
SITE = Path.home() / 'Desktop/p/masih-website/public/rag-mcp'
(SITE / 'retrieval-comparison.svg').write_text(a)
(SITE / 'retrieval-structured-parsing.svg').write_text(b)
print('written: README assets and website public/rag-mcp')
