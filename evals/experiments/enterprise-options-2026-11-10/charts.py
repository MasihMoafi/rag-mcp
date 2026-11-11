"""Draw the README charts for this experiment from results.json.

    python evals/experiments/enterprise-options-2026-11-10/charts.py

Writes assets/enterprise-answers.svg and assets/enterprise-categories.svg. Colours follow the
README's existing charts; the teal is stepped to pass the dark-surface palette checks.
"""

import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[2] / "assets"
SURFACE, INK, MUTED, GRID = "#100f0e", "#efeae2", "#9a9188", "#2c2924"
SHIPPED, CONTEXT = "#1fa396", "#6f675d"  # in rag 0.3 / baseline or rejected
SANS = "ui-sans-serif,system-ui,sans-serif"
STYLE = (
    f".ax{{fill:{MUTED};font-size:11px;font-family:{SANS}}}"
    f".lab{{fill:{INK};font-size:12.5px;font-family:{SANS}}}"
    f".val{{fill:{MUTED};font-size:12px;font-family:{SANS}}}"
    f".grid{{stroke:{GRID};stroke-width:1}}"
)
NAMES = {
    "basic": "Basic",
    "semantic": "Semantic",
    "intra_document_reasoning": "Reasoning within a document",
    "project_related": "Project related",
    "constrained": "Constrained",
    "completeness": "Completeness",
    "miscellaneous": "Miscellaneous",
    "info_not_found": "Info not found",
    "conflicting_info": "Conflicting info",
    "high_level": "High level",
}


def svg(width, height, title, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title><style>{STYLE}</style>"
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>{body}</svg>\n'
    )


def percent(value):
    """Whole percent, halves rounded up (92.5 -> 93), as in the experiment record."""
    return int(value + 0.5)


def heading(title, subtitle):
    return (
        f'<text x="30" y="34" fill="{INK}" font-family="ui-serif,Georgia,serif" font-size="22" '
        f'font-weight="600">{escape(title)}</text>'
        f'<text x="30" y="56" class="val">{escape(subtitle)}</text>'
    )


def legend(x, y, items):
    out = ""
    for colour, text in items:
        out += (
            f'<rect x="{x}" y="{y - 10}" width="11" height="11" rx="3" fill="{colour}"/>'
            f'<text x="{x + 16}" y="{y}" class="val">{escape(text)}</text>'
        )
        x += 16 + 7 * len(text) + 24
    return out


def axis(left, right, top, bottom):
    """Gridlines at 0, 25, 50, 75 and 100 percent; returns the body and the x scale."""
    scale = lambda value: left + (right - left) * value / 100
    body = ""
    for tick in (0, 25, 50, 75, 100):
        x = scale(tick)
        body += (
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>'
            f'<text x="{x:.1f}" y="{bottom + 16}" class="ax" text-anchor="middle">{tick}%</text>'
        )
    return body, scale


def answers_chart(results):
    width, height = 860, 350
    left, right, top = 300, 800, 118
    rows = [
        ("A", "rag 0.2", False),
        ("B", "+ titles, stemming, balanced shortlist", True),
        ("C", "+ short documents read whole", True),
        ("D", "C + lenient refusal rule", False),
    ]
    pitch, bar = 38, 22
    bottom = top + pitch * len(rows) + 4
    body, scale = axis(left, right, top - 8, bottom)
    paper = results["paper_reference"]["correct"]
    x = scale(paper)
    for index, (arm, label, shipped) in enumerate(rows):
        value = results["arms"][arm]["correct"]
        y = top + index * pitch
        end = scale(value)
        colour = SHIPPED if shipped else CONTEXT
        # A value label the reference line would cross goes inside its bar.
        inside = end < x < end + 52
        place = (
            f'x="{end - 8:.1f}" text-anchor="end"' if inside else f'x="{end + 8:.1f}"'
        )
        body += (
            f'<text x="{left - 12}" y="{y + 15.5}" class="lab" text-anchor="end">'
            f"{arm} · {escape(label)}</text>"
            f'<rect x="{left}" y="{y}" width="{end - left:.1f}" height="{bar}" rx="3" fill="{colour}">'
            f"<title>{arm}: {value}% correct</title></rect>"
            f'<text {place} y="{y + 15.5}" class="lab">{value}%</text>'
        )
    body += (
        f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{bottom}" stroke="{INK}" '
        f'stroke-width="1.5" stroke-dasharray="4 4"/>'
        f'<text x="{x:.1f}" y="{top - 18}" class="val" text-anchor="middle">'
        f"Dataset paper's best: {paper}%</text>"
    )
    notes = (
        (
            f"{results['questions']} questions, {results['documents']:,} documents. "
            "Each step adds to the one above; B and C each beat the step before (paired test, p < 0.001)."
        ),
        (
            "The paper's best is BM25 over all 511,962 documents with GPT-5.4; a larger corpus is harder, "
            "so it is a reference, not a like-for-like score."
        ),
    )
    for offset, note in enumerate(notes):
        body += f'<text x="30" y="{bottom + 44 + 18 * offset}" class="ax">{escape(note)}</text>'
    body += legend(30, 86, [(SHIPPED, "in rag 0.3"), (CONTEXT, "before, or rejected")])
    title = "EnterpriseRAG-Bench: correct answers"
    subtitle = "Qwen3 embedding 0.6B and reranker 4B; answers by GPT-6.1 Sol, graded with the benchmark's own judge prompt."
    return svg(width, height, title, heading(title, subtitle) + body)


def categories_chart(results):
    categories = sorted(results["categories"].items(), key=lambda item: -item[1]["n"])
    width, pitch = 860, 30
    left, right, top = 290, 740, 100
    bottom = top + pitch * len(categories)
    height = bottom + 60
    body, scale = axis(left, right, top - 14, bottom - 6)
    for index, (name, row) in enumerate(categories):
        y = top + index * pitch
        before, after = scale(row["A"]), scale(row["C"])
        body += (
            f'<text x="{left - 12}" y="{y + 4.5}" class="lab" text-anchor="end">'
            f'{escape(NAMES[name])} <tspan class="ax">({row["n"]})</tspan></text>'
            f'<line x1="{before:.1f}" y1="{y}" x2="{after:.1f}" y2="{y}" stroke="{CONTEXT}" stroke-width="2"/>'
            f'<circle cx="{before:.1f}" cy="{y}" r="5.5" fill="{CONTEXT}" stroke="{SURFACE}" stroke-width="2">'
            f"<title>{escape(NAMES[name])}, rag 0.2: {row['A']}%</title></circle>"
            f'<circle cx="{after:.1f}" cy="{y}" r="5.5" fill="{SHIPPED}" stroke="{SURFACE}" stroke-width="2">'
            f"<title>{escape(NAMES[name])}, rag 0.3: {row['C']}%</title></circle>"
            f'<text x="{right + 22}" y="{y + 4.5}" class="val">{percent(row["A"])} → '
            f'<tspan fill="{INK}">{percent(row["C"])}</tspan></text>'
        )
    body += legend(560, 82, [(CONTEXT, "rag 0.2 (A)"), (SHIPPED, "rag 0.3 (C)")])
    body += (
        f'<text x="30" y="{bottom + 40}" class="ax">Question types as the benchmark defines them; '
        "number of questions in brackets. Info-not-found questions have no answer in the corpus.</text>"
    )
    title = "Correct answers by question type"
    subtitle = f"Same {results['questions']} questions and judge; percentage of each type answered correctly."
    return svg(width, height, title, heading(title, subtitle) + body)


def main():
    results = json.loads((HERE / "results.json").read_text())
    (ASSETS / "enterprise-answers.svg").write_text(answers_chart(results))
    (ASSETS / "enterprise-categories.svg").write_text(categories_chart(results))


if __name__ == "__main__":
    main()
