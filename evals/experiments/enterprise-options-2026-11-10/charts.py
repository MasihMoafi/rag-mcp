"""Draw the README charts for this experiment from results.json, in light and dark versions.

    python3 evals/experiments/enterprise-options-2026-11-10/charts.py   # needs matplotlib and Inter

Writes assets/enterprise-{answers,categories}-{light,dark}.svg. Text is converted to outlines, so
GitHub shows the same font everywhere. Colours passed the dataviz palette checks for each surface;
the light grey sits below 3:1 contrast, so every mark carries a direct label.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[2] / "assets"
THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "grid": "#e6e8eb",
        "before": "#aeb4bd",
        "step": "#7fbfb5",
        "shipped": "#0a7a6d",
        "line": "#8c959f",
    },
    "dark": {
        "surface": "#0d1117",
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "grid": "#262c36",
        "before": "#5c6370",
        "step": "#14796d",
        "shipped": "#1aa391",
        "line": "#6e7681",
    },
}
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


def setup():
    if not any("Inter" in font.name for font in font_manager.fontManager.ttflist):
        raise SystemExit("Install the Inter font first")
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "svg.fonttype": "path",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )


def percent(value):
    """Whole percent, halves rounded up (92.5 -> 93), as in the experiment record."""
    return int(value + 0.5)


def frame(theme, size):
    figure = plt.figure(figsize=size, dpi=100)
    figure.patch.set_facecolor(theme["surface"])
    return figure


def titles(figure, theme, title, subtitle):
    figure.text(
        0.045, 0.94, title, fontsize=19, fontweight=650, color=theme["ink"], va="top"
    )
    figure.text(0.045, 0.865, subtitle, fontsize=11.5, color=theme["muted"], va="top")


def footnotes(figure, theme, lines, top):
    for index, line in enumerate(lines):
        figure.text(
            0.045,
            top - index * 0.042,
            line,
            fontsize=9.5,
            color=theme["muted"],
            va="top",
        )


def rounded_bar(axes, x, height, width, colour, edge=None, hatch=None):
    """A bar with rounded top corners only, anchored to the baseline."""
    if edge:  # an outlined reference bar: square, so the hatch reads cleanly
        axes.add_patch(
            plt.Rectangle(
                (x - width / 2, 0),
                width,
                height,
                facecolor=colour,
                edgecolor=edge,
                linewidth=1.4,
                hatch=hatch,
            )
        )
        return
    radius = 0.06
    axes.add_patch(
        FancyBboxPatch(
            (x - width / 2, 0),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            mutation_aspect=40,
            facecolor=colour,
            edgecolor=edge or colour,
            linewidth=1.4,
            hatch=hatch,
        )
    )
    axes.add_patch(
        plt.Rectangle(
            (x - width / 2, 0),
            width,
            min(height, 3),
            facecolor=colour,
            edgecolor=edge or colour,
            linewidth=1.4 if edge else 0,
            hatch=hatch,
        )
    )


def answers(results, theme, path):
    arms = results["arms"]
    paper = results["paper_reference"]["correct"]
    bars = [
        ("rag 0.2", "before", arms["A"]["correct"], theme["before"], None),
        (
            "+ titles, stemming,\nbalanced shortlist",
            "step 1",
            arms["B"]["correct"],
            theme["step"],
            None,
        ),
        (
            "+ short documents\nread whole = rag 0.3",
            "step 2",
            arms["C"]["correct"],
            theme["shipped"],
            None,
        ),
        (
            "Dataset paper's best\nBM25, full corpus †",
            "reference",
            paper,
            theme["surface"],
            "///",
        ),
    ]
    figure = frame(theme, (10, 6.2))
    titles(
        figure,
        theme,
        "Correct answers on EnterpriseRAG-Bench",
        f"{results['questions']} questions over company documents (Slack, email, tickets, docs, code); "
        f"{results['documents']:,}-document sample",
    )
    axes = figure.add_axes((0.06, 0.3, 0.9, 0.48))
    axes.set_facecolor(theme["surface"])
    width = 0.56
    for index, (label, _, value, colour, hatch) in enumerate(bars):
        if hatch:
            plt.rcParams["hatch.color"] = theme["line"]
            rounded_bar(
                axes, index, value, width, colour, edge=theme["line"], hatch=hatch
            )
        else:
            rounded_bar(axes, index, value, width, colour)
        axes.text(
            index,
            value + 2.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight=650,
            color=theme["muted"] if hatch else theme["ink"],
        )
    for start, end in ((0, 1), (1, 2)):
        gain = bars[end][2] - bars[start][2]
        axes.annotate(
            "",
            xy=(end - width / 2 - 0.04, bars[end][2] - 6),
            xytext=(start + width / 2 + 0.04, bars[start][2] - 6),
            arrowprops={
                "arrowstyle": "-|>",
                "color": theme["muted"],
                "lw": 1.2,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 11,
            },
        )
        axes.text(
            (start + end) / 2,
            (bars[start][2] + bars[end][2]) / 2 - 3.2,
            f"+{gain:.1f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight=600,
            color=theme["muted"],
        )
    axes.set_xlim(-0.55, len(bars) - 0.45)
    axes.set_ylim(0, 100)
    axes.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    axes.set_xticks(range(len(bars)), [label for label, *_ in bars])
    axes.tick_params(axis="x", length=0, pad=10, labelsize=11, labelcolor=theme["ink"])
    axes.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=theme["muted"])
    axes.grid(axis="y", color=theme["grid"], linewidth=1)
    axes.set_axisbelow(True)
    axes.spines["bottom"].set_color(theme["line"])
    footnotes(
        figure,
        theme,
        [
            "Both rag steps beat the one before on the same questions (paired McNemar test, p < 0.001).",
            "Answers by GPT-6.1 Sol, graded with the benchmark's own judge prompt.",
            (
                "† Not like for like: the paper searches all 511,962 documents with GPT-5.4 answering; "
                "a larger corpus is harder."
            ),
        ],
        0.125,
    )
    figure.savefig(path, facecolor=theme["surface"])
    plt.close(figure)


def categories(results, theme, path):
    rows = sorted(results["categories"].items(), key=lambda item: item[1]["n"])
    figure = frame(theme, (10, 6.6))
    titles(
        figure,
        theme,
        "Where rag 0.3 gains",
        "Share of each question type answered correctly, rag 0.2 → 0.3 (same questions and judge)",
    )
    axes = figure.add_axes((0.3, 0.12, 0.52, 0.66))
    axes.set_facecolor(theme["surface"])
    for y, (name, row) in enumerate(rows):
        before, after = row["A"], row["C"]
        axes.plot(
            [before, after],
            [y, y],
            color=theme["before"],
            lw=2.4,
            solid_capstyle="round",
            zorder=1,
            clip_on=False,
        )
        axes.scatter(
            [before],
            [y],
            s=90,
            color=theme["before"],
            edgecolor=theme["surface"],
            linewidth=2,
            zorder=2,
            clip_on=False,
        )
        axes.scatter(
            [after],
            [y],
            s=110,
            color=theme["shipped"],
            edgecolor=theme["surface"],
            linewidth=2,
            zorder=3,
            clip_on=False,
        )
        gain = percent(after) - percent(before)
        axes.text(
            103,
            y,
            f"{percent(after)}%",
            va="center",
            fontsize=11,
            fontweight=650,
            color=theme["ink"],
        )
        axes.text(
            113,
            y,
            f"+{gain}" if gain else "±0",
            va="center",
            fontsize=10,
            color=theme["muted"],
        )
        if before != after:
            axes.text(
                before - 2.2,
                y,
                f"{percent(before)}",
                va="center",
                ha="right",
                fontsize=9.5,
                color=theme["muted"],
            )
    axes.set_yticks(range(len(rows)), [f"{NAMES[name]}  " for name, _ in rows])
    for tick, (_, row) in zip(axes.get_yticklabels(), rows):
        tick.set_color(theme["ink"])
    for y, (_, row) in enumerate(rows):
        axes.text(
            -1.5,
            y - 0.36,
            f"{row['n']} questions  ",
            ha="right",
            va="center",
            fontsize=8.5,
            color=theme["muted"],
            transform=axes.transData,
        )
    axes.set_xlim(0, 100)
    axes.set_ylim(-0.7, len(rows) - 0.4)
    axes.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    axes.tick_params(axis="y", length=0, labelsize=11.5, pad=4)
    axes.tick_params(axis="x", length=0, labelsize=9.5, labelcolor=theme["muted"])
    axes.grid(axis="x", color=theme["grid"], linewidth=1)
    axes.set_axisbelow(True)
    axes.spines["bottom"].set_visible(False)
    axes.set_clip_on(False)
    for x, colour, label in (
        (0.3, theme["before"], "rag 0.2"),
        (0.42, theme["shipped"], "rag 0.3"),
    ):
        figure.add_artist(
            Line2D(
                [x],
                [0.808],
                marker="o",
                markersize=9,
                color=colour,
                linestyle="",
                transform=figure.transFigure,
            )
        )
        figure.text(
            x + 0.014, 0.808, label, fontsize=10.5, color=theme["ink"], va="center"
        )
    for x, header in ((103, "rag 0.3"), (113, "gain")):
        axes.text(
            x, len(rows) - 0.25, header, fontsize=9, color=theme["muted"], va="center"
        )
    footnotes(
        figure,
        theme,
        [
            (
                "Question types as the benchmark defines them. "
                "Info-not-found questions have no answer in the corpus; rag declines them all."
            )
        ],
        0.06,
    )
    figure.savefig(path, facecolor=theme["surface"])
    plt.close(figure)


def main():
    setup()
    results = json.loads((HERE / "results.json").read_text())
    for name, theme in THEMES.items():
        answers(results, theme, ASSETS / f"enterprise-answers-{name}.svg")
        categories(results, theme, ASSETS / f"enterprise-categories-{name}.svg")


if __name__ == "__main__":
    main()
