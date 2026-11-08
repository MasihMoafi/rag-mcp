import json

import pytest
from evals.benchmarks.answers_codex import (
    ANSWER_SCHEMA,
    GRADE_SCHEMA,
    agreement,
    run,
)
from evals.benchmarks.quality import QUALITY_SCHEMA
from evals.benchmarks.report import build_report, percentile, ranking_metrics, sign_test


def test_precision_and_map_follow_their_definitions():
    scores = ranking_metrics(["a", "x", "b", "y", "z", "c"], ["a", "b", "c"])
    assert scores["precision@5"] == pytest.approx(2 / 5)
    assert scores["MAP"] == pytest.approx((1 + 2 / 3 + 3 / 6) / 3)


def test_a_gold_document_never_found_lowers_map():
    assert ranking_metrics(["a"], ["a", "b"])["MAP"] == pytest.approx(0.5)


def test_percentile_is_nearest_rank():
    assert percentile([1, 2, 3, 4], 0.5) == 2
    assert percentile(list(range(1, 21)), 0.95) == 19


def test_sign_test_is_exact_and_drops_ties():
    assert sign_test([1, 1, 1, 0.5], [0, 0, 0, 0.5]) == {
        "first_better": 3,
        "second_better": 0,
        "p": 0.25,
    }
    assert sign_test([1], [1])["p"] == 1.0


def test_report_compares_two_arms_on_shared_questions():
    def event(arm, question, ranked):
        return {
            "event": "complete",
            "arm": arm,
            "question_id": question,
            "ranked_doc_ids": ranked,
            "latency_seconds": 1.0,
        }

    events = [
        event("bm25", "q1", ["x"]),
        event("hybrid", "q1", ["a"]),
        event("hybrid", "q2", ["a"]),
    ]
    report = build_report(events, {"q1": ["a"], "q2": ["a"]})
    assert (
        report["arms"]["hybrid"]["questions"] == 2
        and report["arms"]["bm25"]["recall@5"] == 0
    )
    assert report["paired_nDCG@10"]["second_better"] == 1


def test_agreement_reports_chance_corrected_kappa():
    rows = [
        {"label": "correct", "blind_label": "correct"},
        {"label": "wrong", "blind_label": "wrong"},
        {"label": "correct", "blind_label": "wrong"},
        {"label": "wrong", "blind_label": None},
    ]
    result = agreement(rows)
    assert result["graded_twice"] == 3 and result["agreement"] == pytest.approx(0.6667)
    assert result["cohen_kappa"] == pytest.approx(0.4)
    assert agreement([{"label": "correct"}]) is None


def test_blind_grader_sees_no_first_grade_and_only_the_sample_is_graded_twice(tmp_path):
    prompts = []

    def call(prompt, schema, model, _effort):
        prompts.append((model, schema is GRADE_SCHEMA, prompt))
        if schema is ANSWER_SCHEMA:
            return {"text": "Ten [d1].", "citations": ["d1"]}
        if schema is QUALITY_SCHEMA:
            return {"claims": [], "relevance": "full", "passages": [True], "facts": []}
        return {"label": "correct", "reason": "FIRST-JUDGE-REASON"}

    jobs = [
        {
            "id": f"q{n}",
            "arm": "hybrid_qwen",
            "question": "What?",
            "gold_answer": "Ten.",
            "passages": [{"doc_id": "d1", "text": "Ten."}],
        }
        for n in range(3)
    ]
    summary = run(
        jobs,
        tmp_path,
        "writer",
        "e",
        1,
        call=call,
        judge="terra",
        blind_judge="blind-sol",
        blind_sample=1,
    )
    blind = [p for p in prompts if p[0] == "blind-sol"]
    assert len(blind) == 1 and blind[0][1]
    assert (
        "FIRST-JUDGE-REASON" not in blind[0][2]
        and "hybrid_qwen" not in blind[0][2]
        and "writer" not in blind[0][2]
    )
    assert summary["judge_agreement"]["graded_twice"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "answers.jsonl").read_text().splitlines()
    ]
    assert all(len(row["prompts_sha256"]) == 64 for row in rows)
