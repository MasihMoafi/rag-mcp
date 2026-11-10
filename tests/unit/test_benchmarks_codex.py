import json

from evals.benchmarks.answers_codex import (
    ANSWER_SCHEMA,
    DEFAULT_JUDGE,
    GRADE_SCHEMA,
    run,
    subscription_env,
)
from evals.benchmarks.quality import QUALITY_SCHEMA, average_precision, quality_scores


def job(number):
    return {
        "id": f"q{number}",
        "arm": "hybrid_qwen",
        "question": "What?",
        "gold_answer": "Ten.",
        "passages": [{"doc_id": "d1", "text": "Ten."}],
    }


def good_call(_prompt, schema, _model, _effort):
    if schema is ANSWER_SCHEMA:
        return {"text": "Ten [d1].", "citations": ["d1"]}
    if schema is QUALITY_SCHEMA:
        return {
            "claims": [{"claim": "Ten", "supported": True}],
            "relevance": "full",
            "passages": [True],
            "facts": [],
        }
    return {"label": "correct", "reason": "matches"}


def test_child_environment_has_no_api_keys():
    environ = {"OPENAI_API_KEY": "x", "OPENROUTER_API_KEY": "y", "HOME": "/h"}
    assert subscription_env(environ) == {"HOME": "/h"}


def test_child_environment_never_uses_the_vpn():
    # 9 Nov: inherited proxy variables sent two benchmark runs through his paid VPN.
    environ = {
        "HTTPS_PROXY": "p",
        "https_proxy": "p",
        "HTTP_PROXY": "p",
        "http_proxy": "p",
        "ALL_PROXY": "p",
        "all_proxy": "p",
        "PATH": "/b",
    }
    assert subscription_env(environ) == {"PATH": "/b"}


def test_rerun_skips_finished_questions(tmp_path):
    calls = []

    def counting(prompt, schema, model, effort):
        calls.append(prompt)
        return good_call(prompt, schema, model, effort)

    assert run([job(1)], tmp_path, "m", "e", 1, call=counting)["correct"] == 1
    summary = run([job(1), job(2)], tmp_path, "m", "e", 1, call=counting)
    assert summary["answered"] == 2 and len(calls) == 6


def test_a_different_model_judges_the_answer(tmp_path):
    used = {}

    def recording(prompt, schema, model, effort):
        used[id(schema)] = model
        return good_call(prompt, schema, model, effort)

    run([job(1)], tmp_path, "writer", "e", 1, call=recording)
    assert used[id(ANSWER_SCHEMA)] == "writer"
    assert (
        used[id(GRADE_SCHEMA)] == used[id(QUALITY_SCHEMA)] == DEFAULT_JUDGE != "writer"
    )


def test_quality_metrics_follow_their_definitions():
    case = dict(
        job(1),
        passages=[{"doc_id": "d1", "text": "a"}, {"doc_id": "d2", "text": "b"}],
        answer_facts=["f1", "f2"],
        gold_doc_ids=["d1"],
    )
    verdict = {
        "claims": [
            {"claim": "x", "supported": True},
            {"claim": "y", "supported": False},
        ],
        "relevance": "partial",
        "passages": [False, True],
        "facts": [
            {"in_passages": True, "in_answer": False},
            {"in_passages": True, "in_answer": True},
        ],
    }
    scores = quality_scores(
        verdict, case, {"text": "x [d1] y [d2]", "citations": ["d1", "d2"]}
    )
    assert scores == {
        "faithfulness": 0.5,
        "answer_relevance": 0.5,
        "context_relevance": 0.5,
        "context_precision": 0.5,
        "context_recall": 1.0,
        "completeness": 0.5,
        "citation_precision": 0.5,
    }
    assert average_precision([True, False, True]) == (1 + 2 / 3) / 2


def test_refusal_has_no_faithfulness_and_miscounted_verdicts_fail():
    import pytest

    case = job(1)
    verdict = {"claims": [], "relevance": "none", "passages": [False], "facts": []}
    refused = {"text": "Insufficient evidence", "citations": []}
    assert quality_scores(verdict, case, refused)["faithfulness"] is None
    with pytest.raises(ValueError, match="wrong number"):
        quality_scores(dict(verdict, passages=[]), case, refused)


def test_uncited_answer_is_retried_then_recorded_as_failed(tmp_path):
    def uncited(_prompt, schema, _model, _effort):
        return (
            {"text": "Ten.", "citations": []}
            if schema is ANSWER_SCHEMA
            else {"label": "correct", "reason": ""}
        )

    summary = run([job(1)], tmp_path, "m", "e", 1, call=uncited)
    row = json.loads((tmp_path / "answers.jsonl").read_text())
    assert (
        summary["failed"] == 1 and summary["answered"] == 0 and row["status"] == "error"
    )


def test_repeated_failures_stop_early_and_keep_results(tmp_path):
    seen = []

    def failing(prompt, schema, model, effort):
        seen.append(prompt)
        if len(seen) > 3:  # one question takes three calls: answer, grade, quality
            raise RuntimeError("limit reached")
        return good_call(prompt, schema, model, effort)

    summary = run([job(n) for n in range(10)], tmp_path, "m", "e", 1, call=failing)
    assert (
        summary["stopped_early"] and summary["answered"] == 1 and summary["failed"] == 3
    )


def test_answer_only_makes_one_call_per_question(tmp_path):
    calls = []

    def counting(prompt, schema, model, effort):
        calls.append(schema)
        return good_call(prompt, schema, model, effort)

    summary = run([job(1)], tmp_path, "m", "e", 1, call=counting, judge=None)
    assert summary["answered"] == 1 and calls == [ANSWER_SCHEMA]


def test_short_documents_are_read_whole_once_and_long_ones_keep_passages():
    from evals.benchmarks.answers_codex import whole_documents

    passages = [
        {"doc_id": "short", "text": "part one"},
        {"doc_id": "long", "text": "a slice"},
        {"doc_id": "short", "text": "part two"},
    ]
    documents = {"short": "part one ... part two", "long": "x" * 100}
    assert [p["text"] for p in whole_documents(passages, documents, 50)] == [
        "part one ... part two",
        "a slice",
    ]
