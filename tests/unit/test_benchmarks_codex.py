import json

from evals.benchmarks.answers_codex import ANSWER_SCHEMA, run, subscription_env


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
    return {"label": "correct", "reason": "matches"}


def test_child_environment_has_no_api_keys():
    environ = {"OPENAI_API_KEY": "x", "OPENROUTER_API_KEY": "y", "HOME": "/h"}
    assert subscription_env(environ) == {"HOME": "/h"}


def test_rerun_skips_finished_questions(tmp_path):
    calls = []

    def counting(prompt, schema, model, effort):
        calls.append(prompt)
        return good_call(prompt, schema, model, effort)

    assert run([job(1)], tmp_path, "m", "e", 1, call=counting)["correct"] == 1
    summary = run([job(1), job(2)], tmp_path, "m", "e", 1, call=counting)
    assert summary["answered"] == 2 and len(calls) == 4


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
        if len(seen) > 2:
            raise RuntimeError("limit reached")
        return good_call(prompt, schema, model, effort)

    summary = run([job(n) for n in range(10)], tmp_path, "m", "e", 1, call=failing)
    assert (
        summary["stopped_early"] and summary["answered"] == 1 and summary["failed"] == 3
    )
