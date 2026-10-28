# Parent–child answers (28 Oct 2026)

**Status:** done, 28 Oct 2026. Parent–child did not help: 21 of 22 correct with it off and with it on. By the rule set before the run, the work is dropped; Masih decides.

## Question

Do answers improve when the answer model reads the whole section around each passage search found (parent–child), instead of the passage alone?

## Decision it informs

Whether to finish and commit the parent–child work that has sat uncommitted since 10 Oct, or drop it. Set now, before any result: **keep** if "on" answers at least 2 more questions correctly and gets none wrong that "off" got right; **drop** if it answers the same number or fewer correctly. Anything between is no evidence either way, and Masih decides.

## What is compared

One setting, `parent_context`, off against on. Everything else is fixed. One confounder comes with the setting itself: "on" re-cuts the book into smaller pieces for search, a quarter of each section, so search sees different pieces as well. The comparison is the parent–child mode as a whole.

## Before the run

The parent–child code (12 changed and 5 new files) has never been tested. It is copied into a worktree and its tests run first. If they fail, the experiment stops there with no API spend, and the failure is reported.

## Inputs

- **Questions:** the 22 anchored questions on `~/Desktop/books/rag`, the same as the 29 Sep run (`../book-answers-2026-09-29/run.py`).
- **Search:** BM25 plus `qwen/qwen3-embedding-8b`, RRF to 50, `cohere/rerank-4-fast` top 5.
- **Answers:** `deepseek/deepseek-v4.1-flash`, through `rag query` exactly as a user runs it.
- **Isolation and key:** as in `../search-quality-2026-10-28/experiment.md`.

## Measures

- Opus 5.5 subagents on Masih's Claude subscription grade with the 29 Sep rubric (that run's judge was Sonnet 5.5 through the API): **correct**, **partial**, **wrong** or **abstained**, against the book passage around each question's anchor. The graders are not told which mode wrote an answer, and the order is shuffled.
- Reported: counts per mode, every question whose grade changed, and input tokens per answer (what "on" costs).
- Reference, 29 Sep, older setup: 17 correct, 4 partial, 0 wrong, 1 abstained. With 22 questions, a one-question difference is noise.

## Cost

Reranking 44 × $0.002 = $0.09; answers about $0.07; embedding under $0.01. Grading runs on the subscription. **About $0.17; stop at $0.50.**

## Evidence

`results.jsonl` (question, mode, answer, grade and reason, tokens), `run.py`, `manifest.json`. Book text stays out of git.

## Result

| Setting | Correct | Partial | Wrong | Abstained |
|---|---|---|---|---|
| Off | 21 | 1 | 0 | 0 |
| On | 21 | 0 | 0 | 1 |

- **Changed grades:** two of 22. Question p4 rose from partial to correct with parent–child on; question 10b fell from correct to abstained ("the sources do not answer").
- **Decision:** the rule set before the run said keep only if "on" gained at least 2 correct answers and lost none. It gained none and lost one, so the rule says drop.
- **Limits:** with parent–child off the system already answered 21 of 22, so these questions leave almost no room to show a gain. A harder set (long passages, answers spread over a section) could still favour it. "On" answers were a little longer (median 391 characters against 321).
- **Graders and spend:** one Opus 5.5 subagent graded blind with the 29 Sep rubric, word for word; the mode behind each answer stayed in `key.json`, out of the grader's reach until grading ended. About $0.13 on the OpenRouter key across all five attempts (two lost to the engine lock, one stopped when the experiment was tabled, the smoke test and the full run), by the upper end of OpenRouter's usage readings, which lag; `run.log` holds the per-answer checkpoints. Grades: `grades.jsonl`; unblinded counts: `summary.json`.
