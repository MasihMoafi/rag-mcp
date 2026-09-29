# Book answer grounding and bounded abstention — 2026-09-23

Decision: the old answer instruction could give a misleading corpus-wide
abstention, and a first wording change produced an unsupported answer. The
revised evidence-specific instruction abstained correctly on that known
failure in one replay. This is a targeted fix and check, **not** a measured
grounding rate or evidence that the generator is generally safe.

The [initial manifest](../../evals/experiments/book-answer-grounding-2026-09-23/manifest.json)
fixed three questions on the published nine-chapter, 884-chunk book index.
The same imported workspace and OpenRouter `qwen/qwen3-embedding-4b` query
embedding fed 50 candidates to equal-weight fusion and retained five, with
reranking and contextual retrieval off. Studio-style synthesis used
`deepseek/deepseek-v4.1-flash`, a 240-token answer cap, and bracket citations.
Manual inspection found the scan/OCR and offline-evaluation answers supported
by their cited passages. The uncertain entity-match answer abstained but
falsely claimed the *indexed material* lacked the rule; that rule exists
elsewhere in the book. It also cited passages that only established adjacent
topics. The original three successful calls took 25.42 seconds, peaked at
687,972 KiB RSS, and reported $0.00037125.

The [spec](../specs-2026-09-23-honest-abstention.md) changed the no-passage
fallback and prompt to scope insufficiency to supplied evidence. A first
live replay of the known miss then answered with entity-merge logic and cited
passage [4], although the question concerned *uncertain matching* and that
passage covered a *known duplicate*. This was an unsupported substitution,
not progress. After a generic instruction to check the requested relation
and conditions, a [registered follow-up](../../evals/experiments/book-answer-grounding-2026-09-23/strict-support-followup.manifest.json)
produced:

| Question | Registered expectation | Observed, manually checked |
| --- | --- | --- |
| 9b, uncertain match | Exact abstention, no citation | Exact `I don't have enough information to answer that question.`; no citation. The rule was still absent from the five passages. |
| 9a, particular historical date | Abstention, no citation | **Expectation failed.** The answer cited [1] and [2] for a valid alternate interpretation: vector search can fail to link the year to the correct entity in a multi-hop movie example. These passages did not support the intended *past-versus-present blurring* explanation, but the question did not distinguish the two. |
| 4b, live feedback | Supported cited answer | Feedback and investigation of bad outputs were directly supported by cited [1]. |

Thus the registered all-three success condition was **not met**. The 9a
answer is not automatically a hallucination: its cited passages support the
answer it actually gave. The label was too narrow for the query wording.
The three successful follow-up calls took 15.15 seconds, peaked at 680,436
KiB RSS, and reported $0.00063342. A separate
[exploratory clarification](../../evals/experiments/book-answer-grounding-2026-09-23/temporal-clarification.manifest.json)
asked explicitly about current information for a past-date query. Retrieval
then put the intended past/present passage at rank one and the cited answer
was supported. It took 11.46 seconds, peaked at 640,048 KiB RSS, and
reported $0.00014290. That after-the-fact rewrite cannot validate the
registered negative case.

The [runner](../../evals/experiments/book-answer-grounding-2026-09-23/run.py)
checks source SHA, source count/bytes, chunk count, workspace revision and
run settings before each batch. It now accepts an optional `EVAL_MANIFEST`
path while retaining the original default; each redacted report records its
manifest SHA. Private answers and retrieved evidence, progress, and traces
remain under `/tmp/rag-book-hosted-20260923/`. Trace content capture was off;
the provider key was not written to reports. Calls used only CPU cores 4–5,
low scheduling priority, 8 GiB virtual-memory and 120-second outer caps;
there was no local Ollama run. Across initial, first replay, follow-up, and
clarification, reported provider cost was $0.00129052.

The adjacent hosted/prompt/settings tests passed (53/53). A final offline
wheel, SHA-256
`b36f69b092bba1a2665cd83190871bb41da661b6ec5c9886cecde78481e33d79`,
was rebuilt and installed into the previously clean Python 3.13 base venv;
inspection of the installed function confirmed the new instruction and
zero-passage fallback. No new full install, browser check, or broad answer
benchmark followed that prompt-only change. Prompt compliance on one known
negative and two positives does not guarantee citation support elsewhere.
The next quality gate needs user-reviewed answer-support labels and a fresh,
unambiguous holdout before any promotion claim.
