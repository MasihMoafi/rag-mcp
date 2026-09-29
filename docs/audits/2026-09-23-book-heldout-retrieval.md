# Held-out book retrieval comparison — 2026-09-23

Decision: the hosted hybrid configuration did **not** meet the preregistered
bar for further promotion on this book. The
[manifest](../../evals/experiments/book-heldout-2026-09-23/manifest.json)
fixed 18 new questions, two from each of nine Markdown chapters, before
either ranking was observed. Its rule required at least four net additional
exact-anchor hits in the top five, no more than two regressions, and all
queries within a 240-second, two-core, 8 GiB virtual-memory cap. The
[result](../../evals/experiments/book-heldout-2026-09-23/result.json) records
11/18 lexical hits and 12/18 hybrid hits: one win, no losses. The quality
threshold failed.

Both rankings used the same previously published 884-chunk LanceDB index
for the same nine chapters (477,021 bytes; aggregate source SHA-256
`2da1ed67ae230992ab2688ccde904e9226b049658d63c5f04b294db1a3426d3e`).
The lexical baseline was Tantivy FTS on that index. The variant added
OpenRouter `qwen/qwen3-embedding-4b` dense retrieval and the shipped
equal-weight reciprocal-rank fusion (`k=60`). Each stage retrieved 50
candidates; both rankings retained five. Reranking, contextual retrieval,
and answer generation were absent. This isolates the ranking choice on a
shared index; it is not a second full BM25-only engine execution.

The target anchor appeared in the dense top 50 for all 18 questions, in the
lexical top 50 for 14, and in the fused top 50 for 17. Dense-only hit@5 was
11/18. Hybrid rescued one scan/OCR question whose anchor lexical search
ranked 14th, but an example about time-specific semantic search had its
anchor dense rank one and fused rank 14. These observations suggest final
ordering is an important limitation here. They do not justify tuning fusion
weights on these same 18 questions or claiming a reranker will help.

All 18 provider calls and retrieval spans succeeded, each with 50 candidates;
traces captured no passage content. OpenRouter reported 274 query input
tokens and $0.00000548 in query costs. The low-priority process ran on CPU
cores 4 and 5, completed in 14.57 seconds wall time, and peaked at 420,264
KiB RSS. The key was absent from process logs. The manifest and runner live
under `../../evals/experiments/book-heldout-2026-09-23/`; private ranked
IDs/scores and traces remain under `/tmp/rag-book-hosted-20260923/heldout-raw`
and `/tmp/rag-book-hosted-20260923/heldout-traces`. A separate recount of
those private rank files reproduced both hit totals and verified all 18
traces as successful.

These questions were manually written from source passages after an earlier
four-question book check, but before seeing results for this set. One book
and 18 hand-authored questions cannot establish general retrieval quality.
An exact supporting phrase is a narrow relevance label; different retrieved
text may also support an answer. No answer was generated or judged. The next
quality change needs a separate confirmation set and human evidence review,
especially for the six hybrid top-five misses.

## Exploratory support review

An [agent review of the seven questions with an exact-anchor miss in either
ranking](../../evals/experiments/book-heldout-2026-09-23/posthoc-support-review.json)
checked the actual retained passages. Four questions had alternate supporting
text in both top-five sets: RAG's retrieval/generation order, the purpose of
two-stage retrieval, a sample production latency KPI, and user feedback for
bad outputs. The scan/OCR question had support only in hybrid. Neither
ranking supplied the needed temporal-similarity explanation or the rule to
escalate uncertain entity matches. This suggests the strict anchor metric
undercounts answerable questions while preserving the same one-question
relative advantage. The review was after outcomes were known, by one agent,
and anchor-positive passages were not all re-judged; it is diagnostic, not a
validated relevance score or a substitute for user labels.
