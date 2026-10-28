# Structured parsing and book windows (28 Oct 2026)

**Status:** done, 28 Oct 2026. The branch answered 48 of 60 exactly against 39, with no misses; one question fell from exact to partial, so the merge rule's "loses none" clause was not met; Masih chose to merge on 28 Oct (main `7d1f502`, `f39038b`).

## Question

On the two narrative books, does retrieval improve when PDFs are parsed into paragraphs and chapter headings (running headers and page numbers removed, ligature glyphs fixed) and book-length text is chunked in 1,600-character windows with 250 overlap, instead of today's flattened text in 700-character windows?

## Why only these questions

The change acts on book-length PDFs. Napoleon (30 questions) and Fire and Blood (30) are the two such collections, and they hold most of today's partial grades ("right scene, missed the key fact"). The Attention paper is short, the codebase is not PDF, and Brain and Behavior enters as extracted text past the 16 MB import limit, so none of them would exercise the change.

## Decision it informs

Merge branch `feat/structured-pdf-chunking` (`dd28cd2`) into main if the variant finds the answer exactly for at least 3 more of the 60 questions and loses none that the baseline found; otherwise keep the branch unmerged and report what it changed.

## What is compared

Baseline: today's search results for these 60 questions (`../search-quality-2026-10-28/results.jsonl`, main at `0ad942a`). Variant: the same search on the branch. Fixed: questions, documents, Qwen3 Embedding 8B, Cohere Rerank 4 Fast, top 5. Changed together: the PDF parser and the chunk windows, so the result credits the pair.

## Measures

One Opus 5.5 subagent grades each question's two top-5 sets side by side, labelled A and B in random order, with the rubric of the search-quality run (exact, partial, miss); the mapping stays out of its reach until grading ends. Reported: counts per setting and every question that changed.

## Cost and safety

OpenRouter key: 60 searches × $0.002 = $0.12 plus embedding the two books (about 2 million characters, under $0.01). A 5-question smoke test runs first; any error stops the run. Isolated settings, index, workspaces, traces and lock.

## Result

| | Exact | Partial | Miss |
|---|---|---|---|
| Baseline (today's main) | 39 | 20 | 1 |
| Branch (`dd28cd2`) | **48** | 12 | 0 |
| Napoleon, baseline → branch | 21 → 27 | 8 → 3 | 1 → 0 |
| Fire and Blood, baseline → branch | 18 → 21 | 12 → 9 | 0 → 0 |

- **Per question:** 11 better (10 partial to exact, Napoleon #8 miss to partial), 1 worse: Fire and Blood #9 fell from exact to partial (its answer split across passages: the Shepherd's words are in rank 4 without the full point).
- **Decision:** the gain (+9 exact) clears the 3-question bar, but the rule also asked for no loss, and one exact answer became partial. Merging is Masih's call.
- **How it ran:** smoke test of 5 per book, then the rest; no errors. The packet (678k characters) was graded by two blind Opus 5.5 subagents, 30 questions each, instead of one, with the search-quality rubric; the A/B key stayed in `~/.cache` until grading ended. Grades: `grades-*.jsonl`; key: `key.json`; counts: `summary.json`.
- **Spend:** 60 searches, about $0.12 in reranking plus embedding the two books, on the OpenRouter key.
- **Limits:** parsing and windows changed together, so the gain belongs to the pair. Fire and Blood's chapter titles are artwork, so its chapters are numbered from their decorated initials.
