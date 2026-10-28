# Search quality today (28 Oct 2026)

**Status:** done, 28 Oct 2026. Today's search beats September's on the same 153 questions: 83.0% exact against 73.9%.

## Question

How often does today's search put the passage that answers a question among its top 5? The README's figures come from September, when search ran locally with no reranker.

## Decision it informs

The result replaces the README's retrieval figures and becomes the baseline every later change is measured against. If the overall exact rate falls more than 5 points below September's rate on the same questions, the reranker or today's chunking gets its own experiment.

## What is compared

Today's search against the September baseline on the same 153 questions. The reranker, the chunking and the embedding host have all changed since, so this is a comparison of two setups, not a test of one cause. The grader also differs (see Measures).

## Inputs

- **Questions:** 153 over the five collections whose documents were kept, in `../experiment-isolated-scaling/data/`: Attention paper 30, Brain and Behavior 30, Napoleon 30, Fire and Blood 30, mixed codebase 33.
- **Left out:** Elpis memories crate (15), rag-mcp codebase (15) and notebook (10), because their documents were not kept or have changed since; Napoleon V1 (30), an older chunking of the same book. The README's rows add up to 223, not the 203 it states.
- **Deviation:** Brain and Behavior's PDF is 141 MB, past rag's 16 MB import limit, so it is imported as the text rag's own parser extracts from it (`rag.parsing.extract_content`), as a text source rather than a PDF.
- **Search:** BM25 plus `qwen/qwen3-embedding-8b` (OpenRouter), fused by RRF to 50 candidates; `cohere/rerank-4-fast` picks the top 5. Today's default chunking; parent–child off; no answer model.
- **Code:** rag-mcp `main` in a worktree; the commit is recorded in `manifest.json`.
- **Isolation:** its own settings (`RAG_MCP_CONFIG`) and indexes (`RAG_MCP_DATA_DIR`) in this folder. Masih's settings and the main checkout are untouched.
- **Key:** the OpenRouter key (`OPENROUTER_API_KEY`).

## Measures

- Opus 5.5 subagents on Masih's Claude subscription grade each question's top 5 against the expected answer: **exact** (a passage states the answer), **partial** (on topic but incomplete) or **miss**.
- Reported: the exact rate and the exact-or-partial rate, per collection and overall, beside September's: 113/153 exact (73.9%) and 141/153 exact or partial (92.2%).
- September's grades came from a different grader, so a difference of a few points may be the grader's.

## Cost

Reranking 153 × $0.002 = $0.31; embedding about 1 M tokens at $0.01 per million, $0.01. Grading runs on the subscription, not the API. **About $0.32; stop at $1.**

## Evidence

`results.jsonl` (each question, its five passages with ranks and scores, latency, grade and the judge's reason), `run.py`, `manifest.json` (commit, settings, model names, date, spend). Book text and indexes stay out of git.

## Result

| Collection | Questions | Exact, today | Exact, Sept | Exact or partial, today | Exact or partial, Sept |
|---|---|---|---|---|---|
| attention | 30 | 28 (93%) | 26 (87%) | 30 (100%) | 29 (97%) |
| brain-and-behavior | 30 | 25 (83%) | 23 (77%) | 29 (97%) | 28 (93%) |
| napoleon | 30 | 22 (73%) | 20 (67%) | 29 (97%) | 27 (90%) |
| fire-and-blood | 30 | 21 (70%) | 14 (47%) | 30 (100%) | 24 (80%) |
| mixed-codebase | 33 | 31 (94%) | 30 (91%) | 32 (97%) | 33 (100%) |
| **all** | 153 | **127 (83.0%)** | 113 (73.9%) | **150 (98.0%)** | 141 (92.2%) |

- **Observation:** exact rose in every collection, most in Fire and Blood (47% to 70%); exact or partial reached 150 of 153. Three misses: Brain and Behavior #25 (the Ultimatum Game), mixed codebase #12 (developer-role messages), Napoleon #8.
- **Limits:** two Opus 5.5 subagents graded today's results; September's grader was different, so part of the gain may be the grader's. Seven answers split across two passages were graded exact. Several test rows look faulty: unquoted commas in seven codebase rows; doubtful expected answers in Fire and Blood #7, #15, #19, Napoleon #11, #20 and Attention #23, #26; codebase #1 and #2 name the wrong file. Grades with reasons: `grades-1.jsonl`, `grades-2.jsonl`.
- **Decision:** no 5-point drop, so no reranker experiment. These figures replace the README's retrieval table once the parent–child README edits in the main checkout are settled.
- **Same graders on August's runs:** August's saved top 5 passages (one index per collection, and one merged index) were regraded by four Opus 5.5 subagents with the same rubric: 107 and 105 exact, 143 and 145 exact or partial, against today's 127 and 150. Per question, 31 improved and 7 got worse. Grades: `regrade-august/`; numbers: `comparison.json`; report: `../report-2026-10-28/report.html`.
- **Grader instructions:** identical rubric for all six graders. One wording difference: today's first grader was told not to fill gaps from its knowledge "of the book", the others "of the book or code"; its packet held no code questions. Graders differed on answers split across two passages (some exact, some partial), which moves a few exact counts; exact-or-partial is unaffected.
- **Spend:** about $0.29 on the OpenRouter key by OpenRouter's count. Refused calls (`engine_busy`, missing documents) never reached OpenRouter and cost nothing.
