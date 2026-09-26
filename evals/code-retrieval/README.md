# Code retrieval eval

Does `rag search` put the right passage in front of a coding agent? Eighteen
questions about a 413-file Rust source tree (`codex-rs/core/src` of the Elpis
repository). Each names the file that holds the answer and an identifier that
must appear in the passage. Every expected file was checked to contain its
identifier before the questions were used.

```bash
python evals/code-retrieval/run.py --corpus ~/Desktop/p/Elpis/codex-rs/core/src --type rs --label baseline
```

Retrieval only: no answer is synthesized, so a run costs embeddings and reranking
but no generation. The first run indexes the corpus (about four minutes here).

## Baseline (2026-09-26)

| | top-1 | top-5 |
| --- | --- | --- |
| expected file | 11 / 18 | 15 / 18 |
| answer identifier in a returned passage | 9 / 18 | 15 / 18 |

Mean reciprocal rank 0.70; median search 5–6 s from a cold CLI, on
`qwen/qwen3-embedding-4b` with the `cohere/rerank-4-fast` reranker. Four runs gave
10, 11, 11 and 11 of 18 for the top-1 file (the 10 came from a separately built
index), so **treat a difference of one or two questions as noise**.

## What the misses look like

- **The right file, the wrong chunk.** For "how is the audit record published
  atomically" and "where is admission state stored", the top passage is from the
  correct file but does not contain the function or file name asked about; the
  answer sits in a neighbouring chunk.
- **Tests crowd out the implementation.** For "how does the exec policy turn rule
  decisions into an approval requirement", three of five passages were from
  `exec_policy_tests.rs`.
- **A busy neighbour outranks the target.** `hook_runtime.rs` took three of five
  slots for a question about `registry.rs`.

These are observations, not fixes; nothing here has been tried yet. Plausible next
experiments: split camelCase identifiers for lexical search, prefix each chunk with
its file path and enclosing item, and rank test files below the code they test when
the question does not mention tests. Any of them should be judged on this set and on
a second corpus, because eighteen questions written by the person who knew the
answers can be overfit.

## Limits

One corpus, one language, eighteen questions, one author. It shows a direction, not
a benchmark.
