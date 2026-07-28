# Retrieval evaluations

Recorded retrieval experiments against real documents, kept as evidence rather than as an
automated suite. Nothing here runs in CI — `tests/` holds the automated checks.

Each folder is one corpus with its own `queries.csv`:

| Corpus | Material |
| --- | --- |
| `attention/` | the *Attention Is All You Need* paper |
| `brain-and-behavior/` | neuroscience text |
| `fire-and-blood/` | long-form narrative fiction |
| `napoleon/` | a book-length biography |

`napoleon/` is the most complete run and the one to read first. Its `experiment_log.md`
records the question, the constraints, and the outcome; `rag-evaluation.csv` and
`results_unified.md` hold the graded results.

That experiment tested raw retrieval in isolation — `top_k=5`, reranking disabled,
`qwen3-embedding:8b` through Ollama — to establish whether embeddings plus BM25 can pull
the right chunk into the top 5 without a reranker compensating for them. Read the log for
what it found; the point of keeping these files is that the numbers came from real runs
and can be re-derived.

`notebooklm-prompt.md` is the prompt used to produce comparison answers from an external
tool.

These evaluations moved here from the Elpis repository, where the retrieval engine used to
live. They are about this engine, so they belong beside it.
