# Book retrieval baseline — 2026-09-23

Decision: keep the zero-setup BM25-only path for this nine-chapter book if at least three of four preregistered questions return the exact supporting passage from its expected chapter within the top five, with no run errors. This checks retrieval only; it cannot establish answer quality, paraphrase robustness, or performance on other corpora.

The isolated `personal` workspace contained nine Markdown chapters imported from `/home/masih/Desktop/books/rag`. Its existing LanceDB/Tantivy index held 884 chunks (700 characters, 100 overlap). The workspace revision was `1790162757560-fcd39655c51c`. The run selected all nine sources, used no embedding model, no reranker, no contextual retrieval, `candidate_k=50`, and `top_k=5`. Both provider API key variables were blank. It reused the published index; it did not rebuild it or call an answer model. The process was limited to CPU cores 4 and 5 at lower priority.

| Question | Expected chapter and passage anchor | Chapter first rank | Anchor first rank | Duration |
| --- | --- | ---: | ---: | ---: |
| How does the book define precision and recall for retrieval? | Ch. 6, “precision focuses on the quality of what was found” | 1 | 4 | 3504 ms |
| How does overlap help fixed-size chunking preserve context across boundaries? | Ch. 2, “preserving local context across boundaries” | 1 | 2 | 18 ms |
| What is the conversion approach to multimodal RAG, and how does it differ from native processing? | Ch. 8, “translate non-text data into text-based structured representations” | 1 | 1 | 26 ms |
| When is the production complexity of a knowledge graph justified for RAG? | Ch. 9, “materially affects business outcomes” | 1 | 1 | 16 ms |

Observed result: batch `387c53ee30b6440c8b5f61c9906fb35e` succeeded, with four of four exact anchor hits and four of four expected chapter hits at rank five or better. The first duration includes process/index startup; these four measurements are not a latency benchmark. The precision/recall result's first passage was also from the correct chapter and discussed the right metrics, but the preregistered exact anchor did not appear until rank four.

The preregistration, preflight, batch, bounded report, script, and copies of the four private raw run results are at `/tmp/rag-studio-book-dJqD9w/eval/`. The raw records contain book passages and should stay private. The manifest was written before execution. To repeat, use its script from the `rag-mcp` repository with the same `RAG_MCP_WORKSPACE_DIR`, `RAG_MCP_CONFIG`, `RAG_MCP_DATA_DIR`, and `RAG_MCP_TRACE_DIR` values under `/tmp/rag-studio-book-dJqD9w`; set `RAG_MCP_EMBED_PROVIDER=none`, `RAG_MCP_RERANKER_TYPE=off`, `RAG_MCP_CONTEXTUAL_RETRIEVAL=off`, `RAG_MCP_RERANK_TOP_K=5`, `RAG_MCP_CANDIDATE_K=50`, and blank `OPENROUTER_API_KEY` and `COHERE_API_KEY`. The isolated data directory must still exist. A subsequent run is a new sample and should retain this batch as its baseline.

The threshold was met, so this check gives no reason to change the BM25 default for these literal questions. Its four questions are source-grounded and lexically close to the passages; they do not probe synonym-heavy requests, long-form answer synthesis, or citations. Those are separate open checks.

## Paraphrase stress check

A separate manifest was registered before running four paraphrases of the same source-grounded questions. The corpus revision, selected sources, index, chunking, retrieval configuration, CPU limit, and expected chapter anchors stayed fixed. Only question wording changed. The prespecified threshold was at least three of four exact supporting passages in the top five; missing or failed runs would count as misses. This is an illustrative paired comparison, not an estimate of general retrieval accuracy.

| Paraphrased question | Expected chapter first rank | Exact anchor first rank | Duration |
| --- | ---: | ---: | ---: |
| What does the text say about balancing finding all useful documents against keeping returned documents relevant? | — | — | 3593 ms |
| Why repeat some material where consecutive text segments meet? | 1 | — | 26 ms |
| How do systems turn pictures and speech into material that ordinary text search can use? | 1 | 1 | 26 ms |
| At what point is the extra engineering burden of graph-based knowledge worthwhile? | 3 | — | 26 ms |

All four runs succeeded, but only one returned the exact supporting passage. Three returned at least one passage from the expected chapter; in two of those cases, it was a different passage. The first question retrieved no passage from its expected chapter. The first duration again includes startup, so these times are not a latency comparison. The preregistered threshold failed. This establishes a concrete limitation for these paraphrases under BM25-only retrieval; it does not prove that any particular embedding model would fix them.

The second preregistration, preflight, bounded report, runner, and four private raw result copies are in the same `/tmp/rag-studio-book-dJqD9w/eval/` directory, named `book-paraphrase-stress.json`, `paraphrase-preflight.json`, `paraphrase-report.json`, `run_paraphrase_stress.py`, and `paraphrase-raw-results/`. The paired Studio settings now warns that exact-word search can miss paraphrases and directs users to find and verify a local embedding model. That text was visually checked in English and Persian on a temporary isolated companion; it is an explanation of the limitation, not a retrieval-quality fix. Model-based retrieval and answer grounding remain to be tested separately.

After that UI check, the bounded Next production build passed. The offline Studio bundle was restaged and `uv build --offline` rebuilt the wheel and source distribution. A later copy correction changed answer status labels from claims of grounding to descriptions of citations, because citation presence alone does not verify support. A subsequent bounded production build and package rebuild passed. Its JavaScript assets contained both the search notice and corrected answer labels. That archive was superseded by a later reranker-inspection build described in `../specs-2026-09-23-reranker-score-trace.md`. The corrected answer labels were not separately visually inspected in a rendered answer.
