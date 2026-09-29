# Retrieval pipeline efficiency — 2026-09-13

Scope: the maintained generic RAG engine. Existing dirty experiments and implementation edits were preserved. No deployment, credentials, active indexes or model settings were changed.

## Changes and evidence

- API embeddings send each exact text once per batch, restore input order and independent vector lists, and retain response validation. This reduces billed input only when a batch has duplicates; it does not eliminate repeated requests across separate batches or queries.
- The Qdrant branch's in-memory BM25 index keeps term postings and only scores matching documents. Formula, repeated query terms, stable ties and incremental additions are unchanged. Posting lists add memory per distinct document/term pair.
- LanceDB projects source fields and explicit scores before converting results to Python, avoiding unused 4096-dimensional vector lists. Vector search still uses the full stored vectors.

Warm medians, milliseconds (15 measured calls after one warm-up):

| Fixture | Before | After |
| --- | ---: | ---: |
| Vector search, 100 hits | 29.264 | 13.760 |
| FTS, 100 hits | 18.623 | 2.998 |
| In-memory BM25, sparse query | 18.565 | 0.048 |
| In-memory BM25, common word | 18.222 | 18.891 |

The common-word case is slightly slower in this run; it matches all documents and cannot benefit from sparse candidate selection. These timings are synthetic stage benchmarks, not full-corpus question/answer latency or a relevance improvement. LanceDB uses 1000 deterministic records with 4096-dimensional vectors and top100; BM25 uses 20000 records and top20. All before/after result hashes match exactly. Raw samples and hashes are adjacent JSON files.

## Verification and reproduction

The original focused suite passed 60 tests. Two new tests reproduced duplicate API input and exhaustive sparse-query scoring before the change. The final generic regression suite passed 523 tests. Existing dependency deprecation warnings remain. The diff passes whitespace checks. API tests use fixtures; no paid calls or corpus re-embedding were performed.

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_search_materialization.py /tmp/rag-efficiency.json
```

## Remaining limits

Existing pipeline caches reuse unchanged indexes; parsers already use a thread pool, ingestion sends batches, and query lifecycle tracking permits concurrent searches. More API threads or changed GPU batch sizes were not added without throughput/rate-limit measurements. There is no new live-provider, GPU saturation or full answer-quality claim.

Changes are local and uncommitted. Already-running MCP processes must restart to import them; these edits do not require a corpus rebuild.
