# Concurrent queries (28 Oct 2026)

**Status:** done, 28 Oct 2026. Before, 1 of N simultaneous queries was answered and the rest refused; after, all N are answered, one after another.

## Question

How many queries arriving at the same moment does rag-mcp answer, before and after commit `d2e428e` (a second query waits for the engine lock instead of being refused)?

## Decision it informs

Whether separate sessions and agents can search at the same time without failures. If any query still fails under load after the change, the lock needs more work.

## What is compared

One variable: the code, `0ad942a` (before) against `d2e428e` (after). Fixed: the same document (the Attention paper), the same questions, keyword search only (empty settings, so no embedding, reranker or answer model), the same machine, and separate processes started together, as separate sessions would be.

## Measures

For 1, 2, 5, 10 and 20 simultaneous queries: how many return passages, how many are refused (`engine_busy`), and the wall time until the last one finishes. Each level runs once; the index is built before the first level.

## Cost

None: no API key is present in the processes, so no paid call can happen.

## Result

| At once | Before (`0ad942a`) | After (`d2e428e`) | Time to the last answer, after |
|---|---|---|---|
| 1 | 1 answered, 0 refused | 1 answered, 0 refused | 4.89 s |
| 2 | 1 answered, 1 refused | 2 answered, 0 refused | 8.99 s |
| 5 | 1 answered, 4 refused | 5 answered, 0 refused | 21.32 s |
| 10 | 1 answered, 9 refused | 10 answered, 0 refused | 41.66 s |
| 20 | 1 answered, 19 refused | 20 answered, 0 refused | 82.26 s |

The change queues queries; it does not run them in parallel. Each takes about 4 s here, so 20 at once finish after 82 s. Chart: `../report-2026-10-28/report.html`, section 4.
