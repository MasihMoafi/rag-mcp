# Shared backend embedding dispatch

The LanceDB and Qdrant adapters duplicated their single/batch embedding dispatch.
Qdrant's callable-batch fallback also caught `KeyboardInterrupt` and `SystemExit`,
then started per-query embedding work. A cancellation should stop that work.

Both adapters now use `rag/backend_embeddings.py`. Existing method names and
override hooks remain. Batch-method precedence, API no-fan-out policy, two fallback
workers, input ordering and backend-specific diagnostic labels are preserved.
Cancellation now propagates through both adapters. No provider, database default,
index schema, ranking, citation or model setting changed. This reduces maintenance
duplication and unnecessary work after cancellation; no normal-query speedup is claimed.

Verification on the existing local environment, without model/API calls:

- Baseline backend/API/log-privacy selection: 66 passed.
- New cancellation regression before implementation: 2 failed for Qdrant,
  2 passed for LanceDB. Both interruption types triggered unwanted fallback.
- Final backend/API/log-privacy/cache/LanceDB/search selection: 97 passed,
  64 existing dependency warnings, 6.76 seconds.
- Before this change, the previously stalled cache/LanceDB selection passed
  all 22 checks in 4.41 seconds. The old connection stall did not reproduce
  in the current unrestricted execution environment; its root cause is not proven.
- A separately maintained downstream record/filter/storage selection passed 37 tests.

Logs are adjacent. The source working tree already contained unrelated efficiency
and experiment edits. They were preserved; the two existing LanceDB result-projection
lines are not part of this change. No supported backend or downstream specialization
was removed. Private full-corpus relevance evaluation remains outside this checkpoint.
