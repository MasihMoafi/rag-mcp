# Eliminate the unused LanceDB dimension probe

The default LanceDB pipeline sent `embed_query("dimension probe")` on every
initialization, including reopening an unchanged index. Its backend does not use
the supplied `vector_size` to construct or search the table: ingestion infers the
Arrow vector schema from actual document embeddings, and reopening reads storage.

The pipeline now probes only for Qdrant, which needs the dimension to create its
collection. LanceDB receives the existing zero hint instead. Its actual vector
schema, embeddings, ranking, citations and persisted data remain unchanged.

## Evidence

A synthetic integration test builds and reopens real local storage for both
backends while recording all calls to a fixture embedding provider. It verifies
exact retrieved-record equality, original source text, and one document-embedding
batch across the two opens. No actual provider request or model download occurs.

| Startup work | Before | After |
| --- | --- | --- |
| LanceDB initial build | 1 dimension query + 1 document batch | 1 document batch |
| LanceDB unchanged reopen | 1 dimension query | No embedding call |
| Qdrant build / reopen | 1 dimension query each | Unchanged |

The test failed on LanceDB's extra queries before the change; Qdrant passed.
The fixture uses an exactly normalized vector so Qdrant normalization rounding
does not confound strict record equality. The first draft used a non-unit vector
and was corrected before the implementation change.

Final selection: **113 passed**, 425 existing dependency warnings, 35.80 seconds.
It covers the request-count regression, embedding-client lifecycle, mocked real
SDK API retrieval/failure handling, Qdrant lifecycle, document refresh, generation
recovery and module contracts. Logs are adjacent.

## Limits and operational effect

This removes one unnecessary embedding request per LanceDB initialization. It is
not a measured end-to-end latency percentage or a relevance improvement. Provider
availability is now checked when document/query embeddings are actually requested,
rather than by an unrelated startup query. Existing tests still verify provider
errors and client cleanup. Qdrant retains its startup probe and probe-failure cleanup.

All validation used synthetic local data and mocked providers. Earlier uncommitted
efficiency and experiment work remains intact. No deployment, live process restart,
private upload, corpus rebuild or embedding-provider change was performed.
