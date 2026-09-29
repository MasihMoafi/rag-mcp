# Bounded book semantic smoke and published-index scope — 2026-09-23

Decision question: before indexing all nine book chapters with a local model,
can the cached `BAAI/bge-small-en-v1.5` recover one preregistered paraphrase's
exact supporting passage from chapter six within the top five, under a
90-second, two-core, low-priority, 8 GiB virtual-memory cap? The
[manifest](../../evals/experiments/book-semantic-smoke-2026-09-23/manifest.json)
was written to the isolated `/tmp/rag-semantic-smoke-38iZ18Ur` area before
either query. It records the 79,909-byte chapter SHA-256, question, short
anchor, models, and success threshold. This is a go/no-go resource and
retrieval probe on one chapter, not an independent quality estimate.

The isolated workspace held only that chapter (148 indexed chunks). Both
runs used `candidate_k=50`, final top five, contextual retrieval off, reranker
off, blank provider keys, and no answer generation. The BM25-only run
`1be2f75ec7b74c9599f6584d6f9a0206` finished in 3.48 seconds, with
216,804 KiB peak resident memory, and missed the exact anchor in its top
five. The hybrid run `d5afc37fa8d44218a740ee8099f78472` ran inside a
network-unshared sandbox with the cached local model. It finished in 86.19
seconds with 1,344,816 KiB peak resident memory; its measured index-build
span took 81.25 seconds, and its dense stage succeeded. It also missed the
anchor in the top five. A warm top-50 diagnostic run
`6bc4ffd808fb4a5790c859467ffd29cd` reopened the index in 3.95 seconds
and placed the anchor at rank 43. The preregistered threshold failed, so this
does not justify a larger build with this model under the current resource
constraint. It does not show that other models, query formulations, rerankers,
or the full nine-chapter corpus would behave the same way.

The diagnostic also found a separate inspection defect: after the successful
workspace query, installed `rag chunks --view indexed` said `not_indexed`.
Workspace queries hash their exact selected-file list into the index scope,
but the published-chunk reader looked only for the older unselected root
scope. A new real two-source integration test failed before the fix. Query
and inspection now use one selected-file scope function; inspection prefers
the default all-source index and retains a fallback for older root-scope
indexes. The focused index-inspection, API-retrieval, and MCP-isolation set
passed 11 tests. Against the existing chapter-six hybrid index, source-tree
and rebuilt installed CLI both reported `current`, 148 published chunks,
without loading a model or exposing text by default. Installed MCP returned
the same status/count, zero stderr bytes, and no page text.

The rebuilt wheel SHA-256 is
`038043ba65c99463a2ed19269b4149deff952e5c71db119ef8ffc09d1e222e9d`;
its Studio archive remained SHA-256
`72b4e190b2c27ea1c6739964371ff0be7e50a59e85a3bcef28c8cf228ad4cdf6`.
Subset-only query indexes are not discoverable from the current source-only
inspection API; `not_indexed` refers to the default all-source scope when
neither that nor a legacy root-scope index exists. Fresh-machine installation,
full-book semantic quality, and answer grounding remain open. The private
result and trace files stayed under `/tmp`; no provider call, remote network
access during the hybrid runs, full-book build, commit, push, or deployment.

## Recommended query-prefix follow-up

The saved warm trace showed the anchor at lexical rank 18 and fused rank 43.
A direct read of the same 148-chunk index confirmed it was absent from the
dense top 50. FastEmbed's installed `query_embed` implementation delegates
to raw text embedding for this ONNX model. The [model author recommends a
query instruction](https://huggingface.co/BAAI/bge-small-en-v1.5) for passage
retrieval, so a follow-up manifest fixed the original question, index,
lexical ranking, 50-candidate limits, and equal-weight RRF, changing only
the dense query text. The manifest, executable procedure, and observed
result are in `../../evals/experiments/book-semantic-smoke-2026-09-23/`
under `query-prefix-followup.*`.

Raw and instructed dense queries both left the exact anchor outside their
top 50; fused rank stayed 43, failing the prespecified top-five threshold.
The warm offline run took 2.69 seconds and peaked at 336,860 KiB RSS on two
low-priority cores, without rebuilding the index or generating an answer.
The first procedure attempt used FastEmbed's wrong default cache path and
failed before a query; using the engine's cache path produced the reported
run. This result does not justify changing the shipped query embedding
behavior. Model discovery found no cached local reranker; a later quality
test needs a different candidate and resource budget, not a claim that this
small embedding model solved the paraphrase.

## Company OpenRouter embedding follow-up

Masih authorized using the configured company OpenRouter key. A separate
manifest fixed the same 79,909-byte chapter-six source, paraphrased question,
148 chunks, 50 candidates, five final passages, equal-weight RRF, and no
reranker, contextual retrieval, or answer generation. Only the embedding
model/provider changed to `qwen/qwen3-embedding-4b` via OpenRouter. Its
[published input price](https://openrouter.ai/qwen/qwen3-embedding-4b) was
$0.02 per million tokens at the time of the run. The manifest and redacted
result are under `../../evals/experiments/book-semantic-smoke-2026-09-23/`
as `openrouter-embedding-followup.*`; private output and trace remain in the
isolated `/tmp/rag-semantic-smoke-38iZ18Ur` area.

Run `efe01837ea844d4080e197df8a285814` succeeded and returned the exact
anchor at rank four, verified source offsets 31,674–32,165. Dense, lexical,
and fused stages each produced 50 candidates successfully. The provider reported 17,924 document
tokens and 18 query tokens, with a $0.00035884 cost subtotal. Wall time was
15.52 seconds and peak RSS 684,976 KiB under the two-core low-priority
limit. The saved trace has content capture off; the key was absent from the
process logs. This meets the preregistered top-five threshold for this
paraphrase and chapter. It is one positive retrieval example, not a
full-book quality or answer-grounding result.
