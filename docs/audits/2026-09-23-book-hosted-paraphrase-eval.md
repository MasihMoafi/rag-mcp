# Hosted embedding, full-book paraphrase check — 2026-09-23

Decision: do not claim that the tested hosted embedding model improves the
nine-chapter retrieval experience. The [preregistered manifest](../../evals/experiments/book-hosted-paraphrases-2026-09-23/manifest.json)
required at least three of four exact supporting passages in the fused top
five. The [redacted result](../../evals/experiments/book-hosted-paraphrases-2026-09-23/result.json)
records one of four, so the threshold failed. The historical BM25-only
paraphrase check also found one of four, though its temporary corpus was not
retained with a full source fingerprint; this is not a controlled claim of
equivalence between the models.

The isolated workspace imported nine top-level Markdown chapters from
`/home/masih/Desktop/books/rag`: 477,021 bytes, aggregate source SHA-256
`2da1ed67ae230992ab2688ccde904e9226b049658d63c5f04b294db1a3426d3e`.
The new index held the expected 884 chunks. Retrieval used OpenRouter
`qwen/qwen3-embedding-4b`, 50 candidates per dense/lexical stage, equal-weight
RRF, five final passages, no reranker, no contextual retrieval, and no answer
generation. All four CPU-retry runs completed successfully; their dense,
lexical, and fusion traces each held 50 candidates and selected five.

| Paraphrase's target chapter | Expected chapter first rank | Exact anchor first rank |
| --- | ---: | ---: |
| 6, precision and recall | 4 | — |
| 2, overlap across chunks | 1 | — |
| 8, non-text conversion | 2 | 2 |
| 9, graph engineering | 1 | — |

The first attempt built the index, but aborted on the third question: the
engine auto-selected CUDA and LanceDB's Rust layer failed a 9,052,160-byte
allocation under the 8 GiB virtual-memory cap. It finished two queries and
did not yield a complete four-question result. A retry hid CUDA, reused the
same published index, and completed all four questions in 42.97 seconds with
714,532 KiB peak resident memory on two low-priority CPU cores. The initial
aborted process lasted 25.93 seconds and peaked at 758,360 KiB RSS. The
provider reported 66,763 document-embedding input tokens; across both
attempts, the six persisted traces reported $0.00133706 total cost. This is
reported provider cost, not an independently audited bill.

A targeted [candidate diagnostic](../../evals/experiments/book-hosted-paraphrases-2026-09-23/diagnostic.json)
then searched up to 200 passages per stage against the same published index
and applied the shipped equal-weight fusion rule at both the configured 50
and diagnostic 200 candidate limits. Exact-anchor ranks were:

| Chapter | Dense | Lexical | Fused, 50 per stage | Fused, 200 per stage |
| --- | ---: | ---: | ---: | ---: |
| 6 | 18 | 79 | 35 | 14 |
| 2 | 57 | 50 | 92 | 20 |
| 8 | 22 | 1 | 2 | 2 |
| 9 | 1 | 44 | 8 | 8 |

Raising only the candidate limit to 200 would still leave the same one of
four anchors in the final five. Fusion demotes even the chapter-nine dense
rank-one anchor to rank eight; chapter two also suffers from the 50-candidate
cutoff. These four labeled questions are too few to tune fusion weights or
declare a reranker effective. Two direct-backend diagnostic invocations made
additional OpenRouter query-embedding calls whose provider cost was not
captured in a trace. The saved full-ranked invocation took 8.52 seconds and
peaked at 327,796 KiB RSS; its charges are excluded from the $0.00133706
traced subtotal above.

The manifest, runner, and redacted result are kept under
`../../evals/experiments/book-hosted-paraphrases-2026-09-23/`. Private logs,
source passages, and trace records remain under
`/tmp/rag-book-hosted-20260923/`. All saved traces have content capture off;
the key was absent from process logs. The experiment evaluates exact-anchor
retrieval on four questions from one book. It does not establish answer
grounding, citation correctness, or broader model quality. The safe
conclusion is to keep the lexical default and avoid advertising this hosted
model as a proven upgrade for this book.

## Remote-only device follow-up

The [device selection contract](../specs-2026-09-23-remote-only-device-selection.md)
now makes OpenAI-compatible embeddings with reranking off or hosted
OpenRouter reranking select CPU without probing local CUDA/MPS. Local model
paths keep their previous device selection. The new hosted-only test failed
before the change; the device, pipeline, embedding lifecycle, and startup
suite then passed 88 tests. A warm query against the published 884-chunk book
index after the change (`4052b5e8a02d432eac2170980060b8b8`) succeeded
with five passages; its process log selected CPU and never selected CUDA.
It took 14.49 seconds and peaked at 638,872 KiB RSS under the same two-core,
8 GiB virtual-memory bounds. Its trace reported an additional $0.00000036
query cost and had content capture off. This confirms the selected device
for this route; it does not establish that CUDA alone caused the earlier
allocation failure or that all memory-pressure cases are resolved.
An offline wheel rebuild succeeded. Its SHA-256 is
`43b82e75906db46856d80f9bb8302517d8e72052906caca22e1e2ab85ab921b1`;
reinstalling it in the isolated package-QA environment and importing from
`/tmp` returned `cpu` for the hosted/off configuration. The wheel was not
published or installed into the user's main environment.
