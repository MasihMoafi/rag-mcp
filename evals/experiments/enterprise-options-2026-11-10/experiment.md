# EnterpriseRAG-Bench: titles, stemming, balanced shortlist, whole documents (10 Nov 2026)

**Question:** do four retrieval ideas from Masih's earlier search work raise rag-mcp's answer correctness on EnterpriseRAG-Bench?

## Setup, fixed before any result

- Corpus: the 100k-document sample of 9 Nov (every gold document plus 100,000 others, `normalize(..., 500, 100000, True)`), all 499 questions with or without gold.
- Models: Qwen3-Embedding-0.6B, Qwen3-Reranker-4B, 100 candidates, chunk size 700 (as on 9 Nov).
- Answers: Sol through Codex on Masih's ChatGPT login, direct (no VPN), the 9 Nov answer prompt, as A (corrected 11 Nov before any B or C answer: the partial-answer wording had not helped and would have confounded A); graded by Onyx's own judge prompt (`regrade_onyx.py`).

| Arm | Retrieval | Answer reads |
|---|---|---|
| A | 9 Nov run (`run-100k-out`) | 20 passages, one neighbour each side |
| B | A + `--option titles --option stemming --option balanced` (new Kaggle run) | as A |
| C | B's retrieval | documents of up to 8,000 characters whole, longer ones as passages |
| D | B's retrieval | as C, with `--prompt lenient`: answer from partial evidence, refuse only off-subject (added 11 Nov, before any B or C answer was graded) |

## Decision rule

An option stays on by default if its arm is more often correct than the previous arm on the same 499 questions (paired exact McNemar test, p < 0.05) and Info Not Found does not fall. One result per arm; no settings changed after seeing answers.

## Prior evidence

On the books (28 Oct, `parent-child-answers-2026-10-28/`) whole sections did not help (21 of 22 either way); here the 9 Nov failure analysis found 42 refusals with every gold document already retrieved.

## Results (11 Nov 2026)

Kaggle run B: `masihmaafi/rag-bench-enterprise-100k-b-20261110`, rag commit ab229f3. Answers and grades: Sol, Onyx's judge prompt, 499 of 499 questions in every arm, no failed calls.

| Arm | Hybrid recall@10 | Correct | Info Not Found | Refusals |
|---|---|---|---|---|
| A | 80.2% | 66.3% | 100% | 114 |
| B | 84.6% | 75.2% | 100% | 80 |
| C | 84.6% | **81.0%** | 100% | 65 |
| D | 84.6% | 78.6% | 100% | 1 |

Paired exact McNemar (questions gained / lost):

- A to B: 57 / 13, p = 1.0e-7. Titles, stemming and the balanced shortlist pass.
- B to C: 35 / 6, p = 4.9e-6. Whole short documents pass.
- C to D: 3 / 15, p = 0.008. The lenient prompt fails: it hurts.

By type, C over A: basic 81 to 93, semantic 54 to 68, intra-document reasoning 63 to 93, constrained 57 to 87, completeness 35 to 50, project related 30 to 48.

For reference only (different corpus size): the dataset paper's best system, BM25 over all 511,962 documents with GPT-5.4, scores 68.8% correct and 68.4% document recall. This sample has 100,721 documents, and the paper shows recall falls as the corpus grows; a like-for-like claim needs the full corpus.

Caveat: B and C were designed after reading A's failures on these questions.

## rag's own answer step (12 Nov 2026)

C's retrieval and whole documents, but answers written by rag's own answer step (`synthesize_answer`,
its prompt and evidence format, as `rag query` runs it), still with Sol; harness `--prompt rag`.

| Arm | rag prompt | Correct | Refusals |
|---|---|---|---|
| E | 0.3.0 | 79.1% (498 graded; one harness failure) | 34 |
| E2 | 0.3.1 | 80.9% (498 graded; one Codex timeout) | 31 |

- C to E: 2 / 11, p = 0.02. The 0.3.0 prompt hedged settled facts ("pending", "unconfirmed") and
  added conditions the documents do not state.
- E to E2: 12 / 3, p = 0.035. 0.3.1 drops that wording.
- C to E2: 6 / 6. rag's own answer step now matches the benchmark prompt.

