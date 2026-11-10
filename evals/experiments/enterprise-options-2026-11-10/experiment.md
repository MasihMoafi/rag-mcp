# EnterpriseRAG-Bench: titles, stemming, balanced shortlist, whole documents (10 Nov 2026)

**Question:** do four retrieval ideas from Masih's earlier search work raise rag-mcp's answer correctness on EnterpriseRAG-Bench?

## Setup, fixed before any result

- Corpus: the 100k-document sample of 9 Nov (every gold document plus 100,000 others, `normalize(..., 500, 100000, True)`), all 499 questions with or without gold.
- Models: Qwen3-Embedding-0.6B, Qwen3-Reranker-4B, 100 candidates, chunk size 700 (as on 9 Nov).
- Answers: Sol through Codex on Masih's ChatGPT login, direct (no VPN), the partial-answer prompt; graded by Onyx's own judge prompt (`regrade_onyx.py`).

| Arm | Retrieval | Answer reads |
|---|---|---|
| A | 9 Nov run (`run-100k-out`) | 20 passages, one neighbour each side |
| B | A + `--option titles --option stemming --option balanced` (new Kaggle run) | as A |
| C | B's retrieval | documents of up to 8,000 characters whole, longer ones as passages |

## Decision rule

An option stays on by default if its arm is more often correct than the previous arm on the same 499 questions (paired exact McNemar test, p < 0.05) and Info Not Found does not fall. One result per arm; no settings changed after seeing answers.

## Prior evidence

On the books (28 Oct, `parent-child-answers-2026-10-28/`) whole sections did not help (21 of 22 either way); here the 9 Nov failure analysis found 42 refusals with every gold document already retrieved.

## Results

Pending.
