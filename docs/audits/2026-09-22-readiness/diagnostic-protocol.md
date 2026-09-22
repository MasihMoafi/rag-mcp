# Chapter 4 development diagnostic — prepared, not executed

Decision: locate the first stage that loses the evidence needed by the previously
failed production question. This is B0 attribution, not a Jev comparison or a
held-out quality result. No retrieval/decomposition/prompt settings change.

`diagnostic-inputs.json` freezes three questions, the chapter hash and eleven
claim anchors with inclusive line ranges and hashes. It stores paraphrased labels,
not the book's text. The original broad question is retained verbatim; focused
architecture and transition questions are controls. The entire book family is
development data. Labels are agent-drafted and require independent adjudication
before use as a release gate. They must never enter the held-out split.

Proposed run: one fresh isolated LanceDB index, one pass over each case, current
Qwen4B embeddings, 50 RRF candidates per facet, Cohere4fast reranking, five final
passages and DeepSeek4.1Flash synthesis (temperature0, output cap700). Contextual
retrieval is disabled. Preserve existing decomposition: the broad question yields
three facets; each control yields one. Five rerank calls and three synthesis calls
are expected. Embedding's existing one-retry policy is included in the estimate;
rerank/synthesis have no automatic retries. Do not add repeats silently.

Before execution, retain exact runtime/config/dependency identities and the
source/frozen-input hashes, enable local content capture in a dedicated private
directory, and ensure no existing user index is touched. Preserve raw traces and
public answers privately. Score each required claim as supported, absent or
contradicted at dense/lexical candidates, RRF, rerank, final five, and answer;
use source spans to adjudicate rather than word overlap. Multiple chunks may
jointly support a claim. Record the first absent stage; if it is unknown, say so.
Do not attribute a missing answer claim to retrieval when the selected evidence
supports it. Broad-case diagnostic success requires all eleven supported claims
and no contradiction; focused controls require every assigned claim. Failures,
timeouts and empty answers stay in the denominator. One pass offers no stable
latency estimate or statistical quality conclusion. Do not tune and re-score
these same questions as a generalization claim.

## Offline cost preflight

Run from the repository root (no network or credentials):

```sh
.venv/bin/python docs/audits/2026-09-22-readiness/diagnostic_preflight.py \
  '/home/masih/Desktop/books/rag/4. Deploying RAG to Production _ Hands-On RAG for Production.md'
```

`diagnostic-preflight.json` verifies source/anchor hashes and records121 chunks,
five facets and a **$0.02229014 planning estimate**. This uses byte-based token
allowances, embedding retries, the five largest chunks for synthesis input plus
2048 framing tokens per answer, and700 output tokens per answer. It is not a
provider-enforced hard cap. The current endpoint snapshot's highest synthesis
rates are used, including a provider above the commonly advertised minimum.

Pricing checked2026-09-22: [Qwen4B](https://openrouter.ai/qwen/qwen3-embedding-4b/apps)
$0.02/M input tokens;
[Cohere4fast](https://openrouter.ai/cohere/rerank-4-fast/pricing) $0.002/search;
[DeepSeek endpoint data](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash/endpoints)
up to $0.375/M input and $1.50/M output tokens in the captured response. Raw public
endpoint JSON is retained under `pricing/`. Cohere's endpoint token prices are
zero because it bills searches; zero token prices must not be read as free.

**Do not execute yet.** The earlier $0.05 allowance was estimated at $0.0496 spent,
leaving only about $0.0004; this is insufficient even for five rerank calls.
A renewed budget and a request-level spending guard are still needed. Before
requesting approval of a paid run, finish the guard, verify billing-unit treatment
for50 candidates and provider/token limits, refresh prices, and present its
concrete maximum. The preflight deliberately cannot submit paid requests.

No new paid experiment, held-out evaluation, Jev implementation or quality claim
is authorized by this preparation artifact.
