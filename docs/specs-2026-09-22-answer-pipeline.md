# Answer pipeline

## Goal

Make one deliberately scoped document query run this internal pipeline:

`BM25 + Qwen3 Embedding 4B -> RRF top 50 -> Cohere Rerank 4 Fast -> top 5 -> DeepSeek V4.1 Flash -> answer`

The CLI prints the answer. The MCP tool returns the answer text, not passages,
paths, citations, retrieval scores, model metadata, or intermediate tool data.

## Fixed choices

- Embeddings: OpenRouter `qwen/qwen3-embedding-4b`.
- Candidate retrieval: BM25 and dense retrieval, fused with RRF; at most 50
  fused candidates reach the reranker.
- Reranker: OpenRouter `cohere/rerank-4-fast`; at most five passages survive.
- Synthesis: OpenRouter `deepseek/deepseek-v4.1-flash`.
- The supported document formats remain available. A caller can restrict a
  directory by extension (for example, `.md` only). The workspace/repository
  root is never an implicit query scope.
- One OpenRouter key is reused for embedding, reranking, and synthesis. It is
  read from configuration or `OPENROUTER_API_KEY` and never returned to a caller.

## Required behavior

- Provider failures are errors. No model is silently replaced and reranking is
  never silently skipped.
- Retrieved text is treated as untrusted evidence, not instructions.
- Synthesis may use only the five reranked passages. When they do not support an
  answer, it says so plainly.
- The synthesized answer contains no source labels or citations.
- Low-level Python retrieval may retain passage metadata for internal use; the
  public CLI and MCP answer boundary does not expose it.
- The public boundary accepts only a nonempty string `result` or `error` envelope.
  Passage arrays, extra metadata fields, malformed JSON, and empty output fail closed.
- Existing indexes are reused when their source and embedding manifest match.

## Provider verification

On 2026-09-22, live OpenRouter requests found no endpoint for either
`qwen/qwen3-reranker-4b` or `qwen/qwen3-reranker-0.6b`. The selected fallback,
`cohere/rerank-4-fast`, returned a successful ranked result at $0.002 per search.
Provider failures must still stop the pipeline rather than silently bypassing
reranking.

The complete configured chain was then verified with one isolated 60-chunk
synthetic Markdown corpus. Qwen3 Embedding 4B embedded 1,949 corpus tokens and 13
query tokens; BM25 and dense retrieval supplied 50 fused candidates; Cohere
Rerank 4 Fast scored all 50 and retained five; DeepSeek V4.1 Flash used 294 input
and 18 output tokens and returned the expected answer. The exact public envelope
was a single `result` string with no paths, chunks, scores, citations, or model
metadata. Provider-reported costs were $0.00003924 for embeddings, $0.002 for
reranking, and $0.0001098 for synthesis: $0.00214904 total. This verifies wiring
and the public boundary for that fixture, not general answer quality.

The first connection attempt was rejected by Cloudflare before reaching the API
because the engine removed the user's HTTP proxy. The engine now preserves
HTTP(S) proxy variables for cloud providers while removing only the unsupported
SOCKS catch-all; a free endpoint check and the successful full-chain run verified
that route.

## Non-goals

- Client-specific integration, hooks, pruning, or transcript rewriting.
- Hiding an MCP invocation from the host model; that requires harness-level
  integration outside this repository. This repository can hide only the
  retrieval evidence and metadata in the tool result.
- Implicit whole-repository indexing. Callers may exclude HTML or any other
  supported format with the per-query extension filter.
- A new broad automated test suite.
- Deployment, publication, or changing provider credentials.
