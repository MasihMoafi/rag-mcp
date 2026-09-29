# Provider-reported embedding usage in private traces

## Goal

An embedding request may report token usage. The engine should retain that
measurement beside its existing embedding timing/count span instead of saying
all embedding usage is unknown. Unknown cost must stay unknown.

## Contract

- For OpenAI-compatible embedding responses, copy only provider-reported
  `prompt_tokens`, `completion_tokens`, `cost`, `search_units`, and cached-token
  details into the existing validated trace usage path.
- Query and indexing batches use the same wrapper, so each request records
  usage on its active embedding span. Deduplicated inputs retain the provider's
  actual count; the engine does not invent a per-document allocation.
- If usage or a field is absent, invalid, or a provider call fails, its value
  remains unknown. `total_tokens` alone is not treated as output tokens or
  money. No query text, embedding vector, credential, or arbitrary response
  field enters normal metadata traces.
- Direct Ollama `/api/embed` reports an optional aggregate `prompt_eval_count`
  ([official schema](https://github.com/ollama/ollama/blob/main/docs/openapi.yaml)).
  Record it as input tokens only after a successful validated response. Sum
  counts across batches only when every batch reports a valid count; otherwise
  leave the whole span's token count unknown. Do not infer output tokens, cost,
  cached tokens, or per-document allocations.

## Verification

Use an in-memory OpenAI-compatible response with `usage` for a real dense
search and a mocked batch response containing an extra private field. Check
recorded input tokens, unknown output/cost where omitted, optional reported
cost, and privacy. Run adjacent offline retrieval/trace tests; no paid call.
Mock direct Ollama query and 513-document batch responses to check summed
counts, omitted/invalid counts, and response-field privacy without loading a model.
