# Remote-only retrieval device selection — 2026-09-23

Trigger: a retrieval-only run uses an OpenAI-compatible hosted embedding
model with reranking off, or a hosted OpenRouter reranker. The pipeline
currently probes and selects local CUDA even though neither stage uses a
local torch model. In a capped nine-chapter book run, CUDA was selected and
the process later aborted on a Rust allocation; hiding CUDA let the same
published index complete all four questions. This association does not prove
CUDA caused the allocation failure.

Contract:

- For hosted OpenAI-compatible embeddings paired with off/none/disabled or
  OpenRouter reranking, select CPU without probing local CUDA or MPS.
- Configurations with a local embedding or local/auto reranker keep the
  existing device-selection behavior.
- Retrieval output, index format, model choice, and provider calls do not
  change. No new user-facing setting is introduced.

Verify with a focused failing test for remote-only selection and a local
path regression, then the device and adjacent pipeline suites. A bounded
real-book CPU run has already completed but is not proof of every runtime
resource behavior after this change.
