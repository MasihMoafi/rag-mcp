# One paired conversation query

## Goal

Studio, CLI, and MCP should use the engine's recorded conversation context.
A paired Studio answer should enter its shared conversation through one gateway
request, with the run and turn linked by one ID.

## Contract

- Studio sends a raw question, conversation ID, expected conversation revision,
  selected source IDs, and workspace revision. The gateway checks the revision
  before executing and uses the same prior-question context as CLI and MCP.
- A successful run appends its saved result to the conversation and returns
  the new revision. A stale revision fails before another model call.
- Retrying the same run ID and identical request never executes the query
  twice and can repair an interrupted conversation append. A different request
  cannot reuse a run ID.
- Cancellation and failed runs never append. If storage fails after an answer,
  the saved run stays available and the response clearly reports the append
  failure for manual recovery.
- Run and conversation files are separate writes, not an atomic disk
  transaction. The gateway saves the run first and makes the append idempotent
  so the same request can repair a partial failure.
- The first persisted run record uses the gateway's fingerprint of the raw
  request, including conversation ID and expected revision. If the service
  stops before its later save or conversation append, a same-ID retry after
  restart can still validate the request and attach the saved answer without
  another model call. A different request remains rejected.

## Verification

Use injected offline execution through the loopback gateway to check context,
source/revision propagation, saved turn, stale conflict, and same-ID retry.
Exercise Studio's request/response contract and the adjacent Python and web
tests. Do not call a paid provider or index the real book.
Inject a failure after the engine's first run save, restart the gateway, and
prove that the same-ID retry attaches the turn without executing again.
