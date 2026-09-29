# Release the run lock before responding

## Goal

A client that receives a completed run, configuration write, or evaluation
batch response must be able to submit its next run immediately. The gateway
currently sends those responses while holding `run_lock`, so a sequential
client can receive a spurious `engine_busy` 409 from another handler thread.

## Contract

- Keep one active engine execution at a time. Truly overlapping executions
  still receive `engine_busy`.
- Finish each operation and its persistence under the lock. Release the lock
  and clear active state before sending its HTTP response.
- Preserve response status/body, idempotent retries, conversation revision
  conflicts, persistence errors, and privacy behavior.
- Deterministic tests hold each operation's response after its body is sent
  and prove a new sequential run succeeds while that response handler is paused.
- The bounded companion and adjacent engine tests pass without external calls.

## Boundary

No queue, retry policy, or increase in concurrent engine executions. A slow
client must not extend the lock lifetime after the run is terminal.

## Verified 2026-09-23

- Three deterministic tests failed before the change and passed after: a run,
  a settings write, and an evaluation batch each hold their response handler
  after sending the body while the next sequential run succeeds.
- The existing truly overlapping run still receives 409 and cancellation works.
- The formerly intermittent conversation continuation passed with all test
  modules collected. The final offline suite passed 775/775 in 88.60 seconds
  with two low-priority CPU cores. No provider call.
