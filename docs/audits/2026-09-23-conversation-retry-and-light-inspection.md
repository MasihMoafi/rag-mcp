# Conversation retry and lightweight inspection — 2026-09-23

Two bounded failures were reproduced before changes. First, a gateway test
saved a conversation answer once, injected failure on the second run-result
save, restarted the server, and retried the same run ID. It returned HTTP 409
because the first saved result held the fingerprint of the contextual query,
while the gateway retry checked the raw request including conversation ID and
expected revision. The gateway now passes that raw fingerprint to the engine
for its first save. The same test passes after the change: the retry attaches
the turn without a second execution. An actual `execute_run` test confirms
its first disk record carries the supplied fingerprint. A different request
still conflicts; run and conversation remain separate files, so this does
not claim a fully atomic transaction.

Second, a cold chunk-preview HTTP test repeatedly timed out at two seconds.
A thread dump showed the request importing PyTorch through `rag.fetch` just
to read index defaults. A fresh-process test confirmed both preview and
published-index inspection imported the retrieval pipeline. The shared
index defaults, scope name, and source stamp now live in lightweight
`rag/index_config.py`; `rag.fetch` reexports its existing helper names.
Both cold reads now complete without importing `rag.fetch`, `rag.core`, or
PyTorch, and the loopback preview test passes under the same two-core
low-priority cap. This removes a needless heavy import; it does not measure
the latency of a full query or a populated index inspection.

The focused conversation, execution, chunk, index-inspection, fetch, and
format-contract suite passed 89 tests in 18.24 seconds. `compileall` and
`git diff --check` passed. Existing LanceDB deprecation warnings remain.
After a final unused-import cleanup, 41 focused tests passed. A cached-offline
wheel build succeeded (SHA-256
`6cb9c8f91fa1780042b4905ab2a14016c5bfa22ef43c96f8a0adbaf3b8c970f8`).
Installed outside the source tree, it previewed one source and read its
current one-chunk published index without importing `rag.core` or PyTorch.
No model load, provider call, commit, push, or deployment was needed for
these two fixes.
