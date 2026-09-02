# Session Checkpoint

- Date: 2026-08-13
- Intent: reduce repeated directory enumeration in the MCP query path without weakening scope limits.
- Status: verified locally
- Baseline: 15 tests passed; local suite elapsed 17.5 seconds.
- Change: reuse a validated scope manifest and invalidate it when watched files or directories change.
- Verification: focused scope/bootstrap tests 7 passed; full suite 16 passed; MCP initialize/tools-list smoke passed.
- Measurement: synthetic 1,000-file scope was 1.72x faster over 10 repeated validations (0.1003s vs 0.1728s).
- Known gap: no production corpus benchmark or real embedding query was run.
- Next action: review and commit only if desired.

## 2026-09-02 — installable CLI and local Codex companion

- Intent: turn the existing retrieval MCP server into an installable `rag` CLI, then connect the hosted Studio to an authenticated loopback companion that reuses an existing Codex login with GPT-5.6 Luna.
- Acceptance: `rag doctor/query/connect-codex/studio` work; the companion binds only to loopback, requires an in-memory token, and passes a real grounded document query through Codex without exposing credentials.
- Safety: preserve unrelated benchmark work; no TUI, background service, credential copy, or browser-source synchronization in this slice.
- Result: checkpoints `2841e79`, `a847ae6`, `a84d9f5`, `afaa8f7`, `b0705f5`, and `b5fef64` add the installable CLI, idempotent Codex registration, token-and-origin protected loopback companion, CUDA/MPS/CPU selection, corrected test checkout, and clear Codex answer label.
- Verification: an isolated wheel installed and exposed `rag`; `rag doctor` detected the existing ChatGPT Codex login; a real local file query retrieved `Silver Kestrel 73` from `/tmp/rag-live-fact.md`; direct and HTTP companion calls returned `Silver Kestrel 73. [1]` through GPT-5.6 Luna.
- Regression: 31 pytest cases pass. LanceDB reports three existing deprecation warnings; no test fails.
- Boundary: the production machine selected CUDA automatically. CPU selection is unit-tested, but the full CPU query could not complete inside Codex's restricted syscall sandbox because LanceDB itself cannot open there. The companion remains a foreground process, and browser-uploaded Studio sources are not copied into the Python index.
