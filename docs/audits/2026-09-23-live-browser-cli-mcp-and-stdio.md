# Installed browser/CLI/MCP parity and MCP stdio isolation — 2026-09-23

An isolated, cached-offline wheel install under `/tmp/ragmcp-package-qa-B8OWEj`
served the bundled Studio and local companion on loopback with provider keys,
embedding, reranking, and contextual retrieval disabled. CLI imported one
synthetic 42-byte chapter into workspace `personal`. In the built-in browser,
Sources showed the CLI-imported `chapter.txt` as a shared local source. A
retrieval-only question returned its one passage at rank 1, source offsets
0–41. Observability listed browser run
`888fb8fe38c04a53abadc9cbb835763e` with a measured 1→1 retrieval funnel,
skipped dense stage, and no reported answer stage. Installed CLI run
`61411943521f487baa0a5263c3149914` and MCP run
`9bfa332ed5ba4fe0bc7472d47374e53a` returned the same rank and source ID
`file-11551f5d80212d133c59fe0aea89739f2341585e5fd663130b953fb72e63d8c5`.
MCP default output omitted passage text and scores. The browser screenshot
showed a legible Observability run list and inspection pane. The Sources
"Browser workspace" message seen immediately after reload was transient;
after sync, it read "Sources shared with CLI and MCP".

This walkthrough exposed two actual package issues. Immediately restarting
`rag studio` could report a closed web port as occupied while connections
remained in `TIME_WAIT`. A socket-based regression failed before and passed
after the POSIX port probe enabled `SO_REUSEADDR`; the installed launcher then
served HTTP 200 twice on port 3022 with a stop and immediate same-port restart.
MCP `search_workspace` also routed Python index logs, including private local
paths, to host stderr. On a first build, LanceDB wrote a path-bearing native
warning directly to file descriptor 2. A red regression covered Python and
native stdout/stderr writes. The MCP tool boundary now suppresses both Python
streams and native stdout/stderr during search, source import, evaluation,
conversation, and answer-query execution; the JSON-RPC response is written
after those streams are restored. Thirty focused MCP/launcher tests passed.

The final rebuilt wheel SHA-256 is
`f1104a54108d4e6a24566925e45b1115f377bdbd4f87eec950fb877f5218650a`.
After offline reinstall, a new first-build MCP search in workspace
`native-quiet-final` returned one metadata-only passage, exactly two JSON-RPC
stdout lines, zero stderr bytes, and saved run
`f6aa6ac98c0d4cf0ac695f4bf3439180`. The wheel contains the port and MCP
output fixes. No listener remained on the test ports. The browser parity run
preceded the final output-isolation rebuild; the Studio archive itself did not
change between those wheels.

This is a Linux, cached-dependency, tiny-fixture check. Fresh-machine and
cross-platform installation, full-book semantic recall, and generated-answer
grounding remain unverified. No paid provider call, book re-index, commit,
push, or deployment occurred.
