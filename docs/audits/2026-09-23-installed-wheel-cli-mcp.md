# Installed wheel acceptance — 2026-09-23

This records the earlier installed wheel. The later
[live-client audit](2026-09-23-live-browser-cli-mcp-and-stdio.md) covers the
newer wheel with browser parity, same-port restart, and native MCP output
isolation.

The rebuilt local wheel, SHA-256
`66765b3a367f176e14886ded3d75b739d773c0360c0e85461c5af9f2bbd6f989`,
was installed into a new temporary Python 3.13 venv outside both source trees
with `uv pip install --offline`. It installed 51 cached distributions. The
installer warned that a newer `filelock` needed registry access and selected
the cached 3.32.5 release; this is **not** a fresh-machine resolution check.
The embedded Studio archive retained measured build ID
`FwXJMXviGtG6TuyWlgK4_` and SHA-256
`72b4e190b2c27ea1c6739964371ff0be7e50a59e85a3bcef28c8cf228ad4cdf6`.

From outside the repositories, installed `rag search --help` started. The
installed runtime manifest found 57 engine files, 51 installed distributions,
Git state `unavailable`, and the measured Studio attestation. Its bundle served
HTTP 200 for `/studio` and `/fa/studio/observability?mode=advanced`, then
exited. Installed MCP stdio advertised 17 tools, including `search_workspace`,
with clean JSON-RPC stdout.

The first installed `rag sources add --json` succeeded but printed a PyMuPDF
legacy-`fitz` warning before its JSON. This made the machine-readable result
invalid. The official [PyMuPDF installation guidance](https://pymupdf.readthedocs.io/en/latest/installation.html)
recommends `import pymupdf`; PDF parsing now uses it. CLI source import also
redirects parser stdout to local stderr, and MCP import discards parser stdout
so private diagnostic text cannot enter the JSON-RPC stream or host stderr.
Both new regression tests failed before these changes and passed afterward.
The adjacent workspace-import, MCP-isolation/stdio, parsing, PDF, document
refresh, and source-citation selection passed 88 tests.

After the offline wheel was rebuilt and reinstalled, the installed CLI imported
one synthetic chapter with a single valid JSON stdout value. `rag search`
and MCP `search_workspace` each returned one passage with matching rank and
source ID; MCP's default result omitted passage text and score. Their saved
retrieval-only runs were `91fe2d1312c443adb3d8ac35bffd0158` (CLI) and
`f2d7ad692e5040ddb45e3157aaf24422` (MCP). Each normal trace's runtime
fingerprint matched its private saved result, and neither trace had a synthesis
stage. Embedding, reranking, contextual retrieval, and provider keys were off
for this fixture. No paid provider call, book re-index, commit, push, or deploy.

This proves a cached offline local install and installed CLI/MCP retrieval on
this Linux host. A fresh-cache or different-machine install, cross-platform
Node behavior, browser visual QA of this wheel, full-book semantic quality,
and generated-answer grounding remain unverified. The temporary venv and its
synthetic workspace live under `/tmp` and may disappear; this audit preserves
only nonprivate observations.
