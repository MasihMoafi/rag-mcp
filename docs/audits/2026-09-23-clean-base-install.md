# Clean base environment, packaged Studio — 2026-09-23

The wheel at this checkpoint, SHA-256
`43b82e75906db46856d80f9bb8302517d8e72052906caca22e1e2ab85ab921b1`,
was installed with dependencies into a new Python 3.13 venv under
`/tmp/ragmcp-clean-20260923`, outside both source trees. `uv pip install
--offline` resolved and installed 51 distributions from this host's cache;
`uv pip check` found them compatible. It warned that a newer `filelock`
would require the registry and selected cached 3.32.5, which satisfies the
declared dependency. PyTorch was absent, as intended for the base install.
No package was downloaded.

With provider keys blank, embeddings disabled, and a separate temporary
workspace/config/index, the installed CLI imported one 64-byte synthetic
text source as one valid JSON value. Installed `rag search --json` indexed it
and returned the unique passage at rank one, with verified source offsets
0–63. The bounded process took 5.21 seconds and peaked at 219,880 KiB RSS
on two low-priority cores. Installed MCP stdio advertised 17 tools; its
`search_workspace` call returned the same source ID and offsets, omitted
private passage text and score by default, and wrote zero bytes to stderr.
Both clients ran from `/tmp`, not the repositories.

The wheel contains Studio archive SHA-256
`afdc5924c6a5ebfd1099bfec08bbb5acbc780600ff3af3a9e5480847aedba58f`.
Its installed launcher served HTTP 200 for English Playground and Persian
Observability on temporary loopback ports. Both pages were visually checked
in the built-in browser: the Playground input and controls were legible,
and Persian text, cards, and RTL layout were visible without clipping in
the viewport. The direct browser URLs were unpaired, as expected without
the launcher's pairing token, so this is a rendering smoke rather than a
browser-to-engine query test. The temporary tab was closed, the launcher
terminated, and neither port remained listening.

This proves a new environment can resolve cached base dependencies and run
that wheel's CLI, MCP, and bundled web routes on this Linux host. It
does not prove resolution from an empty cache or installation on a different
machine/OS, nor does it prove generated-answer grounding. Private local
logs and the synthetic fixture remain under `/tmp/ragmcp-clean-20260923`.
No provider call, commit, push, or deployment occurred.

A later prompt-only wheel was rebuilt and reinstalled in this same venv;
its SHA and targeted installed-code check are in the
[answer-grounding audit](2026-09-23-book-answer-grounding.md). The full
CLI/MCP/browser smoke above refers to the earlier hash.
