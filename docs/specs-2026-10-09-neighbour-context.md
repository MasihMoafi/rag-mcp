# Engine neighbouring-chunk answer context — 9 October 2026

Goal: extend the already implemented browser small-to-large context to the shared engine used by Studio, CLI and MCP. This is a bounded neighbouring window, not a parent-child hierarchy.

Contract:
- Saved `context_radius` is a whole number 0–3, defaults to 0, respects engine environment overrides and revision conflicts.
- Retrieve and rerank the same small passages. Only answer synthesis expands each hit using indexed neighbours from the same selected source.
- Expand at most three neighbours each side and at most 16,000 characters per window. Preserve rank/score/hit identity and number of passages. Source offsets, locations and retained synthesis evidence describe the exact expanded span.
- Verify indexed text at exact offsets before expansion; ambiguous/legacy offsets, changed source text, inaccessible sources or metadata leave the original passage intact. Never read an excluded source. No inferred offsets.
- Retrieval-only queries remain unchanged. No re-embedding or reindexing for a radius change. Trace the expansion step without exposing content by default.
- Settings are accessible through existing CLI/MCP control and Studio shared engine settings, with English/Persian copy and existing presentation.

Verify first with failing unit tests, then real local lexical workspace queries plus a fake answer provider; test defaults, boundaries, source isolation, stale content, overlapping chunks, Unicode, exact citations, settings conflicts and retrieval-only behavior. No model download, paid request or public deployment. Roll back focused commits; do not alter other sessions’ pending conversation work.

Verified locally: real LanceDB lexical workspace with a fake answer provider; unchanged index generation/ranking/retrieve-only evidence, selected-source isolation and UTF-16 citations. 18 window/fallback unit cases pass; full engine suite 910 pass with two unchanged CLI bundled-startup failures reproduced from HEAD. Studio saves radius 2, rejects 4, and retains the value after reload. Web: 293 behavior tests, typecheck, targeted lint and production build pass. No real provider quality claim.
