# Local shared-workspace release check — 2026-09-23

Decision: the current Linux **local candidate** passed its offline engine and
web gates and installed-package smoke. This closes a local acceptance pass,
not the user's broader request for a deployed, clean-machine, generally
grounded product.

| Gate | Current evidence |
| --- | --- |
| Engine | 805 collected; 804 passed, one skipped after two stale test expectations were corrected. Related MCP/cache tests: 42 passed. Full run 94.45 s, peak 1,173,236 KiB RSS. Provider keys were unset; no local model was run. |
| Web | 302 behavior tests passed; TypeScript passed; full ESLint passed with zero warnings after declaring `RAG_STUDIO_BUNDLE` as a build input in `turbo.json`. |
| Production build | Next standalone build passed in 38.55 s, peak 1,005,848 KiB RSS, with two low-priority CPU cores and a 12 GiB virtual-memory cap. An earlier 2 GiB Node heap limit killed a build worker; the bounded 3 GiB retry passed. |
| Package | The staging script rebuilt the web app and recorded a measured source snapshot, build ID `UqWlr79XGHkNfl6Eko_LW`, and bundle attestation. Offline sdist and wheel builds passed. Wheel SHA-256 `a5576404c141be02129a3bb53f9634baf51c3f23de2522d317b98ec8596c5792`; sdist SHA-256 `5636a2e3b5adc644437123345cb2fa47f9352aa26b8add289d408cefc0d8bf3b`. |
| Installed package | Reinstalled the wheel into the separate Python 3.13 base venv; all 51 installed packages remained compatible. Installed code has the bounded no-evidence fallback and evidence-specific prompt. Installed CLI returned the single synthetic passage at rank one with offsets 0–63. Installed MCP returned the same source ID and offsets, omitted content and score by default, emitted two JSON-RPC lines and zero stderr bytes. |
| Bundled Studio | Installed `rag studio` served English Playground and Persian Observability HTTP 200 from the wheel. An earlier installed-wheel walkthrough exercised paired browser/CLI/MCP source and retrieval parity on a synthetic fixture. A later [final-wheel browser walkthrough](2026-09-23-final-wheel-book-browser.md) opened nine CLI-imported book chapters, generated a supported cited answer, highlighted the exact source span, and inspected measured usage. Both temporary ports were closed afterward. |

The engine test fixes changed only expectations: MCP intentionally keeps
private execution diagnostics off host streams; a cache-replacement test now
changes `candidate_k`, which is effective even when reranking is off. The
web config fix makes the standalone-build flag part of Turbo's build cache
key; it changes no product UI. No provider call, commit, push, deployment,
model download, or book re-index occurred in this release check.

This candidate remains in two shared, dirty working trees. The base install
used this host's cached dependencies, so an empty-cache install or different
OS is unproven. The final wheel's one-question browser walkthrough proves a
specific paired UI flow, not broad quality. Full-book hosted hybrid retrieval missed its predeclared
18-question quality threshold, and the targeted answer-abstention repair
does not establish a broad answer-grounding rate. See the
[retrieval audit](2026-09-23-book-heldout-retrieval.md) and
[answer audit](2026-09-23-book-answer-grounding.md). Publication still needs
a separately reviewed scope and user authorization.

Local logs: `/tmp/rag-final-suite2-20260923.log`,
`/tmp/rag-web-release-tests-20260923.log`,
`/tmp/rag-web-release-typecheck-20260923.log`,
`/tmp/rag-web-release-lint2-20260923.log`,
`/tmp/rag-web-release-build2-20260923.log`,
`/tmp/rag-final-stage-20260923.log`, and
`/tmp/rag-final-package-20260923.log`; corresponding resource files are
alongside them. Installed smoke artifacts are under
`/tmp/ragmcp-clean-20260923/`.
