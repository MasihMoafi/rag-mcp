# RAG CLI, MCP and Studio audit — 2026-09-08

The CLI now presents concrete embedding models and selectable installed Ollama models. Reproducible startup, pairing and retrieval-contract failures were repaired in rag-mcp and rag-studio-next. Existing unrelated working-tree changes were preserved. Following the user's deployment instruction, the tested Studio working tree was deployed to production on 2026-09-08. No commit, push or package publication was performed.

Production: https://rag.masihmoafi.com, deployment `dpl_ACWrKzzoYQ13uLmya3UZH8fpFZF8` (`rag-8nh8uqvkk-masihmoafis-projects.vercel.app`), READY. Previous deployment retained as rollback checkpoint: `dpl_3HmMHMMCTwsgPQs1jH6tpZZ984tG` (`rag-nhyty8dq6-masihmoafis-projects.vercel.app`).

Post-deployment browser verification passed on the public domain: custom-port pairing without Codex login, real Ollama passage/query embeddings, three-file upload, explicit hybrid search, cited answer, citation opening and reload persistence. Chromium 152 required its loopback-network permission; granting that permission (equivalent to approving local access) resolved the initial blocked health request.

## Changes

- Shortened the main menu and removed the introductory “meaning-based search” explanation. CPU presets name the model, language coverage and approximate download size. Ollama names the installed recommended model and provides a numbered model list. API setup has separate provider choices and detects the selected provider’s environment key even when Ollama is recommended.
- Settings offers numbered categories and reuses the model chooser. Interactive searches respect the saved result count. Invalid zero/negative menu indexes no longer select an unintended item.
- Detection and automatic setup preserve the configured Ollama URL instead of probing and saving only localhost.
- Fixed `rag serve` accessing an undefined `studio_url`. Non-default companion ports are included in pairing URLs, validated as numeric loopback ports, and preserved across browser reloads.
- Companion availability no longer depends on Codex authentication. Studio can use embeddings without Codex login while still reporting generation availability separately. Pairing state is cleared when the active storage scope has no pairing.
- Studio now sends query/passage mode to `/embed`; the companion uses the model’s query encoder for questions. Older clients that omit the field retain passage behavior.
- CLI query errors now return a failure exit code; MCP tool errors now set `isError`.
- Removed the API-key prompt from connected local embedding selections and corrected the companion description: its configured provider may itself be remote. Updated Persian copy and installer smoke-test inputs.

## Verification

| Check | Result |
| --- | --- |
| Python full suite | 58 passed, 1 skipped; final companion/CLI/MCP subset after the encoder change: 17 passed |
| Studio behavior tests | 264 passed across 67 files |
| Studio TypeScript | Passed |
| Studio ESLint | Zero errors; one existing unused-parameter warning in the hosted API embedder |
| Studio production build | Passed |
| Diff whitespace checks; installer shell syntax | Passed |
| Actual no-argument `uv run --no-sync rag` in a pseudo-terminal | Menu → installed Ollama selection → successful 1024-dimension probe → isolated saved config |
| Real local search | English and Persian BM25; Ollama qwen3-embedding:0.6b hybrid search retrieved physician/patient/hospital for a doctor/sick query |
| Real stdio MCP | Initialize, tool listing, settings write/read and synthetic-corpus search passed |
| Custom API setup | Local mock endpoint verified the chosen endpoint, model, dummy key, embedding probe and subsequent search |
| Browser companion | Actual Ollama embeddings with Codex unavailable, port 47839, token removal from URL and reload persistence passed |
| Browser workflow | Three English/Persian Markdown uploads, passage and query embeddings, a cited local-fallback answer, citation opening and source/answer persistence passed |
| Browser hybrid search | Manual configuration → Both, balanced; doctor/sick query retrieved physician/patient/hospital, with citation and reload persistence |
| Browser navigation/layout | Homepage plus all nine Studio destinations, English/Persian, at 1440×1000 and 390×844: 40 HTTP-200 checks, no page errors or horizontal overflow |
| Public website | Deployment READY and production alias confirmed; real companion/upload/hybrid retrieval/citation/reload workflow passed on the live domain |

The initial browser upload failure was a harness issue: Snap Chromium could not read host `/tmp` file paths. Uploading the same file bytes through Playwright resolved it. The test did not require an application ingestion change.

## Limits

- The skipped Python test requires active CUDA. GPU/OOM behavior was not proven here. Existing LanceDB and Node deprecation warnings remain.
- Live Google OAuth, paid embedding/generation providers, real Codex-backed generation and browser model downloads were not exercised. Their existing behavior/contract tests do not establish live service availability.
- Existing agent registrations were not replaced. Registration paths are covered by tests; MCP itself was exercised over stdio.
- The Docker/npm clean-install matrix was not rerun. Its scripted menu inputs were updated and shell syntax checked; existing built wheel/tarball artifacts were not republished or rebuilt.
- This is functional regression and integration evidence using a small synthetic corpus, not a retrieval-quality benchmark or an assurance about every model, file or workload.
- Website fixes are deployed. The user's existing `uv run rag` command uses the updated CLI checkout; existing long-running MCP sessions need a restart to load server changes.

## Evidence

- `terminal-preview.txt`: actual revised terminal flow.
- `web-tests.log`: individual Studio tests and totals.
- `results.json`: browser route/viewport results and page-error list.
- `paired-studio.png`, `studio-playground-390.png`: desktop/mobile workflow screenshots.
- Reproduction scripts, additional screenshots and pre-edit backups: `/tmp/rag-cli-audit/`.
- Both repositories' `ES.md` files record the session.
