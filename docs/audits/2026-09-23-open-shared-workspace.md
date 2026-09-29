# Open CLI-created workspaces in paired Studio — 2026-09-23

The engine already listed shared workspaces, but Studio loaded only the active
ID in its browser-local catalog and had no selector. In an earlier walkthrough
the browser could not reach a CLI-created `scope-test` workspace until a copy
of the fixture was placed in `personal`. Paired Sources now reads the bounded,
metadata-only `/v1/workspaces` list, supports Refresh, and opens a chosen ID.
Before switching, it validates any existing browser snapshot for the target,
saves the outgoing browser snapshot under its old ID, then saves the new active
ID and reloads. Existing revisioned source reconciliation loads the target.
The companion token is not saved in the workspace catalog. Switching is
disabled during source sync, an answer, comparison, unfinished answer save, or
active file import. Failed reads/writes leave the active ID unchanged and show
an error.

The installed CLI imported one distinct synthetic file into `cli-distinct`:
`phoenix-only.txt`, 51 bytes, with the exact passage “Phoenix Observatory:
unique CLI workspace passage.” No query was run for it. The installed-wheel
browser listed the new workspace after Refresh, switched from `personal`,
loaded its single source, showed that passage in the source inspector, and
included the source in Playground queries. A same-tab reload kept the active
ID. Switching back restored `personal`'s two sources. CLI reported unchanged
revisions throughout: `cli-distinct` was
`1790181005620-157f905cd3ac` and `personal` was
`1790180200225-88360586838c`. Published-index inspection for the new CLI
workspace still returned `not_indexed`, confirming that opening it did not
build an engine index. English and Persian Sources layouts were visually
checked; the imported-source detail is translated in Persian. Guest browser
tabs have distinct local scopes, so a fresh tab starts at its own saved active
ID and can choose any listed engine workspace.

The final packaged wheel was rebuilt and reinstalled from cached dependencies
outside both repositories. Final wheel SHA-256:
`6e80ebeb7a44da798b3c50709c76cac29473d38aefe25fde355254cbc7327b48`;
embedded Studio bundle SHA-256:
`afdc5924c6a5ebfd1099bfec08bbb5acbc780600ff3af3a9e5480847aedba58f`.
Twenty-four focused browser tests, web typecheck, targeted ESLint, bounded
Next production build, wheel build, and `git diff --check` passed. The final
wheel repeated the live CLI-workspace switch and unique source check. The
browser also now accepts an 80-character Unicode workspace name, matching
the Python engine's code-point limit; the pre-fix focused test failed and
the corrected test passed. The test browser tabs and loopback servers were
closed. This was a Linux cached-
dependency fixture check, not fresh-machine or full-book quality acceptance.
No paid provider call, model load, engine index build, commit, push, or deploy.

## Retrieval parity follow-up

Using that same installed wheel and `cli-distinct` source, a bounded
`Phoenix Observatory` retrieval ran with embeddings, reranking, contextual
retrieval, and the hosted API key off. The CLI built a one-chunk lexical index
and returned one rank-1 passage from `phoenix-only.txt`, source ID
`file-f4d9522688c0018c956ee2bbd1d357bcefe56f39a7a98a6dc3388049e2ad8b2d`,
verified offsets 0–50, and score 0.5753642320632935. Installed-wheel MCP
stdio `search_workspace` returned the same source, rank, offsets, passage,
and score; its two JSON-RPC replies were the only stdout lines and stderr
was empty. The paired installed Studio switched from `personal` to
`cli-distinct`, ran its Retrieval view, and displayed one rank-1 passage
with score rounded to 0.575. Opening the evidence highlighted the exact
passage at offsets 0–50 in the original source; this was also visually
checked. No answer generation or semantic-recall quality is implied by this
one-file lexical fixture. The local server and test tab were closed after
verification.
