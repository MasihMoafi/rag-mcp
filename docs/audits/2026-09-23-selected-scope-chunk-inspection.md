# Selected-scope published chunk inspection — 2026-09-23

The bug was a false `not_indexed` in Published chunks after querying only a
subset of saved sources. Inspection could name a source but not the query's
source set. The engine now accepts the selected IDs, hashes the same file set
as the query, echoes canonical IDs, and keeps the old root-scope fallback only
for all-source inspection. Preview rejects a scope. CLI `--scope-source`, MCP
`scope_source_ids`, and repeated HTTP `scope_source_id` use that reader.
The browser sends its current included IDs, rejects a response from another
selection, and explains when the inspected source is excluded. Private chunk
text remains opt-in for CLI/MCP; browser Published chunks explicitly requests
it to show the saved passage and exact source range.

An isolated installed-wheel run under `/tmp/rag-scope-ui-r8pcvm0l` used two
synthetic text sources, no embedding model, no reranker, no contextual retrieval,
and no provider key. A retrieval-only query selected `chapter` and succeeded.
All-source inspection returned `not_indexed`; selected inspection returned
`current`, one chunk, offsets 0–29. The installed CLI returned the same selected
page. In the packaged Studio, the English Published chunks view showed that
chunk and highlighted `Cedar Lantern meeting Friday.` in the source. Including
both sources showed the absent all-source index. Excluding `chapter` showed
an explicit selection message, not the old page. The Persian exclusion view
rendered legibly in RTL. The temporary browser tab and both loopback servers
were closed afterward.

Verification: six focused Python tests passed, including real query/index
construction and CLI/MCP/HTTP equality; three browser transport tests passed.
The browser typecheck, targeted ESLint, bounded Next production build, wheel
build, and `git diff --check` passed. Installed wheel SHA-256:
`a6bd9e281f900df5e966db7b3674de69679f326bc61afb7e9e982741c463fae2`.
Embedded Studio bundle SHA-256:
`fc9d837416cd53807b57b90be9b8426e413adb9905126f743a91904db84c9acd`.
The test used cached dependencies on this Linux host, not fresh-machine
resolution. It proves source-set inspection, not vector quality, generated
answer grounding, full-book semantic recall, or a release. No paid provider
call, full-book build, commit, push, or deployment occurred.
