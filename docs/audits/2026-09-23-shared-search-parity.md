# Shared search parity check — 2026-09-23

The browser had already completed a retrieval-only run over a nine-chapter,
884-chunk BM25 book index. This check added ad hoc retrieval-only CLI and MCP
calls through the same `execute_run(mode='retrieve')` path.

The focused synthetic integration test exercises a keyless workspace through
`rag search` and stdio MCP `search_workspace`: both return the same source,
MCP omits passage text/scores by default, explicit content returns it, and
stale revisions fail. The CLI/MCP/execution regression selection passed
58 tests. After a final CLI privacy assertion, its focused test passed again.
The offline wheel/sdist build passed, and inspection of the wheel found the
CLI command, MCP tool, and shared response projection. Both repositories'
`git diff --check` passed.

For the real book, the first new CLI invocation used no explicit source IDs.
It therefore chose a different cache scope than Studio and created a second
884-chunk lexical index in the isolated QA directory. This exposed the cache
identity mismatch; `execute_run` now treats omitted workspace source IDs as
the same all-source selection Studio sends. A subsequent CPU-limited CLI run
opened the existing Studio table (884 rows), loaded its chunks, and returned
five passages. The exact precision/recall anchor was rank four. A CPU-limited
stdio MCP call with default metadata-only output also opened that existing
table, returned five passages, and matched CLI rank, source ID, and offsets in
order. Its output had no passage text or score. The CLI run IDs were
`fd5a0754960f44a0b15330ed514ebac5` (pre-fix new-scope run) and the final
MCP run was `135a82f605b34c7aabd54495992504f8`; the post-fix CLI run ID was
recorded in the then-local metadata file.

Both book processes had embedding provider `none`, reranker `off`, contextual
retrieval `off`, blank OpenRouter/OpenAI keys, and CPU affinity 4,5 at lower
priority. Their diagnostic output showed lexical search only and no synthesis.
The original book QA directory under `/tmp/rag-studio-book-dJqD9w` was removed
by the environment after these checks, so its private JSON results and index
cannot be re-opened for this audit. The conclusions above reflect the observed
command outputs, not surviving raw artifacts. No live answer quality, semantic
retrieval, clean-machine installation, or remote publication is established.

Follow-up: a legacy repeated-passage fixture showed that the engine selected a
passage but search returned zero because it could not establish a unique source
offset. A failing test captured this. The saved run now retains that passage in
`retrieved_passages` with null offsets and `location_verified: false`, while its
clickable `evidence` remains empty. CLI/MCP projection uses the complete
retrieved list and still excludes text/scores from default MCP output. The
affected execution, storage, CLI, MCP, and evaluation selection passed 67
tests after this change. This fixture result does not replace the missing
real-book raw artifacts.
