# CLI and MCP source import into the shared workspace

## Problem

Paired Studio can publish source snapshots to the engine, but `rag workspaces`
only reads them. A person at the terminal cannot add a book chapter to the
same source set that Studio and MCP subsequently inspect or query.

## Contract

- `rag sources add WORKSPACE_ID --path FILE_OR_FOLDER [--type EXT] [--json]`
  imports extracted text into the existing private `WorkspaceStore`; it does
  not build an index, load an embedding model, query a provider, or send text
  outside the machine. A new workspace uses its ID as its display name.
- A folder uses the same bounded scope discovery and extension filter as
  `rag query`; an oversized mixed-format book folder is rejected before
  parsing. No files or unreadable text is an error, not a partial import.
- Reimporting a path updates its stable source ID in place. Other sources are
  preserved. The full replacement uses the observed workspace revision, so a
  concurrent Studio edit causes a conflict instead of losing that edit.
- A successful response returns metadata only: workspace ID, revision, source
  count, imported source count, and bytes. It never prints source text or
  private stored paths. Browser and MCP read the same resulting snapshot.
- With `--json`, CLI stdout contains exactly one parseable JSON value; parser
  diagnostics go to local stderr. MCP stdout contains only JSON-RPC messages,
  and parser diagnostics cannot reach the MCP host's stderr.
- MCP `add_workspace_sources` calls that same importer for one explicit local
  path. It requires the caller's observed workspace revision (`null` for a new
  workspace) and rejects stale calls even if another edit completed while the
  source was being parsed. It returns metadata only and declares itself a
  mutating local tool.

## Verification

Use isolated storage and tiny Markdown/HTML fixtures. Prove folder filtering,
reimport without duplicate IDs, preservation of a Studio-authored source,
conflict rejection, and metadata-only output. Run the real book folder through
the `--type md` import into an isolated workspace, then read it through the
existing shared-workspace interfaces without starting a model.

Verified locally: the full Python suite passed 766/766. An added failure case
confirmed an unreadable chapter leaves the prior revision intact; adjacent
tests passed 46/46 afterward. The offline wheel contains the import module and
CLI command. The nine real Markdown chapters imported into temporary storage
as 477,021 bytes; no active workspace or index was changed.
Studio's existing source-reconciliation tests also passed 10/10, including
independent remote additions and edits.

MCP acceptance: the tool is advertised as mutating, an actual stdio MCP process
imports one source, `get_workspaces` sees the same revision, a stale retry fails
before parsing, and public MCP output contains no source text. The final full
offline Python suite passed 769/769; the rebuilt wheel contains this tool.

Installed-wheel acceptance: a new offline venv outside both source trees
revealed a legacy PyMuPDF warning before CLI JSON. After switching the PDF
import to `pymupdf` and isolating parser output in both transports, the
installed CLI imported a synthetic chapter with one valid JSON stdout value;
installed CLI and MCP search returned the same passage rank and source ID.
Two red-before/green-after regression tests cover the transport boundary, and
88 affected tests passed. The wheel SHA-256, run IDs, and limitations are
recorded in [the installed-wheel audit](audits/2026-09-23-installed-wheel-cli-mcp.md).
