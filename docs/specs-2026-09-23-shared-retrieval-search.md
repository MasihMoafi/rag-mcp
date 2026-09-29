# Shared retrieval-only search

## Goal

An ad hoc question against a saved workspace should retrieve the same ranked passages through Studio, CLI, and MCP without generating an answer or requiring an answer provider.

## Contract

- `rag search QUESTION --workspace ID` runs the existing `retrieve` execution mode and prints ranked passages plus the saved run ID. `--json` returns one JSON value; `--no-content` suppresses passage text and scores.
- MCP `search_workspace` runs that same mode for an explicit workspace. Its default output contains only source IDs, names, ranks, offsets, counts, and revision/run IDs. `include_content: true` explicitly releases passage text and scores to the MCP host.
- Engine diagnostics, including native-library writes during an index build, stay off MCP stdout and stderr; the host receives only JSON-RPC results unless content is explicitly requested.
- A retrieved passage remains in CLI/MCP search when an older index cannot prove its exact source offset. Its offsets are null and `location_verified` is false; Studio's clickable `evidence` remains limited to passages whose offsets are verified against the saved source.
- Both clients accept optional source IDs and an expected workspace revision. A stale revision or invalid source produces a visible error and no generated answer.
- Neither client changes workspace content, provider settings, or an existing index. Normal engine run results and traces remain local and inspectable.

## Verification

- Unit checks cover metadata-only and explicit-content projections, including errors.
- A repeated-passage fixture proves unlocated retrieval is reported and saved without creating a false source link.
- An isolated CLI/MCP integration check proves both expose the same retrieved rank/source and the MCP default omits private text.
- One real-book CLI and MCP run reopens the existing index and matches the browser result without a provider call.

## Limits

This does not improve semantic recall, answer quality, model selection, or the larger release manifest.

Installed acceptance: a CLI-imported synthetic source appeared in paired
Studio, browser Retrieval returned its rank-1 passage, and CLI/MCP returned the
same source ID and rank. A first-build MCP run in the final offline wheel
produced two JSON-RPC lines and zero stderr bytes. Evidence and limits:
[installed live-client audit](audits/2026-09-23-live-browser-cli-mcp-and-stdio.md).
