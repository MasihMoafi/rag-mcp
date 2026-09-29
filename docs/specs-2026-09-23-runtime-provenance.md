# Private runtime provenance for saved runs

## Goal

A saved run should identify the engine files, installed dependency versions,
local Git state when available, and bundled Studio artifact observed by the
running process. A metadata-only trace should expose only a stable fingerprint
of that private manifest.

## Contract

- The first query in a process captures one bounded runtime manifest. Each
  saved private run result includes it; the root metadata trace includes its
  SHA-256 fingerprint, not the dependency list or source paths.
- The manifest contains relative engine-code file names and hashes, installed
  distribution names/versions, Python/platform identifiers, bundled Studio
  archive hash when present, and Git HEAD plus `clean`, `dirty`, or
  `unavailable` state. It contains no absolute path, document text, key, query,
  or package installation URL.
- The manifest's fingerprint is deterministic for identical observations.
  Running source edits or dependency changes require a process restart to
  recapture; an installed wheel has no Git state unless its runtime directory
  is itself a repository.
- Discovery is local, bounded, and never calls a provider or network service.
  A provenance failure marks the manifest unavailable and cannot fail a query.

## Verification

- Focused tests prove stable fingerprints, clean/dirty/unavailable Git
  reporting, private/public separation, and success when capture fails.
- The affected execution and trace checks pass, and an offline wheel contains
  the provenance implementation.

## Limit

This describes the Python runtime and shipped Studio archive. It does not yet
prove the web source commit used to produce that archive or a fresh-machine
dependency resolution, so it is not a complete two-repository release manifest.

## Observed verification — 2026-09-23

The synthetic capture tests showed stable fingerprints, changed fingerprints
for dirty Git state and edited code, unavailable Git outside a repository,
and rejection of a symlinked source file. A run test showed the full manifest
in the private saved result, only its digest in the normal trace, and successful
retrieval when capture raised an error. The focused provenance tests and the
85-test affected execution/trace/CLI/MCP selection passed. In the current
dirty checkout, first capture found 58 engine files and 182 installed
distributions in about 188 ms; the manifest contained no home path or file URL.
The offline wheel/sdist build passed. Wheel inspection found the new module and
the staged Studio archive with SHA-256
`ee550f9e3fe93d7685423c3c805f2d6953651777b05c9bdecfaad0fc4353e5f0`.
No clean-machine install or web-source provenance check was performed.
