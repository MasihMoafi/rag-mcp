# Studio bundle build provenance

## Goal

The bundled Studio artifact should say whether it was built from a measured
web source snapshot or merely staged from pre-existing Next output. A runtime
run manifest should carry that distinction and the build identity.

## Contract

- The bundle archive contains a bounded `rag-studio-build-manifest.json` with
  its Next build ID, source-attestation state, and staged artifact identity.
- A fresh `build_studio_bundle.py` build hashes build-relevant web and shared
  package files plus the root pnpm lock before and after `pnpm build`. A changed
  source snapshot or Git HEAD aborts staging. The installed pnpm lock must
  match the root lock. It records only relative file names and hashes, local
  Git HEAD/state when available, and names of present local env files, never
  their contents or a key/value hash.
- `--skip-build` marks the source unattested. It never associates the current
  source tree with an output that might have been compiled earlier.
- The stage and package hooks reject mismatched or missing server/build IDs.
  The Python private runtime manifest embeds the bounded bundle attestation;
  the normal trace still exposes only the runtime-manifest fingerprint.

## Verification

- Fixture tests prove fresh versus skipped attestation, source mutation
  rejection, installed-lock mismatch rejection, and build-ID mismatch rejection.
- A CPU-limited real build, staged archive inspection, offline wheel build,
  and focused engine tests verify the release path.

## Limits

Next may read `.env.local` or process environment at build time. The manifest
records only that such files were present, not their values, so it does not
prove every environment input or authorize sharing secret build settings.

## Observed verification — 2026-09-23

Five fixture tests passed for measured and skipped staging, source mutation,
stale installed lock, mismatched build IDs, tampered archive metadata, and
symlink-safe runtime provenance. The affected engine execution/trace selection
passed 61 tests. A real two-core, lower-priority Next build completed and
staged build ID `FwXJMXviGtG6TuyWlgK4_`. Its measured manifest contains 189
relative source-file hashes, matching root/installed pnpm locks, local Git
state `dirty`, and the presence of one `.env.local` file without its value.
The source fingerprint was
`9b045882af1f1e70b5c2b55a4b8aa67ee45ebe93e2bf226e35d831d33ba6e962`
and still matched the source tree after staging.

An initial offline wheel build exposed an isolated Hatchling import failure;
the hook now loads its validator from the source file. The second offline
wheel/sdist build passed. Wheel inspection found the exact staged archive,
SHA-256 `72b4e190b2c27ea1c6739964371ff0be7e50a59e85a3bcef28c8cf228ad4cdf6`.
The bundled server returned HTTP 200 for English `/studio` and Persian
`/fa/studio/observability?mode=advanced` on loopback, then exited. This is
local artifact verification, not a clean-machine install or production deploy.
