# Run revision provenance in local traces

## Goal and contract

An inspected local run should identify the safe engine configuration and immutable workspace snapshot it actually used. The metadata-only trace for a shared `execute_run` records `configuration_revision` and, when a workspace snapshot was loaded, `workspace_revision` on its root span. They match the private saved run result. A per-run answer-model override is still shown by the existing effective `model` attribute; the configuration revision refers to the underlying shared settings snapshot.

The public trace reader selects these two identifiers and nothing from the configuration body, source documents, question, answer, paths, or credentials. Studio's span details shows both identifiers in English and Persian. CLI and MCP continue to use the same trace reader. Missing/invalid workspaces and disabled tracing preserve current behavior. No index, provider, or public answer behavior changes.

## Verification and boundary

- A simulated workspace run asserts the metadata and saved result revisions agree; its public trace excludes a private document marker.
- A model-override run keeps the shared configuration revision and reports the effective model independently.
- Focused engine tests and web typecheck/lint validate the exposed fields; a rendered browser check remains outstanding for this small inspector change.

These revisions are comparison identifiers, not a full release manifest or proof of answer quality. A full Git/dependency manifest remains open.

## Verified 2026-09-23

The two new assertions failed before implementation because the public trace lacked `configuration_revision`. After implementation, 22 focused execution, reader, and trace-pipeline tests passed. A simulated workspace answer matched both private revisions, excluded its document marker from the public trace, and an overridden answer model remained independently visible. Web typecheck, targeted ESLint, and both repositories' `git diff --check` passed. A bounded offline Next build and wheel/sdist build then passed. The wheel contains the revised Python code and a byte-identical copy of the staged Studio bundle (SHA-256 `8aab9190b77bba69eaf419fc20d4ab7db64c3603d6ebbbc8cb06fb9dc7f4c525`, build ID `cYThWDDUM9EzOs_RzCtSb`). In an isolated temporary venv with cached dependencies, the installed `rag` command started from outside the source tree and its bundled Studio returned HTTP 200 for `/studio` and `/fa/studio/observability?mode=advanced`. This is not a different-machine install or a visual browser check of the inspector fields.
