# CLI contract refactor

## Goal

Make the existing `rag` CLI predictable for agents while keeping a deliberate,
inspectable human terminal UI.

## Invariants

- Retrieval, MCP protocol, companion security, and Studio behavior do not change.
- Existing settings and agent registrations are preserved unless an explicit command changes them.
- The human UI remains real terminal interaction, not a mocked presentation.

## Acceptance criteria

- `rag menu` launches the interactive menu; bare `rag` prints help in both TTY and non-TTY contexts.
- `rag query ... --format json` always emits the machine-readable result contract; `--format text` always emits readable passages.
- `rag setup` never silently registers an MCP host.
- `rag doctor` reports optional MCP-host status but does not fail solely because no host is installed or signed in.
- Existing focused CLI, MCP, settings, and companion tests remain green.
- A PTY walkthrough demonstrates menu navigation, cancellation, and quitting.

## Non-goals

- Migrating from argparse to Click.
- Rewriting retrieval or moving storage.
- Changing MCP registration semantics beyond removing hidden setup side effects.
- Deployment or publication.
