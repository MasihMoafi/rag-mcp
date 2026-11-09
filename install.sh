#!/bin/sh
# Install the `rag` command (CLI and MCP server) for the current user.
#   curl -LsSf https://raw.githubusercontent.com/MasihMoafi/rag-mcp/main/install.sh | sh
# Installs uv (https://docs.astral.sh/uv/) first if it is missing; uv provides Python.
# RAGMCP_REF: tag, branch or commit to install (default: main).
# RAGMCP_EXTRAS: optional extras, comma-separated, e.g. "office" or "office,qdrant".
# RAGMCP_SOURCE: a git URL or local checkout to install from instead of GitHub.
set -eu

source="${RAGMCP_SOURCE:-https://github.com/MasihMoafi/rag-mcp}"
ref="${RAGMCP_REF:-main}"
extras="${RAGMCP_EXTRAS:-}"
bin_dir="${XDG_BIN_HOME:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv, which provides Python for rag..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$bin_dir:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        say "uv was installed but this shell cannot find it. Open a new terminal and run this again." >&2
        exit 1
    fi
fi

case "$source" in
    http://* | https://* | git@* | file://*) from="git+$source@$ref" ;;
    *) from="$source" ;; # a local checkout: install it as it is
esac
package="ragmcp-cli"
[ -n "$extras" ] && package="ragmcp-cli[$extras]"

say "Installing rag from $from..."
uv tool install --force "$package @ $from"

if command -v rag >/dev/null 2>&1 || [ -x "$bin_dir/rag" ]; then
    say ""
    say "rag is installed. Next:"
    say "  rag setup                                   # pick a provider and embedding model"
    say "  rag query \"What changed?\" --path ./docs     # ask your documents"
    say "  rag connect claude                          # or: codex, copilot, gemini (adds the MCP server)"
fi
case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) say "" && say "Add $bin_dir to your PATH (run: uv tool update-shell), then open a new terminal." ;;
esac
