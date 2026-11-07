#!/usr/bin/env bash
# Exercise a built wheel and packed npm launcher without provider requests.
set -euo pipefail

WHEEL=${WHEEL:-$(ls /work/ragmcp-*.whl | head -1)}
NPM_TGZ=${NPM_TGZ:-$(ls /work/ragmcp-*.tgz | head -1)}
FACTS=${FACTS:-/work/install/facts.md}
scratch=$(mktemp -d)
trap 'python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$scratch"' EXIT
export RAG_MCP_CONFIG="$scratch/config.json" RAG_MCP_DATA_DIR="$scratch/index"
export RAG_MCP_WORKSPACE_DIR="$scratch/workspaces" RAG_MCP_TRACE_DIR="$scratch/traces"
export RAG_MCP_EMBED_PROVIDER=none RAG_MCP_RERANKER_TYPE=off RAG_MCP_CONTEXTUAL_RETRIEVAL=off
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1
export RAGMCP_UVX_FROM="$WHEEL" npm_config_offline=true npm_config_cache="$scratch/npm-cache"
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY

npm install --offline --no-audit --no-fund --prefix "$scratch/launcher" "$NPM_TGZ" >/dev/null
rag="$scratch/launcher/node_modules/.bin/rag"
"$rag" --help | python3 -c 'import sys; assert "setup" in sys.stdin.read()'
"$rag" sources add smoke --path "$FACTS" --json > "$scratch/import.json"
"$rag" search 'launch codename' --workspace smoke --json > "$scratch/search.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert any("Cedar Lantern" in p["content"] for p in d["passages"])' "$scratch/search.json"
if "$rag" query 'launch codename' --path "$FACTS" --format json > "$scratch/answer.json" 2> "$scratch/answer.err"; then
  echo 'FAIL: an answer with evidence succeeded without an OpenRouter key' >&2
  exit 1
fi
python3 -c 'import json,sys; assert "RAG answer synthesis failed (ValueError)" in json.load(open(sys.argv[1]))["error"]' "$scratch/answer.json"
printf '%s\n%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize"}' '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' |
  "$rag" mcp > "$scratch/mcp.jsonl"
python3 -c 'import json,sys; r=[json.loads(x) for x in open(sys.argv[1])]; assert any(t["name"]=="query_knowledge_base" for t in r[1]["result"]["tools"])' "$scratch/mcp.jsonl"
echo 'PASS: built npm launcher and wheel: keyless BM25, key-required answer, MCP'
