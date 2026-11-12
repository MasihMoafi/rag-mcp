#!/usr/bin/env bash
# Exercise a built wheel and packed npm launcher without provider requests.
set -euo pipefail

WHEEL=${WHEEL:-$(ls /work/ragmcp_cli-*.whl | head -1)}
NPM_TGZ=${NPM_TGZ:-$(ls /work/ragmcp-*.tgz | head -1)}
FACTS=${FACTS:-/work/install/facts.md}
scratch=$(mktemp -d)
mock_pid=''
cleanup() {
  if [ -n "$mock_pid" ]; then kill "$mock_pid" 2>/dev/null || true; wait "$mock_pid" 2>/dev/null || true; fi
  python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$scratch"
}
trap cleanup EXIT
export RAG_MCP_CONFIG="$scratch/config.json" RAG_MCP_DATA_DIR="$scratch/index"
export RAG_MCP_WORKSPACE_DIR="$scratch/workspaces" RAG_MCP_TRACE_DIR="$scratch/traces"
export RAG_MCP_EMBED_PROVIDER=none RAG_MCP_RERANKER_TYPE=off RAG_MCP_CONTEXTUAL_RETRIEVAL=off
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1
export RAGMCP_UVX_FROM="$WHEEL" npm_config_offline=true npm_config_cache="$scratch/npm-cache"
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY

npm install --offline --no-audit --no-fund --prefix "$scratch/launcher" "$NPM_TGZ" >/dev/null
rag="$scratch/launcher/node_modules/.bin/rag"
"$rag" --help | python3 -c 'import sys; assert "setup" in sys.stdin.read()'
npx --yes "file:$NPM_TGZ" --help | python3 -c 'import sys; assert "setup" in sys.stdin.read()'
pnpm --package="$NPM_TGZ" dlx rag --help | python3 -c 'import sys; assert "setup" in sys.stdin.read()'
node_bin=$(command -v node)
mkdir "$scratch/no-uv"
if PATH="$scratch/no-uv" "$node_bin" "$scratch/launcher/node_modules/ragmcp-cli/bin/rag.js" --help 2> "$scratch/no-uv.err"; then
  echo 'FAIL: launcher succeeded without uv' >&2
  exit 1
fi
python3 -c 'import sys; assert "astral.sh/uv" in open(sys.argv[1]).read()' "$scratch/no-uv.err"
"$rag" sources add smoke --path "$FACTS" --json > "$scratch/import.json"
"$rag" search 'launch codename' --workspace smoke --json > "$scratch/search.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert any("Cedar Lantern" in p["content"] for p in d["passages"])' "$scratch/search.json"
python3 -c 'import re,sys; q=set(re.findall(r"\w+", "nocturnal beacon")); words=set(re.findall(r"\w+", open(sys.argv[1]).read().lower())); assert q.isdisjoint(words)' "$FACTS"
"$rag" search 'nocturnal beacon' --workspace smoke --json > "$scratch/lexical-miss.json"
python3 -c 'import json,sys; assert not json.load(open(sys.argv[1]))["passages"]' "$scratch/lexical-miss.json"
if "$rag" query 'launch codename' --path "$FACTS" --format json > "$scratch/answer.json" 2> "$scratch/answer.err"; then
  echo 'FAIL: an answer with evidence succeeded without an OpenRouter key' >&2
  exit 1
fi
python3 -c 'import json,sys; assert "RAG answer synthesis failed (ValueError)" in json.load(open(sys.argv[1]))["error"]' "$scratch/answer.json"
printf '%s\n%s\n%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize"}' '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_workspace","arguments":{"workspace_id":"smoke","query":"launch codename","include_content":true}}}' |
  "$rag" mcp > "$scratch/mcp.jsonl"
python3 -c 'import json,sys; r=[json.loads(x) for x in open(sys.argv[1])]; assert any(t["name"]=="query_knowledge_base" for t in r[1]["result"]["tools"]); result=r[2]["result"]; assert not result.get("isError",False); assert "Cedar Lantern" in result["content"][0]["text"]' "$scratch/mcp.jsonl"
mock_script="$(dirname "$0")/mock_embeddings.py"
python3 "$mock_script" 18089 "$scratch/mock.jsonl" &
mock_pid=$!
python3 -c 'import time,urllib.request; url="http://127.0.0.1:18089/v1/models"; exec("for i in range(50):\n try:\n  urllib.request.urlopen(url,timeout=1).close(); break\n except OSError:\n  time.sleep(.1)\nelse:\n raise RuntimeError(\"mock did not start\")")'
env -u RAG_MCP_EMBED_PROVIDER "$rag" setup --provider openai_compatible --base-url http://127.0.0.1:18089/v1 --model mock-embed --api-key local > "$scratch/setup.log"
env -u RAG_MCP_EMBED_PROVIDER "$rag" search 'nocturnal beacon' --workspace smoke --json > "$scratch/vector-search.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert any("Cedar Lantern" in p["content"] for p in d["passages"])' "$scratch/vector-search.json"
python3 -c 'import json,sys; rows=[json.loads(line) for line in open(sys.argv[1])]; assert any(row["model"]=="mock-embed" and row["path"].endswith("/embeddings") and "nocturnal beacon" in row["input"] for row in rows)' "$scratch/mock.jsonl"
if env -u RAG_MCP_EMBED_PROVIDER RAG_MCP_CONFIG="$scratch/bad-config.json" "$rag" setup --provider openai_compatible --base-url http://127.0.0.1:1/v1 --model nope --api-key local > "$scratch/bad-setup.log" 2>&1; then
  echo 'FAIL: unreachable provider setup succeeded' >&2
  exit 1
fi
test ! -f "$scratch/bad-config.json"
echo 'PASS: built npm launcher and wheel: keyless BM25, key-required answer, MCP'
