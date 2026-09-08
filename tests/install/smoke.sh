#!/usr/bin/env bash
# Acceptance checks for the npm launcher and the light Python install.
# Runs inside the image built from tests/install/Dockerfile. Every check prints
# PASS or FAIL; the script exits 1 if any check failed.
set -uo pipefail

WHEEL=$(ls /work/ragmcp-*.whl | head -1)
TGZ=$(ls /work/ragmcp-*.tgz | head -1)
export RAGMCP_UVX_FROM="$WHEEL"
export HOME=/root
export RAG_MCP_CONFIG=/root/.config/rag/config.json
MOCK_PORT=8089
MOCK_LOG=/tmp/mock_requests.jsonl
FACTS=/work/install/facts.md
failures=0

pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; [ -n "${2:-}" ] && echo "      $2"; failures=$((failures + 1)); }
check() { # check <name> <condition-exit-code> [detail]
  if [ "$2" -eq 0 ]; then pass "$1"; else fail "$1" "${3:-}"; fi
}
reset_config() { rm -f "$RAG_MCP_CONFIG"; }

echo "== wheel: $WHEEL"; echo "== npm tarball: $TGZ"; echo

# 1. Global npm install exposes `rag` and delegates to the Python CLI through uvx.
npm install -g "$TGZ" >/tmp/npm-install.log 2>&1
check "npm install -g exposes rag" $? "$(tail -3 /tmp/npm-install.log)"
help=$(rag --help 2>&1); rc=$?
check "rag --help runs through uvx and lists setup" $([ $rc -eq 0 ] && echo "$help" | grep -q "setup"; echo $?) "$help"

# 2. Light install: no PyTorch in a fresh environment that installs the wheel.
uv venv -q /tmp/light && uv pip install -q --python /tmp/light/bin/python "$WHEEL" >/tmp/light-install.log 2>&1
torch_check=$(/tmp/light/bin/python -c "import importlib.util as u; print('torch' if u.find_spec('torch') else 'no-torch')" 2>&1)
check "light install has no torch" $([ "$torch_check" = "no-torch" ]; echo $?) "$torch_check $(tail -2 /tmp/light-install.log)"

# 3. Doctor before any choice: provider none, not configured. (exit 1 is expected: no Codex login here)
reset_config
doctor=$(rag doctor --json 2>/dev/null)
check "doctor reports embedding provider none before setup" $(echo "$doctor" | python3 -c 'import json,sys; d=json.load(sys.stdin)["embedding"]; sys.exit(0 if d["provider"]=="none" and d["configured"] is False and d["full_extra"] is False else 1)'; echo $?) "$doctor"

# 4. Exact words only via flags; file is owner-only; query returns the fact plus the notice.
rag setup --provider none --no-verify >/tmp/setup-none.log 2>&1
check "setup --provider none saves config" $([ -f "$RAG_MCP_CONFIG" ]; echo $?) "$(cat /tmp/setup-none.log)"
mode=$(stat -c %a "$RAG_MCP_CONFIG" 2>/dev/null)
check "config file is mode 600" $([ "$mode" = "600" ]; echo $?) "mode=$mode"
out=$(rag query "What is the launch codename?" --path "$FACTS" 2>/tmp/query-none.err); rc=$?
check "BM25-only query finds Cedar Lantern" $([ $rc -eq 0 ] && echo "$out" | grep -q "Cedar Lantern"; echo $?) "$out"
check "BM25-only query carries the notice in JSON and on stderr" $(echo "$out" | grep -q '"notice"' && grep -q "rag setup" /tmp/query-none.err; echo $?) "$(cat /tmp/query-none.err | tail -2)"

# 5. Mock OpenAI-compatible + Ollama endpoints on a custom port.
rm -f "$MOCK_LOG"
python3 /work/install/mock_embeddings.py $MOCK_PORT "$MOCK_LOG" &
MOCK_PID=$!
sleep 1

# 5a. API provider via flags: base URL, model and key must all reach the mock.
reset_config
rag setup --provider openai_compatible --base-url "http://127.0.0.1:$MOCK_PORT/v1" --model mock-embed --api-key sk-install-test >/tmp/setup-api.log 2>&1
check "setup openai_compatible verifies through the mock endpoint" $(grep -q "Works: 8 dimensions" /tmp/setup-api.log; echo $?) "$(cat /tmp/setup-api.log)"
check "mock received the configured key and model" $(grep -q '"authorization": "Bearer sk-install-test"' "$MOCK_LOG" && grep -q '"model": "mock-embed"' "$MOCK_LOG"; echo $?) "$(tail -2 "$MOCK_LOG" 2>/dev/null)"
out=$(rag query "What is the launch codename?" --path "$FACTS" 2>/tmp/query-api.err); rc=$?
check "API-embedded query finds Cedar Lantern without a notice" $([ $rc -eq 0 ] && echo "$out" | grep -q "Cedar Lantern" && ! echo "$out" | grep -q '"notice"'; echo $?) "$out $(tail -3 /tmp/query-api.err)"

# 5b. Interactive chooser through piped answers: option 4 (API), then endpoint, model, key.
reset_config
printf '4\nhttp://127.0.0.1:%s/v1\nmock-embed-interactive\nsk-interactive\n' "$MOCK_PORT" | rag setup >/tmp/setup-interactive.log 2>&1
check "interactive setup lists six choices" $(grep -q "6) Exact words only" /tmp/setup-interactive.log; echo $?) "$(head -20 /tmp/setup-interactive.log)"
check "interactive setup saved the API choice" $(python3 -c "import json,sys; c=json.load(open('$RAG_MCP_CONFIG')); sys.exit(0 if c['embed_provider']=='openai_compatible' and c['embed_model']=='mock-embed-interactive' and c['embed_api_key']=='sk-interactive' and c['embed_base_url'].endswith(':$MOCK_PORT/v1') else 1)"; echo $?) "$(cat "$RAG_MCP_CONFIG" 2>/dev/null)"

# 5c. Ollama provider with a custom URL.
reset_config
rag setup --provider ollama --ollama-url "http://127.0.0.1:$MOCK_PORT" --model mock-ollama >/tmp/setup-ollama.log 2>&1
check "setup ollama verifies through the custom URL" $(grep -q "Works: 8 dimensions" /tmp/setup-ollama.log && grep -q '"path": "/api/embed"' "$MOCK_LOG"; echo $?) "$(cat /tmp/setup-ollama.log)"
out=$(rag query "Who owns packaging?" --path "$FACTS" 2>/dev/null); rc=$?
check "Ollama-embedded query finds the platform group" $([ $rc -eq 0 ] && echo "$out" | grep -q "platform group"; echo $?) "$out"

# 5d. A failing verification saves nothing.
reset_config
rag setup --provider openai_compatible --base-url "http://127.0.0.1:1/v1" --model nope --api-key sk-x >/tmp/setup-bad.log 2>&1; rc=$?
check "unreachable endpoint: setup fails and saves nothing" $([ $rc -ne 0 ] && [ ! -f "$RAG_MCP_CONFIG" ]; echo $?) "$(tail -2 /tmp/setup-bad.log)"

kill $MOCK_PID 2>/dev/null

# 6. Small local model (real download from Hugging Face). Skip with SKIP_FASTEMBED=1.
if [ "${SKIP_FASTEMBED:-0}" != "1" ]; then
  reset_config
  rag setup --provider fastembed >/tmp/setup-fastembed.log 2>&1; rc=$?
  check "setup fastembed downloads bge-small and verifies 384 dimensions" $([ $rc -eq 0 ] && grep -q "Works: 384 dimensions" /tmp/setup-fastembed.log; echo $?) "$(tail -3 /tmp/setup-fastembed.log)"
  out=$(rag query "When does the release train leave?" --path "$FACTS" 2>/dev/null); rc=$?
  # No notice proves the embedding path ran; a BM25-only fallback would carry one.
  check "fastembed query finds the release schedule without a notice" $([ $rc -eq 0 ] && echo "$out" | grep -q "first Tuesday" && ! echo "$out" | grep -q '"notice"'; echo $?) "$out"
else
  echo "SKIP  fastembed download (SKIP_FASTEMBED=1)"
fi

# 6b. Automatic setup: nothing local is running in the container, so it must fall back to the small local model.
reset_config
rag setup --auto >/tmp/setup-auto.log 2>&1; rc=$?
check "setup --auto picks the small local model when nothing else is available" $([ $rc -eq 0 ] && grep -q "Chosen: local model · BAAI/bge-small-en-v1.5" /tmp/setup-auto.log && grep -q "Works: 384 dimensions" /tmp/setup-auto.log; echo $?) "$(tail -3 /tmp/setup-auto.log)"

# 7. npx and pnpm dlx from the local tarball.
reset_config
# "file:" marks the tarball as a package spec; a bare path would be read as the command name.
npx_out=$(npx --yes "file:$TGZ" doctor --json 2>/dev/null)
check "npx <tarball> doctor --json returns JSON" $(echo "$npx_out" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; echo $?) "$npx_out"
pnpm_out=$(pnpm --package="$TGZ" dlx rag doctor --json 2>/dev/null | tail -1)
check "pnpm dlx doctor --json returns JSON" $(echo "$pnpm_out" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; echo $?) "$pnpm_out"

# 7b. Permanent install with uv: `rag` lands on PATH and runs.
uv tool install -q "$WHEEL" >/tmp/uv-tool-install.log 2>&1
tool_rag=$(ls /root/.local/bin/rag 2>/dev/null)
tool_doctor=$(/root/.local/bin/rag doctor --json 2>/dev/null)
check "uv tool install puts rag on PATH and doctor answers" $([ -n "$tool_rag" ] && echo "$tool_doctor" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; echo $?) "$(tail -2 /tmp/uv-tool-install.log) $tool_doctor"

# 8. Without uv on PATH the launcher explains how to install it and exits 1.
missing=$(PATH=/usr/local/bin:/usr/bin:/bin rag doctor 2>&1); rc=$?
check "missing uv: exit 1 with install hint" $([ $rc -eq 1 ] && echo "$missing" | grep -q "astral.sh"; echo $?) "$missing"

echo
if [ $failures -eq 0 ]; then echo "ALL CHECKS PASSED"; exit 0; fi
echo "$failures CHECK(S) FAILED"; exit 1
