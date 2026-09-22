"""Repeat selected existing contracts with external Python sockets blocked."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=HERE)
HERE = parser.parse_args().output.resolve()
HERE.mkdir(parents=True, exist_ok=True)
TESTS = [
    "tests/unit/test_query_decomposition.py",
    "tests/unit/test_server_settings_contract.py",
    "tests/unit/test_pipeline_config_validation.py",
    "tests/unit/test_search_contract.py",
    "tests/unit/test_retrieval_log_privacy.py",
    "tests/integration/test_rag_mcp_host.py",
    "tests/integration/test_rag_scope.py",
    "tests/integration/test_incremental_index.py",
    "tests/test_companion.py",
]
GUARD = '''import socket
_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
def checked(original):
    def call(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
            raise RuntimeError('readiness baseline blocks external network')
        return original(self, address)
    return call
socket.socket.connect = checked(_connect)
socket.socket.connect_ex = checked(_connect_ex)
'''

with tempfile.TemporaryDirectory(prefix="rag-readiness-") as temporary:
    scratch = Path(temporary)
    (scratch / "sitecustomize.py").write_text(GUARD)
    env = dict(os.environ)
    # Avoid selecting configured paid providers or writing into user indexes.
    for name in list(env):
        if name.startswith("RAG_MCP_") or name in {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}:
            env.pop(name)
    env.update(
        PYTHONPATH=os.pathsep.join([str(scratch), str(ROOT)]),
        RAG_MCP_CONFIG=str(scratch / "config.json"),
        RAG_MCP_DATA_DIR=str(scratch / "index"),
        XDG_STATE_HOME=str(scratch / "state"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    command = [str(ROOT / ".venv/bin/python"), "-m", "pytest", *TESTS, "-q", "--tb=short"]
    started = time.time()
    with (HERE / "pytest.log").open("w") as output:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    record = {
        "command": command, "started_unix": started,
        "elapsed_seconds": time.time() - started, "exit_code": result.returncode,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "platform": platform.platform(),
        "log_sha256": hashlib.sha256((HERE / "pytest.log").read_bytes()).hexdigest(),
        "scope": "Existing selected contracts; fake providers and temporary indexes. Not held-out quality or hosted latency.",
    }
    (HERE / "pytest-run.json").write_text(json.dumps(record, indent=2) + "\n")
    print((HERE / "pytest.log").read_text())
    raise SystemExit(result.returncode)
