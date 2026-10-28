"""A query that arrives while another is running waits its turn instead of failing with engine_busy
(Masih, 28 Oct 2026: "make it accept multiple requests")."""
import json
import threading
import time

from filelock import FileLock
import pytest

from rag import execution, server
from rag.workspaces import WorkspaceStore


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(tmp_path / 'workspaces'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path / 'traces'))
    monkeypatch.setattr(server, '_query_knowledge_base', lambda *a, **k: json.dumps({'result': 'Friday'}))
    store = WorkspaceStore()
    store.root.mkdir(parents=True, exist_ok=True)
    return store


def hold(lock_path, seconds):
    """Another process's query: holds the engine lock, then releases it."""
    held = threading.Event()
    def run():
        with FileLock(lock_path):
            held.set()
            time.sleep(seconds)
    thread = threading.Thread(target=run)
    thread.start()
    held.wait()
    return thread


def test_a_second_query_waits_for_the_running_one_and_succeeds(engine, tmp_path):
    other = hold(engine.root / '.execution.lock', 1.0)
    began = time.monotonic()
    result = execution.execute_run('When?', str(tmp_path), workspace_store=engine)
    other.join()
    assert result.get('code') != 'engine_busy', result
    assert result['status'] == 'success' and result['answer'] == 'Friday'
    assert time.monotonic() - began >= 0.8  # it queued behind the other query


def test_a_query_still_reports_busy_after_the_wait_limit(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(execution, 'ENGINE_WAIT_SECONDS', 0.2)
    other = hold(engine.root / '.execution.lock', 1.0)
    result = execution.execute_run('When?', str(tmp_path), workspace_store=engine)
    other.join()
    assert result['code'] == 'engine_busy'
