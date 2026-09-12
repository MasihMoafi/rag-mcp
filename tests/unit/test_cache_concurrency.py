from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from rag import fetch


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'none')
    monkeypatch.setenv('RAG_MCP_RERANKER_TYPE', 'off')
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', {})
    paths = [tmp_path / 'one.txt', tmp_path / 'two.txt']
    for path in paths:
        path.write_text('Cedar evidence')
    return [str(path) for path in paths]


def test_same_scope_initializes_once(sources, monkeypatch):
    entered, finish = Event(), Event()
    calls = []
    def build(config):
        calls.append(config)
        if len(calls) == 1:
            entered.set()
            assert finish.wait(5)
        return SimpleNamespace(config=config)
    monkeypatch.setattr(fetch, 'RAGPipelineV2', build)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(fetch.get_rag_pipeline_v2, sources[0])
        try:
            assert entered.wait(5)
            second = workers.submit(fetch.get_rag_pipeline_v2, sources[0])
            with pytest.raises(TimeoutError):
                second.result(timeout=0.05)
        finally:
            finish.set()
        assert first.result(timeout=5) is second.result(timeout=5)
    assert len(calls) == 1


def test_relative_index_directory_reuses_pipeline(sources, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('RAG_MCP_DATA_DIR', 'index')
    first = fetch.get_rag_pipeline_v2(sources[0])
    try:
        second = fetch.get_rag_pipeline_v2(sources[0])
        assert second is first
        monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
        assert fetch.get_rag_pipeline_v2(sources[0]) is first
        assert first.search('Cedar')[0]['content'] == 'Cedar evidence'
    finally:
        fetch.reset_pipelines()


def test_different_scopes_initialize_in_parallel(sources, monkeypatch):
    entered = Barrier(2)
    def build(config):
        entered.wait(timeout=5)
        return SimpleNamespace(config=config)
    monkeypatch.setattr(fetch, 'RAGPipelineV2', build)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(fetch.get_rag_pipeline_v2, source) for source in sources]
        results = [future.result(timeout=5) for future in futures]
    assert results[0] is not results[1]


def test_reset_waits_for_inflight_initialization_then_retires_it(sources, monkeypatch):
    entered, finish, closed = Event(), Event(), Event()
    def build(config):
        entered.set()
        assert finish.wait(5)
        return SimpleNamespace(config=config, close=closed.set)
    monkeypatch.setattr(fetch, 'RAGPipelineV2', build)
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending = workers.submit(fetch.get_rag_pipeline_v2, sources[0])
        try:
            assert entered.wait(5)
            reset = workers.submit(fetch.reset_pipelines)
            with pytest.raises(TimeoutError):
                reset.result(timeout=0.05)
        finally:
            finish.set()
        pending.result(timeout=5)
        reset.result(timeout=5)
    assert closed.is_set()
    assert not fetch._rag_system_v2_instances


def test_failed_lancedb_settings_replacement_keeps_previous_searchable(sources, monkeypatch):
    previous = fetch.get_rag_pipeline_v2(sources[0])
    monkeypatch.setenv('RAG_MCP_RERANKER_MODEL', 'changed-setting')

    def fail(config):
        raise RuntimeError('replacement setup failed')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(fetch, 'RAGPipelineV2', fail)
            with pytest.raises(RuntimeError, match='replacement setup failed'):
                fetch.get_rag_pipeline_v2(sources[0])
        assert previous.search('Cedar')[0]['original_content'] == 'Cedar evidence'
        replacement = fetch.get_rag_pipeline_v2(sources[0])
        assert replacement is not previous
        assert replacement.search('Cedar')[0]['original_content'] == 'Cedar evidence'
        with pytest.raises(RuntimeError, match='closing or closed'):
            previous.search('Cedar')
    finally:
        fetch.reset_pipelines()
