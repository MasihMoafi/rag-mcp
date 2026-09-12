from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from rag import fetch


@pytest.mark.parametrize('error', [RuntimeError('provider failed'), fetch.PipelineClosedError('retired')])
def test_query_retry_is_bounded_and_only_for_closed_pipeline(monkeypatch, error):
    from unittest.mock import Mock

    search = Mock(side_effect=error)
    acquire = Mock(return_value=SimpleNamespace(search=search))
    monkeypatch.setattr(fetch, 'get_rag_pipeline_v2', acquire)
    with pytest.raises(type(error)) as caught:
        fetch.query_documents('question')
    assert caught.value is error
    expected = 2 if isinstance(error, fetch.PipelineClosedError) else 1
    assert acquire.call_count == search.call_count == expected


def test_reset_attempts_other_pipelines_after_close_failure(monkeypatch):
    calls = []
    failure = RuntimeError('first close failed')

    def close():
        calls.append(True)
        if len(calls) == 1:
            raise failure

    cache = {name: SimpleNamespace(close=close) for name in ('first', 'second')}
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', cache)
    with pytest.raises(RuntimeError) as caught:
        fetch.reset_pipelines()
    assert caught.value is failure
    assert len(calls) == 2
    assert len(cache) == 1
    fetch.reset_pipelines()
    assert not cache


def test_cache_trim_retains_failed_cleanup_and_continues(monkeypatch):
    from unittest.mock import Mock

    failed = Mock(side_effect=[OSError('cleanup failed'), True])
    retired = Mock(return_value=True)
    cache = {'failed': SimpleNamespace(close_if_idle=failed),
             'idle': SimpleNamespace(close_if_idle=retired), 'keep': SimpleNamespace()}
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', cache)
    monkeypatch.setenv('RAG_MCP_CACHE_SIZE', '1')
    fetch._trim_cached_pipelines('keep')
    assert list(cache) == ['failed', 'keep']
    retired.assert_called_once_with()
    fetch._trim_cached_pipelines('keep')
    assert list(cache) == ['keep']


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


def test_cache_retires_idle_pipeline_and_reopens_without_reindex(sources, monkeypatch):
    monkeypatch.setenv('RAG_MCP_CACHE_SIZE', '1')
    first = fetch.get_rag_pipeline_v2(sources[0])
    try:
        fetch.get_rag_pipeline_v2(sources[1])
        assert first._queries.closed
        assert len(fetch._rag_system_v2_instances) == 1
        def unexpected_reindex(*args):
            pytest.fail('Eviction must retain reusable on-disk evidence')
        monkeypatch.setattr(fetch.RAGPipelineV2, '_index_documents', unexpected_reindex)
        reopened = fetch.get_rag_pipeline_v2(sources[0])
        assert reopened.search('Cedar')[0]['content'] == 'Cedar evidence'
        assert reopened is not first
    finally:
        fetch.reset_pipelines()


def test_cache_skips_active_pipeline_then_retires_it_when_idle(sources, monkeypatch):
    monkeypatch.setenv('RAG_MCP_CACHE_SIZE', '1')
    first = fetch.get_rag_pipeline_v2(sources[0])
    try:
        with first._queries.query():
            second = fetch.get_rag_pipeline_v2(sources[1])
            assert len(fetch._rag_system_v2_instances) == 2
            assert not first._queries.closing
        assert fetch.get_rag_pipeline_v2(sources[1]) is second
        assert first._queries.closed
        assert len(fetch._rag_system_v2_instances) == 1
    finally:
        fetch.reset_pipelines()


def test_cache_retirement_uses_last_acquisition_order(sources, monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MCP_CACHE_SIZE', '2')
    third_source = tmp_path / 'third.txt'
    third_source.write_text('Cedar evidence')
    try:
        first = fetch.get_rag_pipeline_v2(sources[0])
        second = fetch.get_rag_pipeline_v2(sources[1])
        assert fetch.get_rag_pipeline_v2(sources[0]) is first
        third = fetch.get_rag_pipeline_v2(str(third_source))
        assert second._queries.closed
        assert not first._queries.closed and not third._queries.closed
        assert len(fetch._rag_system_v2_instances) == 2
    finally:
        fetch.reset_pipelines()


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


def test_query_recovers_when_reset_closes_pipeline_before_search(sources, monkeypatch):
    original_get = fetch.get_rag_pipeline_v2
    calls = []

    def acquire(doc_path):
        pipeline = original_get(doc_path)
        calls.append(pipeline)
        if len(calls) == 1:
            fetch.reset_pipelines()
        return pipeline

    monkeypatch.setattr(fetch, 'get_rag_pipeline_v2', acquire)
    try:
        assert fetch.query_documents('Cedar', sources[0])[0]['content'] == 'Cedar evidence'
        assert len(calls) == 2
        assert calls[0] is not calls[1]
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
