from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Barrier, Event
from unittest.mock import Mock

import pytest

from rag.query_lifecycle import QueryLifecycle


def test_idle_close_does_not_interrupt_or_mark_active_queries_closing():
    lifecycle = QueryLifecycle()
    release = Mock()
    with lifecycle.query():
        assert lifecycle.close_if_idle(release) is False
        assert not lifecycle.closing
        release.assert_not_called()
        with lifecycle.query():
            assert lifecycle.active == 2
    assert lifecycle.close_if_idle(release) is True
    assert lifecycle.close_if_idle(release) is True
    release.assert_called_once_with()
    with pytest.raises(RuntimeError):
        with lifecycle.query():
            pytest.fail('Closed pipeline accepted a query')


def test_idle_close_failure_can_be_retried():
    lifecycle = QueryLifecycle()
    release = Mock(side_effect=[OSError('close failed'), None])
    with pytest.raises(OSError):
        lifecycle.close_if_idle(release)
    assert lifecycle.closing and not lifecycle.closed
    assert lifecycle.close_if_idle(release) is True
    assert release.call_count == 2


def test_idle_close_skips_contended_lifecycle_lock():
    lifecycle = QueryLifecycle()
    release = Mock()
    with ThreadPoolExecutor(max_workers=1) as workers:
        with lifecycle.condition:
            result = workers.submit(lifecycle.close_if_idle, release)
            assert result.result(timeout=1) is False
    release.assert_not_called()
    assert not lifecycle.closing


def test_close_waits_for_parallel_queries_and_releases_only_once():
    lifecycle = QueryLifecycle()
    entered, finish = Barrier(3), Event()
    release = Mock()
    def query():
        with lifecycle.query():
            entered.wait(timeout=5)
            assert finish.wait(5)
    with ThreadPoolExecutor(max_workers=3) as workers:
        queries = [workers.submit(query) for _ in range(2)]
        try:
            entered.wait(timeout=5)  # Both searches must run concurrently.
            closing = workers.submit(lifecycle.close, release)
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.05)
            release.assert_not_called()
        finally:
            finish.set()
        for result in queries:
            result.result(timeout=5)
        closing.result(timeout=5)
    lifecycle.close(release)
    release.assert_called_once_with()
    with pytest.raises(RuntimeError, match='closing or closed'):
        with lifecycle.query():
            pytest.fail('Closed pipeline accepted a search')


def test_failed_query_does_not_block_close():
    lifecycle = QueryLifecycle()
    with pytest.raises(ValueError):
        with lifecycle.query():
            raise ValueError('search failed')
    release = Mock()
    lifecycle.close(release)
    release.assert_called_once_with()


def test_failed_release_can_be_retried_without_accepting_new_queries():
    lifecycle = QueryLifecycle()
    release = Mock(side_effect=[OSError('close failed'), None])
    with pytest.raises(OSError):
        lifecycle.close(release)
    with pytest.raises(RuntimeError):
        with lifecycle.query():
            pytest.fail('Closing pipeline accepted a search')
    lifecycle.close(release)
    assert release.call_count == 2
