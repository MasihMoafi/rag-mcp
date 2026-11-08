from urllib.error import HTTPError

import pytest
from evals.benchmarks.datasets import with_retries


def http_error(code):
    return HTTPError("https://example.test", code, "error", {}, None)


def flaky(errors):
    remaining = list(errors)

    def action():
        if remaining:
            raise remaining.pop(0)
        return "done"

    return action


def test_timeouts_and_server_errors_are_retried_with_growing_waits():
    waits = []
    action = flaky([TimeoutError("read timed out"), http_error(503)])
    assert with_retries(action, wait=20, sleep=waits.append) == "done"
    assert waits == [20, 40]


def test_a_client_error_fails_at_once():
    waits = []
    with pytest.raises(HTTPError):
        with_retries(flaky([http_error(404)]), sleep=waits.append)
    assert waits == []


def test_it_gives_up_after_the_last_attempt():
    waits = []
    with pytest.raises(TimeoutError):
        with_retries(flaky([TimeoutError("x")] * 3), attempts=3, sleep=waits.append)
    assert len(waits) == 2
