"""Drain active searches before closing their shared storage connection."""

from contextlib import contextmanager
from threading import Condition


class QueryLifecycle:
    def __init__(self):
        self.condition = Condition()
        self.active = 0
        self.closing = False
        self.closed = False

    @contextmanager
    def query(self):
        with self.condition:
            if self.closing:
                raise RuntimeError('Pipeline is closing or closed; retry with a new pipeline.')
            self.active += 1
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def close(self, release):
        with self.condition:
            self.closing = True
            self.condition.wait_for(lambda: self.active == 0)
            if not self.closed:
                release()
                self.closed = True
