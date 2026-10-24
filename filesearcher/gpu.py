"""Jedna karta, dwóch użytkowników: zapytania mają pierwszeństwo przed fazą 2."""
import threading
from contextlib import contextmanager


class GpuArbiter:
    def __init__(self):
        self._cond = threading.Condition()
        self._busy = False
        self._waiting_queries = 0

    def _release(self):
        with self._cond:
            self._busy = False
            self._cond.notify_all()

    @contextmanager
    def priority(self):
        with self._cond:
            self._waiting_queries += 1
            while self._busy:
                self._cond.wait()
            self._waiting_queries -= 1
            self._busy = True
        try:
            yield
        finally:
            self._release()

    @contextmanager
    def background(self):
        with self._cond:
            while self._busy or self._waiting_queries:
                self._cond.wait()
            self._busy = True
        try:
            yield
        finally:
            self._release()
