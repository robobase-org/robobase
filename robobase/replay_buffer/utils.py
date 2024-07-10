"""Collection of replay buffer utils."""
from multiprocessing import Condition, Lock


class ReadWriteLock:
    """Allows many simultaneous read locks, but only one write lock."""

    def __init__(self):
        self._read_ready = Condition(Lock())
        self._readers = 0

    def acquire_read(self):
        """Acquire a read lock. Blocks only if a thread has acquired the write lock."""
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """Release a read lock."""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """Acquire a write lock. Blocks until there are no acquired read/write locks."""
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """Release a write lock."""
        self._read_ready.release()
