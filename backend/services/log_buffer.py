import logging
from collections import deque
from threading import Lock
from typing import Any


class RingLogHandler(logging.Handler):
    """Small in-memory log ring for the web GUI runtime log viewer."""

    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self.capacity = capacity
        self._records: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            item = {
                "created": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }
            with self._lock:
                self._records.append(item)
        except Exception:
            self.handleError(record)

    def tail(self, limit: int = 300) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._records)
        return records[-max(1, min(limit, self.capacity)) :]


log_buffer = RingLogHandler()
log_buffer.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
