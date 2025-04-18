import asyncio
import threading
from typing import Optional

class _Handle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handle: Optional[asyncio.Handle] = None
        self._cancelled = False

    def set_handle(self, handle: asyncio.Handle) -> None:
        with self._lock:
            if self._cancelled:
                handle.cancel()
                self._handle = None
            else:
                self._handle = handle

    def cancel(self) -> None:
        with self._lock:
            assert not self._cancelled, "timer can only be cancelled once"
            self._cancelled = True
            if self._handle is not None:
                self._handle.cancel()
                self._handle = None

class _TimerHandle(_Handle):
    def __init__(self) -> None:
        super().__init__()

    def set_timer_handle(self, handle: asyncio.TimerHandle) -> None:
        super().set_handle(handle)

    def cancel(self) -> None:
        super().cancel()
