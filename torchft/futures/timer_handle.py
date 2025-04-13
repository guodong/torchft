import asyncio
import threading
from typing import Optional

class _TimerHandle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timer_handle: Optional[asyncio.TimerHandle] = None
        self._cancelled = False

    def set_timer_handle(self, timer_handle: asyncio.TimerHandle) -> None:
        with self._lock:
            if self._cancelled:
                timer_handle.cancel()
                self._timer_handle = None
            else:
                self._timer_handle = timer_handle

    def cancel(self) -> None:
        with self._lock:
            assert not self._cancelled, "timer can only be cancelled once"
            self._cancelled = True
            if self._timer_handle is not None:
                self._timer_handle.cancel()
                self._timer_handle = None

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