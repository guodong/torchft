import asyncio
from math import e
import queue
from contextlib import contextmanager
from datetime import timedelta
from typing import Callable, Generator, TypeVar, Optional, Dict, Any, TYPE_CHECKING
from unittest.mock import Mock
from torchft.futures.events_manager import _EventsManager
import torch
from torch.futures import Future
from torchft.futures.timer_handle import _TimerHandle, _Handle

if TYPE_CHECKING:
    from torchft.process_group import ProcessGroupBaby

T = TypeVar("T")

TIMEOUT_EVENT = "timeout"
IMMEDIATE_INTERRUPT_EVENT = "immediate_interrupt"

class _FailureMetadata:
    def __init__(self, pg: "ProcessGroupBaby", id: int):
        self._pg = pg
        self._id = id

class ImmediateInterruptException(Exception):
    """Exception raised when a future is immediately interrupted."""
    pass

class _FailureManager(_EventsManager):
    """
    This class manages timeouts for code blocks, futures and CUDA events. It
    uses a background thread with an event loop to schedule the timeouts and
    call the callback function when the timeout is reached.

    Generally there is a single instance of this class that is used for all
    timeouts. The callbacks should not block otherwise other timeouts may not
    be processed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._next_timer_id = 0

        # This queue is used to delete events on the main thread as cudaEventDestroy
        # can block if the CUDA queue is full.
        self._del_queue: queue.SimpleQueue[object] = queue.SimpleQueue()

        self._timer_handle = None
        self._metadata_dict: Dict[int, _FailureMetadata] = {}

    def on_event(self, fut: Future[T], event_type: Optional[str], pg: Optional["ProcessGroupBaby"] = None, timeout: Optional[timedelta] = None) -> Future[T]:
        """
        Registers the event on the future and returns a new future
        that will complete when the original future completes or the event fires.

        If the future is already done, returns the future immediately.
        """
        # bypass timeout for mock futures
        if isinstance(fut, Mock):
            return fut

        if fut.done():
            return fut

        self._clear_del_queue()

        loop = self._maybe_start_event_loop()

        if event_type == TIMEOUT_EVENT:
            return self.register_timeout(loop, fut, timeout)
        if event_type == IMMEDIATE_INTERRUPT_EVENT:
            if pg is not None:
                return self.immediate_interrupt(fut, pg)
            else:
                raise ValueError("Process group is required for immediate interrupt")
        else:
            raise ValueError(f"Invalid event type: {event_type}")

    def immediate_interrupt(self, fut: Future[T], pg: Optional["ProcessGroupBaby"] = None) -> Future[T]:
        """
        Sets the future to raise ImmediateInterruptException.
        """
        # pyre-fixme[29]: Future is not a function
        _timer_handle = self._timer_handle
        if _timer_handle is not None:
            _timer_handle.cancel()
        fut.set_exception(ImmediateInterruptException("immediate interrupt"))
        if pg is not None:
            self.rollback(event_type=IMMEDIATE_INTERRUPT_EVENT, pg=pg)
        else:
            raise ValueError("Process group is required for immediate interrupt")
        return fut

    def rollback(self, event_type: Optional[str], pg: Optional["ProcessGroupBaby"]):
        if event_type == IMMEDIATE_INTERRUPT_EVENT:
            pg.abort()
        else:
            raise ValueError(f"Rollback logic not implemented for event type: {event_type}")

    def register_timeout(self, loop: asyncio.AbstractEventLoop, fut: Future[T], timeout: timedelta) -> None:
        """
        Register a timeout for a future.
        Overwrites any existing timeout.
        
        If the future is already done, the timed_fut will be done immediately.
        """
        # pyre-fixme[29]: Future is not a function
        timed_fut: Future[T] = Future()
        handle: _TimerHandle = _TimerHandle()

        loop.call_soon_threadsafe(
            self._register_callback,
            loop,
            TIMEOUT_EVENT,
            lambda: timed_fut.set_exception(
                # pyre-fixme[6]: e is not T
                TimeoutError(f"future did not complete within {timeout}")
            ),
            handle,
            timeout,
        )

        def callback(fut: Future[T]) -> None:
            handle.cancel()
            try:
                timed_fut.set_result(fut.wait())
            except Exception as e:
                try:
                    # this can throw if the future is already done
                    # pyre-fixme[6]: e is not T
                    timed_fut.set_exception(e)
                except Exception:
                    pass

        fut.add_done_callback(callback)
        return timed_fut

    @classmethod
    def _register_callback(
        cls,
        loop: asyncio.AbstractEventLoop,
        event_type: Optional[str],
        callback: Callable[[], None],
        handle: _Handle,
        timeout: Optional[timedelta] = None,
    ) -> None:
        if event_type == TIMEOUT_EVENT:
            if timeout is None:
                raise ValueError("Timeout is required for timeout event")
            cls._register_callback_timeout(loop, callback, timeout, handle)
        elif event_type == IMMEDIATE_INTERRUPT_EVENT:
            cls._register_callback_immediate_interrupt(loop, callback, handle)
        else:
            raise ValueError(f"Invalid event type: {event_type}")

    @classmethod
    def _register_callback_immediate_interrupt(
            cls,
            loop: asyncio.AbstractEventLoop,
            callback: Callable[[], None],
            handle: _Handle
    ) -> None:
        
        interrupt_handle = loop.call_soon_threadsafe(
            callback,
        )
        handle.set_handle(interrupt_handle)
        

    @classmethod
    def _register_callback_timeout(
            cls,
            loop: asyncio.AbstractEventLoop,
            callback: Callable[[], None],
            timeout: timedelta,
            handle: _Handle
    ) -> None:

        timer_handle = loop.call_later(
            timeout.total_seconds(),
            callback,
        )
        if not isinstance(timer_handle, asyncio.TimerHandle):
            raise TypeError("timer_handle must be an instance of asyncio.TimerHandle")
        handle.set_timer_handle(timer_handle)

    def stream_timeout(self, callback: Callable[[], None], timeout: timedelta) -> None:
        self._clear_del_queue()

        loop = self._maybe_start_event_loop()

        event: torch.cuda.Event = torch.cuda.Event()
        event.record()

        def handler() -> None:
            if not event.query():
                callback()

            # cudaEventDestroy can block so we never want to delete in the event
            # loop. Put it on the del queue so we can delete it in the main
            # thread.
            self._del_queue.put(event)

        loop.call_soon_threadsafe(
            self._register_callback, loop, TIMEOUT_EVENT, handler, _TimerHandle(), timeout
        )

    @contextmanager
    def context_timeout(
        self, callback: Callable[[], None], timeout: timedelta
    ) -> Generator[None, None, None]:
        self._clear_del_queue()

        loop = self._maybe_start_event_loop()
        handle = _TimerHandle()

        loop.call_soon_threadsafe(
            self._register_callback, loop, TIMEOUT_EVENT, callback, handle, timeout
        )

        yield

        handle.cancel()

_FAILURE_MANAGER = _FailureManager()



