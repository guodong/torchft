import asyncio
import queue
from contextlib import contextmanager
from datetime import timedelta
from typing import Callable, Generator, TypeVar, Optional
from unittest.mock import Mock

from attr import Out
from torchft.futures.events_manager import _EventsManager
import torch
from torch.futures import Future
from torchft.futures.timer_handle import _TimerHandle, _Handle

T = TypeVar("T")

TIMEOUT_EVENT = "timeout"
IMMEDIATE_INTERRUPT_EVENT = "immediate_interrupt"

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

    def register(self, fut: Future[T], event_type: Optional[str], timeout: timedelta) -> Future[T]:
        """
        Registers a future that will be cancelled after the specified timeout.
        """
        # bypass timeout for mock futures
        if isinstance(fut, Mock):
            return fut

        self._clear_del_queue()

        loop = self._maybe_start_event_loop()

        if event_type == TIMEOUT_EVENT:
            self.register_timeout(loop, fut, timeout)
        if event_type == "immediate_interrupt":
            self.register_immediate_interrupt(loop, fut)
        else:
            raise ValueError(f"Invalid event type: {event_type}")

    def register_immediate_interrupt(self, loop: asyncio.AbstractEventLoop, fut: Future[T]) -> None:
        # pyre-fixme[29]: Future is not a function
        interrupted_fut: Future[T] = Future()
        handle: _Handle = _Handle()
        loop.call_soon_threadsafe(
            self._register_callback,
            IMMEDIATE_INTERRUPT_EVENT,
            loop,
            lambda: interrupted_fut.set_exception(
                TimeoutError("immediate interrupt")
            ),
            None,
            handle,
        )

        def callback(fut: Future[T]) -> None:
            handle.cancel()
            try:
                interrupted_fut.set_result(fut.wait())
            except Exception as e:
                try:
                    # this can throw if the future is already done
                    # pyre-fixme[6]: e is not T
                    interrupted_fut.set_exception(e)
                except Exception:
                    pass

        fut.add_done_callback(callback)
        return interrupted_fut


    def register_timeout(self, loop: asyncio.AbstractEventLoop, fut: Future[T], timeout: timedelta) -> None:
        # pyre-fixme[29]: Future is not a function
        timed_fut: Future[T] = Future()
        handle: _TimerHandle = _TimerHandle()
        loop.call_soon_threadsafe(
            self._register_callback,
            TIMEOUT_EVENT,
            loop,
            lambda: timed_fut.set_exception(
                # pyre-fixme[6]: e is not T
                TimeoutError(f"future did not complete within {timeout}")
            ),
            timeout,
            handle,
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
        timeout: timedelta,
        handle: _TimerHandle,
    ) -> None:
        if event_type == TIMEOUT_EVENT:
            cls._register_callback_timeout(loop, callback, timeout, handle)
        elif event_type == "immediate_interrupt":
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
            handle: _TimerHandle
    ) -> None:

        timer_handle = loop.call_later(
            timeout.total_seconds(),
            callback,
        )
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
            self._register_callback, loop, handler, timeout, _TimerHandle()
        )

    def context_timeout(
        self, callback: Callable[[], None], timeout: timedelta
    ) -> Generator[None, None, None]:
        self._clear_del_queue()

        loop = self._maybe_start_event_loop()
        handle = _TimerHandle()

        loop.call_soon_threadsafe(
            self._register_callback, loop, callback, timeout, handle
        )

        yield

        handle.cancel()

_FAILURE_MANAGER = _FailureManager()



