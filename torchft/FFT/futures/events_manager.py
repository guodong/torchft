import asyncio
import queue
import sys
import threading
from typing import Optional, TypeVar
from abc import ABC, abstractmethod
from torch.futures import Future

T = TypeVar("T")

class _EventsManager(ABC):
    """
    Abstract base class for registering events on Pytorch Futures
    and CUDA events, involving a background event loop thread.

    Provides the core mechanism for starting, managing, and shutting down
    an asyncio event loop in a separate thread. Subclasses should implement
    specific logic for interacting with futures and the event loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_loop_thread: Optional[threading.Thread] = None
        self._del_queue: queue.SimpleQueue[object] = queue.SimpleQueue()

    def _maybe_start_event_loop(self) -> asyncio.AbstractEventLoop:
        """
        Start the event loop if it has not already been started.
        """
        with self._lock:
            if self._event_loop is None:
                self._event_loop = asyncio.new_event_loop()
                self._event_loop_thread = threading.Thread(
                    target=self._event_loop.run_forever,
                    daemon=True,
                    name=f"{self.__class__.__name__}EventLoop",
                )
                self._event_loop_thread.start()
            return self._event_loop


    @abstractmethod
    def register(self, fut: Future[T], event_type: Optional[str]) -> Future[T]:
        """
        Registers an event on the event loop with a callback that augments
        the future's completion.

        This allows for communication of background threads with the main thread
        when the main thread calls fut.wait() on the returned future.
        """
        pass

    def _clear_del_queue(self) -> int:
        """
        Clear the queue of futures to be deleted.

        Returns the number of items deleted.
        """
        count = 0
        while True:
            try:
                # get and immediately discard item
                item = self._del_queue.get_nowait()
                refcount = sys.getrefcount(item)
                assert (
                    # 1 from item, 1 from getrefcount
                    refcount
                    == 2
                ), f"items in del_queue reference should not have other references, found {refcount=}"
                del item

                count += 1
            except queue.Empty:
                break

        return count
    
    def shutdown(self) -> None:
        """
        Shutdown the event loop and cancel all pending timeouts.
        """
        with self._lock:
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)
                self._event_loop_thread.join()
                self._event_loop = None
                self._event_loop_thread = None