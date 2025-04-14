from datetime import timedelta
import threading
from typing import Any, Optional, TypeVar, Callable, Dict
from torchft.futures.failure_manager import _FAILURE_MANAGER, TIMEOUT_EVENT, IMMEDIATE_INTERRUPT_EVENT, ImmediateInterruptException
from torchft.futures.timeout import future_timeout, future_wait
from torchft.process_group import ProcessGroupBaby
from torch.futures import Future
T = TypeVar("T")
S = TypeVar("S")

class InterruptableFuture:
    """
    A future that can be interrupted.
    This future cannot be cancelled since it wraps around a cuda event, and we cannot roll back a cuda event.

    # Example Usage

    # Manager:
    # def __init__(self):
    #     self._pending_work = []
    #     self._pg = pg

    # def allreduce(tensor):
    #     fut = self._pg.allreduce(tensor)
    #     interruptable_future = InterruptableFuture(fut, metadata={"allreduce": True, "timeout": self.timeout})
    #     self._pending_work.append(interruptable_future)
    #     return interruptable_future.future

    # Listening Thread:
    # on_message_received(message: Message) -> Future[Message]:
    #     for interruptable_future in self._pending_work:
    #         if interruptable_future.metadata["allreduce"] == True:
    #             interruptable_future.immediate_interrupt()
    #         elif interruptable_future.metadata["timeout"] is not None:
    #             interruptable_future.add_timeout(interruptable_future.metadata["timeout"])
    #     return interruptable_future.future
    """
    def __init__(self, future: Future[T], pg: ProcessGroupBaby):
        self.future = future
        self._pg = pg
        self._lock = threading.Lock()
        self._interrupted = False

    def immediate_interrupt(self):
        """
        Sets self.future to raise ImmediateInterruptException.
        """
        with self._lock:
            # Pass process group to ensure proper interruption
            self.future = _FAILURE_MANAGER.on_event(
                self.future, 
                IMMEDIATE_INTERRUPT_EVENT, 
                pg=self._pg, 
                timeout=None
            )
            self._interrupted = True

    def add_timeout(self, timeout: timedelta):
        """
        Replace the underlying future with one that raises TimeoutError after the specified duration.
        Note that this will not put the timeout on the original future, it will only raise an exception on the returned future.
        """
        if self._interrupted:
            pass
        else:
            self.future = _FAILURE_MANAGER.on_event(self.future, 
                                                    TIMEOUT_EVENT, 
                                                    pg=None,
                                                    timeout=timeout)

    def wait(self, timeout: Optional[timedelta] = None) -> T:
        """
        Wait for the future to complete, releasing the internal lock during the wait.
        """
        # Wait outside the lock
        if timeout is not None:
            # Use the specific future_wait utility for timeout handling
            return future_wait(self.future, timeout)
        else:
            return self.future.wait()

    def done(self) -> bool:
        """Check if the future is done, releasing the internal lock during the check."""
        with self._lock:
            current_future = self.future
        # Check done status outside the lock
        return current_future.done()

    def value(self) -> T:
        """Get the future's value, releasing the internal lock."""
        with self._lock:
            current_future = self.future
        # Get value outside the lock
        return current_future.value()

    def then(self, callback: Callable[[Future[T]], S]) -> Future[S]:
        """Add a callback via then, releasing the internal lock. Returns a new torch.Future."""
        with self._lock:
            current_future = self.future
        # Call then outside the lock
        # Returns a new torch.Future, does not modify self.future
        return current_future.then(callback)

    def add_done_callback(self, callback: Callable[[Future[T]], None]) -> None:
        """Add a done callback, releasing the internal lock."""
        with self._lock:
            current_future = self.future
        # Add callback outside the lock
        current_future.add_done_callback(callback)

    def set_result(self, result: T) -> None:
        """Set the future's result, releasing the internal lock."""
        with self._lock:
            current_future = self.future
        # Set result outside the lock
        current_future.set_result(result)

    def set_exception(self, exception: Exception) -> None:
        """Set the future's exception, releasing the internal lock."""
        with self._lock:
            current_future = self.future
        # Set exception outside the lock
        current_future.set_exception(exception)
            
