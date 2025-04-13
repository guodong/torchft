from datetime import timedelta
import threading
from typing import Any, Optional, TypeVar, Callable, Dict
from torchft.futures.failure_manager import _FAILURE_MANAGER, TIMEOUT_EVENT, IMMEDIATE_INTERRUPT_EVENT, ImmediateInterruptException
from torchft.futures.timeout import future_timeout, future_wait
from torch.futures import Future

T = TypeVar("T")
S = TypeVar("S")

class InterruptableFuture:
    def __init__(self, future: Future[T], metadata: Optional[Dict[str, Any]] = None):
        self.future = future
        self.metadata = metadata
        self._lock = threading.Lock()

    def immediate_interrupt(self):
        """
        Replace the underlying future with one that immediately raises ImmediateInterruptException.
        """
        with self._lock:
            current_future = self.future
        # Register outside the lock
        new_wrapped_future = _FAILURE_MANAGER.register(current_future, IMMEDIATE_INTERRUPT_EVENT)
        with self._lock:
            self.future = new_wrapped_future

    def add_timeout(self, timeout: timedelta):
        """
        Replace the underlying future with one that raises TimeoutError after the specified duration.
        """
        with self._lock:
            current_future = self.future
        # Register outside the lock
        new_wrapped_future = _FAILURE_MANAGER.register(current_future, TIMEOUT_EVENT, timeout)
        with self._lock:
            self.future = new_wrapped_future

    def wait(self, timeout: Optional[timedelta] = None) -> T:
        """
        Wait for the future to complete, releasing the internal lock during the wait.
        """
        with self._lock:
            current_future = self.future
        # Wait outside the lock
        if timeout is not None:
            # Use the specific future_wait utility for timeout handling
            return future_wait(current_future, timeout)
        else:
            return current_future.wait()

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
            
