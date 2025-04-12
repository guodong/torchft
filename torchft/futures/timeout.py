import threading
from contextlib import contextmanager
from datetime import timedelta
from typing import Callable, Generator, TypeVar
from torch.futures import Future
from torchft.futures.failure_manager import _FAILURE_MANAGER, TIMEOUT_EVENT

T = TypeVar("T")

def future_timeout(fut: Future[T], timeout: timedelta) -> Future[T]:
    """
    Return a Future that completes with the result of the given Future within
    the given timeout or with a TimeoutError.

    Args:
        fut: The Future to wait for
        timeout: The timeout to wait for the Future to complete

    Returns:
        The future with a timeout
    """
    return _FAILURE_MANAGER.register(fut, TIMEOUT_EVENT, timeout)


def future_wait(fut: Future[T], timeout: timedelta) -> T:
    """
    Wait for a Future to complete up to a timeout.

    Args:
        fut: The Future to wait for
        timeout: The timeout to wait for the Future to complete

    Returns:
        The result of the Future if it completed within the timeout.

    Raises:
        TimeoutError if the Future did not complete within the timeout.
        Any other exception that occurred in the Future.
    """

    event: threading.Event = threading.Event()

    def callback(fut: Future[T]) -> T:
        event.set()
        return fut.wait()

    fut = fut.then(callback)

    if not event.wait(timeout=timeout.total_seconds()):
        raise TimeoutError(f"future did not complete within {timeout}")

    return fut.wait()


def stream_timeout(callback: Callable[[], None], timeout: timedelta) -> None:
    """
    Registers a callback that will be called after the specified timeout if
    the current stream doesn't complete in time.

    This uses a cuda Event to track the completion of the current stream. If
    the stream is not complete after the timeout, the callback is called.

    Args:
        callback: The callback to call if the stream doesn't complete in time.
        timeout: The timeout to wait for the stream to complete.
    """
    _FAILURE_MANAGER.stream_timeout(callback, timeout)


@contextmanager
def context_timeout(
    callback: Callable[[], None], timeout: timedelta
) -> Generator[None, None, None]:
    """
    Registers a callback that will be called after the specified timeout if
    the current contextmanager doesn't exit in time.

    Args:
        callback: The callback to call if we time out.
        timeout: How long to wait for the contextmanager to exit.
    """

    with _FAILURE_MANAGER.context_timeout(callback, timeout):
        yield
