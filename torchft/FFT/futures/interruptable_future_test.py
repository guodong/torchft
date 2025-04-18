import threading
from datetime import timedelta
from time import sleep
from unittest import TestCase, skipUnless

import torch
from torch.futures import Future

from torchft.futures import (
    context_timeout,
    future_timeout,
    future_wait,
    stream_timeout,
)
from torchft.futures.failure_manager import _FAILURE_MANAGER, ImmediateInterruptException
from torchft.futures.interruptable_future import InterruptableFuture

class InterruptableFutureTest(TestCase):
    def test_future_wait_timeout(self) -> None:
        """Tests original future_wait for reference."""
        fut = Future()
        with self.assertRaisesRegex(TimeoutError, "future did not complete within"):
            future_wait(fut, timeout=timedelta(seconds=0.01))

    def test_future_wait_success(self) -> None:
        """Tests original future_wait for reference."""
        fut = Future()
        fut.set_result(1)
        self.assertEqual(future_wait(fut, timeout=timedelta(seconds=1.0)), 1)

    def test_future_wait_exception(self) -> None:
        """Tests original future_wait for reference."""
        fut = Future()
        fut.set_exception(RuntimeError("test"))
        with self.assertRaisesRegex(RuntimeError, "test"):
            future_wait(fut, timeout=timedelta(seconds=1.0))

    def test_interruptable_future_basic_set_result_wait(self) -> None:
        """Tests basic set_result and wait functionality."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        self.assertFalse(int_fut.done())

        int_fut.set_result(42)
        self.assertTrue(int_fut.done())
        self.assertEqual(int_fut.wait(), 42)
        self.assertEqual(int_fut.value(), 42)

    def test_interruptable_future_basic_set_exception_wait(self) -> None:
        """Tests basic set_exception and wait functionality."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        self.assertFalse(int_fut.done())

        test_exception = ValueError("test error")
        int_fut.set_exception(test_exception)
        self.assertTrue(int_fut.done())
        with self.assertRaisesRegex(ValueError, "test error"):
            int_fut.wait()
        with self.assertRaisesRegex(ValueError, "test error"):
            int_fut.value()

    def test_interruptable_future_wait_with_timeout(self) -> None:
        """Tests wait with a timeout on an incomplete future."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        with self.assertRaisesRegex(TimeoutError, "future did not complete within"):
            int_fut.wait(timeout=timedelta(seconds=0.01))

    def test_interruptable_future_then_success(self) -> None:
        """Tests the then method for successful completion."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        callback_called = threading.Event()

        def callback(f: Future[int]) -> int:
            self.assertTrue(f.done())
            self.assertEqual(f.value(), 10)
            callback_called.set()
            return f.value() + 1

        then_fut = int_fut.then(callback)
        self.assertFalse(callback_called.is_set())
        self.assertFalse(then_fut.done())

        int_fut.set_result(10)

        self.assertTrue(callback_called.wait(timeout=1.0))
        self.assertTrue(int_fut.done())
        self.assertTrue(then_fut.done())
        self.assertEqual(then_fut.wait(), 11)

    def test_interruptable_future_then_exception(self) -> None:
        """Tests the then method when the original future has an exception."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        callback_called = threading.Event()
        then_callback_called = threading.Event()

        test_exception = RuntimeError("original error")

        def callback(f: Future) -> None:
            callback_called.set()
            # This callback runs *after* the original future (int_fut) is done.
            # Accessing the value/exception should propagate correctly.
            try:
                f.wait() # Call wait to trigger the exception
                # Should not reach here if original future had exception
                self.fail("Original future did not raise expected exception in callback")
            except RuntimeError as e:
                 # Check if it's the expected exception
                 self.assertEqual(str(e), "original error")
                 # Re-raise the caught exception to propagate it to then_fut
                 raise e
            except Exception as e:
                # Catch any other unexpected exception during wait
                self.fail(f"Unexpected exception in callback: {e}")

        # The `then` method should return a new future that eventually
        # completes with the result of the callback (or its exception).
        then_fut = int_fut.then(callback)

        def then_callback(f_then: Future) -> None:
             then_callback_called.set()
             self.assertTrue(f_then.done())
             # This callback runs on the future returned by `then` (then_fut).
             # It should also have the exception propagated from the first callback.
             with self.assertRaisesRegex(RuntimeError, "original error"):
                 f_then.wait() # Check the exception on the 'then_fut'

        then_fut.add_done_callback(then_callback)

        self.assertFalse(callback_called.is_set())
        self.assertFalse(then_callback_called.is_set())
        self.assertFalse(then_fut.done())

        int_fut.set_exception(test_exception)

        self.assertTrue(callback_called.wait(timeout=1.0))
        self.assertTrue(then_callback_called.wait(timeout=1.0))
        self.assertTrue(int_fut.done())
        self.assertTrue(then_fut.done())
        with self.assertRaisesRegex(RuntimeError, "original error"):
            then_fut.wait()

    def test_interruptable_future_add_done_callback_success(self) -> None:
        """Tests add_done_callback for successful completion."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        callback_called = threading.Event()
        callback_value = None

        def callback(f: Future[int]) -> None:
            nonlocal callback_value
            self.assertTrue(f.done())
            callback_value = f.value()
            callback_called.set()

        int_fut.add_done_callback(callback)
        self.assertFalse(callback_called.is_set())

        int_fut.set_result(20)

        self.assertTrue(callback_called.wait(timeout=1.0))
        self.assertEqual(callback_value, 20)
        self.assertTrue(int_fut.done())

    def test_interruptable_future_add_done_callback_exception(self) -> None:
        """Tests add_done_callback when the future has an exception."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        callback_called = threading.Event()
        callback_exception = None

        test_exception = TypeError("callback test error")

        def callback(f: Future) -> None:
            nonlocal callback_exception
            self.assertTrue(f.done())
            # callback_exception = f.exception()
            try:
                f.wait()
            except Exception as e:
                callback_exception = e
            callback_called.set()

        int_fut.add_done_callback(callback)
        self.assertFalse(callback_called.is_set())

        int_fut.set_exception(test_exception)

        self.assertTrue(callback_called.wait(timeout=1.0))
        self.assertIsInstance(callback_exception, TypeError)
        self.assertEqual(str(callback_exception), "callback test error")
        self.assertTrue(int_fut.done())

    def test_interruptable_future_add_done_callback_after_done(self) -> None:
        """Tests adding a callback after the future is already done."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        int_fut.set_result(30)

        callback_called = threading.Event()
        callback_value = None

        def callback(f: Future[int]) -> None:
            nonlocal callback_value
            callback_value = f.value()
            callback_called.set()

        # Callback should run immediately (inline)
        int_fut.add_done_callback(callback)

        # No wait needed, should be set already
        self.assertTrue(callback_called.is_set())
        self.assertEqual(callback_value, 30)

    def test_interruptable_future_thread_safety_basic(self) -> None:
        """Tests basic thread safety with concurrent wait and set_result."""
        num_threads = 10
        results = [None] * num_threads
        exceptions = [None] * num_threads
        threads = []
        barrier = threading.Barrier(num_threads + 1)

        fut = Future()
        int_fut = InterruptableFuture(fut)

        def worker(idx: int):
            try:
                barrier.wait()
                results[idx] = int_fut.wait() # type: ignore
            except Exception as e:
                exceptions[idx] = e

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        barrier.wait()
        sleep(0.1) # Give workers a chance to block on wait()
        int_fut.set_result(55)

        for t in threads:
            t.join()

        for i in range(num_threads):
            self.assertIsNone(exceptions[i], f"Thread {i} had exception: {exceptions[i]}")
            self.assertEqual(results[i], 55)

    def test_interruptable_future_thread_safety_callbacks(self) -> None:
        """Tests thread safety with concurrent add_done_callback/then and set_result."""
        num_threads = 10
        then_results = [None] * num_threads
        done_cb_events = [threading.Event() for _ in range(num_threads)]
        threads = []
        barrier = threading.Barrier(num_threads + 1)

        fut = Future()
        int_fut = InterruptableFuture(fut)

        def then_worker(idx: int):
            def callback(f: Future[int]) -> int:
                return f.value() * 2
            barrier.wait()
            then_fut = int_fut.then(callback)
            then_results[idx] = then_fut.wait() # type: ignore

        def done_cb_worker(idx: int):
            def callback(f: Future[int]) -> None:
                done_cb_events[idx].set()
            barrier.wait()
            int_fut.add_done_callback(callback)

        # Mix then and add_done_callback workers
        for i in range(num_threads):
            if i % 2 == 0:
                t = threading.Thread(target=then_worker, args=(i,))
            else:
                t = threading.Thread(target=done_cb_worker, args=(i,))
            threads.append(t)
            t.start()

        barrier.wait()
        sleep(0.1) # Give workers a chance to add callbacks
        int_fut.set_result(66)

        for t in threads:
            t.join()

        for i in range(num_threads):
            if i % 2 == 0:
                self.assertEqual(then_results[i], 66 * 2)
            else:
                self.assertTrue(done_cb_events[i].wait(timeout=0.1), f"Done callback {i} did not fire")

    def test_interruptable_future_add_timeout_triggers(self) -> None:
        """Tests that add_timeout causes wait() to timeout."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        timeout_duration = timedelta(seconds=0.1)
        wait_exception = None
        wait_event = threading.Event()

        def wait_thread():
            nonlocal wait_exception
            try:
                # This wait should timeout because of add_timeout
                int_fut.wait()
            except Exception as e:
                wait_exception = e
            finally:
                wait_event.set()

        # Start thread that will wait
        t = threading.Thread(target=wait_thread)
        t.start()

        sleep(0.01) # Ensure wait_thread gets to wait()
        # Add timeout from another thread
        int_fut.add_timeout(timeout_duration)

        # Wait for the wait_thread to finish or timeout itself
        self.assertTrue(wait_event.wait(timeout=timeout_duration.total_seconds() + 0.5))
        t.join()

        # Check that the waiting thread received a TimeoutError
        self.assertIsInstance(wait_exception, TimeoutError)
        self.assertTrue(int_fut.done()) # The timeout future should be done
        with self.assertRaises(TimeoutError):
            int_fut.value() # Accessing value should also raise

    def test_interruptable_future_add_timeout_completes_first(self) -> None:
        """Tests that if future completes before timeout, the correct value is returned."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        timeout_duration = timedelta(seconds=0.5)
        expected_result = 77
        wait_result = None
        wait_exception = None
        wait_event = threading.Event()

        # Add timeout first
        int_fut.add_timeout(timeout_duration)

        def wait_thread():
            nonlocal wait_result, wait_exception
            try:
                # This wait should succeed as set_result happens before timeout
                wait_result = int_fut.wait() # type: ignore
            except Exception as e:
                wait_exception = e
            finally:
                wait_event.set()

        # Start thread that will wait
        t = threading.Thread(target=wait_thread)
        t.start()

        sleep(0.1) # Ensure wait_thread gets to wait()
        # Set the result before the timeout expires
        int_fut.set_result(expected_result)

        # Wait for the wait_thread to finish
        self.assertTrue(wait_event.wait(timeout=0.5))
        t.join()

        # Check that the waiting thread received the correct result
        self.assertIsNone(wait_exception, f"Wait thread raised: {wait_exception}")
        self.assertEqual(wait_result, expected_result)
        self.assertTrue(int_fut.done())
        self.assertEqual(int_fut.value(), expected_result)

    def test_interruptable_future_immediate_interrupt_during_wait(self) -> None:
        """Tests immediate_interrupt called while another thread is waiting."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        wait_exception = None
        wait_event = threading.Event()

        def wait_thread():
            nonlocal wait_exception
            try:
                # This wait should be interrupted
                int_fut.wait()
            except Exception as e:
                wait_exception = e
            finally:
                wait_event.set()

        # Start thread that will wait
        t = threading.Thread(target=wait_thread)
        t.start()

        sleep(0.01) # Ensure wait_thread gets to wait()
        # Interrupt from another thread
        int_fut.immediate_interrupt()

        # Wait for the wait_thread to finish (should be quick due to interrupt)
        self.assertTrue(wait_event.wait(timeout=0.5))
        t.join()

        # Check that the waiting thread received an ImmediateInterruptException
        self.assertIsInstance(wait_exception, ImmediateInterruptException)
        self.assertTrue(int_fut.done()) # Future is marked done by interrupt
        with self.assertRaises(ImmediateInterruptException):
            int_fut.value() # Accessing value should also raise

    def test_interruptable_future_immediate_interrupt_before_wait(self) -> None:
        """Tests immediate_interrupt called before any thread waits."""
        fut = Future()
        int_fut = InterruptableFuture(fut)

        # Interrupt the future immediately
        int_fut.immediate_interrupt()

        # Wait should immediately raise the exception
        with self.assertRaises(ImmediateInterruptException):
            int_fut.wait()

        # Check future state
        self.assertTrue(int_fut.done())
        with self.assertRaises(ImmediateInterruptException):
            int_fut.value()

    def test_interruptable_future_interrupt_then_timeout(self) -> None:
        """Tests calling immediate_interrupt followed by add_timeout."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        timeout_delay = timedelta(seconds=0.5)

        # Interrupt first
        int_fut.immediate_interrupt()

        # Then add a timeout (this should ideally be a no-op on the already interrupted future)
        # The current implementation might replace the future, let's test current behavior
        int_fut.add_timeout(timeout_delay)

        # Wait should still raise the ImmediateInterruptException because it happened first
        with self.assertRaises(ImmediateInterruptException):
             int_fut.wait()
        self.assertTrue(int_fut.done())
        with self.assertRaises(ImmediateInterruptException):
             int_fut.value()

    def test_interruptable_future_timeout_then_interrupt(self) -> None:
        """Tests calling add_timeout followed by immediate_interrupt."""
        fut = Future()
        int_fut = InterruptableFuture(fut)
        timeout_delay = timedelta(seconds=0.5)

        # Add timeout first
        int_fut.add_timeout(timeout_delay)

        # Then interrupt
        int_fut.immediate_interrupt()

        # Wait should raise the ImmediateInterruptException because it's more immediate
        with self.assertRaises(ImmediateInterruptException):
            int_fut.wait()
        self.assertTrue(int_fut.done())
        with self.assertRaises(ImmediateInterruptException):
            int_fut.value()

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



