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
from torchft.process_group import ProcessGroupDummy

class InterruptableFutureTest(TestCase):
    def setUp(self) -> None:
        # Create a dummy process group for testing
        self.dummy_pg = ProcessGroupDummy(rank=0, world=1)
        
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
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        self.assertFalse(int_fut.done())

        int_fut.set_result(42)
        self.assertTrue(int_fut.done())
        self.assertEqual(int_fut.wait(), 42)
        self.assertEqual(int_fut.value(), 42)

    def test_interruptable_future_basic_set_exception_wait(self) -> None:
        """Tests basic set_exception and wait functionality."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        with self.assertRaisesRegex(TimeoutError, "future did not complete within"):
            int_fut.wait(timeout=timedelta(seconds=0.01))

    def test_interruptable_future_then_success(self) -> None:
        """Tests the then method for successful completion."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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
        
    def test_interruptable_future_immediate_interrupt(self) -> None:
        """Tests the immediate_interrupt functionality."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        self.assertFalse(int_fut.done())
        
        # Spy on the abort method to verify it was called
        original_abort = self.dummy_pg.abort
        abort_called = False
        
        def mock_abort():
            nonlocal abort_called
            abort_called = True
            original_abort()
            
        self.dummy_pg.abort = mock_abort
        
        # Trigger immediate interrupt
        int_fut.immediate_interrupt()
        
        # Future should be done with an ImmediateInterruptException
        self.assertTrue(int_fut.done())
        with self.assertRaises(ImmediateInterruptException):
            int_fut.wait()
            
        # Verify process group abort was called
        self.assertTrue(abort_called)
        
        # Restore original abort method
        self.dummy_pg.abort = original_abort