from datetime import timedelta
from unittest import TestCase

from torch.futures import Future
from torchft.futures.failure_manager import ImmediateInterruptException
from torchft.futures.interruptable_future import InterruptableFuture
from torchft.process_group import ProcessGroupDummy

class InterruptableFutureEventsTest(TestCase):
    def setUp(self) -> None:
        # Create a dummy process group for testing
        self.dummy_pg = ProcessGroupDummy(rank=0, world=1)
        
    def test_immediate_interrupt_basic(self) -> None:
        """Tests basic immediate_interrupt functionality."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        self.assertFalse(int_fut.done())
        
        int_fut.immediate_interrupt()
        
        self.assertTrue(int_fut.done())
        with self.assertRaises(ImmediateInterruptException):
            int_fut.wait()
            
    def test_immediate_interrupt_after_completion(self) -> None:
        """Tests immediate_interrupt on an already completed future."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        int_fut.set_result(42)
        
        # Future is already done with a result
        self.assertTrue(int_fut.done())
        self.assertEqual(int_fut.value(), 42)
        
        # Interrupting should not change the result
        int_fut.immediate_interrupt()
        self.assertTrue(int_fut.done())
        self.assertEqual(int_fut.value(), 42)
        
    def test_immediate_interrupt_calls_pg_abort(self) -> None:
        """Tests that process group's abort is called during immediate_interrupt."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        
        # Track if abort was called
        abort_called = False
        original_abort = self.dummy_pg.abort
        
        def mock_abort():
            nonlocal abort_called
            abort_called = True
            original_abort()
            
        self.dummy_pg.abort = mock_abort
        
        int_fut.immediate_interrupt()
        
        # Verify abort was called
        self.assertTrue(abort_called)
        
        # Restore original method
        self.dummy_pg.abort = original_abort
        
    def test_add_timeout_raises_after_duration(self) -> None:
        """Tests that add_timeout raises TimeoutError after the specified duration."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        
        # Add a short timeout
        timeout_duration = timedelta(seconds=0.01)
        int_fut.add_timeout(timeout_duration)
        
        # Wait for timeout to occur
        with self.assertRaises(TimeoutError):
            int_fut.wait()
            
    def test_add_timeout_completes_before_duration(self) -> None:
        """Tests that a future completes normally if it returns before timeout."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        
        # Add a longer timeout
        timeout_duration = timedelta(seconds=0.1)
        int_fut.add_timeout(timeout_duration)
        
        # Complete the future before timeout
        expected_value = 100
        int_fut.set_result(expected_value)
        
        # Should return the expected value, not timeout
        self.assertEqual(int_fut.wait(), expected_value)
        
    def test_add_timeout_on_completed_future(self) -> None:
        """Tests adding timeout to an already completed future."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        expected_value = 200
        int_fut.set_result(expected_value)
        
        # Add timeout after completion
        int_fut.add_timeout(timedelta(seconds=0.01))
        
        # Should still return the original value
        self.assertEqual(int_fut.value(), expected_value)
        
    def test_immediate_interrupt_then_add_timeout(self) -> None:
        """Tests calling immediate_interrupt followed by add_timeout."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        
        # Interrupt first
        int_fut.immediate_interrupt()
        
        # Then add timeout
        int_fut.add_timeout(timedelta(seconds=0.01))
        
        # Should still have the interrupt exception
        with self.assertRaises(ImmediateInterruptException):
            int_fut.wait()
            
    def test_add_timeout_then_immediate_interrupt(self) -> None:
        """Tests calling add_timeout followed by immediate_interrupt."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        
        # Add timeout first
        int_fut.add_timeout(timedelta(seconds=0.1))
        
        # Then interrupt
        int_fut.immediate_interrupt()
        
        # Should have the interrupt exception overriding the timeout
        with self.assertRaises(ImmediateInterruptException):
            int_fut.wait()
            
    def test_add_timeout_multiple_times(self) -> None:
        """Tests calling add_timeout multiple times (should replace the timeout)."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
        
        # Add a very short timeout that would trigger immediately
        int_fut.add_timeout(timedelta(seconds=0.01))
        
        # Then replace with a longer timeout
        int_fut.add_timeout(timedelta(seconds=0.1))
        
        # Complete the future before the second timeout
        expected_value = 300
        int_fut.set_result(expected_value)
        
        # Should complete normally with the value
        self.assertEqual(int_fut.wait(), expected_value) 