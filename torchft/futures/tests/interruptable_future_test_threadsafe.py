import threading
from datetime import timedelta
from time import sleep
from unittest import TestCase
from torch.futures import Future
from torchft.futures.failure_manager import ImmediateInterruptException
from torchft.futures.interruptable_future import InterruptableFuture
from torchft.process_group import ProcessGroupDummy

class InterruptableFutureTest(TestCase):
    def setUp(self) -> None:
        # Create a dummy process group for testing
        self.dummy_pg = ProcessGroupDummy(rank=0, world=1)
        
    def test_interruptable_future_thread_safety_basic(self) -> None:
        """Tests basic thread safety with concurrent wait and set_result."""
        num_threads = 10
        results = [None] * num_threads
        exceptions = [None] * num_threads
        threads = []
        barrier = threading.Barrier(num_threads + 1)

        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)

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
        int_fut = InterruptableFuture(fut, self.dummy_pg)

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

    def test_interruptable_future_add_timeout_completes_first(self) -> None:
        """Tests that if future completes before timeout, the correct value is returned."""
        fut = Future()
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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
        int_fut = InterruptableFuture(fut, self.dummy_pg)

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
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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
        int_fut = InterruptableFuture(fut, self.dummy_pg)
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