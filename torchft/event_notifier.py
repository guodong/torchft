import queue
import threading
import time

class EventNotifier:
    def __init__(self, name="EventNotifier", debug=False, daemon=True, queue_size=0):
        # Condition variable for thread synchronization
        self.condition = threading.Condition()
        # Callback function to handle messages
        self.callback = None
        # Thread object for running the loop
        self.thread = None
        # Flag to indicate if the thread is running
        self.running = False
        # Queue to store messages
        self.message_queue = queue.Queue(queue_size)
        # Name of the thread
        self.name = name
        # Debug flag to print additional information
        self.debug = debug
        # Daemon flag for the thread
        self.daemon = daemon
        # Flag to request the thread to stop
        self._stop_requested = False
        # Flag to indicate if the stop is graceful
        self.graceful = True

    def _run_loop(self):
        """
        The main loop that processes messages from the queue.
        This method runs in a separate thread and waits for messages to be added
        to the queue. When a message is received, it calls the registered callback
        function with the message as an argument.
        """
        while True:
            with self.condition:
                self.condition.wait_for(lambda: not self.message_queue.empty() or self._stop_requested)
                should_stop = self._stop_requested
                graceful = self.graceful
                if should_stop and not graceful:
                    break
            
            # Execute the callback outside the lock
            while not self.message_queue.empty():
                msg = self.message_queue.get()
                if self.callback:
                    try:
                        self.callback(msg)
                    except Exception as e:
                        print(f"[{self.name}] Callback error: {e}")
                self.message_queue.task_done()

            if should_stop:
                with self.condition:
                    self.condition.notify_all()
                    break

    def run(self):
        """
        Start the event loop thread.
        """
        if self.running:
            return self
        
        with self.condition:
            self.running = True
            self.thread = threading.Thread(
                target=self._run_loop,
                name=self.name,
                daemon=self.daemon
            )
            self.thread.start()
        return self

    def stop(self, timeout=5, graceful=True):
        """Safely stop the thread
        Args:
            timeout: Maximum waiting time (seconds) - only used when graceful=True
            graceful: False=immediate stop, True=wait for messages to process
        Returns:
            bool: Whether the thread was successfully stopped
        """
        self.graceful = graceful
        with self.condition:
            self._stop_requested = True
            self.running = False
            self.condition.notify_all()  # wake up event loop
        
        if graceful:
            # Graceful stop: wait for all messages to be processed until timeout
            start_time = time.time()
            with self.condition:
                if not self.condition.wait_for(
                    lambda: self.message_queue.empty(),
                    timeout=timeout
                ):
                    print(f"[{self.name}] Timeout with {self.message_queue.qsize()} pending messages")
            
            # Remain time to wait for the thread to finish
            remaining_time = max(0, timeout - (time.time() - start_time))
            self.thread.join(remaining_time)
        else:
            # Stop immediately
            self.thread.join(timeout=0.1)
        
        stopped = not self.thread.is_alive()
        if self.debug:
            mode = "Graceful" if graceful else "Forced"
            print(f"[{self.name}] {mode} stop {'succeeded' if stopped else 'timeout'}")
        return stopped

    def notify(self, msg):
        """Send a message and wake up event loop
        Args:
            msg: Message to be sent
        Returns:
            self: EventNotifier instance
        """
        if not self.running:
            raise RuntimeError("Thread is not running!")
        
        with self.condition:
            self.message_queue.put(msg)
            self.condition.notify_all()
                
        return self

    def register(self, callback):
        """Register a callback (ensure it is a callable object)"""
        if not callable(callback):
            raise TypeError("Callback must be callable!")
        
        with self.condition:
            self.callback = callback
            if self.debug:
                print(f"[{self.name}] Callback registered: {callback.__name__}")
        return self
    
    def clear(self):
        with self.condition:
            while not self.message_queue.empty():
                self.message_queue.get()
                self.message_queue.task_done()
            if self.debug:
                print(f"[{self.name}] Messages cleared.")
        return self
    
    @property
    def is_running(self):
        return self.running and self.thread is not None and self.thread.is_alive()
    
    @property
    def pending_count(self):
        """Return the number of unprocessed messages"""
        with self.condition:
            return self.message_queue.qsize()

# Example usage
if __name__ == '__main__':
    def message_handler(msg):
        time.sleep(2)
        print(f"[Callback] Received message: {msg}")

    manager = EventNotifier(name="EvtNotifier",debug=True)

    manager.register(message_handler).run()
    manager.notify("First message")
    manager.notify("Second message")
    print('exiting')
    manager.stop(timeout=10, graceful=False)
    