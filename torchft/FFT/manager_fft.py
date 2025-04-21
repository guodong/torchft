import os
import signal
import threading
from torchft.FFT.messaging.error_bus import ErrorBus
from torchft.FFT.messaging.message import ErrorMessage, GPUErrorMessage, Message, NodeErrorMessage
from torchft.manager import Manager

class ManagerFFT(Manager):
    """
    ManagerFFT is a subclass of torchft.manager that adds a method to report errors.
    ManagerFFT = Manager + ErrorBus
    """
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.fft = FFT(self)

    def report_error(self, e: Exception, broadcast=False) -> None:
        """
        Report an error to the manager.

        At the same time, format the error into a FFT.GPUErrorMessage format and broadcast it.

        This will cause the manager to skip the current step and will be
        reconfigured on the next step.

        This should be called when an error occurs that leads to a corrupted
        gradient that needs to be discarded.
        """
        super().report_error(e)
        if broadcast:
            message = ErrorMessage(reason=str(e))
            self.fft.eb.broadcast(message)
        
    def reconfigure(self):
        """
        Reconfigure the process when participant changes.
        This is called when an error occurs that leads to a corrupted replica groups
        that needs to be discarded.
        """
        raise NotImplementedError("TODO: implemente")
class FFT:
    def __init__(self, manager: ManagerFFT):
        """
        Performs error handling and broadcasting.
        It uses the ErrorBus to send error messages to all nodes in the cluster.
        """
        self.manager = manager
        self.error_msg = None
        self._interrupt_by_signal = threading.Event()

        signal.signal(signal.SIGUSR1, self._signal_handler)

        self.eb = ErrorBus(
            host_name=self.manager._store.host(),
            port=self.manager._store.port(),
        )

        self.eb.subscribe(self.on_message)

    def _signal_handler(self, signum, frame):
        if self._interrupt_by_signal.is_set():
            return
        
        self.abort()
        
    def on_message(self, message: Message):
        """
        This callback runs in a thread, here we notify the main thread.
        """
        self.error_msg = message
        os.kill(os.getpid(), signal.SIGUSR1)
        if not self._interrupt_by_signal.is_set():
            self.abort()

        self._interrupt_by_signal.clear()
        
    
    def abort(self):
        # Short circuit subsquent procs in current step
        self.manager.report_error(Exception(self.error_msg))
        self.manager._quorum_future.set_exception(InterruptedError(self.error_msg))
        for work in self.manager._pending_work:
            work.set_exception(InterruptedError(self.error_msg))

        self.manager._pending_work.clear()

        if self.should_reconfigure():
            self.manager.reconfigure()

    def should_reconfigure(self):
        """
        Check if the error message causes reconfigure of the running settings.
        """
        return isinstance(self.error_msg, GPUErrorMessage) or isinstance(self.error_msg, NodeErrorMessage)

        

    




        

        