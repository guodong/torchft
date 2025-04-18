from torchft import torchft

class ManagerFFT(torchft.manager):
    """
    ManagerFFT is a subclass of torchft.manager that adds a method to report errors.
    ManagerFFT = Manager + ErrorBus
    """
    def __init__(self, ...):
        super.__init__(...)
        self.init_signal_handler()
        self.

    def run_error_bus():
        self.eb.register_callback(self.FFT.on_message)

    def report_error(self, e: Exception) -> None:
        """
        Report an error to the manager.

        At the same time, format the error into a FFT.GPUErrorMessage format and broadcast it.

        This will cause the manager to skip the current step and will be
        reconfigured on the next step.

        This should be called when an error occurs that leads to a corrupted
        gradient that needs to be discarded.
        """
        super.report_error(e)
        message = self.FFT.format_GPU_message(e, self._rank) # Could be GPU index or something else
        self.eb.broadcast(message)
        

class FFT:
    """
    stateless class. Consists of pure functions. Defines a namespace of functions.
    """
    def format_message(e, rank):
        # TODO: Add proto
        # Put into Proto format, or something like that
        # Proto vs. Dataclass vs. Tuple?

    def on_message():
        
    def init_signal_handler():

    




        

        