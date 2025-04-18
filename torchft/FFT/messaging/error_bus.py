

from datetime import timedelta
import time
from torchft.FFT.messaging.message_queue import MessageQueue
from torchft.FFT.messaging.message import Message


class ErrorBus(MessageQueue):
    """
    Error bus for distributed systems using PyTorch's TCPStore.
    This class provides a simple interface for broadcasting and receiving messages
    across multiple nodes.

    Note: c10d uses `activeOpLock_` to guarantee that only one op is active at a time. Thus the `get()` operation
    will block the compare_set operation, we use two TCPStore clients as a work around, 
    one for broadcasting and one for receiving.
    """

    # Prefix for TCPStore, used to avoid key collision
    PREFIX = '__error_bus__'

    # Must be empty string, otherwise the compare_set will not work due to key existance check
    NONCE = ''

    def __init__(
        self,
        host_name: str = '127.0.0.1',
        port: int = 0,
        is_master: bool = False,
        timeout: timedelta = timedelta(seconds=10),
        skip_outdated: bool = True,
    ) -> None:
        """
        Similar to TCPStore, but with a callback function for receiving messages.
        :param host_name: Host name of the master TCPStore
        :param port: Port number of the master TCPStore
        :param is_master: Whether this node is the master node
        :param timeout: Timeout for TCPStore operations
        :param callback: Callback function for receiving messages
        :param skip_outdated: Whether to skip outdated messages
        
        :return: None
        """
        super().__init__(host_name, port, is_master, timeout, skip_outdated, self.PREFIX)

        # Set the default serializer and deserializer for the EB
        self.set_encoder(Message.serialize)
        self.set_decoder(Message.deserialize)

    def broadcast(self, message: Message) -> str | None:
        """
        Broadcast message to the error bus.
        Returns the key of the set value, or None on error.
        """
        return self.send(message)
    

if __name__ == '__main__':
    import sys
    from torchft.FFT.messaging.message import GPUErrorMessage
    def cb(data):
        print(f"Callback received data: {data}")

    port = 22223
    if sys.argv[1] == 'master':
        try:
            eb = ErrorBus(port=port, is_master=True)
            
            eb.subscribe(cb).join()
        except KeyboardInterrupt:
            eb.shutdown()
    else:
        eb = ErrorBus(port=port, is_master=False)
        eb.subscribe(cb)
        m_idx = 0
        while True:
            try:
                msg = GPUErrorMessage(
                    reason='Test GPU error',
                    gpu_index=m_idx,
                )
                eb.broadcast(msg)
                m_idx += 1
                time.sleep(1)
            except KeyboardInterrupt:
                eb.shutdown()
                break
            
        