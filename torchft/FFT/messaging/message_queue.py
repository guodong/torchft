

from datetime import timedelta
import logging
import threading
import torch.distributed as dist

class MessageQueue():
    """
    Message queue for distributed systems using PyTorch's TCPStore.
    This class provides a simple interface for broadcasting and receiving messages
    across multiple nodes.

    Note: c10d uses `activeOpLock_` to guarantee that only one op is active at a time. Thus the `get()` operation
    will block the compare_set operation, we use two TCPStore clients as a work around, 
    one for broadcasting and one for receiving.
    """

    # Must be empty string, otherwise the compare_set will not work due to key existance check
    NONCE = ''

    def __init__(
        self,
        host_name: str = '127.0.0.1',
        port: int = 0,
        is_master: bool = False,
        timeout: timedelta = timedelta(seconds=10),
        skip_outdated: bool = True,
        topic: str = '__message_queue__',
    ) -> None:
        """
        Similar to TCPStore, but with a callback function for receiving messages.
        :param host_name: Host name of the master TCPStore
        :param port: Port number of the master TCPStore
        :param is_master: Whether this node is the master node
        :param timeout: Timeout for TCPStore operations
        :param skip_outdated: Whether to skip outdated messages
        :param topic: Topic name for the message queue
        
        :return: None
        """
        self._skip_outdated = skip_outdated
        self._topic = topic

        # TCPStore backend for broadcasting messages
        self._broadcast_backend = dist.TCPStore(host_name, port, None, is_master, timeout)
        self._broadcast_backend = dist.PrefixStore(self._topic, self._broadcast_backend)

        # TCPStore backend for receiving messages
        self._recv_backend = dist.TCPStore(host_name, port, None, False, timeout)
        self._recv_backend = dist.PrefixStore(self._topic, self._recv_backend)

        self._current_write_index = 0
        self._current_read_index = 0

        # For graceful shutdown
        self._shutdown_flag = threading.Event()

        # For thread safe write_index mutation in recv and broadcast, or multiple broadcasts
        self._write_index_lock = threading.Lock()

        self._encoder = None
        self._decoder = None

        self._init_indices()

    def _init_indices(self) -> None:
        """
        Initialize the read and write indices.
        This is used to avoid key collision when broadcasting messages.
        """
        keys_count = self._broadcast_backend.num_keys()
        self._current_write_index = keys_count
        if self._skip_outdated:
            self._current_read_index = keys_count

    def set_encoder(self, encoder: callable) -> None:
        self._encoder = encoder

    def set_decoder(self, decoder: callable) -> None:
        self._decoder = decoder

    def send(self, message: str) -> str | None:
        try:
            with self._write_index_lock:
                if self._encoder:
                    message = self._encoder(message)
                key = self._safe_set(message)
                return key
        except Exception as e:
            logging.error(f"Error: {e}")
            return None
        
    def poll(self) -> str:
        """
        Blocking receive from the MQ.
        It will keep trying to get the message from the backend until it succeeds.
        """
        while not self._shutdown_flag.is_set():
            try:
                result = self._recv_backend.get(f'{self._current_read_index}')
                self._current_read_index += 1

                ####
                # If no exception is raised, we can safely move to the next index for writing
                # Warnning! Invoking in multi thread may lead to write_index plus twice
                ####
                # self._current_write_index = self._current_read_index

                if self._decoder:
                    return self._decoder(result.decode())
                
                return result.decode()
            except Exception as e:
                logging.debug(f"Pooling message: {e}")
        
    def subscribe(self, callback: callable) -> threading.Thread:
        """
        Subscribe to the message queue.
        This will start a new thread that will listen for messages and call the callback function.
        :param callback: Callback function to be called when a message is received
        :return: None
        """
        if not callable(callback):
            raise ValueError("Callback must be a callable function")
        
        self._callback = callback
        sub_thread = threading.Thread(target=self._subscribe, daemon=True)
        sub_thread.start()
        return sub_thread

    def register_callback(self, callback: callable) -> None:
        self._callback = callback

    def _safe_set(self, value: str) -> str:
        """
        Safely set the value to the MQ with a unique key.

        Returns the key of the set value.
        
        The compare_set method is used to ensure that the message is only set if the current value is equal to the nonce.
        Note: We found that the desired value is only set if 
            1) the key is not set (by setting expected_value to empty string "")
            2) or key is set and the value is equal to the expected_value.
        Thus we can simply set NONCE to an empty string and use it as the expected value to cover the two cases.
        See https://pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore.
        """
        while not self._shutdown_flag.is_set():
            key = f'{self._current_write_index}'

            # Short circuit for existing keys, recudes overhead for comparing compare_set result
            if self._broadcast_backend.check([key]):
                # Key already exists, increment the index and try again
                self._current_write_index += 1
                continue
            result = self._broadcast_backend.compare_set(key, self.NONCE, value)
            self._current_write_index += 1

            # We tested on python 3.12 that decode is faster than encode for the same string, 
            if result.decode() == value:
                return key
            else:
                logging.debug('key is taken')

    def _subscribe(self) -> None:
        while not self._shutdown_flag.is_set():
            try:
                message = self.poll()
                if self._callback and callable(self._callback):
                    self._callback(message)
            except Exception as e:
                logging.error(f"Error in run loop: {e}")
                break

    def shutdown(self) -> None:
        """Gracefully shutdown the MQ."""
        self._shutdown_flag.set()            
        