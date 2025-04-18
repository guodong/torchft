```python
import cmd
import argparse
import sys
import torch.distributed as dist

from torchft.distributed.messaging.error_bus import ErrorBus
from torchft.distributed.messaging.message import ErrorMessage, GPUErrorMessage


class TorchftShell(cmd.Cmd):
    """
    Custom shell for torchft management.
    Example usage:
        $ python cli/torchft_shell.py -h 127.0.0.1 -p 22223
        torchft> broadcast 0
        torchft> listen
        torchft> quit
    """
    intro = 'Torchft management CLI'
    prompt = 'torchft> '

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        try:
            self.eb = ErrorBus(host_name=self.host, port=self.port, is_master=False, callback=lambda x: None)
            print(f"Connect to {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect {self.host}:{self.port}: {e}")
            sys.exit(1)

    def do_broadcast(self, arg):
        """Broadcast a message to the error bus"""
        if not arg:
            print("Please provide a message to broadcast.")
            return
        try:
            msg = GPUErrorMessage(gpu_index=arg)
            self.eb.broadcast(msg.serialize())
            print(f"Broadcasted: {msg}")
        except Exception as e:
            print(f"Failed to broadcast message: {e}")

    def do_listen(self, arg):
        self.eb.register_callback(lambda msg: print(f"Received message: {msg}"))
        self.eb.run()

    def do_quit(self, arg):
        """Exit torchft shell"""
        self.eb.shutdown()
        print('Bye！')
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Torchft shell')
    parser.add_argument('-H', '--host', default='127.0.0.1', help='ErrorBus address')
    parser.add_argument('-p', '--port', type=int, default=22223, help='ErrorBus port')
    args = parser.parse_args()

    TorchftShell(args.host, args.port).cmdloop()
    
```


```python


from datetime import timedelta
import logging
import threading
import time
import torch.distributed as dist

from torchft.distributed.messaging.message import Message


class ErrorBus():
    """
    Error bus for distributed systems using PyTorch's TCPStore.
    This class provides a simple interface for broadcasting and receiving messages
    across multiple nodes.

    Note: c10d uses `activeOpLock_` to guarantee that only one op is active at a time. Thus the `get()` operation
    will block the compare_set operation, we use two TCPStore clients as a work around, 
    one for broadcasting and one for receiving.
    """

    # Prefix for TCPStore, used to avoid key collision
    PREFIX = 'error_bus'

    # Must be empty string, otherwise the compare_set will not work due to key existance check
    NONCE = ''

    def __init__(
        self,
        host_name: str = '127.0.0.1',
        port: int = 0,
        is_master: bool = False,
        timeout: timedelta = timedelta(seconds=10),
        callback: callable = None,
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
        self._callback = callback
        self._skip_outdated = skip_outdated

        # TCPStore backend for broadcasting messages
        self._broadcast_backend = dist.TCPStore(host_name, port, None, is_master, timeout)
        self._broadcast_backend = dist.PrefixStore(self.PREFIX, self._broadcast_backend)

        # TCPStore backend for receiving messages
        self._recv_backend = dist.TCPStore(host_name, port, None, False, timeout)
        self._recv_backend = dist.PrefixStore(self.PREFIX, self._recv_backend)

        self._current_write_index = 0
        self._current_read_index = 0

        # For graceful shutdown
        self._shutdown_flag = threading.Event()

        # For thread safe write_index mutation in recv and broadcast, or multiple broadcasts
        self._write_index_lock = threading.Lock()

        self._init_indces()

    def _init_indces(self) -> None:
        """
        Initialize the read and write indices.
        This is used to avoid key collision when broadcasting messages.
        """
        keys_count = self._broadcast_backend.num_keys()
        self._current_write_index = keys_count
        if self._skip_outdated:
            self._current_read_index = keys_count

    def register_callback(self, callback: callable) -> None:
        self._callback = callback

    def broadcast(self, message: str) -> str | None:
        """
        Broadcast message to the error bus.
        Returns the key of the set value, or None on error.
        """
        try:
            with self._write_index_lock:
                key = self._safe_set(message)
                return key
        except Exception as e:
            logging.error(f"Error in broadcast: {e}")
            return None

    def _safe_set(self, value: str) -> str:
        """
        Safely set the value to the error bus with a unique key.

        Returns the key of the set value.
        
        The compare_set method is used to ensure that the message is only set if the current value is equal to the nonce.
        Note: We found that the desired value is only set if 
            1) the key is not set (by setting expected_value to empty string "")
            2) or key is set and the value is equal to the expected_value.
        Thus we can simply set NONCE to an empty string and use it as the expected value to cover the two cases.
        The document from https://pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore seems outdated.
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

    def recv(self) -> str:
        """
        Blocking receive from the error bus.
        It will keep trying to get the message from the backend until it succeeds.
        """
        while not self._shutdown_flag.is_set():
            try:
                result = self._recv_backend.get(f'{self._current_read_index}')
                self._current_read_index += 1

                ####
                # If no exception is raised, we can safely move to the next index for broadcasting
                # Warnning! Broadcast is invoked in main thread, may lead to write_index plus twice
                ####
                # self._current_write_index = self._current_read_index

                return result.decode()
            except Exception as e:
                logging.debug(f"Pooling message: {e}")
                
    def run(self) -> None:
        """
        Start recv messages in a separate thread, returns the thread object.
        """
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        return thread

    def _run(self) -> None:
        while not self._shutdown_flag.is_set():
            try:
                message = self.recv()
                parsed = Message.deserialize(message)
                if self._callback and callable(self._callback):
                    self._callback(parsed)
            except Exception as e:
                logging.error(f"Error in run loop: {e}")
                break

    def shutdown(self) -> None:
        """Gracefully shutdown the error bus."""
        self._shutdown_flag.set()

    

if __name__ == '__main__':
    import sys
    def cb(data):
        print(f"Callback received data: {data}")

    port = 22223
    if sys.argv[1] == 'master':
        try:
            eb = ErrorBus(port=port, is_master=True, callback=cb)
            eb.run().join()
        except KeyboardInterrupt:
            eb.shutdown()
    else:
        eb = ErrorBus(port=port, is_master=False, callback=cb)
        eb.run()
        m_idx = 0
        while True:
            try:
                eb.broadcast(f'err {m_idx}')
                m_idx += 1
                time.sleep(1)
            except KeyboardInterrupt:
                eb.shutdown()
                break
            
        
```