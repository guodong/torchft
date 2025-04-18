import cmd
import argparse
import os
import signal
import sys
import socket
import time
import torch.distributed as dist

from torchft.distributed.messaging.error_bus import ErrorBus
from torchft.distributed.messaging.message import ErrorMessage, GPUErrorMessage, NodeErrorMessage


class SignalHandler:
    """
    Handler for testing signal functionality.
    """
    def __init__(self):
        self.error_msg = None
        self.signal_received = False
        
        # Set up signal handler
        def on_signal(signum, frame):
            print(f"Received signal {signum}")
            self.signal_received = True
            self.handle_signal()
        
        signal.signal(signal.SIGUSR1, on_signal)
    
    def error_bus_callback(self, error_msg: ErrorMessage):
        """
        Callback for the error bus that stores the message and sends a signal.
        """
        print(f"Received message from error bus: {error_msg}")
        self.error_msg = error_msg
        # Send a signal to self to trigger the signal handler
        os.kill(os.getpid(), signal.SIGUSR1)
    
    def handle_signal(self):
        """
        Handle the received signal based on the error message.
        """
        if self.error_msg is None:
            print("No error message received")
            return
        
        if isinstance(self.error_msg, GPUErrorMessage):
            print(f"GPU error on {self.error_msg.host_name}, GPU {self.error_msg.gpu_index}")
            print(f"SIGNAL HANDLER: Would abort step due to GPU error")
        elif isinstance(self.error_msg, NodeErrorMessage):
            print(f"Node error on {self.error_msg.host_name}")
            print(f"SIGNAL HANDLER: Would abort step due to node error")
        else:
            print(f"SIGNAL HANDLER: Would abort step due to unknown error: {self.error_msg}")
        
        self.signal_received = False


class TorchftShell(cmd.Cmd):
    """
    Custom shell for torchft management.
    Example usage:
        # On one shell, start the error_bus
        $ python -m torchft.distributed.messaging.error_bus master 

        # Connect through another shell
        $ python cli/torchft_shell.py -H 127.0.0.1 -p 22223
        torchft> broadcast 0
        torchft> listen
        torchft> quit
    """
    intro = 'Torchft management CLI'
    prompt = 'torchft> '

    def __init__(self, host, port, test_signals=False):
        super().__init__()
        self.host = host
        self.port = port
        self.test_signals = test_signals
        self.signal_handler = SignalHandler() if test_signals else None
        
        try:
            callback = self.signal_handler.error_bus_callback if self.test_signals else lambda x: None
            self.eb = ErrorBus(host_name=self.host, port=self.port, is_master=False, callback=callback)
            print(f"Connected to error bus at {self.host}:{self.port}")
            
            if self.test_signals:
                print("Signal testing mode enabled. Error messages will trigger SIGUSR1 signals.")
        except Exception as e:
            print(f"Failed to connect to error bus at {self.host}:{self.port}: {e}")
            sys.exit(1)

    def do_broadcast(self, arg):
        """
        Broadcast a GPU error message to the error bus
        
        Usage: broadcast <gpu_index> [reason]
        Example: broadcast 0 Out of memory
        """
        args = arg.split(maxsplit=1)
        if not args:
            print("Please provide a GPU index to broadcast.")
            return
        
        try:
            gpu_index = int(args[0])
            reason = args[1] if len(args) > 1 else "Test GPU error"
            
            msg = GPUErrorMessage(
                reason=reason, 
                gpu_index=gpu_index, 
                host_name=socket.gethostname()
            )
            self.eb.broadcast(msg.serialize())
            print(f"Broadcasted GPU error: {msg}")
        except ValueError:
            print("GPU index must be an integer.")
        except Exception as e:
            print(f"Failed to broadcast message: {e}")

    def do_broadcast_node(self, arg):
        """
        Broadcast a node error message to the error bus
        
        Usage: broadcast_node <hostname> [reason]
        Example: broadcast_node localhost Node crashed
        """
        args = arg.split(maxsplit=1)
        if not args:
            print("Please provide a hostname to broadcast.")
            return
        
        try:
            hostname = args[0]
            reason = args[1] if len(args) > 1 else "Test node error"
            
            msg = NodeErrorMessage(
                reason=reason, 
                host_name=hostname
            )
            self.eb.broadcast(msg.serialize())
            print(f"Broadcasted node error: {msg}")
        except Exception as e:
            print(f"Failed to broadcast message: {e}")
    
    def do_broadcast_general(self, arg):
        """
        Broadcast a general error message to the error bus
        
        Usage: broadcast_general [reason]
        Example: broadcast_general Training process failed
        """
        reason = arg if arg else "Test general error"
        
        try:
            msg = ErrorMessage(reason=reason)
            self.eb.broadcast(msg.serialize())
            print(f"Broadcasted general error: {msg}")
        except Exception as e:
            print(f"Failed to broadcast message: {e}")

    def do_listen(self, arg):
        """Start listening for error messages from the error bus"""
        if self.test_signals:
            print("Listening for error messages with signal handling...")
            # Already set up callback in __init__
        else:
            print("Listening for error messages without signal handling...")
            self.eb.register_callback(lambda msg: print(f"Received message: {msg}"))
        
        self.eb.run()
        print("Listening for error bus messages. Press Ctrl+C to stop.")

    def do_quit(self, arg):
        """Exit torchft shell"""
        self.eb.shutdown()
        print('Bye！')
        return True
    
    # Aliases for convenience
    do_exit = do_quit
    do_q = do_quit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Torchft shell')
    parser.add_argument('-H', '--host', default='127.0.0.1', help='ErrorBus address')
    parser.add_argument('-p', '--port', type=int, default=22223, help='ErrorBus port')
    parser.add_argument('-s', '--test-signals', action='store_true', help='Enable signal testing mode')
    args = parser.parse_args()

    TorchftShell(args.host, args.port, args.test_signals).cmdloop()
    