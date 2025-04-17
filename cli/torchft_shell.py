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
    