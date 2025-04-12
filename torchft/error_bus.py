from abc import ABC, abstractmethod
from collections import namedtuple
import logging
from typing import TYPE_CHECKING
from torch.distributed import Store, PrefixStore
from torchft.event_notifier import EventNotifier

Message = namedtuple('Message', ['sender_type', 'error_replica_id', 'type'])

class ErrorBus(EventNotifier):
    """
    Short circuit spreads errors to the world, and gracefully handle error recovery.
    """
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.replica_id = kwargs.pop('replica_id', None)
        super().__init__(*args, **kwargs)   
        self.register(self.on_message).run()

    def on_message(self, msg: Message) -> None:
        print(f"Received message: {msg}")
        raise NotImplementedError("not implemented")

    def _abort_local(self, msg: Message) -> None:
        print(f"Aborting local process for message: {msg}")
        raise NotImplementedError("not implemented")

    def _broadcast(self, msg: Message) -> None:
        print(f"Broadcasting message: {msg}")
        raise NotImplementedError("not implemented")

    def get_channel(self, comm_context) -> "ErrorChannel":
        raise NotImplementedError("not implemented")

if TYPE_CHECKING:
    from torchft.manager import Manager

class ManagedErrorBus(ErrorBus):
    def __init__(self, manager: "Manager", *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.manager = manager

    def on_message(self, msg: Message) -> None:
        # LOGIC TO BE REWRITTEN
        if msg.sender_type is None or msg.error_replica_id is None:
            logging.error(f"Received invalid message: {msg}")
            return
        if self.replica_id == msg.error_replica_id:
            self._abort_local(msg)
        elif msg.sender_type == "listening_thread":
            return
        else:
            self._broadcast(msg)

    def _abort_local(self, msg: Message) -> None:
        self.manager._abort_local(msg)

    def _broadcast(self, msg: Message) -> None:
        self.manager._broadcast(msg)

    def get_channel(self, store: Store) -> PrefixStore:
        prefix = "error_bus_channel"
        store = PrefixStore(prefix, store)
        
        return ErrorChannelKVStore(store)

class ErrorChannel(ABC):
    @abstractmethod
    def recv_message(self) -> Message:
        pass
    
    @abstractmethod
    def broadcast_message(self, msg: Message) -> None:
        pass

# Need to append rather than set
# Goal: When a new event comes in, 
# Need to delete.
# Shared public key
# Compute local hash
# Wait on "error_{local_hash}"
# After received, update local_hash. E.g. Hash(local_hash)


# Alternatively: 1. Use append.
# For each worker, they have their own unique prefix. 
# Everyone writes to their own prefix

# We have a centralized store
# Everyone can append/set to that store
# Everyone can get from that store

# If I have error:
#   Append to key
# Always listening to other people's error

# Desired Behavior:

# - Whenever I have an error, I can immediately append
# - Whenever there is an error, it will ASAP return to me

# We currently have two primitives
#   recv_message and broadcast_message

# Whenever I broadcast a message, I append to a certain key
# Whenever I receive message, 

# When we recieve, we listen to our replica_id's error

# When I broadcast, I broadcast to the TCPStore, but the prefix is the error_replica_id. 

# Logical ordering of error. If all the errors are in the form error_1, error_2, etc..

# Message = namedtuple('Message', ['sender_type', 'error_replica_id', 'type'])

# Only one person has authority to delete. 


# Recv_message returns ASAP
# Broadcast message gives it to everyone

# Counter increment
# Get
# Set
# Wait
# Compare and set
# 

# Compare and set: Before set, check 



class ErrorChannelKVStore(ErrorChannel):
    """
    A wrapper around the channel to automatically use "error" as the key
    It expects that the message is serialized in the following format:
    <sender_type>:<error_replica_id>:<type>
    """
    def __init__(self, channel: PrefixStore):
        self.channel = channel
        
    def recv_message(self) -> Message:
        self._wait()
        return self._get()

    def broadcast_message(self, msg: Message) -> None:
        self._set(msg)

    def _set(self, msg: Message):
        serialized_msg = f"{msg.sender_type}:{msg.error_replica_id}:{msg.type}"
        self.channel.set("error", serialized_msg)
        
    def _wait(self) -> None:
        return self.channel.wait("error")

    def _get(self) -> Message:
        serialized_msg = self.channel.get("error")
        if not serialized_msg:
            logging.info("No message in error bus")
            return Message(sender_type=None, error_replica_id=None, type="unknown")
        try:
            parts = serialized_msg.split(":", 2)
            if len(parts) == 3:
                sender_type, error_replica_id, msg_type = parts
                if error_replica_id == "None" or sender_type == "None":
                    logging.error(f"Incorrect message format in error bus: message={serialized_msg}")
                    return Message(sender_type=None, error_replica_id=None, type="unknown")
                return Message(sender_type=sender_type, error_replica_id=error_replica_id, type=msg_type)
            else:
                logging.error(f"Incorrect message format in error bus: message={serialized_msg}")
                return Message(sender_type=None, error_replica_id=None, type="unknown")
        except Exception:
            logging.error(f"Error parsing message in error bus: message={serialized_msg}")
            return Message(
                sender_type=None,
                error_replica_id=None,
                type=type(serialized_msg).__name__ if serialized_msg is not None else "unknown"
            )

if __name__ == '__main__':
    eb = ErrorBus(debug=True, replica_id=1233)
    msg = Message(type='error', replica_id=123)
    eb.notify(msg)
    eb.stop(graceful=True)