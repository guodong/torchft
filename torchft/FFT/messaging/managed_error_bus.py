from collections import namedtuple
import logging
from typing import TYPE_CHECKING
from torchft.distributed.messaging.error_bus import ErrorBus

if TYPE_CHECKING:
    from torchft.manager import Manager

Message = namedtuple('Message', ['uuid', 'type', 'data'])

class ManagedErrorBus(ErrorBus):
    def __init__(self, manager: "Manager", *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.manager = manager
        self.register_callback = self.on_message

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

    def get_channel(self) -> ErrorBus:
        return self