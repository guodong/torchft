from ast import Dict
from datetime import timedelta
from typing import Any, Optional
from torchft.futures.failure_manager import _FAILURE_MANAGER, TIMEOUT_EVENT, IMMEDIATE_INTERRUPT_EVENT
from torch.futures import Future

class InterruptableFuture:
    def __init__(self, future: Future[T], metadata: Optional[Dict[str, Any]] = None):
        self.future = future
        self.failure_manager = _FAILURE_MANAGER
        self.metadata = metadata

    def immediate_interrupt(self, timeout: timedelta):
       _FAILURE_MANAGER.register(self.future, IMMEDIATE_INTERRUPT_EVENT, timeout)

    def add_timeout(self, timeout: timedelta):
       _FAILURE_MANAGER.register(self.future, TIMEOUT_EVENT, timeout)


# Example Usage

# Manager:
# def __init__(self):
#     self._pending_work = []

# During All Reduce:
# fut = allreduce(tensor)
# interruptable_future = InterruptableFuture(fut, metadata={"allreduce": True})
# self._pending_work.append(interruptable_future)

# Listening Thread:
# on_message_received(message: Message) -> Future[Message]:
#     interruptable_future = InterruptableFuture(future, metadata={"message": message})
#     if message_type == "interrupt":
#         interruptable_future.immediate_interrupt(timedelta(seconds=10))
#     elif message_type == "change_timeout":
#         timeout: timedelta = interruptable_future.metadata["timeout"]
#         interruptable_future.add_timeout(timeout)
#     return interruptable_future.future



