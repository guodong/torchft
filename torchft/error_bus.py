
from collections import namedtuple
import time
from event_notifier import EventNotifier

Message = namedtuple('Message', ['type', 'replica_id'])

class ErrorBus(EventNotifier):
    """
    Short circuit spreads errors to the world, and gracefully handle error recovery.
    """
    def __init__(self, *args: object, **kwargs: object):
        self.replica_id = kwargs.pop('replica_id', None)
        super().__init__(*args, **kwargs)
        self.register(self.on_message).run()


    def on_message(self, msg: Message):
        if self.replica_id == msg.replica_id:
            self._abort_local(msg)
        else:
            self._broadcast(msg)

    def _abort_local(self, msg: Message):
        print(f"Aborting local process for message: {msg}")
        raise NotImplementedError("not implemented")

    def _broadcast(self, msg: Message):
        print(f"Broadcasting message: {msg}")
        raise NotImplementedError("not implemented")
        

if __name__ == '__main__':
    eb = ErrorBus(debug=True, replica_id=1233)
    msg = Message(type='error', replica_id=123)
    eb.notify(msg)
    eb.stop(graceful=True)