TODOs:

1. Add retry logic. This needs to be at the top level


1. Race Condition Handling

If call `interrupt` after `timeout`, then we are interrupting the timed_fut and 

As long as we set exception on the timed_fut, we should be ok?


1. For a single `interruptable_future`, do I need to lock it when modifying the underlying future? Or this is actually not needed?

My current idea is the following: 

I definitely need to lock it when I do _FAILURE_MANAGER.register(IMMEDIATE_INTERRUPT_EVENT), since my IMMEDIATE INTERRUPT event is total
- But for all the other methods acting on the future, then, I need to check whether my current future is good or not. 

For add_timeout, I should also have the lock for the whole thing to go through.

For wait, it is weird because I should have the lock when I do current_future.wait(). The problem here is that it is waiting on the older one.

My current strategy is the following:

Every time I call add_timeout or immediate_interrupt, I return a new future. This new future's 

I think I figured out the strategy: 

Whenever I add timeout, I should return a new timed_fut object. Howver, whenever I am doing anything else. I shoudln't. 




# FAQ: How does the code ensure that the future waits for the cuda stream to finish (since native pytorch futures return once the operation is enqueued onto the stream)

We have the code, deceivingly simple, that is the following:
```python
work = pg._fun_func(...)
fut = work.get_future()
fut.wait()
```

What actually happens is the following:

Here, pg._fun_func(...) returns a _BabyWork object.

When we do work.get_future(), we call 

`self._pg._get_future(self.op_id, self._stream)`

where `_pg` is a ProcessGroupBaby object.

Thus, when we call `_get_future`, we actually communicate to the `_worker` subprocess in ProcessGroupBaby. `_worker`. However, and here is where something interesting happens, `_get_future` returns a brand new future created in the following way:

```python
def _get_future(
    self, op_id: int, stream: Optional[torch.cuda.Stream]
) -> Future[object]:
    with self._futures_lock:
        fut = Future()  # pyre-fixme[29]: is not a function
        self._futures[op_id] = _FutureMetadata(future=fut, stream=stream)
        assert self._pipe is not None
        self._pipe.send(("future", op_id))

    # TODO: return correct tensor instead of None
    return fut
```

We will later on manipulate this new future safed in `self._futures[op_id]` to ensure proper synchronization. 


Now, back to `_get_future`:

This communicates with the `_worker` subprocess, which executes: 

```python
metadata.work.get_future().add_done_callback(
    lambda fut: callback(fut, metadata)
)
```

Note that `metadata.work.get_future()` gets the future associated with the work. 

Now, the callback is where the core of the synchronization logic happens:

```python
def callback(fut: Future[object], metadata: _OpMetadata) -> None:
    try:
        # create an event after the collective has been issued
        # to wait on this before we call "future"
        with metadata.set_stream():
            fut.wait()
            event = (
                torch.cuda.current_stream().record_event(
                    torch.cuda.Event(interprocess=True)
                )
                if metadata.stream is not None
                else None
            )

        future_pipe.send((op_id, _FUTURE_RESULT, None, event))
    except Exception as e:
        future_pipe.send((op_id, _FUTURE_EXCEPTION, e, None))
```

Here, we record a cuda event and send this event through the future pipe. Or, if the operation is not successfully enqueued, then `fut.wait()` will raise an error and we will also send this through the pipe.

Then, the `_future_handler` thread that monitors the pipe can retrieve the result and wait for the event + wait for the future!

So, to summarize:

There are two futures around:

1. A pytorch future associated with a _BabyWork object, where the work is an MPI call.
2. future() defined by ourselves and returned to the main `train.py` thread, and which it waits on.

We handle the creation and enqueing of the pytorch future in the _worker subprocess, which may seem strange. Why do we do so? The answer is in the [torchft design doc](https://docs.google.com/document/d/1OZsOsz34gRDSxYXiKkj4WqcD9x0lP9TcsfBeu_SsOY4/edit?tab=t.0):

```txt
NCCL is prone to deadlocks on errors as well as when calling NCCL comm abort. In OSS version some of these issues have been fixed but it's unknown to what extent at this point since I haven't used it extensively. In addition it sounds like NVIDIA is working on making NCCL safer but it's not fully ready yet.

An alternative to using NCCLs error handling is to simply run it in a subprocess. This subprocess can be managed by the parent process and on error or quorum change, killed on all nodes and recreated.
```

To explain: To prevent NCCL deadlock taking the whole `train.py` down, we handle all the errors related to NCCL inside a `_worker` subprocess. This `worker` process can be killed and reconfigured when our quorum changes or upon unresponsiveness. This prevents the deadlock taking the whole process down!

Now, this is also why we spawn out a `_future_handler` thread to handle the `event.wait()` for the `event` associated with the work that we are getting a future from. 

We do this so that the `_worker` subprocess dono't get blocked waiting for the wait. Here, our `_future_handler`'s setting of the returned future's result is actually the event that we are waiting for when we do `fut.wait()` in our main thread!

(Note that when we do `work.wait()`, we don't need this. Here we directly call `event.wait()` in the main thread in `pg._wait`. The reason here is that waiting for the work is immediately blocking, whereas waiting for the future is more tricky since we can later on wait on this future at any time. So we have a background thread that waits on the cuda event to monitor the progress of that future.)


Tricky things: Uses the _BabyWork object. When I wait for baby work. 


