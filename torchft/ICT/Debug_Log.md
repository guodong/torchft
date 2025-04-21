

## Debug Experiences

## When may the stacking of the transport layers deadlock?

- For GPUA and GPUB, they will deadlock if
    - GPUA is waiting for GPUB
    - Whilst GPUB is waiting for GPUA
- This can happen if GPUA and GPUB are in the same process group twice.
- Thus, our principle is that two GPUs can only be in one process group together (TRUE?)

- Separating out the distinction between inner and outer optimizers are very important.
- A big problem is that one is waiting for the inner step, and one is waiting for the outer step.

## TCPStore

- There is the problem of coordinating the store address and port for all clients to connect to in a process group.
- Within the replica group, this is solved through the quorum mechanism, where one of the local TCPStores from a replica (spun up automatically by `torchrun`) gets assigned to be the TCPStore ran by the whole process group through the quorum mechanism.
- We could do the same for the global TCPStore. Currently, we are directly spinning up our own TCPStore in the shell script

## Broadcast One

Now, we call manager.broadcast_one inside localsgd to broadcast the tensor to all the other local replica groupis who are not doing global all reduce. 

However, we forgot to add a manager.start_quorum() to this.

In fact, from here we realized that we should have used our local_manager to call the allreduce, rather than the global manager, as we did before, which prevented gradient synchronization.

## for p in model.parameters:

Should do p.data rather than directly sending p.

## self.local_step

Before, self.local_step was in the `sync()` function. However, this is only called by one replica group, not all in broadcast_one.




# TO DEBUG

The failed code:

Decided to rewrite. Realized that there were many concurrency problems... Too many in the current state.

```python
class LocalSGD_ICT:
    """
    LocalSGD is a context manager that
    implements the algorithm described in https://arxiv.org/pdf/1805.09767

    This will synchronize the model parameters periodically in a fault tolerant
    way using a torchft Manager. The allreduce on the parameters will happen
    every sync_every steps after the optimizer.step call.

    To implement safe and fault tolerant, this requires a backup copy of the
    weights. By default these are stored in CPU memory. If any error occurs
    during the LocalSGD step, the step will be discarded and the model
    parameters will reset back to the last time LocalSGD synchronized.

    The backup weights could be eliminated by relaxing the guarantee of exactly
    `sync_every` steps but that would diverge from the LocalSGD algorithm.
    DiLoCo also needs this backup copy to compute the delta.

    The torchft quorum is computed at the beginning of ``sync_every`` steps. If
    any error occurs, or a worker fails between syncs, ``sync_every`` steps will be
    discarded and a new quorum will be computed on the next step.

    If running in async mode, on a joining worker the first ``sync_every`` steps
    will discarded as the model will be recovering during that period. When
    using sync mode, the checkpoint will be restored prior to the first step.

    TODO: add a way via Manager to detect workers failing early for shrink only
    TODO: add DiLoCo support
    """

    def __init__(
        self,
        manager: Manager,
        cluster_manager: Manager,
        model: nn.Module,
        optimizer: optim.Optimizer,
        sync_every: int,
        self_rank: int,
        root_rank: int,
    ) -> None:
        """
        Args:
            manager: The manager to use.
            model: The model to wrap.
            optimizer: The optimizer used by the model.
            sync_every: How often to sync the model weights.
            backup_device: The device to store the backup of the model parameters on. (default cpu)
            pin_memory: Whether to pin the memory used for the backup of the model parameters.
        """
        super().__init__()
        self._manager = manager
        self._cluster_manager = cluster_manager
        self._model = model
        self._local_optimizer = optimizer
        self._local_step = 0
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"
        self._self_rank = self_rank
        self._root_rank = root_rank

        self._hooks: List[RemovableHandle] = []

    def __enter__(self) -> "LocalSGD":
        # Add optimizer hook which increments the local step counter and syncs if necessary
        self._hooks.append(
            self._local_optimizer.register_step_post_hook(self._step_post_hook)
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        # Handle any cleanup or error handling here
        # Clean up hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        return False  # Propagate exceptions

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        """
        This hook is registered on the optimizer and is called after the optimizer step.
        """
        # TODO: There should be an optimization where 
        print("FROM LOCAL SGD: _step_post_hook")
        self._local_step += 1
        print(f"FROM LOCAL SGD: local_step: {self._local_step}, sync_every: {self._sync_every}")
        if self._local_step >= self._sync_every:
            print("FROM LOCAL SGD: at step %d, syncing" % self._local_step)

            self._cluster_manager.start_quorum(allow_heal=False) # Start the quorum for the cluster manager

            if self._self_rank == self._root_rank:
                self.sync()
            
            # Broadcast the model parameters to the other replica groups/receive them from the root rank
            for p in self._model.parameters():
                p.data.grad = None
                self._cluster_manager.broadcast_one(p.data, root_rank=self._root_rank, timeout=self._manager._timeout * 2) # *2 because have to wait for self.sync() to finish also

            self._local_step = 0

    def sync(self) -> None:
        """
        Synchronizes and averages the model weights across the manager.
        """
        print("FROM LOCAL SGD: sync")
        self._manager.start_quorum()
        print("FROM LOCAL SGD: start_quorum")
        self._perform_sync()
        print

    def _perform_sync(self) -> None:
        """
        Performs the synchronization of the model weights across the manager.
        """
        averaged_parameters = self._average()
        if self._manager.should_commit():
            # Update the model parameters with the averaged values
            for param, avg_param in zip(self._model.parameters(), averaged_parameters):
                param.data.copy_(avg_param)

    def _average(self) -> list[torch.Tensor]:
        """
        Averages the model parameters across the manager and returns the averaged parameters.
        """
        works = []
        averaged_parameters = []
        for p in self._model.parameters():
            # Create a new tensor to store the averaged parameter
            p.data.grad = None
            avg_param = p.data.clone()
            works.append(self._manager.allreduce(avg_param))
            averaged_parameters.append(avg_param)
            
        for work in works:
            work.wait()
        return averaged_parameters
```

## Cuda IPC errors

```txt
2025-04-20T10:01:10.184 [INFO] [torchft::manager] - [Replica train_localsgd_0_0] should_commit completed should_commit=true
FROM LOCAL SGD: _step_post_hook
FROM LOCAL SGD: local_step: 1, sync_every: 2
got unexpected error in future handler:
Traceback (most recent call last):
  File "/srv/apps/warren/torchft/torchft/process_group.py", line 1362, in _future_handler
    cmd = future_pipe.recv(timedelta(seconds=10))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/srv/apps/warren/torchft/torchft/multiprocessing.py", line 21, in recv
    out = self._pipe.recv()
          ^^^^^^^^^^^^^^^^^
  File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
          ^^^^^^^^^^^^^^^^^^
  File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/multiprocessing/connection.py", line 430, in _recv_bytes
    buf = self._recv(4)
          ^^^^^^^^^^^^^
  File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/multiprocessing/connection.py", line 399, in _recv
    raise EOFError
EOFError
got unexpected error in future handler:
Traceback (most recent call last):
  File "/srv/apps/warren/torchft/torchft/process_group.py", line 1362, in _future_handler
    cmd = future_pipe.recv(timedelta(seconds=10))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/srv/apps/warren/torchft/torchft/multiprocessing.py", line 21, in recv
    out = self._pipe.recv()
          ^^^^^^^^^^^^^^^^^
  File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
          ^^^^^^^^^^^^^^^^^^
  File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/multiprocessing/connection.py", line 430, in _recv_bytes
    buf = self._recv(4)
          ^^^^^^^^^^^^^
  File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/multiprocessing/connection.py", line 399, in _recv
    raise EOFError
EOFError
[W420 10:01:11.996799136 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
(base) root@sz-k8s-master:~#
```

Honestly here not sure at all what is going on. I think I need to redesign LocalSGD if the bugs are so hard to fix lol.

Every time at step 130 or so (not deterministic), get the following error. Terminates one process, the other process keeps on going.

## Checkpoint Transport

When I run 
```bash
(base) root@sz-k8s-master:~# /srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 1 1 0 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000 # Replica Group 1, Cluster 0, wait for 1 second between steps, sync every 2 steps for 1000 stepsclear
```
before 
```bash
(base) root@sz-k8s-master:~# /srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 0 0 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000
```

I get HTTPTransport locks.

## Checkpoint Metadata

After running the following in sequence:

```bash
/srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 0 0 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000 # Cuda Device 0, Replica Group 0, Cluster 0, wait for 1 second between steps, sync every 2 steps for 1000 steps
/srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 1 1 0 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000 # Cuda Device 1, Replica Group 1, Cluster 0, wait for 1 second between steps, sync every 2 steps for 1000 steps
/srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 0 1 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000 # Cuda Device 0, Replica Group 0, Cluster 1, wait for 1 second between steps, sync every 2 steps for 1000 steps
/srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 1 1 1 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000 # Cuda Device 1, Replica Group 1, Cluster 1, wait for 1 second between steps, sync every 2 steps for 1000 steps
```

I got the following error:

============================================================
/srv/apps/warren/torchft/train_localsgd-two_level.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-04-20_07:33:53
  host      : sz-k8s-master
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 65111)
  error_file: /tmp/torchelastic_sio2pi4d/none_cejtyodz/attempt_0/0/error.json
  traceback : Traceback (most recent call last):
    File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 357, in wrapper
      return f(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^
    File "/srv/apps/warren/torchft/train_localsgd-two_level.py", line 281, in main
      managed_inner_optimizer.step()
    File "/srv/apps/warren/torchft/torchft/optim.py", line 55, in step
      self.optim.step()
    File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/optim/optimizer.py", line 512, in wrapper
      post_hook(self, args, kwargs)
    File "/srv/apps/warren/torchft/torchft/local_sgd.py", line 115, in _step_post_hook
      self.sync()
    File "/srv/apps/warren/torchft/torchft/local_sgd.py", line 124, in sync
      self._perform_sync()
    File "/srv/apps/warren/torchft/torchft/local_sgd.py", line 132, in _perform_sync
      averaged_parameters = self._average()
                            ^^^^^^^^^^^^^^^
    File "/srv/apps/warren/torchft/torchft/local_sgd.py", line 148, in _average
      works.append(self._manager.allreduce(avg_param))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/srv/apps/warren/torchft/torchft/manager.py", line 363, in allreduce
      return self._run_collective(tensor, work, post, timeout=self._timeout)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/srv/apps/warren/torchft/torchft/manager.py", line 311, in _run_collective
      self.wait_quorum()
    File "/srv/apps/warren/torchft/torchft/manager.py", line 530, in wait_quorum
      self._quorum_future.result()
    File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/concurrent/futures/_base.py", line 456, in result
      return self.__get_result()
             ^^^^^^^^^^^^^^^^^^^
    File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/concurrent/futures/_base.py", line 401, in __get_result
      raise self._exception
    File "/srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/concurrent/futures/thread.py", line 58, in run
      result = self.fn(*self.args, **self.kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/srv/apps/warren/torchft/torchft/manager.py", line 621, in _async_quorum
      checkpoint_metadata = primary_client._checkpoint_metadata(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  RuntimeError: status: InvalidArgument, message: "rank not found", details: [], metadata: MetadataMap { headers: {"content-type": "application/grpc", "date": "Sat, 19 Apr 2025 23:33:53 GMT", "content-length": "0"} }


 - Further, note that this had a more pernicious effect.
 Although the main process stopped running and erred, it seems like the heartbeat thread has not stopped. This meant that replica group 0 on cluster 0 stays waiting for this process to join quorum (as it needs at least half to join quorum to prevent split-brain issues). 