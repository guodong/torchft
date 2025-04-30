"""
LocalSGD
=========
This module implements a fault tolerant version of LocalSGD and related methods.

This is an enhanced version of LocalSGD that allows for two levels of synchronization.
"""

import logging
from types import TracebackType
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.utils.hooks import RemovableHandle

from torchft.manager import Manager
from torchft.optim import OptimizerWrapper

logger: logging.Logger = logging.getLogger(__name__)

class DiLoCo_ICT:
    """
    DiLoCo is a subclass of LocalSGD that overrides the synchronization
    mechanism to average and synchronize the pseudogradients (delta of the previous global weight and current local weights).

    diloco: https://arxiv.org/pdf/2311.08105
    """

    def __init__(
        self,
        outer_manager: Optional[Manager],
        inner_manager: Manager,
        model: nn.Module,
        inner_optimizer: optim.Optimizer,
        outer_optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        device: Optional[torch.device] = None,
        pin_memory: bool = True,
        debug: bool = False,
    ) -> None:
        if outer_manager is not None and outer_manager._use_async_quorum:
            raise ValueError(
                "Using DiLoCo require synchronous quorum to be enabled. "
                "Ensure that the outer_manager is initialized with use_async_quorum=False"
            )
        self._leader = outer_manager is not None and outer_manager._rank == 0

        super().__init__()
        self._outer_manager = outer_manager
        self._inner_manager = inner_manager
        self._model = model
        self._local_optimizer = inner_optimizer
        self._local_step = 0
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"
        self._backup_device = backup_device
        self._device = device
        self._pin_memory = pin_memory

        self._hooks: List[RemovableHandle] = []
        self._outer_optimizer = outer_optimizer
        self.original_parameters: Dict[str, torch.Tensor] = {}
        for name, p in self._model.named_parameters():
            t = torch.empty(*tuple(p.shape), dtype=p.dtype, device=self._backup_device)
            if (
                self._pin_memory
                and t.device == torch.device("cpu")
                and torch.cuda.is_available()
            ):
                t = t.pin_memory()
            self.original_parameters[name] = t

        # Need to copy the parameters to the host to be safe if we are on the first step.
        self._debug = debug
        self.debug_print(f"DiLoCo_ICT initialized with {self._outer_manager=}")
        self.debug_print(f"{self._inner_manager=}")
        self.debug_print(f"{self._model=}")
        self.debug_print(f"{self._local_optimizer=}")
        self.debug_print(f"{self._outer_optimizer=}")
        self.debug_print(f"{self._sync_every=}")
        self.debug_print(f"{self._backup_device=}")
        self.debug_print(f"{self._pin_memory=}")
        self.debug_print(f"{self._hooks=}")
        self.debug_print(f"{self.original_parameters.keys()=}")

        self._save_parameters()

        self.debug_print(f"{self._leader=}")

        if not self._leader:
            self.debug_print(f"Setting up follower")
            self._start_ckpt_assist_listener()

    def _start_ckpt_assist_listener(self) -> None:
        # TODO: Implement the following algorithm:
        # Start a listening thread that waits for a message on a listening socket. Potentially a message on the TCP Store
        # When the message is received, send checkpoint to the correct location
        # See the live checkpoint recovery logic in manager.py (in _async_quorum)
        # This could be potentially parallelized through this process
        # Need to add a send_checkpoint_assist_msg function to the manager._async_quorum function
        # Anyhow, each listener has to receive one message from the leader. Or else it will not return.
        return

    def _finish_checkpoint_assist(self) -> None:
        # TODO: Wait until the listener receives a negative message (no need to send anything), or if the listener receives a positive message, then send the checkpoint to the correct location
        return

    def debug_print(self, *args, **kwargs) -> None:
        if self._debug:
            if self._leader:
                print("[FROM LOCAL_SGD_ICT, LEADER]", *args, **kwargs)
            else:
                print("[FROM LOCAL_SGD_ICT, FOLLOWER]", *args, **kwargs)

    def _save_parameters(self) -> None:
        self.debug_print(f"Saving parameters")
        with torch.no_grad():
            # TODO: consider running copy on a separate stream
            for name, p in self._model.named_parameters():
                self.original_parameters[name].copy_(p.data, non_blocking=True)

    def _restore_parameters(self) -> None:
        self.debug_print(f"Restoring parameters")
        with torch.no_grad():
            # TODO: consider running copy on a separate stream
            for name, p in self._model.named_parameters():
                p.data.copy_(self.original_parameters[name], non_blocking=False)

    def __enter__(self) -> "DiLoCo_ICT  ":
        self.debug_print(f"Entering DiLoCo")
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
        self.debug_print(f"Exiting DiLoCo")
        # Handle any cleanup or error handling here
        # Clean up hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.debug_print(f"Cleared hooks")

        return False  # Propagate exceptions

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        """
        This hook is registered on the optimizer and is called after the optimizer step.
        """
        self.debug_print(f"Step post hook")
        self._local_step += 1
        self.debug_print(f"self._local_step: {self._local_step}")
        self.debug_print(f"self._sync_every: {self._sync_every}")
        need_global_sync = self._local_step >= self._sync_every
        if need_global_sync:
            self.debug_print(f"Syncing")
            self.sync()
        else:
            self.debug_print(f"Not syncing")

    def sync(self) -> None:
        """
        Synchronizes and averages the model weights across the outer_manager.
        """
        if self._leader and self._outer_manager is not None: # To suppress linting error
            self.debug_print(f"Starting quorum")
            self._outer_manager.start_quorum()

        self.debug_print(f"Performing Outer Sync")
        self._perform_outer_sync() # Different sync for leader and follower
        self._local_step = 0
        self.debug_print(f"Local step reset to 0")

    def _perform_outer_sync(self) -> None:
        """
        Overrides the sync method to calculate the pseugradient, average them across the outer_manager group, and
        step using the outer optimizer.
        """
        self.debug_print(f"Setting pseudogradients")
        if self._leader:
            # Set the .grad field of each parameter to its pseudogradient
            # CALC_PSEUDOGRADS
            self.debug_print("CALC_PSEUDOGRADS")
            for name, p in self._model.named_parameters():
                pseudogradient = p.data - self.original_parameters[name]
                p.grad = pseudogradient
            self.debug_print(f"calc_done")

            self.debug_print(f"ALLREDUCE_PSEUDOGRADS")
            self._average_grads() # synchronous allreduce
            self.debug_print(f"Restoring parameters")

        # Restore the parameters back to the previous state
        self.debug_print(f"RESTORE_PARAMETERS")
        self._restore_parameters() # Potentially asynchronous
        
        self.debug_print(f"BROADCAST_PSEUDOGRADS")
        self.broadcast_pseudograds() # synchronous broadcast (at least currently so) 

        self.debug_print(f"STEP_OUTER_OPTIMIZER")
        if self._leader:
            # Leader decides whether to commit and broadcasts the decision (1 = commit, 0 = skip)
            should_commit_flag = self._outer_manager.should_commit() # Implicitly waits for the live checkpoint recovery to be finished
            commit_tensor = torch.tensor([1 if should_commit_flag else 0], dtype=torch.uint8, device=self._device)
            # Broadcast to everyone in the *inner* replica‑group. Leader is root_rank 0.
            fut = self._inner_manager.broadcast_one(commit_tensor, root_rank=0)
            fut.wait()
            self.debug_print(f"Broadcasted should_commit={should_commit_flag}")
            
            # Use the outer optimizer to update the model parameters
            # Need to check whether should_commit()
            # Currently this will always return true
            if should_commit_flag:
                # Use the outer optimizer to update the model parameters
                self._outer_optimizer.step()
                self.debug_print(f"Commited, saving parameters")
                self._save_parameters() # Currently synchronous
            else:
                self.debug_print(f"Not committing")
            self._outer_optimizer.zero_grad()
        else:
            # Is follower
            # 1. Wait for the checkpoint assist to be finished
            # 2. Step the outer optimiaer
            # 3. Save the parameters
            # Followers receive leader’s decision
            # Wait until the checkpoint‑assist thread (if any) finishes before potentially stepping.
            self.debug_print(f"Waiting for checkpoint assist")
            self._finish_checkpoint_assist()

            # Receive leader’s decision
            commit_tensor = torch.zeros(1, dtype=torch.uint8, device=self._device)
            self._inner_manager.broadcast_one(commit_tensor, root_rank=0).wait()
            should_commit_flag = bool(commit_tensor.item())
            self.debug_print(f"Received should_commit={should_commit_flag} from leader")

            self.debug_print(f"Stepping outer optimizer")
            if should_commit_flag:
                self._outer_optimizer.step()
            self.debug_print(f"Save parameters")
            self._save_parameters() # Currently synchronous

        self.debug_print(f"Zeroing gradients")
        self._outer_optimizer.zero_grad()

    def _average_grads(self) -> None:
        """
        Average the gradients across the diloco group.
        """
        works = []
        for name, p in self._model.named_parameters():
            # Perform allreduce on the pseudogradients
            assert p.grad is not None
            work = self._outer_manager.allreduce(p.grad)
            works.append(work)
        # Wait for all allreduce operations to complete
        self.debug_print(f"Waiting for allreduce operations to complete")
        counter = 0
        for work in works:
            work.wait()
            counter += 1
            self.debug_print(f"Allreduce operation {counter} completed")

    def broadcast_pseudograds(self) -> None:
        """
        Broadcast the pseudograds instead of the updated 
        model parameters because pseudograds can be more 
        aggressively quantized than the model parameters.
        """
        works = []
        for name, p in self._model.named_parameters():
            # Perform allreduce on the pseudogradients
            assert p.grad is not None
            work = self._inner_manager.broadcast_one(p.grad, root_rank=0)
            works.append(work)

        # Wait for all allreduce operations to complete
        self.debug_print(f"Waiting for allreduce operations to complete")
        counter = 0
        for work in works:
            work.wait()
            counter += 1
            self.debug_print(f"Allreduce operation {counter} completed")

