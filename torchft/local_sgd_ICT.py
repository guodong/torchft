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

logger: logging.Logger = logging.getLogger(__name__)

class DiLoCo_ICT:
    """
    DiLoCo is a subclass of LocalSGD that overrides the synchronization
    mechanism to average and synchronize the pseudogradients (delta of the previous global weight and current local weights).

    diloco: https://arxiv.org/pdf/2311.08105
    """

    def __init__(
        self,
        outer_manager: Manager,
        model: nn.Module,
        inner_optimizer: optim.Optimizer,
        outer_optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
        debug: bool = False,
    ) -> None:
        if outer_manager._use_async_quorum:
            raise ValueError(
                "Using DiLoCo require synchronous quorum to be enabled. "
                "Ensure that the outer_manager is initialized with use_async_quorum=False"
            )
        super().__init__()
        self._outer_manager = outer_manager
        self._model = model
        self._local_optimizer = inner_optimizer
        self._local_step = 0
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"
        self._backup_device = backup_device
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
        self.debug_print(f"{self._model=}")
        self.debug_print(f"{self._local_optimizer=}")
        self.debug_print(f"{self._outer_optimizer=}")
        self.debug_print(f"{self._sync_every=}")
        self.debug_print(f"{self._backup_device=}")
        self.debug_print(f"{self._pin_memory=}")
        self.debug_print(f"{self._hooks=}")
        self.debug_print(f"{self.original_parameters.keys()=}")

        self._save_parameters()

    def debug_print(self, *args, **kwargs) -> None:
        if self._debug:
            print("[FROM LOCAL_SGD_ICT]", *args, **kwargs)

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
        if self._local_step >= self._sync_every:
            self.debug_print(f"Syncing")
            self.sync()

    def sync(self) -> None:
        """
        Synchronizes and averages the model weights across the outer_manager.
        """
        self.debug_print(f"Starting quorum")
        self._outer_manager.start_quorum()
        self.debug_print(f"Performing sync")
        self._perform_sync()
        self._local_step = 0
        self.debug_print(f"Local step reset to 0")

    def _perform_sync(self) -> None:
        """
        Overrides the sync method to calculate the pseugradient, average them across the outer_manager group, and
        step using the outer optimizer.
        """
        self.debug_print(f"Setting pseudogradients")
        # Set the .grad field of each parameter to its pseudogradient
        for name, p in self._model.named_parameters():
            pseudogradient = p.data - self.original_parameters[name]
            p.grad = pseudogradient

        self.debug_print(f"Averaging gradients")
        self._average_grads()
        self.debug_print(f"Restoring parameters")
        # Restore the parameters back to the previous state
        self._restore_parameters()
        if self._outer_manager.should_commit():
            self.debug_print(f"Stepping outer optimizer")
            # Use the outer optimizer to update the model parameters
            self._outer_optimizer.step()
            self.debug_print(f"Saving parameters")
            self._save_parameters()
        self.debug_print(f"Zeroing gradients")
        self._outer_optimizer.zero_grad()

    def _average_grads(self) -> None:
        """
        Average the gradients across the diloco group.
        """
        works = []
        self.debug_print(f"Averaging gradients")
        for name, p in self._model.named_parameters():
            # Perform allreduce on the pseudogradients
            assert p.grad is not None
            self.debug_print(f"Allreducing gradient for {name}")
            work = self._outer_manager.allreduce(p.grad)
            works.append(work)
        # Wait for all allreduce operations to complete
        self.debug_print(f"Waiting for allreduce operations to complete")
        for work in works:
            work.wait()