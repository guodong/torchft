# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
LocalSGD
=========
This module implements a fault tolerant version of LocalSGD and related methods.
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

import torch.distributed.checkpoint as dcp

from torch.distributed.tensor import DeviceMesh, distribute_tensor

logger: logging.Logger = logging.getLogger(__name__)

import copy

class LocalSGD:
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
        model: nn.Module,
        optimizer: optim.Optimizer,
        sync_every: int,
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
        self._model = model
        self._local_optimizer = optimizer
        self._local_step = 0
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"

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
        self._local_step += 1
        if self._local_step >= self._sync_every:
            self.sync()

    def sync(self) -> None:
        """
        Synchronizes and averages the model weights across the manager.
        """
        self._manager.start_quorum()
        self._perform_sync()
        self._local_step = 0

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


class DiLoCo:
    """
    DiLoCo is a subclass of LocalSGD that overrides the synchronization
    mechanism to average and synchronize the pseudogradients (delta of the previous global weight and current local weights).

    diloco: https://arxiv.org/pdf/2311.08105
    """

    def __init__(
        self,
        manager: Manager,
        model: nn.Module,
        inner_optimizer: optim.Optimizer,
        outer_optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
        off_load: bool = False,
        if_shard: bool = False,
        rpg_id: int = 0,
        local_rank: int = 0,
        num_rg: int = 2
        # eager_mode: bool = False
    ) -> None:
        if manager._use_async_quorum:
            raise ValueError(
                "Using DiLoCo require synchronous quorum to be enabled. "
                "Ensure that the manager is initialized with use_async_quorum=False"
            )
        super().__init__()
        self._manager = manager
        self._model = model
        self._local_optimizer = inner_optimizer
        self._local_step = 0
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"
        self._backup_device = backup_device
        self._pin_memory = pin_memory
        
        self._off_load = off_load
        self._if_shard = if_shard
        self.async_future = None
        # self._eager = eager_mode
        self.works = []
        self.num_rg = num_rg

        self._hooks: List[RemovableHandle] = []
        self._outer_optimizer = outer_optimizer
        self.original_parameters: Dict[str, torch.Tensor] = {}
        self.original_grads: Dict[str, torch.Tensor] = {}  #用来存放在t-h时刻的model grad
        self.newest_grads: Dict[str, torch.Tensor] = {}    #用来发送和接收pesudogradient
        self.savegrad1time = True
        
        for name, p in self._model.named_parameters():
            t1 = torch.empty(*tuple(p.shape), dtype=p.dtype, device=self._backup_device)
            if (
                self._pin_memory
                and t1.device == torch.device("cpu")
                and torch.cuda.is_available()
            ):
                t1 = t1.pin_memory()
            self.original_parameters[name] = t1


            #here we add the initialization of original_grads and newest_grads
            t2 = torch.empty(*tuple(p.shape), dtype=p.dtype, device=self._backup_device)
            if (
                self._pin_memory
                and t2.device == torch.device("cpu")
                and torch.cuda.is_available()
            ):
                t2 = t2.pin_memory()
            self.original_grads[name] = t2

            t3 = torch.empty(*tuple(p.shape), dtype=p.dtype, device=self._backup_device)
            if (
                self._pin_memory
                and t3.device == torch.device("cpu")
                and torch.cuda.is_available()
            ):
                t3 = t3.pin_memory()
            self.newest_grads[name] = t3

        self._save_parameters(first_time=True)
 

    def get_offloaded_param(outer_optimizer: torch.optim.Optimizer):
        return [
            param.data.detach().clone().to("cpu")
            for group in outer_optimizer.param_groups
            for param in group["params"]
        ]

    def _save_parameters(self,first_time: bool = False) -> None:
        with torch.no_grad():
            #since the p.grad is None at first, we dont initialize original_grads here
            for name, p in self._model.named_parameters():
                self.original_parameters[name].copy_(p.data, non_blocking=True)
                if not first_time:
                    self.original_grads[name].copy_(p.grad,non_blocking=True)

    def _restore_parameters(self) -> None:
        with torch.no_grad():
            # TODO: consider running copy on a separate stream
            for name, p in self._model.named_parameters():
                p.data.copy_(self.original_parameters[name], non_blocking=False)

    def __enter__(self) -> "DiLoCo":
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
        self._local_step += 1
        if self._local_step >= self._sync_every:
            self.sync()

    def sync(self) -> None:
        """
        Synchronizes and averages the model weights across the manager.
        """
        self._manager.start_quorum()

        self._perform_sync()
        self._local_step = 0


    def _perform_sync(self) -> None:
        """
        Overrides the sync method to calculate the pseugradient, average them across the manager group, and
        step using the outer optimizer.
        """

        

        # this is the case where t>h
        if self.works!=[]:
            
            if self._off_load:
                for name, _ in self._model.named_parameters():
                    self.original_parameters[name] = self.original_parameters[name].to("cuda")
                    self.original_grads[name] = self.original_grads[name].to("cuda")


            #wait for the pesudogradient in t-h to come
            for work in self.works:
                work.wait()
            fresh_grad={}
            #recevie the pesudogradient in t-h
            for name, p in self.newest_grads.items():
                fresh_grad[name] = p

            #immediately send pesudogradient asynchronously
            for name, p in self._model.named_parameters():
                pseudogradient =p - self.original_parameters[name]
                self.newest_grads[name] = pseudogradient.detach().clone()
            new_works = self._average_grads(asy=True)
            self.works = new_works

            for name, p in self._model.named_parameters():
                fresh_grad[name] += 1/self.num_rg*(p.grad.detach().clone()-self.original_grads[name])
                p.grad = fresh_grad[name]
                #this grad will be used by outer_optimizer

        #this is the case where t mod h == 0
        else:
            if self._off_load:
                for name, _ in self._model.named_parameters():
                    self.original_parameters[name] = self.original_parameters[name].to("cuda")
                    self.original_grads[name] = self.original_grads[name].to("cuda")

            for name, p in self._model.named_parameters():
                pseudogradient =p - self.original_parameters[name]
                self.newest_grads[name] = pseudogradient.detach().clone()
                self.original_grads[name] = p.grad.detach().clone()
                #we save the firstround's pesudogradient and initialize the original_grads here

            if self._off_load:
                for name, _ in self._model.named_parameters():
                    self.original_parameters[name] = self.original_parameters[name].to("cpu")
                    self.original_grads[name] = self.original_grads[name].to("cpu")
            new_works = self._average_grads(asy=True)
            #outer asyschronization here
            self.works = new_works
            return
            #return here, not using outer optimizer

        # Restore the parameters back to the previous state
        self._restore_parameters()
        

        if self._manager.should_commit():
            # Use the outer optimizer to update the model parameters
            self._outer_optimizer.step()
            self._save_parameters()  #also including original_grads,newest_grads


        if self._off_load:
            for name, _ in self._model.named_parameters():
                self.original_parameters[name] = self.original_parameters[name].to("cpu")
                self.original_grads[name] = self.original_grads[name].to("cpu")

        self._outer_optimizer.zero_grad()
        


    def _average_grads(self,asy:bool = False) :
        """
        Average the gradients across the diloco group.
        """
        works = []
        if not asy:
            for p in self._model.parameters():
                # Perform allreduce on the pseudogradients
                assert p.grad is not None
                work = self._manager.allreduce(p.grad)
                works.append(work)
            # Wait for all allreduce operations to complete

            for work in works:
                work.wait()
        else:
            for name,p in self.newest_grads.items():
                # Perform allreduce on the pseudogradients
                assert p is not None
                work = self._manager.allreduce(p)
                works.append(work)
            return works
