#!/usr/bin/env python3
"""
Fault‑tolerant LocalSGD example that mirrors the logic in
tests/inner_sgd_train_loop but is ready for multi‑node execution.

Launch with torchrun; the script picks everything it needs from the
environment that torchrun and your launcher already set.
"""
import argparse
from functools import partial
import os
from datetime import timedelta
from time import sleep
from typing import Dict, Optional

from urllib3 import PoolManager
from torch.distributed.elastic.multiprocessing.errors import record
from torchdata.stateful_dataloader import StatefulDataLoader
import torch
from torch import nn
import torch.nn.functional as F
from torchft.local_sgd_ICT import DiLoCo_ICT

from torchft import (
    DistributedSampler,
    Manager,
    Optimizer,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
)

def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    return (x - mean) / std

class RandomCIFAR10(torch.utils.data.Dataset):
    """Generates random 32×32 RGB images with CIFAR‑10 label space."""
    def __init__(self, size: int = 50_000, transform=None, num_classes: int = 10):
        self.size = size
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        img = torch.randn(3, 32, 32)
        label = torch.randint(0, self.num_classes, (1,)).item()
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@record
def main(sleep_time: float = 1.0, sync_every: int = 2, steps_to_run: int = 100, debug: bool = True) -> None:
    def debug_print(*args, **kwargs) -> None:
        if debug:
            print("[FROM MAIN:] ", *args, **kwargs)
        return
    
    debug_print(f"sleep_time: {sleep_time}, sync_every: {sync_every}, steps_to_run: {steps_to_run}")
    MIN_SIZE_GLOBAL = 1
    MIN_SIZE_LOCAL = 1
    # Local cluster group ID and number
    CLUSTER_GROUP_ID = int(os.environ.get("CLUSTER_GROUP_ID", 0)) # ID of the inner cluster
    NUM_CLUSTERS = int(os.environ.get("NUM_CLUSTERS", 2)) # number of physical clusters
    # Local replica group ID and number within the inner cluster
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0)) # ID of the inner replica group
    leader = (REPLICA_GROUP_ID == 0)
    NUM_REPLICA_GROUPS_LOCAL_CLUSTER = int(os.environ.get("NUM_REPLICA_GROUPS_LOCAL_CLUSTER", 2)) # DP groups *inside* each cluster
    # Local rank within the inner replica group
    RANK = int(os.environ.get("RANK")) # This is set by torchrun, it is the rank within the inner replica group of the current process
    # Global replica ID and number
    GLOBAL_REPLICA_ID = CLUSTER_GROUP_ID * NUM_REPLICA_GROUPS_LOCAL_CLUSTER + REPLICA_GROUP_ID # Global replica ID
    GLOBAL_REPLICA_NUM = NUM_REPLICA_GROUPS_LOCAL_CLUSTER * NUM_CLUSTERS # Total number of outer replicas
    debug_print(
        f"[Config] INNER_RANK={RANK}, CLUSTER_GROUP_ID={CLUSTER_GROUP_ID}, "
        f"REPLICA_GROUP_ID={REPLICA_GROUP_ID}, "
        f"NUM_CLUSTERS={NUM_CLUSTERS}, NUM_REPLICA_GROUPS_LOCAL_CLUSTER={NUM_REPLICA_GROUPS_LOCAL_CLUSTER}, "
        f"GLOBAL_REPLICA_NUM={GLOBAL_REPLICA_NUM}, "
        f"MIN_SIZE_GLOBAL={MIN_SIZE_GLOBAL}, MIN_SIZE_LOCAL={MIN_SIZE_LOCAL}"
    )

    lighthouse_addr_inner = os.environ["TORCHFT_LIGHTHOUSE_LOCAL"]
    lighthouse_addr_outer = os.environ["TORCHFT_LIGHTHOUSE_GLOBAL"]
    store_addr_inner = os.environ["MASTER_ADDR_LOCAL"]
    store_addr_outer = os.environ["MASTER_ADDR_CLUSTER"]
    store_port_inner = int(os.environ["MASTER_PORT_LOCAL"])
    store_port_outer = int(os.environ["MASTER_PORT_CLUSTER"])
    debug_print(f"lighthouse_addr_inner: {lighthouse_addr_inner}, lighthouse_addr_outer: {lighthouse_addr_outer}")
    debug_print(f"store_addr_inner: {store_addr_inner}, store_port_inner: {store_port_inner}")
    debug_print(f"store_addr_outer: {store_addr_outer}, store_port_outer: {store_port_outer}")
    
    rank_inner = RANK
    outer_manager_rank = REPLICA_GROUP_ID # This is the rank of the outer manager.
    world_size_replica_group = int(os.environ["WORLD_SIZE"]) # Number of gpus in the replica group. Handled by torchrun
    world_size_cluster = 1 # This is always 1, because at the cluster level we are only doing DDP (for now)

    debug_print(f"world_size_replica_group: {world_size_replica_group}, world_size_cluster: {world_size_cluster}")

    mean = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    std = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    transform = partial(normalize, mean=mean, std=std)
    trainset = RandomCIFAR10(transform=transform)

    # This shards the training set across all ranks and replica groups. We manage
    # the dataloaders on a per replica group basis with the assumption that the
    # majority of groups will be available so few batches will be dropped.
    sampler = DistributedSampler(
        trainset,
        replica_group=GLOBAL_REPLICA_ID, # the group ID (0-num_replica_groups) to use for this shard of data.
        num_replica_groups=GLOBAL_REPLICA_NUM, # the max number of outer replica groups
        rank=RANK, # the inner group rank
        # for DDP we can use replica groups of size 1, FSDP/PP/CP would need more.
        num_replicas=1, # the inner group world size
        shuffle=True,
    )

    # This uses the torchdata StatefulDataLoader to be able to checkpoint and
    # restore the per worker dataloader position.
    trainloader = StatefulDataLoader(
        trainset, batch_size=64, num_workers=2, sampler=sampler
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pg_inner = (
        ProcessGroupBabyNCCL(
            timeout=timedelta(seconds=5),
        )
        if torch.cuda.is_available()
        else ProcessGroupGloo(timeout=timedelta(seconds=5))
    )
    pg_outer = (
        ProcessGroupBabyNCCL(
            timeout=timedelta(seconds=5),
        )
        if torch.cuda.is_available()
        else ProcessGroupGloo(timeout=timedelta(seconds=5))
    )

    m = Net().to(device)
    inner_optimizer = torch.optim.AdamW(
        m.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
    )
    outer_optimizer = torch.optim.SGD(
        m.parameters(), lr=0.7, momentum=0.9, nesterov=True
    )

    debug_print(f"m: {m}")
    debug_print(f"inner_optimizer: {inner_optimizer}")
    debug_print(f"outer_optimizer: {outer_optimizer}")

    def inner_load_state_dict(state_dict: Dict[str, Dict[str, object]]) -> None:
        m.load_state_dict(state_dict["model"])
        inner_optimizer.load_state_dict(state_dict["optim"])

    def inner_state_dict() -> Dict[str, Dict[str, object]]:
        return {
            "model": m.state_dict(),
            "optim": inner_optimizer.state_dict(),
        }


    debug_print(f"pg_outer: {pg_outer}")
    debug_print(f"MIN_SIZE_GLOBAL: {MIN_SIZE_GLOBAL}")
    debug_print("inner_load_state_dict: ", inner_load_state_dict)
    debug_print(f"replica_id: {f'train_outermanager_{CLUSTER_GROUP_ID}'}")
    debug_print(f"timeout: {timedelta(seconds=10)}")
    debug_print(f"lighthouse_addr: {lighthouse_addr_outer}")
    debug_print(f"store_addr: {store_addr_outer}")
    debug_print(f"store_port: {store_port_outer}")
    debug_print(f"rank: {outer_manager_rank}")
    debug_print(f"world_size_cluster: {world_size_cluster}")

    inner_manager = Manager(
        pg=pg_inner,
        min_replica_size=MIN_SIZE_LOCAL,
        load_state_dict=inner_load_state_dict,
        state_dict=inner_state_dict,
        replica_id=f"train_outermanager_{CLUSTER_GROUP_ID}_{REPLICA_GROUP_ID}", #TODO: Do we need to have the cluster group id here?
        timeout=timedelta(seconds=10),
        # Different varaibles for outer and inner managers.
        lighthouse_addr=lighthouse_addr_inner,
        store_addr=store_addr_inner,
        store_port=store_port_inner,
        rank=rank_inner,
        world_size=world_size_replica_group,
    )

    outer_manager: Optional[Manager] = None

    if leader:
        # Define the outer manager, only needed by the leader
        def outer_load_state_dict(state_dict: Dict[str, Dict[str, object]]) -> None:
            m.load_state_dict(state_dict["model"])
            m.to(device)
            diloco.original_parameters = state_dict["original_params"]
            for name in diloco.original_parameters.keys():
                diloco.original_parameters[name] = diloco.original_parameters[name].to(
                    device
                )
            inner_optimizer.load_state_dict(state_dict["inner_optim"])
            outer_optimizer.load_state_dict(state_dict["outer_optim"])
            
        def outer_state_dict() -> Dict[str, Dict[str, object]]:  # pyre-ignore[53]
            return {
                "model": m.state_dict(),
                "original_params": diloco.original_parameters,
                "inner_optim": inner_optimizer.state_dict(),
                "outer_optim": outer_optimizer.state_dict(),
            }
        outer_manager = Manager(
            pg=pg_outer,
            min_replica_size=MIN_SIZE_GLOBAL,
            load_state_dict=outer_load_state_dict,
            state_dict=outer_state_dict,
            replica_id=f"train_outermanager_{CLUSTER_GROUP_ID}",
            timeout=timedelta(seconds=10),

            # Different variables for outer and inner managers.
            lighthouse_addr=lighthouse_addr_outer,
            store_addr=store_addr_outer,
            store_port=store_port_outer,
            rank=outer_manager_rank,
            world_size=world_size_cluster,
        )

        debug_print("outer_load_state_dict: ", outer_load_state_dict)
        debug_print("outer_state_dict: ", outer_state_dict)
        debug_print(f"outer_manager: {outer_manager}")
        outer_manager._use_async_quorum = False

    managed_inner_optimizer = Optimizer(manager=inner_manager, optim=inner_optimizer)
    criterion = nn.CrossEntropyLoss()

    current_step = 0
    with DiLoCo_ICT(
        outer_manager=outer_manager,
        inner_manager=inner_manager,
        model=m,
        inner_optimizer=inner_optimizer,
        outer_optimizer=outer_optimizer,
        sync_every=sync_every,
        backup_device=device, #TODO: Make this CPU for CPU offloading. Currently using GPU for backup device.
        device=device,
        debug=True,
    ) as diloco:
        # If REPLICA_GROUP_ID == 0, then sync with the outer manager.
        # After syncing with the outer manager, needs to broadcast the model parameters to the other inner replica groups.
        for x, y in trainloader:
            current_step += 1
            if outer_manager is not None:
                debug_print(f"outer_manager.current_step(): {outer_manager.current_step()}")
            # debug_print(f"inner_manager.current_step(): {inner_manager.current_step()}")
            x = x.to(device)
            y = y.to(device)
            output = m(x)
            loss = criterion(output, y)
            managed_inner_optimizer.zero_grad()
            loss.backward()
            managed_inner_optimizer.step()

            if current_step >= steps_to_run:
                # complete training
                exit()

            sleep(sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sleep_time",
        type=float,
        nargs="?",
        default=1.0,
        help="Seconds to sleep per training iteration (default: 1.0)",
    )
    parser.add_argument(
        "sync_every",
        type=int,
        nargs="?",
        default=2,
        help="Sync every N steps (default: 2)",
    )
    parser.add_argument(
        "steps_to_run",
        type=int,
        nargs="?",
        default=1000,
        help="Number of steps to run (default: 100)",
    )
    args = parser.parse_args()
    main(args.sleep_time, args.sync_every, args.steps_to_run)