#!/usr/bin/env python3
"""
Fault‑tolerant LocalSGD example that mirrors the logic in
tests/local_sgd_train_loop but is ready for multi‑node execution.

Launch with torchrun; the script picks everything it needs from the
environment that torchrun and your launcher already set.
"""
import argparse
from functools import partial
import os
from datetime import timedelta
from time import sleep
from torch.distributed.elastic.multiprocessing.errors import record
from torchdata.stateful_dataloader import StatefulDataLoader
import torch
from torch import nn, optim
import torch.nn.functional as F

from torchft import (
    DistributedSampler,
    Manager,
    Optimizer,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
)
from torchft.local_sgd import LocalSGD

def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    return (x - mean) / std

def load_state_dict(state_dict, optimizer):
    m.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optim"])

def state_dict(optimizer):
    return {
        "model": m.state_dict(),
        "optim": optimizer.state_dict(),
    }

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
def main(sleep_time: float = 1.0, sync_every: int = 2, steps_to_run: int = 100) -> None:
    print(f"sleep_time: {sleep_time}, sync_every: {sync_every}, steps_to_run: {steps_to_run}")
    MIN_SIZE_GLOBAL = 1
    MIN_SIZE_LOCAL = 1
    # Local cluster group ID and number
    CLUSTER_GROUP_ID = int(os.environ.get("CLUSTER_GROUP_ID", 0)) # ID of the local cluster
    NUM_CLUSTERS = int(os.environ.get("NUM_CLUSTERS", 2)) # number of physical clusters
    # Local replica group ID and number within the local cluster
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0)) # ID of the local replica group
    NUM_REPLICA_GROUPS_LOCAL_CLUSTER = int(os.environ.get("NUM_REPLICA_GROUPS_LOCAL_CLUSTER", 2)) # DP groups *inside* each cluster
    # Local rank within the local replica group
    RANK = int(os.environ.get("RANK")) # This is set by torchrun, it is the rank within the local replica group of the current process
    # Global replica ID and number
    GLOBAL_REPLICA_ID = CLUSTER_GROUP_ID * NUM_REPLICA_GROUPS_LOCAL_CLUSTER + REPLICA_GROUP_ID # Global replica ID
    GLOBAL_REPLICA_NUM = NUM_REPLICA_GROUPS_LOCAL_CLUSTER * NUM_CLUSTERS # Total number of global replicas
    print(
        f"[Config] RANK={RANK}, CLUSTER_GROUP_ID={CLUSTER_GROUP_ID}, "
        f"REPLICA_GROUP_ID={REPLICA_GROUP_ID}, "
        f"NUM_CLUSTERS={NUM_CLUSTERS}, NUM_REPLICA_GROUPS_LOCAL_CLUSTER={NUM_REPLICA_GROUPS_LOCAL_CLUSTER}, "
        f"GLOBAL_REPLICA_NUM={GLOBAL_REPLICA_NUM}, "
        f"MIN_SIZE_GLOBAL={MIN_SIZE_GLOBAL}, MIN_SIZE_LOCAL={MIN_SIZE_LOCAL}"
    )

    lighthouse_addr_local = os.environ["TORCHFT_LIGHTHOUSE_LOCAL"]
    lighthouse_addr_global = os.environ["TORCHFT_LIGHTHOUSE_GLOBAL"]
    store_addr_local = os.environ["MASTER_ADDR_LOCAL"]
    store_addr_global = os.environ["MASTER_ADDR_GLOBAL"]
    store_port_local = int(os.environ["MASTER_PORT_LOCAL"])
    store_port_global = int(os.environ["MASTER_PORT_GLOBAL"])
    print(f"lighthouse_addr_local: {lighthouse_addr_local}, lighthouse_addr_global: {lighthouse_addr_global}")
    print(f"store_addr_local: {store_addr_local}, store_port_local: {store_port_local}")
    print(f"store_addr_global: {store_addr_global}, store_port_global: {store_port_global}")
    
    rank_local = RANK
    rank_global = CLUSTER_GROUP_ID
    world_size_replica_group = int(os.environ["WORLD_SIZE"]) # Number of gpus in the replica group. Handled by torchrun
    world_size_cluster = 1 # This is always 1, because at the cluster level we are only doing DDP (for now)

    print(f"world_size_replica_group: {world_size_replica_group}, world_size_cluster: {world_size_cluster}")

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
        num_replica_groups=GLOBAL_REPLICA_NUM, # the max number of global replica groups
        rank=RANK, # the local group rank
        # for DDP we can use replica groups of size 1, FSDP/PP/CP would need more.
        num_replicas=1, # the local group world size
        shuffle=True,
    )
    """
        Args:
            data: the dataset to use
            replica_group: the group ID (0-num_replica_groups) to use for this shard of data.
    """

    # This uses the torchdata StatefulDataLoader to be able to checkpoint and
    # restore the per worker dataloader position.
    trainloader = StatefulDataLoader(
        trainset, batch_size=64, num_workers=2, sampler=sampler
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pg_local = (
        ProcessGroupBabyNCCL(
            timeout=timedelta(seconds=5),
        )
        if torch.cuda.is_available()
        else ProcessGroupGloo(timeout=timedelta(seconds=5))
    )
    pg_global    = (
        ProcessGroupBabyNCCL(
            timeout=timedelta(seconds=5),
        )
        if torch.cuda.is_available()
        else ProcessGroupGloo(timeout=timedelta(seconds=5))
    )

    m = Net().to(device)
    inner_optimizer = optim.Adam(m.parameters())
    outer_optimizer = optim.Adam(m.parameters())

    print(f"m: {m}")
    print(f"inner_optimizer: {inner_optimizer}")
    print(f"outer_optimizer: {outer_optimizer}")

    manager_local = Manager(
        pg=pg_local,
        min_replica_size=MIN_SIZE_LOCAL,
        load_state_dict=partial(load_state_dict, optimizer=inner_optimizer),
        state_dict=partial(state_dict, optimizer=inner_optimizer),
        replica_id=f"train_localsgd_{CLUSTER_GROUP_ID}_{REPLICA_GROUP_ID}", #TODO: Do we need to have the cluster group id here?
        timeout=timedelta(seconds=10),
        # Different varaibles for global and local managers.
        lighthouse_addr=lighthouse_addr_local,
        store_addr=store_addr_local,
        store_port=store_port_local,
        rank=rank_local,
        world_size=world_size_replica_group,
    )

    print(f"manager_local: {manager_local}")

    print(f"pg_global: {pg_global}")
    print(f"MIN_SIZE_GLOBAL: {MIN_SIZE_GLOBAL}")
    print(f"load_state_dict: {partial(load_state_dict, optimizer=outer_optimizer)}")
    print(f"state_dict: {partial(state_dict, optimizer=outer_optimizer)}")
    print(f"replica_id: {f'train_localsgd_{CLUSTER_GROUP_ID}'}")
    print(f"timeout: {timedelta(seconds=10)}")
    print(f"lighthouse_addr: {lighthouse_addr_global}")
    print(f"store_addr: {store_addr_global}")
    print(f"store_port: {store_port_global}")
    print(f"rank: {rank_global}")
    print(f"world_size: {world_size_cluster}")
    
    manager_global = Manager(
        pg=pg_global,
        min_replica_size=MIN_SIZE_GLOBAL,
        load_state_dict=partial(load_state_dict, optimizer=outer_optimizer),
        state_dict=partial(state_dict, optimizer=outer_optimizer),
        replica_id=f"train_localsgd_{CLUSTER_GROUP_ID}",
        timeout=timedelta(seconds=10),
        # Different varaibles for global and local managers.
        lighthouse_addr=lighthouse_addr_global,
        store_addr=store_addr_global,
        store_port=store_port_global,
        rank=rank_global,
        world_size=world_size_cluster,
    )

    print(f"manager_global: {manager_global}")

    managed_inner_optimizer = Optimizer(manager=manager_local, optim=inner_optimizer) # TODO: Move to DiLoCo, where the global and local optimizers are different
    criterion = nn.CrossEntropyLoss()

    with LocalSGD(manager_global, m, optimizer=outer_optimizer, sync_every=sync_every):
    # with DiLoCo(manager_global, m, inner_optimizer=managed_inner_optimizer, outer_optimizer=outer_optimizer, sync_every=sync_every):
        while True:
            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device)
                managed_inner_optimizer.zero_grad()
                output = m(x)
                loss = criterion(output, y)
                loss.backward()
                managed_inner_optimizer.step()

                if manager_global.current_step() >= steps_to_run:
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