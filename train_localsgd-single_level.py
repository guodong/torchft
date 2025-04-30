#!/usr/bin/env python3
"""
Fault‑tolerant LocalSGD example that mirrors the logic in
tests/local_sgd_train_loop but is ready for multi‑node execution.

Launch with torchrun; the script picks everything it needs from the
environment that torchrun and your launcher already set.
"""
import argparse
import os
import time
from datetime import timedelta
from typing import Dict, Any
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

@record
def main(sleep_time: float = 1.0, sync_every: int = 2, steps_to_run: int = 100) -> None:

    print(f"sleep_time: {sleep_time}, sync_every: {sync_every}, steps_to_run: {steps_to_run}")

    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    NUM_REPLICA_GROUPS_LOCAL_CLUSTER = int(os.environ.get("NUM_REPLICA_GROUPS_LOCAL_CLUSTER", 2))

    mean = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    std = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)

    def normalize(x: torch.Tensor) -> torch.Tensor:
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

    transform = normalize
    trainset = RandomCIFAR10(transform=transform)

    # This shards the training set across all ranks and replica groups. We manage
    # the dataloaders on a per replica group basis with the assumption that the
    # majority of groups will be available so few batches will be dropped.
    sampler = DistributedSampler(
        trainset,
        replica_group=REPLICA_GROUP_ID,
        num_replica_groups=NUM_REPLICA_GROUPS_LOCAL_CLUSTER,
        rank=0,
        # for DDP we can use replica groups of size 1, FSDP/PP/CP would need more.
        num_replicas=1,
        shuffle=True,
    )

    # This uses the torchdata StatefulDataLoader to be able to checkpoint and
    # restore the per worker dataloader position.
    trainloader = StatefulDataLoader(
        trainset, batch_size=64, num_workers=2, sampler=sampler
    )


    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optim"])

    def state_dict():
        return {
            "model": m.state_dict(),
            "optim": optimizer.state_dict(),
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pg = (
        ProcessGroupBabyNCCL(
            timeout=timedelta(seconds=5),
        )
        if torch.cuda.is_available()
        else ProcessGroupGloo(timeout=timedelta(seconds=5))
    )

    lighthouse_addr_global = os.environ.get("TORCHFT_LIGHTHOUSE_GLOBAL")
    manager = Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"train_localsgd_{REPLICA_GROUP_ID}",
        lighthouse_addr=lighthouse_addr_global,
        timeout=timedelta(seconds=10),
    )

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

    m = Net().to(device)
    optimizer = optim.Adam(m.parameters())
    criterion = nn.CrossEntropyLoss()

    with LocalSGD(manager, m, optimizer, sync_every=sync_every):
        while True:
            x = torch.randn(2, 3, 32, 32).to(device)          # N, C, H, W for Conv2d
            y = torch.randint(10, (2,), device=device)        # match Net’s 10‑class output

            # optimizer.zero_grad()
            # loss = criterion(m(x), y)
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()
            loss = criterion(m(x), y)
            loss.backward()
            optimizer.step()

            if manager.current_step() >= steps_to_run:
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