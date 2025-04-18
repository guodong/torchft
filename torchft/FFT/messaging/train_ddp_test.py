# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple training example using ManagerFFT with synthetic data.

This example demonstrates how to use ManagerFFT in a distributed training setup
without requiring torchvision or other external dataset dependencies.

Usage:
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train_ddp_test.py
"""

import logging
import os
import sys
from datetime import timedelta
from time import sleep
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DistributedSampler, StatefulDataLoader
from torch.distributed.elastic.multiprocessing.errors import record

from torchft import (
    DistributedDataParallel,
    Optimizer,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
)
from torchft.manager_fft import ManagerFFT

logging.basicConfig(level=logging.INFO)


# Simple synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, size=10000, dims=32, num_classes=10):
        self.size = size
        self.dims = dims
        self.num_classes = num_classes
        
        # Generate random data and labels
        self.data = torch.randn(size, 3, dims, dims)
        self.labels = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    NUM_REPLICA_GROUPS = int(os.environ.get("NUM_REPLICA_GROUPS", 2))
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    
    # Create synthetic dataset
    trainset = SyntheticDataset(size=5000, dims=32, num_classes=10)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        trainset,
        replica_group=REPLICA_GROUP_ID,
        num_replica_groups=NUM_REPLICA_GROUPS,
        rank=0,
        # for DDP we can use replica groups of size 1, FSDP/PP/CP would need more.
        num_replicas=1,
        shuffle=True,
    )
    
    # Create dataloader
    trainloader = StatefulDataLoader(
        trainset, batch_size=64, num_workers=2, sampler=sampler
    )
    
    # Model state dict functions
    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optim"])

    def state_dict():
        return {
            "model": m.state_dict(),
            "optim": optimizer.state_dict(),
        }

    # Setup device and process group
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pg = (
        ProcessGroupBabyNCCL(
            timeout=timedelta(seconds=5),
        )
        if torch.cuda.is_available()
        else ProcessGroupGloo(timeout=timedelta(seconds=5))
    )

    # Create ManagerFFT instead of Manager 
    manager = ManagerFFT(
        pg=pg,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"train_ddp_test_{REPLICA_GROUP_ID}",
        rank=RANK,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=10),
        error_bus_host='127.0.0.1',
        error_bus_port=22223,
        error_bus_debug=True
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
    m = DistributedDataParallel(manager, m)
    optimizer = Optimizer(manager, optim.AdamW(m.parameters()))
    criterion = nn.CrossEntropyLoss()

    print(m)

    # You can use an epoch based training but with faults it's easier to use step
    # based training.
    while True: 
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # must be called at the beginning of each train loop
            # Quorum computation is triggered here but only needed in the backwards pass.
            optimizer.zero_grad()

            out = m(inputs)
            loss = criterion(out, labels)

            # Gradient allreduce overlaps with the backwards pass.
            loss.backward()

            # must be called at the end of the train loop
            # This may not actually step the optimizer if an error occured during grad allreduce.
            optimizer.step()

            if manager.current_step() % 100 == 0:
                print(f"[{manager.current_step()}] loss = {loss.item()}")

            if manager.current_step() >= 10000:
                # complete training
                exit()
            
            sleep(2)
        
if __name__ == "__main__":
    main() 