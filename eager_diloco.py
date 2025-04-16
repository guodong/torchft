# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from datetime import timedelta
import time
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchft import (
    DistributedDataParallel,
    DistributedSampler,
    Manager,
    Optimizer,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
    
)

from torchft.local_sgd_v2 import DiLoCo

logging.basicConfig(level=logging.INFO)


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    NUM_REPLICA_GROUPS = int(os.environ.get("NUM_REPLICA_GROUPS", 2))

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="../cifar", train=True, download=True, transform=transform
    )

    # This shards the training set across all ranks and replica groups. We manage
    # the dataloaders on a per replica group basis with the assumption that the
    # majority of groups will be available so few batches will be dropped.
    sampler = DistributedSampler(
        trainset,
        replica_group=REPLICA_GROUP_ID,
        num_replica_groups=NUM_REPLICA_GROUPS,
        rank=0,
        # for DDP we can use replica groups of size 1, FSDP/PP/CP would need more.
        num_replicas=1,
        shuffle=True,
    )

    # This uses the torchdata StatefulDataLoader to be able to checkpoint and
    # restore the per worker dataloader position.
    trainloader = StatefulDataLoader(
        trainset, batch_size=16, num_workers=2, sampler=sampler
    )

    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        inner_optimizer.load_state_dict(state_dict["optim_inner"])
        outer_optimizer.load_state_dict(state_dict["optim_outer"])

    def state_dict():
        return {
            "model": m.state_dict(),
            "optim_inner": inner_optimizer.state_dict(),
            "optim_outer": outer_optimizer.state_dict()
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pg = (
        ProcessGroupBabyNCCL(
            timeout=timedelta(seconds=5),
        )
        if torch.cuda.is_available()
        else ProcessGroupGloo(timeout=timedelta(seconds=5))
    )

    manager = Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"train_ddp_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=10),
        use_async_quorum=False
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
    inner_optimizer =torch.optim.AdamW(
            m.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )

    outer_optimizer =torch.optim.SGD(
            m.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )
    criterion = nn.CrossEntropyLoss()
    manager.start_quorum()

    with DiLoCo(
            manager, m, inner_optimizer, outer_optimizer, sync_every=2,backup_device=torch.device("cpu"),off_load=True,num_rg=NUM_REPLICA_GROUPS
        ) as diloco:
        while True:
            for i, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                inner_optimizer.zero_grad()
                out = m(inputs)
                loss = criterion(out, labels)
                loss.backward()
                inner_optimizer.step()


                print(f"[{manager.current_step()}] loss = {loss.item()}")
                # if manager.current_step() >= 10000:
                #     exit()
                time.sleep(2)

if __name__ == "__main__":
    main()
