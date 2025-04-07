import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.distributed as dist
import wandb
from cyclopts import App
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
import logging
import os
from datetime import timedelta
import torch.nn.functional as F
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

from torchft.local_sgd import DiLoCo
from torchft.process_group import ManagedProcessGroup, ft_init_device_mesh

from torchft.checkpointmanager import CkptManager

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy

logging.basicConfig(level=logging.INFO)

# def get_offloaded_param(outer_optimizer: torch.optim.Optimizer):
#     return [
#         param.data.detach().clone().to("cpu")
#         for group in outer_optimizer.param_groups
#         for param in group["params"]
#     ]


# app = App()


# @app.default
def main() -> None:
    batch_size: int = 512
    per_device_train_batch_size: int = 32
    seq_length: int = 1024
    warmup_steps: int = 1000
    total_steps: int = 88_000
    project: str = "diloco"
    config_path: str = "/srv/apps/danny/config/config_14m.json"
    lr: float = 4e-4
    outer_lr: float = 0.7
    local_steps: int = 6
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    NUM_REPLICA_GROUPS = int(os.environ.get("NUM_REPLICA_GROUPS", 2))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_replicas = int(os.environ.get("NUM_REPLICAS", 1))
    
    assert batch_size % per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size / per_device_train_batch_size
    world_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if local_rank == 0:
    #     wandb.init(project=project)

    # Load model configuration and tokenizer
    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path=config_path)

    m = LlamaForCausalLM(config).to(device)

    # for param in model.parameters():
    #     # this make sure all device have the same weight init
    #     dist.broadcast(param.data, src=0)

    # Setup optimizers
    

    tokenizer = AutoTokenizer.from_pretrained(
        "/srv/apps/danny/models/mistralai/Mistral-7B-v0.1", use_fast=True
    )
    tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

    ds = load_dataset("/srv/apps/danny/data/PrimeIntellect/c4-tiny")

    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=seq_length)
        return outputs

    tokenized_datasets = ds.map(
        tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"]
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    sampler = DistributedSampler(
        tokenized_datasets["train"],
        replica_group=REPLICA_GROUP_ID,
        num_replica_groups=NUM_REPLICA_GROUPS,
        rank=local_rank,
        num_replicas=num_replicas,
        shuffle=True,
    )
    trainloader = StatefulDataLoader(
        tokenized_datasets["train"], batch_size=per_device_train_batch_size, num_workers=2, sampler=sampler, collate_fn=data_collator
    )

    


    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        inner_optimizer.load_state_dict(state_dict["optim_inner"])
        outer_optimizer.load_state_dict(state_dict["optim_outer"])

    def state_dict():
        return {
            "model": m.state_dict(),
            "optim_inner": inner_optimizer.state_dict(),
            "optim_outer": outer_optimizer.state_dict()}
    
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

    
    device_mesh = ft_init_device_mesh(
            device_type="cuda",
            mesh_shape=(NUM_REPLICA_GROUPS, num_replicas),
            mesh_dim_names=("dp_replicate", "dp_shard"),
            replicate_dim=local_rank,
            manager=manager,
        )   #如果有新的节点加入，或有节点退出，这个会被重新建构吗？
    

    m = fully_shard(m, mesh=device_mesh, offload_policy=CPUOffloadPolicy(pin_memory=True))
    #need to deal with the problem of dtensor and torch.tensor in local_sgd.py

    inner_optimizer = torch.optim.AdamW(
        m.parameters(), weight_decay=0.1, lr=lr, betas=(0.9, 0.95)
    )
    outer_optimizer = torch.optim.SGD(
        m.parameters(), lr=outer_lr, momentum=0.9, nesterov=True
    )

    # params_offloaded = get_offloaded_param(outer_optimizer)

    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    #prime version
    # ckpt_manager=CkptManager(config=config.ckpt,
    #         model=m,
    #         optimizer=inner_optimizer,
    #         scheduler=scheduler,
    #         dataloader=trainloader,
    #         training_progress=training_progress,
    #         data_rank=config.data.data_rank,
    #         diloco_offloaded_optimizer=diloco.outer_optimizer if config.diloco is not None else None, 
    #         diloco_offloaded_param_list=diloco.param_list_cpu if config.diloco is not None else None,  )

    #torchtitan version
    ckpt_manager = CkptManager(
        dataloader=trainloader,
        model_parts=m,
        optimizers=optimizers,
        lr_schedulers=scheduler,
        states={"train_state": train_state},
        job_config=job_config,
    )

    m.train()
    manager.start_quorum() #this line must be added, otherwise, "pseudogradient = p.data - self.original_parameters[name]" could cause semantic error
    loss_batch = 0
    with DiLoCo(
            manager, m, inner_optimizer, outer_optimizer, sync_every=local_steps,backup_device=torch.device("cpu"),off_load=True
        ) as diloco:

        for step, batch in enumerate(iterable=trainloader):
            if step == 0:
                inner_optimizer.zero_grad()
            real_step = (step + 1) // gradient_accumulation_steps
            step_within_grad_acc = (step + 1) % gradient_accumulation_steps
            if step_within_grad_acc == 0:
                inner_optimizer.zero_grad()
            batch = batch.to(device)
            out = m(**batch)
            loss = out.loss / gradient_accumulation_steps
            loss_batch += loss.detach()
            # loss = criterion(out, labels)
            loss.backward()
            if step_within_grad_acc == 0:
                
                if real_step % local_steps == 0:
                    if local_rank == 0:
                        print(f"perform outer step at step {real_step}")
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)  # gradient clipping
                inner_optimizer.step()
                scheduler.step()

                if local_rank == 0:
                    print(
                        f"step: {real_step}, loss: {loss_batch.item()}, lr {[group['lr'] for group in inner_optimizer.param_groups][0]}"
                    )
                    loss_batch = 0
    print("Training completed.")
    # wandb.finish()            
                

if __name__ == "__main__":
    main()
