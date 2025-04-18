Lighthouse Setup:

```python
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000 --bind "127.0.0.1:29510"
```

Node1 Setup:

export REPLICA_GROUP_ID=0
export NUM_REPLICA_GROUPS=3
export TORCHFT_LIGHTHOUSE=http://sz-k8s-master:29510
CUDA_VISIBLE_DEVICES=0 TORCHFT_LIGHTHOUSE=http://127.0.0.1:29510 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29502 train_ddp.py

Node2 Setup:

export REPLICA_GROUP_ID=1
export NUM_REPLICA_GROUPS=3
CUDA_VISIBLE_DEVICES=1 TORCHFT_LIGHTHOUSE=http://127.0.0.1:29510 torchrun --nnodes=1 --nproc_per_node=1 -master_port=29503 train_ddp.py