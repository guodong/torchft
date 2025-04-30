# ICT Design Doct

ICT is the protocol for Inter-Cluster Machine Learning Transport. Its aim is to address the following difficulty in multi-cluster training:

Long delays, low bandwidth, heterogenous sizes, fault tolerance

Its core concepts are the following:

1. A hierarchy of transport layers. There is a logical separation in each transport layer that enables each transport layer to use different optimizers, synchronization policies, and hyperparameters
2. Autoconfiguration of training hyperparameters
    - A controller that reconfigures the routing within each transport layer to a) Enable fault tolerant routing, and b) minimize communication overhead
        - All the more important in multi-cluster settings with heterogenous networks, long-delays and low-bandwidth
    - Autoconfiguration of dataset partitioning and training hyperparameters
    - 

There are also advanced optimziations that one can configure for ICT:
3. Maximum asynchronous execution
    - All communication is done asynchronously to overlap with computation.
    - Uses new research from multi-cluster training dynamics to take advantage of computation time even when gradients are being synced cross-cluster
        - E.g. Streaming DiLoCo



A simple example of the use case is LocalSGD. In LocalSGD, every step we will call `manager_local.allreduce(params)` to synchronize the gradients within the local group. Then, after every `sync_every` steps, we will call `manager_global.all_reduce(params)` to synchronize the global gradients.

Below is a diagram of a scenario that ICT operates in:
- Centralized controller
- Don't introduce 3rd party dependency unless necessary


```txt
                        ┌────────────┐
                        │ Lighthouse │
                        └────────────┘
                                │
                                │
                        ┌────────────┐
                        │   Store    │
                        └────────────┘
                    ───────┼──────┬──────┼──────
                        │      │      │
                        │      │      │
      ┌────────────┐     ┌────────────┐     ┌────────────┐
      │ Lighthouse │     │ Lighthouse │     │ Lighthouse │
      └────────────┘     └────────────┘     └────────────┘
            │                  │                  │
            │                  │                  │
      ┌────────────┐     ┌────────────┐     ┌────────────┐
      │   Store    │     │   Store    │     │   Store    │
      └────────────┘     └────────────┘     └────────────┘
        │      │           │      │           │      │
        │      │           │      │           │      │
   ┌────┴┐  ┌──┴──┐   ┌────┴┐  ┌──┴──┐   ┌────┴┐  ┌──┴──┐
   │ RG1 │  │ RG2 │   │ RG1 │  │ RG2 │   │ RG1 │  │ RG2 │
   └─────┘  └─────┘   └─────┘  └─────┘   └─────┘  └─────┘
```

ICT is the transport layer for hierarchical, intercluster communication. In an intercluster setting, where there are data parallel replica groups within each cluster, and data parallel replica groups between each cluster, ICT supports:

1. Flexible synchronization policies
2. Flexible Optimizer choice
3. Flexible Collective Communications Library (CCL) choice (e.g., cluster 1 nccl, inter-cluster gloo, inter-cluster use mirror descent)

Under the hood, ICT conducts the following optimizations:

1. Collects cluster information (machine type, data location)
2. Has a collective communications library (CCL) wrapper layer that enables machines that use different CCLs to work together

### Version 1

ICT V1 Has the following model:

- Input
    - Model
    - A set of clusters and the mapping inside the clusters
        - <cluster, cluster-mapping>
- Output
    - Compute the composition structure
        - Dynamically collect the structure of each cluster (need to form inter-cluster peering relationship)
            - Intra-cluster-state: (cluster-id, dp, node ip, model-param)
            - Sub-protocol (ISC): Inter-cluster State Collection : Collect bw and delay
            - Compute inter-cluster sync states

The test cases:

#### Demo 1.1a (intra-state change => inter-reaction

Why this: it is kind of like internet routing, the failure of intra state will be handled nicely

#### Demo 1.1b (intra-state change => inter-reaction

#### Demo 1.2: add a new cluster and the system integrate it into it without restart
Sub cases
- All nvidia, the new set will still use nccl
- But the new one is Huawei, then it should be gloo

#### Demo 1.3: change inter-cluster state (change the bw), and see that the inter-cluster protocol adapt

For example, change the H param of diloco


## To Run the current code:

First, set up the environment.

```bash
# TorchTitan setup
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
python scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=...

# TorchFT setup
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

# Installation from github because don't have sudo
wget https://github.com/protocolbuffers/protobuf/releases/download/v30.2/protoc-30.2-linux-x86_64.zip
unzip protoc-30.2-linux-x86_64.zip -d $HOME/.local
export PATH=$HOME/.local/bin:$PATH
pip install .
```

Then, run the following commands. This script is set up for running localsgd on shenzhen-node2 and shenzhen-nodem
```bash
# Example:
# ./run_server_localsgd.sh <cuda_devices> <replica_group_id> <cluster_group_id> [train_script] [script‑args…]

# For the experimental DiLoCo-ICT, we need to run the following command:

## On Cluster 0:
/srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 0 0 /srv/apps/warren/torchft/train_DiLoCo-ICT.py 1 2 1000 # Replica Group 0, Cluster 0, wait for 1 second between steps, sync every 2 steps for 1000 steps
/srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 1 1 0 /srv/apps/warren/torchft/train_DiLoCo-ICT.py 1 2 1000 # Replica Group 1, Cluster 0, wait for 1 second between steps, sync every 2 steps for 1000 steps

## On Cluster 1:
/root/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 0 1 /root/warren/torchft/train_DiLoCo-ICT.py 1 2 1000 # Replica Group 0, Cluster 1, wait for 1 second between steps, sync every 2 steps for 1000 steps
/root/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 1 1 /root/warren/torchft/train_DiLoCo-ICT.py 1 2 1000 # Replica Group 1, Cluster 1, wait for 1 second between steps, sync every 2 steps for 1000 steps


# Demo 2: Two clusters on a single cluster 2 GPU setup:

# Note that all three lighthouses have to be running for this to work.

# Cluster 0:
/srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 0 0 /srv/apps/warren/torchft/train_DiLoCo-ICT.py 1 2 1000 # Replica Group 0, Cluster 0, wait for 1 second between steps, sync every 2 steps for 1000 steps, running of CUDA_VISIBLE_DEVICES=0

/srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 1 0 1 /srv/apps/warren/torchft/train_DiLoCo-ICT.py 1 2 1000 # Replica Group 0, Cluster 1, wait for 1 second between steps, sync every 2 steps for 1000 steps, running of CUDA_VISIBLE_DEVICES=1
```

## Features

### Routing Information

We want to provide flexible routing.

Ultimately, this should be done at the NCCL level. We should have a NCCL-P Library that enables the routing to be configured.

### Hyperparameter Autoconfig

This is a main feature of ICT. There are three following features that we support. 

One design principle of ICT is extensibility. Therefore, we make it as easy as possible for others to build custom modules on top of our lighthouse module.

1. Dataloader configuration
    - local_batch_size: int
    - global_batch_size: int
2. Hyperparameter Configuration
    - Optimizer hyperparameters
        - momentum: float
        - lr: float
        - weight_decay: float
        - betas: float
        - nesterov: bool
3. Failure configuration
    - Timeout

#### Dataloader information

Abstractly, think about the data as a list of tensors. I want to have a DistributedSampler component that the lighthouse queries.

Now, think about it as the following:



There is a 
Ideally, the global_batch_size=np.dot(local_batch_sizes, sync_everies), local_batch_sizes dot_product iters * the local_batch_size, 

 The sampler does the following:

This needs to be coupled with a data movement service (e.g. Effingle) to move the data to the correct places for the sampling service.

The current torchft distributed sampler does the followingL

```python
class DistributedSampler(data.distributed.DistributedSampler):
    """
    DistributedSampler extends the standard PyTorch DistributedSampler with a
    `num_replica_groups` that is used to shard the data across the fault
    tolerance replica groups.

    torchft doesn't know how many replica groups ahead of time so we need to set
    this to be the max number.

    This sampler is inherently lossy when used with torchft. torchft
    occasionally drops batches on rejoining and if a replica group is down that
    group examples will never be used. This can lead to imbalances if using a
    small dataset.

    This will shard the input dataset into ``num_replicas*num_replica_group``
    number of shards.

    Each shard rank is calculated via: ``rank + num_replicas*replica_group``

    num_replicas and replica_group must be the same on all workers.
    """

    def __init__(
        self,
        dataset: data.Dataset,
        replica_group: int,
        num_replica_groups: int,
        rank: Optional[int] = None,
        num_replicas: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        """
        Args:
            data: the dataset to use
            replica_group: the group ID (0-num_replica_groups) to use for this shard of data.
            num_replica_groups: the max number of global replica groups
            rank: the local group rank
            num_replicas: the local group world size
        """
        if rank is None:
            rank = dist.get_rank()
        if num_replicas is None:
            num_replicas = dist.get_world_size()

        self.global_rank: int = rank + num_replicas * replica_group
        self.global_world_size: int = num_replicas * num_replica_groups

        super().__init__(
            dataset,
            rank=self.global_rank,
            num_replicas=self.global_world_size,
            # pyre-fixme[6]: got object
            **kwargs,
        )
```

Pytorch Datasets:

```python
class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    # def __getitems__(self, indices: List) -> List[T_co]:
    # Not implemented to prevent false-positives in fetcher check in
    # torch.utils.data._utils.fetch._MapDatasetFetcher

    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py

```

```python
dataset = Tensor_List(Length=num_data)
```

## Diagram

The two-level DiLoCo implementation follows the diagram below closely. There is a inner process group within each cluster, and a cross-cluster process group. Only one replica group in each cluster participates in the cross-cluster process group. That replica group is called the "Leader". Information from the cross-cluster process group is broadcasted to the other replica groups ("followers") from the Leader.

```python
from graphviz import Digraph
from IPython.display import Image, display

# ----------------------------------------
# Colors
reconfig_pg_color = '#FFD580'
reconfig_pg_node_color = '#FFF2CC'

training_color = '#ADD8E6'
training_node_color = '#D6EBF2'
compute_color = '#D6EBF9'
compute_node_color = '#EAF5FC'
quorum_color = '#EAF5FD'
quorum_node_color = '#F4FAFE'
healing_color = '#F0E0F8'
healing_node_color = '#F7F0FB'
rollback_color = 'lightpink'
rollback_node_color = '#FFDBE0'
draining_color = 'lightgreen'
draining_node_color = '#C7F6C7'
rejoin_color = 'lightyellow'
rejoin_node_color = '#FFFFEF'
received_join_color = '#FFFACD'
received_join_node_color = '#FFFEF5'

init_color  = '#E8E8E8'
idle_color  = '#E0FFFF'
exit_color  = '#FFE6D6'

global_leader_color = '#DCC6FF'
global_leader_node_color = '#EFE6FF'
follower_color = '#FFEFD5'
follower_node_color = '#FFF8E6'
# ----------------------------------------

def add_cluster(graph, cluster_id, label, fillcolor, nodes, edges, node_fillcolor='white'):
    cluster = Digraph(f'cluster_{cluster_id}')
    cluster.graph_attr.update(label=label, style='filled', fillcolor=fillcolor, rankdir='LR')
    for node_id, node_label in nodes:
        cluster.node(node_id, node_label, shape='box', style='filled', fillcolor=node_fillcolor)
    for src, dst, lbl, attrs in edges:
        cluster.edge(src, dst, lbl, **attrs)
    graph.subgraph(cluster)

def add_top_level(graph):
    top_colors = {
        'INIT': init_color, 'TRAINING': training_color, 'ROLLBACK': rollback_color, 'IDLE': idle_color,
        'DRAINING': draining_color, 'REJOIN': rejoin_color, 'RECEIVED_JOIN': received_join_color, 'EXIT': exit_color
    }
    edges = [
        ('INIT','TRAINING','first quorum / pg.configure()', {}),
        ('TRAINING','ROLLBACK','relevant_fault', {}),
        ('TRAINING','DRAINING','drain_self | drain_noticed', {}),
        ('ROLLBACK','TRAINING','pg_ready (Err‑1)', {}),
        ('ROLLBACK','IDLE','self_fault (Err‑2)', {}),
        ('IDLE','REJOIN','advertise_join', {}),
        ('REJOIN','TRAINING','pg_ready', {}),
        ('TRAINING','RECEIVED_JOIN','received_join', {}),
        ('RECEIVED_JOIN','TRAINING','pg_ready', {}),
        ('IDLE','EXIT','pre‑empted', {}),
        ('DRAINING','EXIT','drain_self', {}),
        ('DRAINING','TRAINING','drain_noticed', {})
    ]
    t = Digraph('cluster_top')
    t.graph_attr.update(style='filled,bold', color='lightsteelblue1', rankdir='LR',
                        label='TOP‑LEVEL FLOW', penwidth='4', fontsize='24', margin='1')
    for n,c in top_colors.items():
        t.node(n, n, shape='box', fillcolor=c, style='filled', fontsize='18', width='2.2', height='0.8')
    for s,d,lbl,attrs in edges:
        t.edge(s, d, lbl, fontsize='12', **attrs)
    graph.subgraph(t)

def add_reconfig_pg(graph):
    edges = [
        ('PG_SHUTDOWN','PG_RECONFIG','pg_down && pg_other_fault (Err 1)', {})
    ]
    add_cluster(graph,'reconfig','RECONFIG PG',reconfig_pg_color,
                [('PG_SHUTDOWN','PG_SHUTDOWN'),('PG_RECONFIG','PG_RECONFIG')],
                edges,reconfig_pg_node_color)

def add_rollback(graph):
    edges = [
        ('RB_BCAST','CANCEL_RELEVANT_FUTURES','rollback_broadcasted', {})
    ]
    add_cluster(graph,'rb','ROLLBACK',rollback_color,
                [('RB_BCAST','BCAST_ROLLBACK'),('CANCEL_RELEVANT_FUTURES','CANCEL_RELEVANT_FUTURES')],
                edges,rollback_node_color)
    graph.edge('CANCEL_RELEVANT_FUTURES','PG_SHUTDOWN','futs_cancelled')

def add_training(graph):
    train = Digraph('cluster_train')
    train.graph_attr.update(label='TRAINING', style='filled', fillcolor=training_color, rankdir='LR')

    # COMPUTE STREAM
    compute_edges = [
        ('FORWARD','BACKWARD','forward_done', {})
    ]
    add_cluster(train,'compute','COMPUTE STREAM',compute_color,
                [('FORWARD','FORWARD'),('BACKWARD','BACKWARD')],
                compute_edges, compute_node_color)

    # LOCAL QUORUM STREAM
    q = Digraph('cluster_quorum')
    q.graph_attr.update(label='LOCAL QUORUM STREAM', style='filled', fillcolor=quorum_color, rankdir='LR')
    q.node('QUORUM_COMPUTE','QUORUM_COMPUTE', shape='box', style='filled', fillcolor=quorum_node_color)
    train.subgraph(q)

    # LOCAL HEALING
    heal_edges = []
    add_cluster(train,'heal','LOCAL HEALING',healing_color,
                [('SEND_CKPT','SEND_CKPT'),('FETCH_CKPT','FETCH_CKPT')],
                heal_edges, healing_node_color)

    # Sync & commit nodes
    train.node('SYNC_INTRA_CLUSTER_DDP','Sync_Intra‑Cluster‑DDP\\n(wait compute + quorum)',
               shape='diamond', peripheries='2', style='bold,filled', fillcolor=training_node_color)
    train.node('SYNC_INTRA_REPLICA_GROUP','Sync_Intra‑Replica‑Group', shape='diamond',
               style='filled', fillcolor=training_node_color)
    train.node('COMMIT','COMMIT', shape='box', style='filled', fillcolor=training_node_color)

    # Edges (sync dotted)
    dotted = {'style':'dotted'}
    heavy = {'style':'dotted', 'penwidth':'3'}

    train.edge('BACKWARD','SYNC_INTRA_CLUSTER_DDP','backward_done')
    train.edge('SEND_CKPT','COMMIT','checkpoint_return')
    train.edge('FETCH_CKPT','COMMIT','checkpoint_return')
    train.edge('QUORUM_COMPUTE','SYNC_INTRA_CLUSTER_DDP','quorum_done', **dotted)
    train.edge('QUORUM_COMPUTE','SEND_CKPT','pg_same & peer_needs_ckpt?', **dotted)
    train.edge('QUORUM_COMPUTE','FETCH_CKPT','pg_same & need_heal?', **dotted)
    train.edge('SYNC_INTRA_CLUSTER_DDP','SYNC_INTRA_REPLICA_GROUP','local_ok (allreduce & quorum)')
    train.edge('SYNC_INTRA_REPLICA_GROUP','COMMIT','group_ok')

    # start next step
    train.edge('COMMIT','FORWARD','step++ (compute)')
    train.edge('COMMIT','QUORUM_COMPUTE','step++ (quorum)')

    # GLOBAL SYNC CHECK
    train.node('GLOBAL_SYNC_CHECK','Need Global Sync?', shape='diamond', style='filled',
               fillcolor=training_node_color)
    train.edge('COMMIT','GLOBAL_SYNC_CHECK','local_commit')
    train.edge('GLOBAL_SYNC_CHECK','FORWARD','no')

    # GLOBAL SYNC – LEADER
    gsl = Digraph('cluster_gsl')
    gsl.graph_attr.update(label='GLOBAL SYNC – LEADER', style='filled',
                          fillcolor=global_leader_color, rankdir='LR')

    # GLOBAL QUORUM STREAM
    gq = Digraph('cluster_gsl_q')
    gq.graph_attr.update(label='GLOBAL QUORUM STREAM', style='filled', fillcolor=quorum_color, rankdir='LR')
    gq.node('L_QUORUM_COMPUTE','QUORUM_COMPUTE_GLOBAL', shape='box', style='filled', fillcolor=quorum_node_color)
    gsl.subgraph(gq)

    # PSEUDO‑GRADIENT
    pg = Digraph('cluster_gsl_pg')
    pg.graph_attr.update(label='PSEUDO‑GRADIENT', style='filled', fillcolor=compute_color, rankdir='LR')
    pg.node('L_CALC_PG','CALC_PSEUDOGRADS', shape='box', style='filled', fillcolor=compute_node_color)
    pg.node('L_ALLREDUCE_PG','ALLREDUCE_PSEUDOGRADS', shape='box', style='filled', fillcolor=compute_node_color)
    pg.node('L_BCAST_PG','BCAST_PSEUDOGRADS', shape='box', style='filled', fillcolor=compute_node_color)
    pg.edge('L_CALC_PG','L_ALLREDUCE_PG','calc_done')
    pg.edge('L_ALLREDUCE_PG','L_BCAST_PG','allreduce_done')
    gsl.subgraph(pg)

    # GLOBAL HEALING
    gh = Digraph('cluster_gsl_h')
    gh.graph_attr.update(label='GLOBAL HEALING', style='filled', fillcolor=healing_color, rankdir='LR')
    gh.node('L_SEND_CKPT','SEND_CKPT_G', shape='box', style='filled', fillcolor=healing_node_color)
    gh.node('L_FETCH_CKPT','FETCH_CKPT_G', shape='box', style='filled', fillcolor=healing_node_color)
    gh.edge('L_SEND_CKPT','L_FETCH_CKPT','', **heavy, dir='none')
    gsl.subgraph(gh)

    # sync replica & commit
    gsl.node('L_SYNC_REPLICA_GROUP','Sync_Intra‑Replica‑Group_G', shape='diamond', style='filled',
             fillcolor=training_node_color)
    gsl.node('L_GCOMMIT','GLOBAL_COMMIT_LEADER', shape='box', style='filled', fillcolor=global_leader_node_color)

    # edges (global dotted sync)
    gsl.edge('L_QUORUM_COMPUTE','L_CALC_PG','quorum_ok', **dotted)
    gsl.edge('L_QUORUM_COMPUTE','L_SEND_CKPT','quorum_ok', **dotted)
    gsl.edge('L_QUORUM_COMPUTE','L_FETCH_CKPT','quorum_ok', **dotted)
    gsl.edge('L_ALLREDUCE_PG','L_SYNC_REPLICA_GROUP','allreduce_done')
    gsl.edge('L_SYNC_REPLICA_GROUP','L_GCOMMIT','group_ok')

    gsl.edge('L_SEND_CKPT','L_GCOMMIT','ckpt_done')
    gsl.edge('L_FETCH_CKPT','L_GCOMMIT','heal_done')

    # reconfig interactions
    gsl.edge('L_QUORUM_COMPUTE','PG_SHUTDOWN','quorum_fault')
    gsl.edge('PG_RECONFIG','L_QUORUM_COMPUTE','pg_ready')

    train.subgraph(gsl)

    # GLOBAL SYNC – FOLLOWER
    gsf_edges = [
        ('F_WAIT_BCAST','F_GCOMMIT','bcast_received', {}),
        ('F_LISTEN_CKPT','F_GCOMMIT','assist_done', {})
    ]
    add_cluster(train,'gsf','GLOBAL SYNC – FOLLOWER',follower_color,
                [('F_WAIT_BCAST','WAIT_BCAST_PSEUDOGRADS'),
                 ('F_LISTEN_CKPT','LISTEN_CKPT_ASSIST'),
                 ('F_GCOMMIT','GLOBAL_COMMIT_FOLLOWER')],
                gsf_edges, follower_node_color)

    # edges from global sync check
    train.edge('GLOBAL_SYNC_CHECK','L_QUORUM_COMPUTE','yes & leader')
    train.edge('GLOBAL_SYNC_CHECK','F_WAIT_BCAST','yes & follower')
    train.edge('GLOBAL_SYNC_CHECK','F_LISTEN_CKPT','yes & follower')
    train.edge('L_GCOMMIT','FORWARD','global_commit')
    train.edge('F_GCOMMIT','FORWARD','global_commit')

    # communication edges (heavy dotted)
    train.edge('SEND_CKPT','FETCH_CKPT','', **heavy, dir='none')
    train.edge('F_WAIT_BCAST','L_BCAST_PG','', **heavy, dir='none')
    train.edge('L_QUORUM_COMPUTE','F_LISTEN_CKPT','send_ckpt_assist_message', **heavy)

    # training loop dotted
    train.edge('FORWARD','BACKWARD','training_loop', style='dotted')

    graph.subgraph(train)

def add_draining(graph):
    edges = [
        ('D_START','D_NOTICE_OTHERS','drain_self', {}),
        ('D_START','D_RUN_UNTIL_COMMIT','drain_noticed', {}),
        ('D_NOTICE_OTHERS','D_COMMIT','local_ok / implicit_sync', {}),
        ('D_RUN_UNTIL_COMMIT','D_COMMIT','proceed_to_commit', {}),
        ('D_COMMIT','D_EXIT','if_drainer', {})
    ]
    add_cluster(graph,'dr','DRAINING',draining_color,
                [('D_START','D_START'),('D_NOTICE_OTHERS','D_NOTICE_OTHERS'),
                 ('D_RUN_UNTIL_COMMIT','D_RUN_UNTIL_COMMIT'),('D_COMMIT','D_COMMIT'),('D_EXIT','D_EXIT')],
                edges,draining_node_color)
    graph.edge('D_COMMIT','PG_SHUTDOWN','if_survivor | collective_timeout')

def add_rejoin(graph):
    edges = [
        ('JOIN_ADVERTISE','JOIN_WAIT','ack_from_peers', {})
    ]
    add_cluster(graph,'join','REJOIN',rejoin_color,
                [('JOIN_ADVERTISE','ADVERTISE_JOIN'),('JOIN_WAIT','WAIT_CURRENT_STEP')],
                edges, rejoin_node_color)
    graph.edge('JOIN_WAIT','PG_SHUTDOWN','all_commits_seen')

def add_received_join(graph):
    edges = [
        ('RJ_RUN_UNTIL_COMMIT','RJ_ADVERTISE_COMMIT','run_until_commit', {})
    ]
    add_cluster(graph,'recvjoin','RECEIVED JOIN',received_join_color,
                [('RJ_RUN_UNTIL_COMMIT','RUN_UNTIL_COMMIT'),('RJ_ADVERTISE_COMMIT','ADVERTISE_COMMIT')],
                edges, received_join_node_color)
    graph.edge('TRAINING','RJ_RUN_UNTIL_COMMIT','received_join')
    graph.edge('RJ_ADVERTISE_COMMIT','PG_SHUTDOWN','advertise_commit')

def add_layout_order(graph):
    order=['TRAINING','ROLLBACK','DRAINING','REJOIN','RECEIVED_JOIN']
    for s,d in zip(order,order[1:]):
        graph.edge(s, d, style='invis', weight='100')

# Build diagram
g = Digraph('TorchFT_FSM_v20_reconfig_cluster_v16', format='png')
g.attr(labelloc='t', fontsize='18',
       label='TorchFT FSM – Rev 20 (Reconfig Cluster + DiLoCo) – v16')

add_top_level(g)
add_reconfig_pg(g)
add_training(g)
add_rollback(g)
add_draining(g)
add_rejoin(g)
add_received_join(g)
g.edge('QUORUM_COMPUTE','PG_SHUTDOWN','pg_diff')
add_layout_order(g)

path = '/mnt/data/torchft_fsm_v20_reconfig_cluster_v16'
g.render(filename=path, format='png', cleanup=True)
display(Image(path + '.png'))
```

## TODOs

Elucidate what each parameter means and does clearly, and how they should be set up in a clear table.

1. Support elasticity with the data sampler. GLOBAL_REPLICA_NUM is not known ahead of time.
2. Set CLUSTER_GROUP_ID in the run_server.sh script
3. The global rank is hard to set/define. The current reliance on rank is restrictive. Makes the programming model fully grid like.
4. Cluster world size should be auto-configurable.
5. Work on the DistributedSampler to make it easier to understand and shard. The current way is really unsustainable.
6. There should be two sets of timeouts
7. Increase the efficiency of broadcast_one in LocalSGD_Two_level. It should be continuously streamed as `avg_param` arrives. To do so, need to do `self.braodcast_one(p.data.clone(), root_rank=self._root_rank)`. Did not implement because it makes the code unclean and unclear. Without a simple `if self._rank == root_rank` type logic.
8. Currently allreduce and broadcast are synchronous in the outer step, same with save and load parameters. These can all be asynchronous. Need to think about the gain from this. If we make them synchronous, currently actually we cannot do more work lol. Potentially change it to asynchronous. However, we could implement the algorithm in DiLoCo streaming to make this worthwhile.
9. Make healing more asynchronous (this needs us to specify a recovery stream)
- Potentially, we need to implement a sync function that doesn't directly call `_average`, but does the following:

```python
    def _perform_sync(self) -> None:
        """
        Performs the synchronization of the model weights across the manager.
        """
        """
        Averages the model parameters across the manager and returns the averaged parameters.
        """
        works = []
        averaged_parameters = []
        for p in self._model.parameters():
            if self.self_rank == self._root_rank:
                # Create a new tensor to store the averaged parameter
                p.data.grad = None
                avg_param = p.data.clone()
                works.append(self._manager.allreduce(avg_param))
                averaged_parameters.append(avg_param)
                works.append(self._manager.broadcast_one(avg_param, root_rank=self._root_rank, timeout=self._manager.timeout * 2)) # Sends to others, multiply timeout by 2 because have to wait for self.sync() to also finish
            else:
                works.append(self._manager.broadcast_one(p.data.grad, root_rank=self._root_rank, timeout=self._manager.timeout * 2)) # Receives avg_param
        for work in works:
            work.wait()
        if self._manager.should_commit():
            # Update the model parameters with the averaged values
            for param, avg_param in zip(self._model.parameters(), averaged_parameters):
                param.data.copy_(avg_param)
```

8. Support broadcast