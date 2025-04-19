#!/bin/bash

# Common configuration
LIGHTHOUSE_PORT=29520
LIGHTHOUSE_HOST="sz-k8s-master"
LIGHTHOUSE_URL="http://${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}"
NUM_REPLICA_GROUPS_LOCAL_CLUSTER=3
NCCL_COMMON_ENV="export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL"
CUDA_DEVICES=0
NNODES=1
NPROC_PER_NODE=1
TRAIN_SCRIPT="train_test.py"

# Master node configuration
MASTER_HOSTNAME="sz-k8s-master"
MASTER_PATH="/srv/apps/warren/torchft"
MASTER_CONDA_PATH="/srv/apps/danny/miniconda3"
MASTER_ENV_NAME="localsgd"
MASTER_REPLICA_GROUP=0
MASTER_NETWORK_INTERFACE="eno2"
MASTER_ADDR="10.0.0.3"
MASTER_PORT=29520

# Node2 configuration
NODE2_HOSTNAME="sz-k8s-node2"
NODE2_PATH="/root/warren/torchft"
NODE2_CONDA_PATH="/usr/local/conda"
NODE2_ENV_NAME="localsgd"
NODE2_REPLICA_GROUP=2
NODE2_NETWORK_INTERFACE="ens81f1"
NODE2_ADDR="10.0.0.2"
NODE2_PORT=29520

TEST_CMD="python -m unittest -v torchft.futures.futures_test"

# Detect and configure based on hostname
HOSTNAME=$(hostname)
if [ "$HOSTNAME" = "$MASTER_HOSTNAME" ]; then
    cd $MASTER_PATH
    source "${MASTER_CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${MASTER_CONDA_PATH}/envs/${MASTER_ENV_NAME}"
    export PATH="${MASTER_CONDA_PATH}/envs/${MASTER_ENV_NAME}/bin:$PATH"
    . "$HOME/.cargo/env"

    eval "$NCCL_COMMON_ENV"
    export REPLICA_GROUP_ID=$MASTER_REPLICA_GROUP
    export NUM_REPLICA_GROUPS_LOCAL_CLUSTER=$NUM_REPLICA_GROUPS_LOCAL_CLUSTER
    export NCCL_SOCKET_IFNAME=$MASTER_NETWORK_INTERFACE

    eval "$TEST_CMD"

elif [ "$HOSTNAME" = "$NODE2_HOSTNAME" ]; then
    cd $NODE2_PATH
    source "${NODE2_CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate $NODE2_ENV_NAME
    export PATH="${NODE2_CONDA_PATH}/envs/${NODE2_ENV_NAME}/bin:$PATH"
    . "$HOME/.cargo/env"

    eval "$NCCL_COMMON_ENV"
    export REPLICA_GROUP_ID=$NODE2_REPLICA_GROUP
    export NUM_REPLICA_GROUPS_LOCAL_CLUSTER=$NUM_REPLICA_GROUPS_LOCAL_CLUSTER
    export NCCL_SOCKET_IFNAME=$NODE2_NETWORK_INTERFACE
fi
