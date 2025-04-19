#!/bin/bash
#
# Launcher for fault‑tolerant LocalSGD with hierarchical (local + global)
# TorchFT lighthouses.  Sets every env‑var expected by `train_localsgd.py`.
#
# ── Usage ───────────────────────────────────────────────────────────────
#   ./run_server_localsgd.sh <cuda_devices> [train_script] [script‑args…]
#
#   <cuda_devices>  – comma‑separated CUDA indices to expose
#   [train_script]  – python entrypoint     (default: train_localsgd.py)
#   [script‑args…]  – forwarded verbatim to the training script
#
# Edit the topology section to match your machines/IPs.
set -euo pipefail

# User Input
CUDA_DEVICES=${1:-0} # CUDA devices 
REPLICA_GROUP_ID=${2:-${CUDA_DEVICES%%,*}} # ID of the local replica group
CLUSTER_GROUP_ID=${3:-0} # ID of the local cluster

TRAIN_SCRIPT=${4:-"train_localsgd.py"}
TRAIN_ARGS="${@:5}"

# --- Python import path -------------------------------------------------
# Ensure the TorchFT repo root (the directory that contains the top‑level
# `torchft/` package) is on PYTHONPATH so that the training script can
# `import torchft` even when launched from a sub‑directory.
REPO_ROOT="$(dirname "$(dirname "$TRAIN_SCRIPT")")"   # strip "…/torchft/localsgd"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"


########################  Topology description  ##########################
NUM_REPLICA_GROUPS_LOCAL_CLUSTER=3                # DP groups *inside* each cluster
NUM_CLUSTERS=2                      # physical clusters

MASTER_HOSTNAME="sz-k8s-master"
MASTER_IP="10.0.0.3"
MASTER_NET_IF="eno2"

NODE2_HOSTNAME="sz-k8s-node2"
NODE2_IP="10.0.0.2"
NODE2_NET_IF="ens81f1"

GLOBAL_LIGHTHOUSE_PORT=29520        # one per experiment (master)
LOCAL_LIGHTHOUSE_PORT=$((29521 + CLUSTER_GROUP_ID))    # one per cluster

GLOBAL_STORE_PORT=29499             # kv‑store for the global Manager
LOCAL_STORE_PORT_BASE=29400         # kv‑store for the local  Manager (+gpu‑idx)
##########################################################################

# Detect and configure based on hostname
HOSTNAME=$(hostname)

case "$HOSTNAME" in
  "$MASTER_HOSTNAME")
    HOST_IP=$MASTER_IP
    NCCL_IF=$MASTER_NET_IF
    ;;
  "$NODE2_HOSTNAME")
    HOST_IP=$NODE2_IP
    NCCL_IF=$NODE2_NET_IF
    ;;
  *)
    echo "Unknown host $HOSTNAME – extend topology in script." >&2
    exit 1
    ;;
esac


# ── env‑vars consumed by train_localsgd.py ──────────────────────────────
export CLUSTER_GROUP_ID
export NUM_CLUSTERS
export REPLICA_GROUP_ID
export NUM_REPLICA_GROUPS_LOCAL_CLUSTER

export TORCHFT_LIGHTHOUSE_LOCAL="http://${HOST_IP}:${LOCAL_LIGHTHOUSE_PORT}"
export TORCHFT_LIGHTHOUSE_GLOBAL="http://${MASTER_IP}:${GLOBAL_LIGHTHOUSE_PORT}"

export MASTER_ADDR_LOCAL="$HOST_IP"
export MASTER_PORT_LOCAL=$((LOCAL_STORE_PORT_BASE + REPLICA_GROUP_ID))
export MASTER_ADDR_GLOBAL="$MASTER_IP"
export MASTER_PORT_GLOBAL=$GLOBAL_STORE_PORT
# ────────────────────────────────────────────────────────────────────────

# NCCL + CUDA housekeeping
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=$NCCL_IF
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

echo "[server] Host=$HOSTNAME  Cluster=$CLUSTER_GROUP_ID  Replica=$REPLICA_GROUP_ID"
echo "         CLUSTER_GROUP_ID: $CLUSTER_GROUP_ID"
echo "         NUM_CLUSTERS: $NUM_CLUSTERS"
echo "         REPLICA_GROUP_ID: $REPLICA_GROUP_ID"
echo "         NUM_REPLICA_GROUPS_LOCAL_CLUSTER: $NUM_REPLICA_GROUPS_LOCAL_CLUSTER"
echo "         TORCHFT_LIGHTHOUSE_LOCAL: $TORCHFT_LIGHTHOUSE_LOCAL"
echo "         TORCHFT_LIGHTHOUSE_GLOBAL: $TORCHFT_LIGHTHOUSE_GLOBAL"
echo "         MASTER_ADDR_LOCAL: $MASTER_ADDR_LOCAL"
echo "         MASTER_PORT_LOCAL: $MASTER_PORT_LOCAL"
echo "         MASTER_ADDR_GLOBAL: $MASTER_ADDR_GLOBAL"
echo "         MASTER_PORT_GLOBAL: $MASTER_PORT_GLOBAL"
echo "         CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "         NCCL_IF: $NCCL_IF"
echo "         NCCL_DEBUG: $NCCL_DEBUG"
echo "         NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
echo "         NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"

# Activate the correct conda environment
MASTER_PATH="/srv/apps/warren/torchft"
MASTER_CONDA_PATH="/srv/apps/danny/miniconda3"
MASTER_ENV_NAME="warren/torchtitan"
NODE2_PATH="/root/warren/torchft"
NODE2_CONDA_PATH="/usr/local/conda"
NODE2_ENV_NAME="localsgd"
if [ "$HOSTNAME" = "$MASTER_HOSTNAME" ]; then
    cd $MASTER_PATH
    source "${MASTER_CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${MASTER_CONDA_PATH}/envs/${MASTER_ENV_NAME}"
    export PATH="${MASTER_CONDA_PATH}/envs/${MASTER_ENV_NAME}/bin:$PATH"
    . "$HOME/.cargo/env"
elif [ "$HOSTNAME" = "$NODE2_HOSTNAME" ]; then
    cd $NODE2_PATH
    source "${NODE2_CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${NODE2_CONDA_PATH}/envs/${NODE2_ENV_NAME}"
    export PATH="${NODE2_CONDA_PATH}/envs/${NODE2_ENV_NAME}/bin:$PATH"
    . "$HOME/.cargo/env"
fi



# --- Start Global TCPStore (only on designated lead process) ---
# We designate Cluster 0, Replica 0 running on the MASTER_HOSTNAME as the leader
# responsible for hosting the *global* store.
if [[ "$HOSTNAME" == "$MASTER_HOSTNAME" && "$CLUSTER_GROUP_ID" -eq 0 && "$REPLICA_GROUP_ID" -eq 0 ]]; then
    # --- Location of the KV store script ---
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
    KV_STORE_SCRIPT="${SCRIPT_DIR}/start_kv_store.py" # Assuming it's in the same dir

    # --- Process Management for background store ---
    PID_DIR="/tmp"
    GLOBAL_STORE_PID_FILE="${PID_DIR}/torchft_global_store.pid"
    GLOBAL_STORE_LOG_FILE="${PID_DIR}/torchft_global_store.log"

    echo ">>> This is the lead process (Cluster 0, Replica 0 on $MASTER_HOSTNAME)."

    # Check if script exists
    if [[ ! -f "$KV_STORE_SCRIPT" ]]; then
        echo "ERROR: KV store script not found at $KV_STORE_SCRIPT"; exit 1;
    fi


    # Check if already running via PID file
    if [[ -f "$GLOBAL_STORE_PID_FILE" ]] && ps -p "$(cat "$GLOBAL_STORE_PID_FILE")" > /dev/null; then
        echo ">>> Global store appears to be running (PID $(cat "$GLOBAL_STORE_PID_FILE")). Skipping start."
    else
         if [[ -f "$GLOBAL_STORE_PID_FILE" ]]; then
             echo ">>> Removing stale PID file $GLOBAL_STORE_PID_FILE"
             rm -f "$GLOBAL_STORE_PID_FILE"
         fi

        echo ">>> Starting GLOBAL KVStore via ${KV_STORE_SCRIPT} in the background..."
        echo ">>> Store Address: ${MASTER_ADDR_GLOBAL}:${MASTER_PORT_GLOBAL}"
        echo ">>> Log File:      ${GLOBAL_STORE_LOG_FILE}"
        echo ">>> PID File:      ${GLOBAL_STORE_PID_FILE}"

        nohup python -u "$KV_STORE_SCRIPT" \
            --host "$MASTER_ADDR_GLOBAL" \
            --port "$MASTER_PORT_GLOBAL" \
            --timeout 3600 \
            --pid-file "$GLOBAL_STORE_PID_FILE" \
            > "$GLOBAL_STORE_LOG_FILE" 2>&1 & # Run in background
            
        GLOBAL_STORE_PID=$!
        echo ">>> Global store background process PID: $GLOBAL_STORE_PID"
        echo ">>> Waiting a few seconds for global store to initialize..."
        sleep 5 # Give the store time to start listening
    fi
fi

# ------------------------------------------------------------------
echo "Launching training script via torchrun..."
# torchrun uses MASTER_ADDR_LOCAL and MASTER_PORT_LOCAL for its own coordination
# and to provide the default store for the *local* process group.

torchrun --nnodes=1 --nproc_per_node=1 \
         --master_addr="$MASTER_ADDR_LOCAL" --master_port="$MASTER_PORT_LOCAL" \
         "$TRAIN_SCRIPT" $TRAIN_ARGS

# Example:
# ./run_server_localsgd.sh <cuda_devices> <replica_group_id> <cluster_group_id> [train_script] [script‑args…]

# /srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 0 0 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000 # Replica Group 0, Cluster 0, wait for 1 second between steps, sync every 2 steps for 1000 steps
# /srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 1 0 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000 # Replica Group 1, Cluster 0, wait for 1 second between steps, sync every 2 steps for 1000 steps
# /srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 0 1 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000 # Replica Group 0, Cluster 1, wait for 1 second between steps, sync every 2 steps for 1000 steps
# /srv/apps/warren/torchft/.shell_scripts/local_sgd/run_server_localsgd.sh 0 1 1 /srv/apps/warren/torchft/train_localsgd-two_level.py 1 2 1000 # Replica Group 1, Cluster 1, wait for 1 second between steps, sync every 2 steps for 1000 steps

# TODO:  
# 1. Support elasticity with the data sampler. GLOBAL_REPLICA_NUM is not known ahead of time.
# 2. Set CLUSTER_GROUP_ID in the run_server.sh script
# 3. The global rank is hard to set/define. The current reliance on rank is restrictive. Makes the programming model fully grid like.
# 4. Cluster world size should be auto-configurable.