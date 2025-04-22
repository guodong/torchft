#!/bin/bash

# Usage:
# ./run_server_localsgd.sh <cuda_devices> <replica_group_id> <cluster_group_id> [train_script] [script-args...]

set -e

CUDA_VISIBLE_DEVICES=$1
REPLICA_GROUP_ID=$2
CLUSTER_GROUP_ID=$3
shift 3
TRAIN_SCRIPT=$1
shift 1

export CUDA_VISIBLE_DEVICES

# Configuration variables (these would normally be set or sourced from a config)
NUM_CLUSTERS=2
NUM_REPLICA_GROUPS_LOCAL_CLUSTER=2

MASTER_ADDR_CLUSTER="10.0.0.3"
MASTER_PORT_CLUSTER=29499

MASTER_ADDR_LOCAL="localhost"  # Overwritten by torchrun normally
MASTER_PORT_LOCAL=29522        # Overwritten by torchrun normally

TORCHFT_LIGHTHOUSE_GLOBAL="http://10.0.0.3:29520"
TORCHFT_LIGHTHOUSE_LOCAL="http://10.0.0.3:29522"

WORLD_SIZE_CLUSTER=1
WORLD_SIZE_REPLICA_GROUP=2

LOCAL_RANK=0
GLOBAL_REPLICA_NUM=$((NUM_CLUSTERS * NUM_REPLICA_GROUPS_LOCAL_CLUSTER))
GLOBAL_REPLICA_ID=$((CLUSTER_GROUP_ID * NUM_REPLICA_GROUPS_LOCAL_CLUSTER + REPLICA_GROUP_ID))

KV_STORE_SCRIPT="/srv/apps/warren/torchft/kv_store.py"

CLUSTER_STORE_PID_FILE="/tmp/cluster_store_${CLUSTER_GROUP_ID}.pid"
CLUSTER_STORE_LOG_FILE="/tmp/cluster_store_${CLUSTER_GROUP_ID}.log"

echo "Starting server with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, REPLICA_GROUP_ID=${REPLICA_GROUP_ID}, CLUSTER_GROUP_ID=${CLUSTER_GROUP_ID}"

if [[ -f "$CLUSTER_STORE_PID_FILE" ]] && ps -p "$(cat "$CLUSTER_STORE_PID_FILE")" > /dev/null 2>&1; then
    echo ">>> Global store already running (PID $(cat "$CLUSTER_STORE_PID_FILE")). Skipping KV‑store startup."
else
    # Clean stale PID file
    if [[ -f "$CLUSTER_STORE_PID_FILE" ]]; then
        echo ">>> Removing stale PID file $CLUSTER_STORE_PID_FILE"
        rm -f "$CLUSTER_STORE_PID_FILE"
    fi

    echo ">>> Starting GLOBAL KVStore in background..."
    echo ">>> Store address: ${MASTER_ADDR_CLUSTER}:${MASTER_PORT_CLUSTER}"
    echo ">>> Log file:      ${CLUSTER_STORE_LOG_FILE}"
    echo ">>> PID file:      ${CLUSTER_STORE_PID_FILE}"

    nohup python -u "$KV_STORE_SCRIPT" \
          --host    "$MASTER_ADDR_CLUSTER" \
          --port    "$MASTER_PORT_CLUSTER" \
          --timeout 3600 \
          --pid-file "$CLUSTER_STORE_PID_FILE" \
          > "$CLUSTER_STORE_LOG_FILE" 2>&1 &

    CLUSTER_STORE_PID=$!
    echo "$CLUSTER_STORE_PID" > "$CLUSTER_STORE_PID_FILE"
    echo ">>> Global store PID: $CLUSTER_STORE_PID"
    echo ">>> Waiting a few seconds for the store to come up..."
    sleep 5
fi

# Start training script
echo ">>> Launching training script: $TRAIN_SCRIPT $@"
python "$TRAIN_SCRIPT" "$@"
