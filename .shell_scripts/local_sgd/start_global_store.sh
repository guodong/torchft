#!/bin/bash
# start_global_store.sh – starts the TorchFT global TCPStore on the lead process

set -euo pipefail

# ------------------------------------------------------------------
# Provide safe defaults so the script can run even if the launcher
# has not exported them (set -u would otherwise abort).
MASTER_HOSTNAME=${MASTER_HOSTNAME:-NULL}      # default leader host sentinel
CLUSTER_GROUP_ID=${CLUSTER_GROUP_ID:-0}       # treat as cluster 0
REPLICA_GROUP_ID=${REPLICA_GROUP_ID:-0}       # treat as replica 0
MASTER_ADDR_CLUSTER=${MASTER_ADDR_CLUSTER:-127.0.0.1}
MASTER_PORT_CLUSTER=${MASTER_PORT_CLUSTER:-29499}
HOSTNAME=${HOSTNAME:-$(hostname)}
# ------------------------------------------------------------------

# Expects the environment vars already set by run_server_localsgd.sh:
#   HOSTNAME MASTER_HOSTNAME CLUSTER_GROUP_ID REPLICA_GROUP_ID
#   MASTER_ADDR_CLUSTER MASTER_PORT_CLUSTER

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
KV_STORE_SCRIPT="${SCRIPT_DIR}/start_kv_store.py"   # adjust if you relocate the .py

PID_DIR="/tmp"
CLUSTER_STORE_PID_FILE="${PID_DIR}/torchft_global_store.pid"
CLUSTER_STORE_LOG_FILE="${PID_DIR}/torchft_global_store.log"

 # Only Cluster‑0 / Replica‑0 launches the store when
 #   – MASTER_HOSTNAME is "NULL"   (no explicit leader)  *or*
 #   – HOSTNAME matches MASTER_HOSTNAME
