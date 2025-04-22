#!/bin/bash
#
# Starts TorchFT lighthouses for hierarchical LocalSGD.
# • GLOBAL lighthouse   – master host only (port 29520)
# • LOCAL  lighthouse   – every cluster leader (port 29521)
#
# Run once per physical host (systemd/tmux/nohup, as you prefer).
set -euo pipefail

WHETHER_GLOBAL=${1:-"global"} # whether to run the global lighthouse
CLUSTER_GROUP_ID=${2:-0} # ID of the local cluster
MASTER_HOSTNAME="sz-k8s-master"
NODE2_HOSTNAME="sz-k8s-node2"
MASTER_IP="10.0.0.3"
NODE2_IP="10.0.0.2"

GLOBAL_LIGHTHOUSE_PORT=29520
LOCAL_LIGHTHOUSE_PORT=$((29521 + CLUSTER_GROUP_ID))
FLAGS="--min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000"
HOSTNAME=$(hostname)

case "$HOSTNAME" in
  "$MASTER_HOSTNAME")
    HOST_IP=$MASTER_IP
    ;;
  "$NODE2_HOSTNAME")
    HOST_IP=$NODE2_IP
    ;;
  *)
    echo "Unknown host $HOSTNAME – extend script." >&2
    exit 1
    ;;
esac

# Activate the correct conda environment
MASTER_PATH="/srv/apps/warren/torchft"
MASTER_CONDA_PATH="/srv/apps/danny/miniconda3"
MASTER_ENV_NAME="warren/torchtitan"
NODE2_PATH="/root/warren/torchft"
NODE2_CONDA_PATH="/usr/local/conda"
NODE2_ENV_NAME="localsgd-warren"
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

launch () {
  local port=$1
  echo "[lighthouse] $(date '+%F %T')  ${HOST_IP}:${port}"
  export RUST_BACKTRACE=1
  exec torchft_lighthouse --bind "[::]:${port}" $FLAGS
}

if [ "$WHETHER_GLOBAL" = "global" ]; then
  # fork global lighthouse in background, keep local one in foreground
  launch $GLOBAL_LIGHTHOUSE_PORT
else
  launch $LOCAL_LIGHTHOUSE_PORT
fi

# Example:
# /srv/apps/warren/torchft/.shell_scripts/local_sgd/run_lighthouse_localsgd.sh global 0 # Global Lighthouse: [lighthouse] 2025-04-19 09:29:20  10.0.0.3:29520
# /srv/apps/warren/torchft/.shell_scripts/local_sgd/run_lighthouse_localsgd.sh local 0 # For Cluster 0: [lighthouse] 2025-04-19 09:29:55  10.0.0.3:29521
# /root/warren/torchft/.shell_scripts/local_sgd/run_lighthouse_localsgd.sh local 1 # For Cluster 1: [lighthouse] 2025-04-19 09:30:09  10.0.0.3:29522
