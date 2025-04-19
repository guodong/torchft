#!/bin/bash

# Where the scripts reside on the local server:
LOCAL_DIR="/srv/apps/warren/torchft/.shell_scripts"

# Where the scripts should go on the remote server:
REMOTE_USER="root"
REMOTE_HOST="10.0.0.2"
REMOTE_DIR="/root/warren/torchft/.shell_scripts"

# Explicit paths for clarity:
LOCAL_LIGHTHOUSE="${LOCAL_DIR}/run_lighthouse.sh"
LOCAL_SERVER="${LOCAL_DIR}/run_server.sh"
LOCAL_CP_SCRIPTS="${LOCAL_DIR}/cp_scripts.sh"

REMOTE_LIGHTHOUSE="${REMOTE_DIR}/run_lighthouse.sh"
REMOTE_SERVER="${REMOTE_DIR}/run_server.sh"
REMOTE_CP_SCRIPTS="${REMOTE_DIR}/cp_scripts.sh"

# Create remote directory if it doesn't exist
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# Copy entire directory
rsync -avz "${LOCAL_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# Optionally, confirm success or do further actions
echo "Directory ${LOCAL_DIR} has been synchronized to ${REMOTE_HOST}:${REMOTE_DIR}."