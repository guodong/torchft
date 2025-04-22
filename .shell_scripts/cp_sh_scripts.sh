#!/bin/bash

# Where the scripts reside on the local server:
LOCAL_DIR="/srv/apps/warren/torchft/.shell_scripts"

# Where the scripts should go on the remote server:
REMOTE_USER="root"
REMOTE_HOST="10.0.0.2"
REMOTE_DIR="/root/warren/torchft/.shell_scripts"

# Create remote directory if it doesn't exist
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# Copy entire directory
rsync -avz "${LOCAL_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# Optionally, confirm success or do further actions
echo "Directory ${LOCAL_DIR} has been synchronized to ${REMOTE_HOST}:${REMOTE_DIR}."