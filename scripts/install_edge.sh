#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/ai_robot}"
CONFIG_DIR="${CONFIG_DIR:-/etc/ai-robot}"
SERVICE_FILE="${SERVICE_FILE:-/etc/systemd/system/ai-robot-edge.service}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ai-robot}"

# Create conda environment with Python 3.10+
if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
fi

# Activate and install dependencies
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"
pip install --upgrade pip
pip install -e "${APP_DIR}[audio,vision]"

# Source CANN environment
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Install config
install -d "${CONFIG_DIR}"
if [[ ! -f "${CONFIG_DIR}/edge.yaml" ]]; then
    install -m 0640 "${APP_DIR}/config/edge.example.yaml" "${CONFIG_DIR}/edge.yaml"
fi

# Install systemd service
install -m 0644 "${APP_DIR}/deploy/ai-robot-edge.service" "${SERVICE_FILE}"
systemctl daemon-reload

echo "Edit ${CONFIG_DIR}/edge.yaml, then run:"
echo "  systemctl enable --now ai-robot-edge"
