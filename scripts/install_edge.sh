#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/ai_robot}"
CONFIG_DIR="${CONFIG_DIR:-/etc/ai-robot}"
SERVICE_FILE="${SERVICE_FILE:-/etc/systemd/system/ai-robot-edge.service}"

python3 -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install -e "${APP_DIR}[audio,vision]"

install -d "${CONFIG_DIR}"
if [[ ! -f "${CONFIG_DIR}/edge.yaml" ]]; then
  install -m 0640 "${APP_DIR}/config/edge.example.yaml" "${CONFIG_DIR}/edge.yaml"
fi

install -m 0644 "${APP_DIR}/deploy/ai-robot-edge.service" "${SERVICE_FILE}"
systemctl daemon-reload

echo "Edit ${CONFIG_DIR}/edge.yaml, then run:"
echo "  systemctl enable --now ai-robot-edge"
