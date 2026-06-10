#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/ai_robot}"
CONFIG_DIR="${CONFIG_DIR:-/etc/ai-robot}"
SERVICE_FILE="${SERVICE_FILE:-/etc/systemd/system/ai-robot-server.service}"
VENV_DIR="${VENV_DIR:-${APP_DIR}/.venv}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -e "${APP_DIR}"

install -d "${CONFIG_DIR}"
if [[ ! -f "${CONFIG_DIR}/server.yaml" ]]; then
    install -m 0640 "${APP_DIR}/config/server.example.yaml" "${CONFIG_DIR}/server.yaml"
fi

install -m 0644 "${APP_DIR}/deploy/ai-robot-server.service" "${SERVICE_FILE}"
systemctl daemon-reload

echo "Edit ${CONFIG_DIR}/server.yaml, then run:"
echo "  systemctl enable --now ai-robot-server"
