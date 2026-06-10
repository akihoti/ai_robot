#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-pretrain/yolov5s-face.onnx}"
OUTPUT_PREFIX="${2:-pretrain/yolov5s-face}"
SOC_VERSION="${SOC_VERSION:-Ascend310B1}"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

atc \
    --framework=5 \
    --model="${ONNX_PATH}" \
    --output="${OUTPUT_PREFIX}" \
    --input_format=NCHW \
    --input_shape="input:1,3,640,640" \
    --soc_version="${SOC_VERSION}"

echo "created ${OUTPUT_PREFIX}.om"
