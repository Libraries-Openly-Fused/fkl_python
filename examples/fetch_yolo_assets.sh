#!/usr/bin/env bash
# Fetch the assets for examples/11_yolo_inference.py:
#   bus.jpg      ultralytics sample image (810x1080)
#   yolov8n.onnx COCO-pretrained YOLOv8n exported to ONNX (1x3x640x640)
set -euo pipefail
cd "$(dirname "$0")/assets" 2>/dev/null || { mkdir -p "$(dirname "$0")/assets"; cd "$(dirname "$0")/assets"; }

if [ ! -f bus.jpg ]; then
  curl -sL -o bus.jpg \
    https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
  echo "downloaded bus.jpg"
fi

if [ ! -f yolov8n.onnx ]; then
  curl -sL -o yolov8n.onnx \
    "https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/main/yolov8n-onnx-web/yolov8n.onnx"
  echo "downloaded yolov8n.onnx"
fi

python3 - <<'EOF'
import struct
ok = open("yolov8n.onnx", "rb").read(4)[:2] == b"\x08\x08"
print("yolov8n.onnx looks like ONNX protobuf:", ok)
EOF
echo "assets ready"
