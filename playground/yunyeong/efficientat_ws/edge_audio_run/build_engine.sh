#!/usr/bin/env bash
set -e
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -x "$TRTEXEC" ]; then
  TRTEXEC="$(which trtexec || true)"
fi
if [ -z "$TRTEXEC" ]; then
  echo "trtexec not found. Check JetPack/TensorRT installation."
  exit 1
fi

mkdir -p deploy
$TRTEXEC \
  --onnx=deploy/efficientat_sound.onnx \
  --saveEngine=deploy/efficientat_sound_fp16.engine \
  --fp16 \
  --workspace=1024 \
  --verbose | tee deploy/trtexec_build.log

ls -lh deploy/efficientat_sound_fp16.engine
