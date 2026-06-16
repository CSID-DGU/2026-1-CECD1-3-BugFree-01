#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

curl -L -o yamnet.tflite \
  "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"

curl -L -o yamnet_class_map.csv \
  "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

ls -lh yamnet.tflite yamnet_class_map.csv
