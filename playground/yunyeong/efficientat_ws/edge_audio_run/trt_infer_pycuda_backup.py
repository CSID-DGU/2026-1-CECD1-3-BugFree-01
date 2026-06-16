import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class TRTClassifier:
    def __init__(self, engine_path: str, labels_path: str, risk_map_path: str = None):
        self.engine_path = engine_path
        self.labels = self._load_labels(labels_path)
        self.risk_map = self._load_risk_map(risk_map_path)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_idx = None
        self.output_idx = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
            else:
                self.output_idx = i
        if self.input_idx is None or self.output_idx is None:
            raise RuntimeError("Failed to find TensorRT input/output bindings")

        self.input_shape = tuple(self.engine.get_binding_shape(self.input_idx))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_idx))
        if len(self.output_shape) == 1:
            self.output_shape = (1, self.output_shape[0])

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))

        self.h_input = cuda.pagelocked_empty(int(np.prod(self.input_shape)), self.input_dtype)
        self.h_output = cuda.pagelocked_empty(int(np.prod(self.output_shape)), self.output_dtype)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_idx] = int(self.d_input)
        self.bindings[self.output_idx] = int(self.d_output)

        print("TensorRT engine loaded")
        print(" input_shape:", self.input_shape, "input_dtype:", self.input_dtype)
        print(" output_shape:", self.output_shape, "output_dtype:", self.output_dtype)

    def _load_engine(self, path: str):
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {path}")
        return engine

    def _load_labels(self, path: str) -> Dict[str, str]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_risk_map(self, path: str):
        if path is None or not Path(path).exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def infer_logits(self, mel: np.ndarray) -> np.ndarray:
        mel = np.ascontiguousarray(mel.astype(self.input_dtype).reshape(-1))
        if mel.size != self.h_input.size:
            raise ValueError(f"Input size mismatch. got={mel.size}, expected={self.h_input.size}")

        np.copyto(self.h_input, mel)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return np.array(self.h_output, dtype=np.float32).reshape(self.output_shape)

    def predict(self, mel: np.ndarray) -> dict:
        logits = self.infer_logits(mel)[0]
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        label = self.labels[str(idx)]
        conf = float(probs[idx])
        return {
            "label": label,
            "confidence": conf,
            "risk_level": self.risk_map.get(label, "info" if label != "background" else "none"),
            "probs": probs,
        }
