import json
import ctypes
import ctypes.util
from pathlib import Path
from typing import Dict

import numpy as np
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _load_cudart():
    candidates = [
        "libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so.10.2",
    ]

    found = ctypes.util.find_library("cudart")
    if found:
        candidates.insert(0, found)

    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass

    raise OSError(
        "libcudart.so를 찾지 못했습니다. "
        "CUDA가 설치되어 있는지 확인하세요: ls /usr/local/cuda/lib64/libcudart*"
    )


_CUDART = _load_cudart()

_CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_CUDART.cudaMalloc.restype = ctypes.c_int

_CUDART.cudaFree.argtypes = [ctypes.c_void_p]
_CUDART.cudaFree.restype = ctypes.c_int

_CUDART.cudaMemcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]
_CUDART.cudaMemcpy.restype = ctypes.c_int

_CUDART.cudaDeviceSynchronize.argtypes = []
_CUDART.cudaDeviceSynchronize.restype = ctypes.c_int

_CUDART.cudaGetErrorString.argtypes = [ctypes.c_int]
_CUDART.cudaGetErrorString.restype = ctypes.c_char_p


def _cuda_check(status: int, message: str):
    if status != 0:
        err = _CUDART.cudaGetErrorString(status)
        err_msg = err.decode("utf-8") if err else "unknown"
        raise RuntimeError(f"{message}: CUDA error {status} ({err_msg})")


def _cuda_malloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _cuda_check(
        _CUDART.cudaMalloc(ctypes.byref(ptr), int(nbytes)),
        f"cudaMalloc failed for {nbytes} bytes",
    )
    return ptr


def _cuda_free(ptr: ctypes.c_void_p):
    if ptr is not None and ptr.value:
        _CUDART.cudaFree(ptr)


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

        self.input_idx = None
        self.output_idx = None

        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
            else:
                self.output_idx = i

        if self.input_idx is None or self.output_idx is None:
            raise RuntimeError("TensorRT input/output binding을 찾지 못했습니다.")

        self.input_shape = tuple(self.engine.get_binding_shape(self.input_idx))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_idx))

        if any(dim < 0 for dim in self.input_shape):
            raise RuntimeError(f"Dynamic input shape은 현재 코드에서 지원하지 않습니다: {self.input_shape}")

        if len(self.output_shape) == 1:
            self.output_shape = (1, self.output_shape[0])

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))

        self.h_input = np.empty(int(np.prod(self.input_shape)), dtype=self.input_dtype)
        self.h_output = np.empty(int(np.prod(self.output_shape)), dtype=self.output_dtype)

        self.d_input = _cuda_malloc(self.h_input.nbytes)
        self.d_output = _cuda_malloc(self.h_output.nbytes)

        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_idx] = int(self.d_input.value)
        self.bindings[self.output_idx] = int(self.d_output.value)

        print("TensorRT engine loaded")
        print(" input_shape:", self.input_shape, "input_dtype:", self.input_dtype)
        print(" output_shape:", self.output_shape, "output_dtype:", self.output_dtype)

    def __del__(self):
        try:
            _cuda_free(getattr(self, "d_input", None))
            _cuda_free(getattr(self, "d_output", None))
        except Exception:
            pass

    def _load_engine(self, path: str):
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(f"TensorRT engine 로드 실패: {path}")

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
            raise ValueError(
                f"Input size mismatch. got={mel.size}, expected={self.h_input.size}, "
                f"engine_input_shape={self.input_shape}"
            )

        np.copyto(self.h_input, mel)

        _cuda_check(
            _CUDART.cudaMemcpy(
                self.d_input,
                ctypes.c_void_p(self.h_input.ctypes.data),
                self.h_input.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy HostToDevice failed",
        )

        ok = self.context.execute_v2(bindings=self.bindings)
        if not ok:
            raise RuntimeError("TensorRT context.execute_v2 failed")

        _cuda_check(
            _CUDART.cudaDeviceSynchronize(),
            "cudaDeviceSynchronize failed",
        )

        _cuda_check(
            _CUDART.cudaMemcpy(
                ctypes.c_void_p(self.h_output.ctypes.data),
                self.d_output,
                self.h_output.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy DeviceToHost failed",
        )

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
