"""Small ONNX Runtime adapter shared by the photo and X-ray predictors."""

from __future__ import annotations

import base64
import io
import json
import pathlib

import numpy as np
from PIL import Image, ImageOps


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess_image(
    image: Image.Image, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    normalized = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
    return normalized.astype(np.float32, copy=False), rgb


def normalize_cam(cam: np.ndarray, width: int, height: int) -> np.ndarray:
    values = np.asarray(cam, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"CAM must be 2D, received {values.shape}")
    values = np.maximum(values, np.float32(0.0))
    resized = np.asarray(
        Image.fromarray(values, mode="F").resize(
            (width, height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    low = float(resized.min())
    high = float(resized.max())
    if not np.isfinite(resized).all() or high - low <= np.finfo(np.float32).eps:
        return np.zeros((height, width), dtype=np.float32)
    return ((resized - low) / (high - low)).astype(np.float32, copy=False)


def sharpen_cam(cam: np.ndarray, percentile: float = 55.0, gamma: float = 1.3) -> np.ndarray:
    threshold = float(np.percentile(cam, percentile))
    maximum = float(cam.max())
    if maximum - threshold <= np.finfo(np.float32).eps:
        return np.zeros_like(cam)
    clipped = np.clip((cam - threshold) / (maximum - threshold), 0.0, 1.0)
    return np.power(clipped, gamma).astype(np.float32, copy=False)


def overlay_data_url(rgb: np.ndarray, cam: np.ndarray, image_weight: float = 0.6) -> str:
    base = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    heat = Image.fromarray(np.clip(cam * 255.0, 0, 255).astype(np.uint8), mode="L")
    colored = ImageOps.colorize(heat, black="#172554", mid="#facc15", white="#dc2626")
    overlay = Image.blend(colored, base, alpha=float(image_weight))
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponentials = np.exp(values[~positive])
    result[~positive] = exponentials / (1.0 + exponentials)
    return result


class ONNXCAMModel:
    def __init__(self, model_path: pathlib.Path, metadata_path: pathlib.Path):
        import onnxruntime as ort

        self.model_path = pathlib.Path(model_path)
        metadata = json.loads(pathlib.Path(metadata_path).read_text(encoding="utf-8"))
        self.diseases = list(metadata["diseases"])
        self.width = int(metadata["width"])
        self.height = int(metadata["height"])
        self.backbone = metadata["backbone"]
        self.val_mean_auc = metadata.get("val_mean_auc")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensor, rgb = preprocess_image(image, self.width, self.height)
        logits, cams = self.session.run(None, {self.input_name: tensor})
        return sigmoid(logits[0]), np.asarray(cams[0], dtype=np.float32), rgb

    def heatmap(self, rgb: np.ndarray, cams: np.ndarray, class_index: int, sharpen=False) -> str:
        cam = normalize_cam(cams[class_index], self.width, self.height)
        if sharpen:
            cam = sharpen_cam(cam)
        return overlay_data_url(rgb, cam)
