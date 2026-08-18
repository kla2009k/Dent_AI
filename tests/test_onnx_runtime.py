import pathlib
import sys

import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from onnx_runtime import normalize_cam, preprocess_image  # noqa: E402


def test_preprocess_respects_rectangular_model_dimensions():
    tensor, rgb = preprocess_image(Image.new("RGB", (900, 500)), width=640, height=320)
    assert tensor.shape == (1, 3, 320, 640)
    assert tensor.dtype == np.float32
    assert rgb.shape == (320, 640, 3)
    assert rgb.dtype == np.float32


def test_cam_normalization_clips_negative_values():
    cam = np.array([[-2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    result = normalize_cam(cam, width=20, height=10)
    assert result.shape == (10, 20)
    assert result.dtype == np.float32
    assert result.min() == 0.0
    assert result.max() == 1.0


def test_constant_cam_returns_zero_map():
    result = normalize_cam(np.ones((8, 8), dtype=np.float32), width=32, height=16)
    assert not result.any()


def test_production_image_uses_onnx_without_pytorch():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8")
    blueprint = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "INFERENCE_BACKEND=onnx" in dockerfile
    assert "torch" not in dockerfile.lower()
    assert "onnxruntime" in requirements
    assert "timm" not in requirements
    assert "plan: free" in blueprint
    assert "ENABLE_HEATMAP" not in blueprint
