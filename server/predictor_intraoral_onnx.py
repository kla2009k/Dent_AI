"""ONNX Runtime predictor for ordinary intraoral smartphone photos."""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

from onnx_runtime import ONNXCAMModel
from predictor_intraoral import DISEASES_TH, apply_fusion, _thr


ROOT = pathlib.Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "onnx" / "intraoral_efficientnet_b0_cam.onnx"
METADATA_PATH = ROOT / "models" / "onnx" / "intraoral_efficientnet_b0_cam.json"

_model = None
_diseases = ["Caries", "Gingivitis", "Calculus"]
_info = {
    "loaded": False,
    "mock": True,
    "modality": "photo",
    "model": None,
    "diseases": _diseases,
    "val_mean_auc": None,
    "reason": "not_loaded",
}


def _load() -> None:
    global _model, _diseases, _info
    if _info["loaded"]:
        return
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        _info = {**_info, "loaded": True, "reason": "ONNX model not found"}
        return
    _model = ONNXCAMModel(MODEL_PATH, METADATA_PATH)
    _diseases = _model.diseases
    _info = {
        "loaded": True,
        "mock": False,
        "modality": "photo",
        "backend": "onnxruntime",
        "model": MODEL_PATH.relative_to(ROOT).as_posix(),
        "diseases": _diseases,
        "val_mean_auc": _model.val_mean_auc,
        "heatmap_enabled": True,
        "tta_enabled": False,
        "attribution_method": "class_activation_mapping",
        "reason": None,
    }


def _mock_predict(image: Image.Image) -> dict:
    seed = int(np.asarray(image.convert("L").resize((16, 16)), dtype=np.int64).sum() % 100)
    generator = np.random.default_rng(seed)
    return {disease: round(float(0.2 + generator.random() * 0.6), 4) for disease in _diseases}


def predict(image: Image.Image, symptoms: dict = None, make_heatmap: bool = True) -> dict:
    _load()
    if _info["mock"]:
        raw = _mock_predict(image)
        fused, notes = apply_fusion(raw, symptoms)
        detected = [disease for disease in _diseases if fused[disease] > _thr(disease)]
        return {
            "model": {"mock": True, "modality": "photo", "note": _info["reason"]},
            "raw_predictions": raw,
            "predictions": fused,
            "detected": detected,
            "detected_th": [DISEASES_TH.get(disease, disease) for disease in detected],
            "is_normal": not detected,
            "fusion_notes": notes,
            "thresholds": {disease: _thr(disease) for disease in _diseases},
            "heatmaps": {},
        }

    probabilities, cams, rgb = _model.infer(image)
    raw = {disease: round(float(value), 4) for disease, value in zip(_diseases, probabilities)}
    fused, notes = apply_fusion(raw, symptoms)
    detected = [disease for disease in _diseases if fused[disease] > _thr(disease)]
    heatmaps = {}
    if make_heatmap:
        for disease in detected:
            index = _diseases.index(disease)
            heatmaps[disease] = _model.heatmap(rgb, cams, index, sharpen=True)
    return {
        "model": {
            "mock": False,
            "modality": "photo",
            "backend": "onnxruntime",
            "model": _info["model"],
            "val_mean_auc": _info["val_mean_auc"],
            "attribution_method": "class_activation_mapping",
        },
        "raw_predictions": raw,
        "predictions": fused,
        "detected": detected,
        "detected_th": [DISEASES_TH.get(disease, disease) for disease in detected],
        "is_normal": not detected,
        "fusion_notes": notes,
        "thresholds": {disease: _thr(disease) for disease in _diseases},
        "heatmaps": heatmaps,
    }


def info() -> dict:
    _load()
    return _info
