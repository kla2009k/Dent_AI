"""ONNX Runtime ensemble predictor for panoramic dental X-rays."""

from __future__ import annotations

import json
import pathlib

import numpy as np
from PIL import Image

from onnx_runtime import ONNXCAMModel
from predictor import DISEASES, DISEASES_TH, DEFAULT_THRESHOLD, apply_fusion


ROOT = pathlib.Path(__file__).parent.parent
CONFIG_PATH = ROOT / "models" / "serving_config.json"
MODEL_SPECS = (
    (
        ROOT / "models" / "onnx" / "xray_v1_efficientnet_b0_cam.onnx",
        ROOT / "models" / "onnx" / "xray_v1_efficientnet_b0_cam.json",
    ),
    (
        ROOT / "models" / "onnx" / "xray_v2_efficientnet_b0_cam.onnx",
        ROOT / "models" / "onnx" / "xray_v2_efficientnet_b0_cam.json",
    ),
)

_models = []
_thresholds = {disease: DEFAULT_THRESHOLD for disease in DISEASES}
_model_info = {
    "loaded": False,
    "mock": True,
    "mode": None,
    "models": [],
    "val_mean_auc": None,
    "tta": False,
    "reason": "not_loaded",
}


def load_model() -> None:
    global _models, _thresholds, _model_info
    if _model_info["loaded"]:
        return
    missing = [str(model) for model, metadata in MODEL_SPECS if not model.exists() or not metadata.exists()]
    if missing:
        _model_info = {**_model_info, "loaded": True, "reason": "ONNX model not found"}
        return
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _thresholds = {
        disease: float(config.get("thresholds", {}).get(disease, DEFAULT_THRESHOLD))
        for disease in DISEASES
    }
    _models = [ONNXCAMModel(model, metadata) for model, metadata in MODEL_SPECS]
    for model in _models:
        if model.diseases != DISEASES:
            raise ValueError(f"class mismatch in {model.model_path}")
    _model_info = {
        "loaded": True,
        "mock": False,
        "mode": "ensemble",
        "backend": "onnxruntime",
        "models": [model.relative_to(ROOT).as_posix() for model, _ in MODEL_SPECS],
        "val_mean_auc": None,
        "reference_val_mean_auc": config.get("val_mean_auc"),
        "evaluation_note": "reference metric used TTA; ONNX no-TTA runtime requires reevaluation",
        "tta": False,
        "source_config_tta": bool(config.get("tta_hflip", False)),
        "thresholds": _thresholds,
        "attribution_method": "class_activation_mapping",
        "reason": None,
    }


def _mock_predict(image: Image.Image) -> dict:
    seed = int(np.asarray(image.convert("L").resize((16, 16)), dtype=np.int64).sum() % 100)
    generator = np.random.default_rng(seed)
    values = {
        "Caries": 0.3 + generator.random() * 0.6,
        "Deep Caries": generator.random() * 0.5,
        "Periapical Lesion": generator.random() * 0.4,
        "Impacted": generator.random() * 0.7,
    }
    return {name: round(float(value), 4) for name, value in values.items()}


def _detect(predictions: dict) -> list:
    return [
        disease
        for disease in DISEASES
        if predictions[disease] > _thresholds.get(disease, DEFAULT_THRESHOLD)
    ]


def predict(image: Image.Image, symptoms: dict = None, make_heatmap: bool = True) -> dict:
    load_model()
    if _model_info["mock"]:
        raw = _mock_predict(image)
        fused, notes = apply_fusion(raw, symptoms)
        detected = _detect(fused)
        return {
            "model": {"mock": True, "note": _model_info["reason"]},
            "raw_predictions": raw,
            "predictions": fused,
            "detected": detected,
            "detected_th": [DISEASES_TH[disease] for disease in detected],
            "is_normal": not detected,
            "fusion_notes": notes,
            "thresholds": _thresholds,
            "heatmaps": {},
        }

    outputs = [model.infer(image) for model in _models]
    probabilities = np.mean([result[0] for result in outputs], axis=0)
    raw = {disease: round(float(value), 4) for disease, value in zip(DISEASES, probabilities)}
    fused, notes = apply_fusion(raw, symptoms)
    detected = _detect(fused)
    _, primary_cams, primary_rgb = outputs[0]
    heatmaps = {}
    if make_heatmap:
        for disease in detected:
            index = DISEASES.index(disease)
            heatmaps[disease] = _models[0].heatmap(primary_rgb, primary_cams, index)
    return {
        "model": {
            "mock": False,
            "mode": "ensemble",
            "backend": "onnxruntime",
            "models": _model_info["models"],
            "tta": False,
            "val_mean_auc": None,
            "reference_val_mean_auc": _model_info["reference_val_mean_auc"],
            "evaluation_note": _model_info["evaluation_note"],
            "attribution_method": "class_activation_mapping",
        },
        "raw_predictions": raw,
        "predictions": fused,
        "detected": detected,
        "detected_th": [DISEASES_TH[disease] for disease in detected],
        "is_normal": not detected,
        "fusion_notes": notes,
        "thresholds": _thresholds,
        "heatmaps": heatmaps,
    }
