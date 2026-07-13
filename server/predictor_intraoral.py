"""
DentScan AI — intraoral-photo predictor (modality="photo").

Separate from predictor.py (panoramic X-ray) so the X-ray serving path is
untouched.  app.py routes by request modality.

- loads models/intraoral/best_model.pth (from train_intraoral.py)
- classes come from the checkpoint (starts as ["Caries"], extends to
  Caries/Gingivitis/Calculus when retrained — no code change needed)
- per-image sigmoid probs + Grad-CAM heatmap (base64) per detected disease
- light symptom fusion (subset relevant to what a photo can show)
- mock fallback so the UI demos even before a model is trained
"""
import io
import json
import base64
import os
import pathlib
from typing import Optional

import numpy as np
from PIL import Image

from model_artifacts import ensure_model_file

ROOT = pathlib.Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "intraoral" / "best_model.pth"

# Thai labels for every disease this track may produce
DISEASES_TH = {
    "Caries": "ฟันผุ",
    "Gingivitis": "เหงือกอักเสบ",
    "Calculus": "หินปูน",
    "Discoloration": "ฟันเปลี่ยนสี",
    "Ulcer": "แผลในปาก",
}
DEFAULT_THRESHOLD = 0.5
# per-class detection thresholds — recall-favoured for screening. Caries/Calculus
# lower because moderate anterior caries and light tartar score below 0.5.
THRESHOLDS = {"Caries": 0.35, "Calculus": 0.45}


def _thr(d):
    return THRESHOLDS.get(d, DEFAULT_THRESHOLD)

_torch = None
_model = None
_cam = None
_cam_layer = None
_device = None
_img_size = 384
_diseases = ["Caries"]
_info = {"loaded": False, "mock": True, "modality": "photo",
         "model": None, "diseases": _diseases, "val_mean_auc": None,
         "reason": "not_loaded"}


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _is_render() -> bool:
    return _env_enabled("RENDER", False)


def _heatmap_enabled() -> bool:
    return _env_enabled("ENABLE_HEATMAP", not _is_render())


def _tta_enabled() -> bool:
    return _env_enabled("ENABLE_TTA", not _is_render())


def _load():
    """Idempotent load. No checkpoint → mock mode."""
    global _torch, _model, _cam_layer, _device, _img_size, _diseases, _info
    if _info["loaded"]:
        return

    def use_mock(reason):
        global _info
        _info = {"loaded": True, "mock": True, "modality": "photo",
                 "model": None, "diseases": _diseases, "val_mean_auc": None,
                 "reason": reason}
        print(f"[predictor_intraoral] {reason} → MOCK mode")

    try:
        resolved_model = ensure_model_file(MODEL_PATH)
    except Exception as exc:
        use_mock(f"model download unavailable ({type(exc).__name__})")
        return

    if resolved_model is None:
        use_mock("no model found")
        return

    try:
        import torch
        import timm
    except ModuleNotFoundError as e:
        use_mock(f"missing dependency: {e.name}")
        return
    _torch = torch
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    _diseases = ck.get("diseases", ["Caries"])
    _img_size = ck.get("cfg", {}).get("img_size", 384)
    backbone = ck.get("cfg", {}).get("backbone", "efficientnet_b0")
    _model = timm.create_model(backbone, pretrained=False, num_classes=len(_diseases))
    _model.load_state_dict(ck["model"])
    _model.to(_device).eval()

    # Keep only the target layer here. Creating HiResCAM attaches activation
    # hooks, so it is deferred until a heatmap is explicitly requested.
    if hasattr(_model, "conv_head"):
        _cam_layer = [_model.conv_head]
    else:
        _cam_layer = [_model.blocks[-1]]

    _info = {"loaded": True, "mock": False, "modality": "photo",
             "model": MODEL_PATH.relative_to(ROOT).as_posix(),
             "diseases": _diseases,
             "val_mean_auc": ck.get("metrics", {}).get("mean_auc"),
             "heatmap_enabled": _heatmap_enabled(),
             "tta_enabled": _tta_enabled(),
             "reason": None}
    print(f"[predictor_intraoral] loaded {backbone} classes={_diseases} "
          f"img={_img_size} auc={_info['val_mean_auc']}")


def _sharpen_cam(g, pct=55, gamma=1.3):
    """Tighten a Grad-CAM map onto its stronger activations: clip everything
    below the `pct`-th percentile, renormalize, apply gamma. Trims the diffuse
    background haze so the heatmap sits on the diseased region — but stays broad
    enough to cover the whole area (pct=55 covers, pct=80 shrinks to a dot)."""
    thr = np.percentile(g, pct)
    g2 = np.clip((g - thr) / (g.max() - thr + 1e-8), 0, 1)
    return g2 ** gamma


def _tensor(pil: Image.Image):
    image = pil.convert("RGB").resize((_img_size, _img_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    arr = ((arr - mean) / std).transpose(2, 0, 1).copy()
    return _torch.from_numpy(arr).unsqueeze(0).to(_device)


def _ensure_cam():
    global _cam
    if _cam is None:
        from pytorch_grad_cam import HiResCAM
        _cam = HiResCAM(model=_model, target_layers=_cam_layer)
    return _cam


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── symptom fusion (subset relevant to a visible photo) ────
def apply_fusion(probs: dict, symptoms: dict) -> tuple:
    adj = dict(probs)
    notes = []
    s = symptoms or {}

    def boost(disease, amount, reason):
        if disease in adj:
            old = adj[disease]
            adj[disease] = float(min(1.0, adj[disease] + amount))
            if adj[disease] - old > 0.01:
                notes.append(reason)

    if s.get("visible_hole"):
        boost("Caries", 0.12, "เห็นรูผุชัด → ฟันผุ")
    if s.get("sensitive_hot_cold"):
        boost("Caries", 0.08, "เสียวร้อน/เย็น → เพิ่มน้ำหนักฟันผุ")
    if s.get("gum_bleeding"):
        boost("Gingivitis", 0.15, "เหงือกเลือดออก → เหงือกอักเสบ")
    if s.get("gum_swelling"):
        boost("Gingivitis", 0.10, "เหงือกบวมแดง → เหงือกอักเสบ")
    if s.get("bad_breath"):
        boost("Calculus", 0.08, "กลิ่นปาก → คราบหินปูน")
        boost("Gingivitis", 0.06, "กลิ่นปาก → เหงือกอักเสบ")
    return adj, notes


def _mock_predict(pil: Image.Image) -> dict:
    h = int(np.array(pil.convert("L").resize((16, 16)), dtype=np.int64).sum() % 100)
    rng = np.random.default_rng(int(h))
    return {d: round(float(0.2 + rng.random() * 0.6), 4) for d in _diseases}


def predict(pil: Image.Image, symptoms: dict = None, make_heatmap=True) -> dict:
    _load()

    if _info["mock"]:
        raw = _mock_predict(pil)
        fused, notes = apply_fusion(raw, symptoms)
        detected = [d for d in _diseases if fused[d] > _thr(d)]
        return {
            "model": {"mock": True, "modality": "photo",
                      "note": "ยังไม่มี trained model — โหมด demo"},
            "raw_predictions": raw, "predictions": fused,
            "detected": detected,
            "detected_th": [DISEASES_TH.get(d, d) for d in detected],
            "is_normal": len(detected) == 0, "fusion_notes": notes,
            "thresholds": {d: _thr(d) for d in _diseases}, "heatmaps": {},
        }

    x = _tensor(pil)
    with _torch.inference_mode():
        probs_t = _torch.sigmoid(_model(x))
        if _tta_enabled():
            # Optional hflip TTA improves framing robustness but costs a second
            # forward pass, so free CPU deployments disable it by default.
            probs_t = (probs_t + _torch.sigmoid(
                _model(_torch.flip(x, dims=[3])))) / 2
        probs = probs_t[0].cpu().numpy()
    raw = {d: round(float(p), 4) for d, p in zip(_diseases, probs)}
    fused, notes = apply_fusion(raw, symptoms)
    detected = [d for d in _diseases if fused[d] > _thr(d)]

    heatmaps = {}
    if make_heatmap and detected and _heatmap_enabled():
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        cam = _ensure_cam()
        rgb = np.array(pil.convert("RGB").resize((_img_size, _img_size))) / 255.0
        for d in detected:
            idx = _diseases.index(d)
            gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(idx)])[0]
            gray = _sharpen_cam(gray)          # crisp, detection-like localization
            overlay = show_cam_on_image(rgb.astype(np.float32), gray, use_rgb=True,
                                        image_weight=0.6)
            heatmaps[d] = _np_to_b64(overlay)

    return {
        "model": {"mock": False, "modality": "photo",
                  "model": _info["model"], "val_mean_auc": _info["val_mean_auc"]},
        "raw_predictions": raw, "predictions": fused,
        "detected": detected,
        "detected_th": [DISEASES_TH.get(d, d) for d in detected],
        "is_normal": len(detected) == 0, "fusion_notes": notes,
        "thresholds": {d: _thr(d) for d in _diseases}, "heatmaps": heatmaps,
    }


def info():
    _load()
    return _info


if __name__ == "__main__":
    import sys
    print("info:", json.dumps(info(), ensure_ascii=False))
    if len(sys.argv) > 1:
        res = predict(Image.open(sys.argv[1]), symptoms={"visible_hole": True})
        res = {k: v for k, v in res.items() if k != "heatmaps"} | {
            "heatmap_count": len(predict(Image.open(sys.argv[1])).get("heatmaps", {}))}
        print(json.dumps(res, indent=2, ensure_ascii=False))
