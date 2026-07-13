"""
DentScan AI — core predictor (ใช้โดย FastAPI app.py)
- เสิร์ฟตาม models/serving_config.json (ensemble v1+v2 + hflip TTA + per-class threshold)
  ถ้าไม่มี config → fallback single best 4-class model
- predict 4 disease + Grad-CAM heatmap (base64) + symptom fusion
- mock fallback ถ้าไม่มี model เลย → demo ได้ทันที
"""
import io
import json
import base64
import pathlib
from typing import Optional

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
SERVING_CONFIG = MODELS_DIR / "serving_config.json"

DISEASES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
DISEASES_TH = {
    "Caries": "ฟันผุ",
    "Deep Caries": "ฟันผุลึก",
    "Periapical Lesion": "รอยโรคปลายราก",
    "Impacted": "ฟันคุด",
}
DEFAULT_THRESHOLD = 0.5
IMG_W, IMG_H = 640, 320

# lazy torch import (frontend dev ไม่ต้องมี torch)
_torch = None
_models = []          # list of (model, (w,h))
_cam = None           # Grad-CAM บน primary model
_device = None
_tta = False
_thresholds = {d: DEFAULT_THRESHOLD for d in DISEASES}
_model_info = {"loaded": False, "mock": True, "mode": None,
               "models": [], "val_mean_auc": None, "tta": False,
               "reason": "not_loaded"}


def _cfg_wh(cfg):
    if "img_w" in cfg and "img_h" in cfg:
        return cfg["img_w"], cfg["img_h"]
    if "img_size" in cfg:
        return cfg["img_size"], cfg["img_size"]
    return IMG_W, IMG_H


def _resolve_model_path(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str)
    return path if path.is_absolute() else ROOT / path


def _find_best_4class() -> Optional[pathlib.Path]:
    """fallback: best 4-class model จาก metrics.json (ข้าม 3-class)"""
    try:
        import torch
    except ModuleNotFoundError:
        return None
    best = None
    for mp in MODELS_DIR.rglob("best_model.pth"):
        mpath = mp.parent / "metrics.json"
        auc = json.load(open(mpath)).get("mean_auc", 0.0) if mpath.exists() else 0.0
        try:
            ck = torch.load(mp, map_location="cpu", weights_only=False)
            if ck.get("diseases") != DISEASES:
                continue
        except Exception:
            continue
        if best is None or auc > best[0]:
            best = (auc, mp)
    return best[1] if best else None


def _load_one(path, torch, timm):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("diseases") != DISEASES:
        raise ValueError(f"class mismatch: {path}")
    backbone = ck["cfg"].get("backbone", "efficientnet_b0")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(DISEASES))
    model.load_state_dict(ck["model"])
    model.to(_device).eval()
    return model, _cfg_wh(ck["cfg"])


def load_model():
    """โหลด serving config (idempotent). ถ้าไม่มี model → mock mode"""
    global _torch, _models, _cam, _device, _tta, _thresholds, _model_info
    if _model_info["loaded"]:
        return

    # หา model paths จาก config หรือ fallback
    paths, tta, thresholds, val_auc, mode = [], False, None, None, "single"
    if SERVING_CONFIG.exists():
        try:
            cfg = json.load(open(SERVING_CONFIG, encoding="utf-8"))
            paths = [_resolve_model_path(p) for p in cfg["models"]
                     if _resolve_model_path(p).exists()]
            tta = bool(cfg.get("tta_hflip", False))
            thresholds = cfg.get("thresholds")
            val_auc = cfg.get("val_mean_auc")
            mode = cfg.get("mode", "single")
        except Exception as e:
            print(f"[predictor] serving_config error: {e}")
    if not paths:
        bm = _find_best_4class()
        if bm:
            paths = [bm]

    def use_mock(reason):
        global _model_info
        _model_info = {"loaded": True, "mock": True, "mode": None,
                       "models": [], "val_mean_auc": None, "tta": False,
                       "reason": reason}
        print(f"[predictor] {reason} → MOCK mode")

    if not paths:
        use_mock("no model found")
        return

    try:
        import torch
        import timm
        from pytorch_grad_cam import GradCAM
    except ModuleNotFoundError as e:
        use_mock(f"missing dependency: {e.name}")
        return
    _torch = torch
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tta = tta
    if thresholds:
        _thresholds = {d: float(thresholds.get(d, DEFAULT_THRESHOLD)) for d in DISEASES}

    _models = [_load_one(p, torch, timm) for p in paths]
    # Grad-CAM บน primary (ตัวแรก = v1, 512² localization ดี)
    primary = _models[0][0]
    layer = [primary.conv_head] if hasattr(primary, "conv_head") else [primary.blocks[-1]]
    _cam = GradCAM(model=primary, target_layers=layer)

    _model_info = {
        "loaded": True, "mock": False, "mode": mode if len(_models) > 1 else "single",
        "models": [p.relative_to(ROOT).as_posix() for p in paths],
        "val_mean_auc": val_auc, "tta": _tta,
        "thresholds": _thresholds,
        "reason": None,
    }
    print(f"[predictor] loaded {len(_models)} model(s) | mode={_model_info['mode']} "
          f"tta={_tta} val_auc={val_auc}")


def _resize_tensor(pil: Image.Image, w, h):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    tfm = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    arr = np.array(pil.convert("RGB"))
    return tfm(image=arr)["image"].unsqueeze(0).to(_device)


def _np_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── ensemble inference ────────────────────────────────────
def _ensemble_probs(pil: Image.Image) -> np.ndarray:
    """เฉลี่ย sigmoid prob ข้าม model (+ hflip TTA) — แต่ละ model ใช้ res ของตัวเอง"""
    acc = np.zeros(len(DISEASES), dtype=np.float64)
    n = 0
    with _torch.no_grad():
        for model, (w, h) in _models:
            x = _resize_tensor(pil, w, h)
            logit = model(x)
            if _tta:
                logit = (logit + model(_torch.flip(x, dims=[3]))) / 2
            acc += _torch.sigmoid(logit)[0].cpu().numpy().astype(np.float64)
            n += 1
    return acc / max(n, 1)


# ── symptom fusion (rule-based) ───────────────────────────
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

    if s.get("sensitive_hot_cold"):
        boost("Caries", 0.08, "เสียวร้อน/เย็น → เพิ่มน้ำหนักฟันผุ")
        boost("Deep Caries", 0.10, "เสียวร้อน/เย็น → ผุอาจถึงโพรงประสาท")
    if s.get("pain_chew"):
        boost("Deep Caries", 0.10, "ปวดเวลาเคี้ยว → ผุลึก/อักเสบ")
        boost("Periapical Lesion", 0.12, "ปวดเวลาเคี้ยว → รอยโรคปลายราก")
    if s.get("spontaneous_pain"):
        boost("Periapical Lesion", 0.15, "ปวดเอง (ไม่กระตุ้น) → ปลายรากอักเสบ")
        boost("Deep Caries", 0.08, "ปวดเอง → ผุถึงโพรงประสาท")
    if s.get("gum_swelling"):
        boost("Periapical Lesion", 0.12, "เหงือกบวม → ฝี/รอยโรคปลายราก")
    if s.get("visible_hole"):
        boost("Caries", 0.12, "เห็นรูผุชัด → ฟันผุ")
    dur = s.get("duration_days", 0) or 0
    if dur >= 14:
        boost("Periapical Lesion", 0.08, "อาการเรื้อรัง >2 สัปดาห์ → รอยโรคสะสม")

    return adj, notes


def _mock_predict(seed_img: Image.Image) -> dict:
    h = int(np.array(seed_img.convert("L").resize((16, 16)), dtype=np.int64).sum() % 100)
    rng = np.random.default_rng(int(h))
    base = {"Caries": 0.3 + rng.random() * 0.6, "Deep Caries": rng.random() * 0.5,
            "Periapical Lesion": rng.random() * 0.4, "Impacted": rng.random() * 0.7}
    return {k: round(float(v), 4) for k, v in base.items()}


def _detect(preds: dict) -> list:
    """ใช้ per-class threshold"""
    return [d for d in DISEASES if preds[d] > _thresholds.get(d, DEFAULT_THRESHOLD)]


def predict(pil: Image.Image, symptoms: dict = None, make_heatmap=True) -> dict:
    load_model()

    if _model_info["mock"]:
        raw = _mock_predict(pil)
        fused, notes = apply_fusion(raw, symptoms)
        detected = [d for d in DISEASES if fused[d] > DEFAULT_THRESHOLD]
        return {
            "model": {"mock": True, "note": "ยังไม่มี trained model — โหมด demo"},
            "raw_predictions": raw, "predictions": fused,
            "detected": detected, "detected_th": [DISEASES_TH[d] for d in detected],
            "is_normal": len(detected) == 0, "fusion_notes": notes,
            "thresholds": _thresholds, "heatmaps": {},
        }

    probs_arr = _ensemble_probs(pil)
    raw = {d: round(float(p), 4) for d, p in zip(DISEASES, probs_arr)}
    fused, notes = apply_fusion(raw, symptoms)
    detected = _detect(fused)

    heatmaps = {}
    if make_heatmap and detected:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        primary, (pw, ph) = _models[0]
        x = _resize_tensor(pil, pw, ph)
        rgb = np.array(pil.convert("RGB").resize((pw, ph))) / 255.0
        for d in detected:
            idx = DISEASES.index(d)
            gray = _cam(input_tensor=x, targets=[ClassifierOutputTarget(idx)])[0]
            overlay = show_cam_on_image(rgb.astype(np.float32), gray, use_rgb=True)
            heatmaps[d] = _np_to_b64(overlay)

    return {
        "model": {"mock": False, "mode": _model_info["mode"],
                  "models": _model_info["models"], "tta": _model_info["tta"],
                  "val_mean_auc": _model_info["val_mean_auc"]},
        "raw_predictions": raw, "predictions": fused,
        "detected": detected, "detected_th": [DISEASES_TH[d] for d in detected],
        "is_normal": len(detected) == 0, "fusion_notes": notes,
        "thresholds": _thresholds, "heatmaps": heatmaps,
    }


if __name__ == "__main__":
    import sys
    load_model()
    print("model_info:", json.dumps(_model_info, ensure_ascii=False))
    if len(sys.argv) > 1:
        res = predict(Image.open(sys.argv[1]),
                      symptoms={"pain_chew": True, "sensitive_hot_cold": True})
        res_print = {k: v for k, v in res.items() if k != "heatmaps"}
        res_print["heatmap_count"] = len(res["heatmaps"])
        print(json.dumps(res_print, indent=2, ensure_ascii=False))
