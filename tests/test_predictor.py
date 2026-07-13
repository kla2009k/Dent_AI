"""
Tests for DentScan AI predictor + fusion logic.
รัน: pytest tests/ -v   (จาก root Project_DentScanAI)
ไม่ต้องโหลด GPU model — เทสต์ fusion/threshold/decode logic ล้วน
"""
import sys
import pathlib
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

import predictor  # noqa: E402


# ── fusion logic ──────────────────────────────────────────
def test_fusion_no_symptoms_unchanged():
    base = {"Caries": 0.4, "Deep Caries": 0.2, "Periapical Lesion": 0.1, "Impacted": 0.3}
    adj, notes = predictor.apply_fusion(base, None)
    assert adj == base
    assert notes == []


def test_fusion_pain_chew_boosts_periapical_and_deep():
    base = {"Caries": 0.4, "Deep Caries": 0.2, "Periapical Lesion": 0.1, "Impacted": 0.3}
    adj, notes = predictor.apply_fusion(base, {"pain_chew": True})
    assert adj["Periapical Lesion"] > base["Periapical Lesion"]
    assert adj["Deep Caries"] > base["Deep Caries"]
    assert adj["Caries"] == base["Caries"]      # ไม่กระทบ caries
    assert len(notes) >= 1


def test_fusion_clamps_at_one():
    base = {"Caries": 0.95, "Deep Caries": 0.5, "Periapical Lesion": 0.5, "Impacted": 0.5}
    adj, _ = predictor.apply_fusion(base, {"visible_hole": True})
    assert adj["Caries"] <= 1.0


def test_fusion_chronic_duration_boosts_periapical():
    base = {"Caries": 0.3, "Deep Caries": 0.3, "Periapical Lesion": 0.3, "Impacted": 0.3}
    adj, _ = predictor.apply_fusion(base, {"duration_days": 30})
    assert adj["Periapical Lesion"] > base["Periapical Lesion"]


# ── threshold detection ───────────────────────────────────
def test_detect_uses_thresholds():
    predictor._thresholds = {"Caries": 0.35, "Deep Caries": 0.3,
                             "Periapical Lesion": 0.3, "Impacted": 0.45}
    preds = {"Caries": 0.4, "Deep Caries": 0.25, "Periapical Lesion": 0.31, "Impacted": 0.4}
    det = predictor._detect(preds)
    assert "Caries" in det                # 0.40 > 0.35
    assert "Deep Caries" not in det       # 0.25 < 0.30
    assert "Periapical Lesion" in det     # 0.31 > 0.30
    assert "Impacted" not in det          # 0.40 < 0.45


# ── cfg resolution derivation ─────────────────────────────
def test_cfg_wh_img_size():
    assert predictor._cfg_wh({"img_size": 512}) == (512, 512)


def test_cfg_wh_img_wh():
    assert predictor._cfg_wh({"img_w": 640, "img_h": 320}) == (640, 320)


def test_resolve_model_path_relative_to_project_root():
    assert predictor._resolve_model_path("models/best_model.pth") == ROOT / "models" / "best_model.pth"


# ── mock predict (no torch model) ─────────────────────────
def test_mock_predict_deterministic():
    img = Image.new("RGB", (100, 100), (123, 50, 200))
    a = predictor._mock_predict(img)
    b = predictor._mock_predict(img)
    assert a == b                          # deterministic จาก hash
    assert all(0 <= v <= 1 for v in a.values())
    assert set(a.keys()) == set(predictor.DISEASES)


def test_b64_roundtrip():
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    s = predictor._np_to_b64(arr)
    assert s.startswith("data:image/png;base64,")
