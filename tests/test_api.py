"""
API smoke tests (FastAPI TestClient — ไม่ต้องเปิด uvicorn จริง)
รัน: pytest tests/test_api.py -v
หมายเหตุ: โหลด torch model จริง (ถ้ามี) — ใช้ CPU ได้
"""
import sys
import io
import base64
import pathlib
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

try:
    from fastapi.testclient import TestClient
    import app as app_module
    client = TestClient(app_module.app)
    HAS_API = True
except Exception:
    HAS_API = False


def _dummy_png_b64():
    img = Image.new("RGB", (640, 320), (90, 90, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.skipif(not HAS_API, reason="fastapi/torch not available")
def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["diseases"] == ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]


@pytest.mark.skipif(not HAS_API, reason="fastapi/torch not available")
def test_predict_shape():
    r = client.post("/api/predict",
                    json={"image": _dummy_png_b64(), "symptoms": {}, "heatmap": False})
    assert r.status_code == 200
    d = r.json()
    assert set(d["predictions"].keys()) == {"Caries", "Deep Caries",
                                            "Periapical Lesion", "Impacted"}
    assert isinstance(d["detected"], list)
    assert isinstance(d["is_normal"], bool)


@pytest.mark.skipif(not HAS_API, reason="fastapi/torch not available")
def test_predict_rejects_garbage():
    r = client.post("/api/predict",
                    json={"image": "data:image/png;base64,not_real_image_xx", "heatmap": False})
    assert r.status_code in (400, 422)


@pytest.mark.skipif(not HAS_API, reason="fastapi/torch not available")
def test_predict_rejects_empty():
    r = client.post("/api/predict", json={"image": "", "heatmap": False})
    assert r.status_code == 422   # pydantic validator


@pytest.mark.skipif(not HAS_API, reason="fastapi/torch not available")
def test_report_endpoint_rule_based_fallback(monkeypatch):
    monkeypatch.setattr(app_module.llm_report, "_provider", lambda: None)
    payload = {
        "predictions": {"Caries": 0.72, "Deep Caries": 0.21},
        "detected": ["Caries"],
        "symptoms": {"visible_hole": True},
        "modality": "xray",
    }
    r = client.post("/api/report", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert d["source"].startswith("rule-based")
    assert d["summary_layperson"]
    assert d["recommendations"]
    assert "urgency" in d


@pytest.mark.skipif(not HAS_API, reason="fastapi/torch not available")
def test_chat_endpoint_rule_based_fallback(monkeypatch):
    monkeypatch.setattr(app_module.llm_report, "_provider", lambda: None)
    payload = {
        "question": "ปวดฟันควรทำยังไง",
        "predictions": {"Caries": 0.72},
        "detected": ["Caries"],
        "symptoms": {"visible_hole": True},
        "history": [],
        "modality": "xray",
    }
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert d["source"].startswith("rule-based")
    assert d["answer"]
