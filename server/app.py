"""
DentScan AI — FastAPI backend
รัน: cd server && python -m uvicorn app:app --port 8000
เปิด: http://localhost:8000

Endpoints:
  GET  /              → serve frontend (web/index.html)
  GET  /api/health    → สถานะ + model/serving info
  POST /api/predict   → {image, symptoms} → predictions + heatmaps + fusion
"""
import io
import base64
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from PIL import Image

import predictor
import predictor_intraoral
import llm_report

ROOT = pathlib.Path(__file__).parent.parent
WEB = ROOT / "web"
MAX_IMAGE_BYTES = 12 * 1024 * 1024   # 12 MB
MAX_DIM = 6000                        # กันภาพใหญ่ผิดปกติ


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_model()             # warm-up X-ray at startup
    predictor_intraoral.info()         # warm-up intraoral (mock until trained)
    yield


app = FastAPI(title="DentScan AI", version="1.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    image: str
    symptoms: dict | None = None
    heatmap: bool = True
    modality: str = "xray"          # "xray" (panoramic) | "photo" (intraoral)

    @field_validator("image")
    @classmethod
    def _check_image(cls, v):
        if not v or len(v) < 32:
            raise ValueError("image data empty/too small")
        return v

    @field_validator("modality")
    @classmethod
    def _check_modality(cls, v):
        v = (v or "xray").lower()
        if v not in ("xray", "photo"):
            raise ValueError("modality must be 'xray' or 'photo'")
        return v


@app.get("/api/health")
def health():
    predictor.load_model()
    return {"status": "ok",
            "xray": {"model": predictor._model_info,
                     "diseases": predictor.DISEASES,
                     "diseases_th": predictor.DISEASES_TH},
            "photo": {"model": predictor_intraoral.info(),
                      "diseases": predictor_intraoral.info()["diseases"],
                      "diseases_th": predictor_intraoral.DISEASES_TH},
            # back-compat: default xray fields at top level
            "model": predictor._model_info,
            "diseases": predictor.DISEASES, "diseases_th": predictor.DISEASES_TH}


def _decode_image(data: str) -> Image.Image:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception as e:
        raise HTTPException(400, f"invalid base64: {e}")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"image too large (> {MAX_IMAGE_BYTES // 1024 // 1024} MB)")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()                          # ตรวจว่าเป็นภาพจริง
        img = Image.open(io.BytesIO(raw))     # reopen หลัง verify
    except Exception as e:
        raise HTTPException(400, f"not a valid image: {e}")
    if max(img.size) > MAX_DIM:
        raise HTTPException(400, f"image dimension too large (max {MAX_DIM}px)")
    return img


@app.post("/api/predict")
def predict(req: PredictRequest):
    pil = _decode_image(req.image)
    engine = predictor_intraoral if req.modality == "photo" else predictor
    try:
        result = engine.predict(pil, symptoms=req.symptoms, make_heatmap=req.heatmap)
    except Exception as e:
        raise HTTPException(500, f"prediction failed: {e}")
    result["modality"] = req.modality
    return JSONResponse(result)


class ReportRequest(BaseModel):
    predictions: dict
    detected: list
    symptoms: dict | None = None
    modality: str = "xray"


@app.post("/api/report")
def report(req: ReportRequest):
    try:
        result = {"predictions": req.predictions, "detected": req.detected,
                  "modality": req.modality}
        return JSONResponse(llm_report.generate_report(result, req.symptoms))
    except Exception as e:
        raise HTTPException(500, f"report failed: {e}")


class ChatRequest(BaseModel):
    question: str
    predictions: dict
    detected: list
    symptoms: dict | None = None
    history: list | None = None
    modality: str = "xray"

    @field_validator("question")
    @classmethod
    def _check_q(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("question empty")
        return v[:1000]


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        result = {"predictions": req.predictions, "detected": req.detected,
                  "modality": req.modality}
        return JSONResponse(llm_report.chat(req.question, result, req.symptoms, req.history))
    except Exception as e:
        raise HTTPException(500, f"chat failed: {e}")


@app.get("/")
def index():
    idx = WEB / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return JSONResponse({"msg": "frontend not built", "api": "/api/health"})


if WEB.exists():
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
