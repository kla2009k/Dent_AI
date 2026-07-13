---
title: DentScan AI
emoji: 🦷
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: other
---

# DentScan AI 🦷

AI คัดกรองโรคฟันจาก panoramic X-ray (hybrid: X-ray + อาการ) สำหรับ **PCSHS Symposium 2026** (Oral Thai · สาขานวัตกรรม)

> **Medical disclaimer:** โครงการนี้เป็นต้นแบบสำหรับคัดกรองเบื้องต้นและงานวิจัย/การศึกษา ไม่ใช่อุปกรณ์การแพทย์หรือการวินิจฉัย ผลลัพธ์ต้องได้รับการยืนยันโดยทันตแพทย์

[![CI](https://github.com/kla2009k/Dent_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/kla2009k/Dent_AI/actions/workflows/ci.yml)

## สถานะปัจจุบัน (2026-07-09)
- ✅ **Phase 0** — EDA DENTEX
- ✅ **Phase 1** — train 3 versions + **serving optimization**
  - single best v1 = 0.692 → **ensemble(v1+v2)+TTA = 0.723** (ทะลุ 0.70 ✅)
  - per-class threshold tuning (F1) + Grad-CAM
- ✅ **Phase 1.5** — intraoral photo track
  - ภาพถ่ายฟันสี 3 โรค: Caries / Gingivitis / Calculus
  - test mean AUC = **0.978** บน 991 ภาพ (ดู `models/intraoral/test_metrics.json`)
- ✅ **Phase 2** — เว็บ + FastAPI + symptom fusion + heatmap
  - รองรับ `modality=xray` และ `modality=photo`
  - input validation, graceful mock fallback, history, i18n, light/dark
- ✅ **Phase 3 partial** — Thai report + dental chat
  - ใช้ Gemini/Claude ได้ถ้ามี key, ไม่มี key จะ fallback เป็น rule-based สำหรับเดโม
  - `/api/report` และ `/api/chat`
- ⏳ **ถัดไป:** deploy/demo hardening, export assets, สไลด์+คลิป+เล่มสำหรับหลายเวที

## คุณภาพ/วิศวกรรม
- `models/serving_config.json` — ensemble + TTA + per-class threshold (จาก `scripts/tune_serving.py`)
- `tests/` — `pytest tests/ -v` (16 tests: fusion, threshold, API, decode, report/chat fallback)
- `scripts/gen_showcase.py` — สร้างภาพ poster (ensemble + ฟอนต์ไทย)

## เปิดเว็บ (Phase 2)
```powershell
cd server
python -m uvicorn app:app --port 8000
# เปิด browser: http://localhost:8000
```

### ติดตั้งจาก GitHub

```powershell
git clone https://github.com/kla2009k/Dent_AI.git
cd Dent_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd server
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

ไฟล์ dataset, API keys และ model checkpoints (`*.pth`) ไม่อยู่ใน repository เนื่องจากขนาดและเงื่อนไขการเผยแพร่ข้อมูล วาง checkpoint ที่ได้รับอนุญาตไว้ที่ `models/intraoral/best_model.pth` และ/หรือ path ตาม `models/serving_config.json` ก่อนเปิดเซิร์ฟเวอร์ หากไม่มี checkpoint ระบบจะแสดงสถานะ `mock` อย่างชัดเจน

### Docker

```powershell
docker build -t dentscan-ai .
docker run --rm -p 7860:7860 --env-file .env dentscan-ai
```

GitHub Pages ให้บริการได้เฉพาะไฟล์ static และไม่สามารถรัน FastAPI/PyTorch ได้ ดังนั้น repository นี้ใช้เก็บ source code และ CI ส่วน public AI demo ต้องนำ Docker image ไปวางบนบริการที่รองรับ Python backend และแนบ model checkpoint ที่มีสิทธิ์เผยแพร่แยกต่างหาก

### Public demo บน Hugging Face Spaces

ใช้ Docker Space บน CPU basic และเก็บ checkpoint ใน private model repository:

1. สร้าง private model repository แล้วอัปโหลด checkpoint เป็น `best_model.pth`
2. สร้าง Space ใหม่ เลือก `Docker` และใช้ port `7860`
3. push source code ชุดนี้ไปยัง Space repository
4. ใน Space Settings เพิ่ม Variables:
   - `HF_MODEL_REPO_ID=<username>/<private-model-repo>`
   - `HF_MODEL_FILENAME=best_model.pth`
   - `HF_REVISION=<commit-sha-or-tag>`
5. เพิ่ม Secret ชื่อ `HF_TOKEN` โดยใช้ fine-grained token ที่อ่านได้เฉพาะ model repository
6. ไม่ต้องใส่ `GEMINI_API_KEY` หรือ `ANTHROPIC_API_KEY` ใน public demo; ระบบรายงานและแชตจะใช้ rule-based fallback เพื่อป้องกันการใช้โควตาจากบุคคลภายนอก

เมื่อ Space เริ่มทำงาน ให้ตรวจ `GET /api/health` ว่า `photo.model.mock` เป็น `false` ก่อนแจก URL ให้กรรมการ

ถ้าเครื่องไม่มี `torch`/`timm`/`grad-cam` ระบบยังเปิดได้ใน mock mode เพื่ออัดเดโมและทดสอบ UI; เมื่อติดตั้ง dependency ครบจะโหลดโมเดลจริงจาก `models/serving_config.json`

## API สำคัญ
| Endpoint | ใช้ทำอะไร |
|---|---|
| `GET /api/health` | ตรวจสถานะ X-ray/photo model และ disease list |
| `POST /api/predict` | วิเคราะห์ภาพ + symptoms + heatmap (`modality=xray` หรือ `photo`) |
| `POST /api/report` | สร้างรายงานภาษาไทยจากผล AI |
| `POST /api/chat` | ถาม-ตอบเรื่องผลคัดกรอง/คำแนะนำทันตกรรม |

## โครงสร้าง
```
Project_DentScanAI/
├── ROADMAP.md              # แผน 6 phase
├── requirements.txt
├── setup_phase1.ps1        # one-shot: ลง torch + deps + โหลด data
├── notebooks/
│   ├── 00_EDA_DENTEX.py    # ✅ EDA (รันแล้ว)
│   ├── chart*.png          # ผล EDA
│   └── dentex_labels.csv
├── scripts/
│   ├── prepare_labels.py       # แตก zip → multi-label CSV
│   ├── train_multilabel.py     # train EfficientNet-B0 + focal loss
│   └── inference_gradcam.py    # predict + Grad-CAM heatmap
├── web/                    # SPA demo
├── server/                 # FastAPI + predictor + report/chat
├── models/                 # X-ray + intraoral checkpoints/metrics
├── TICTA2026/              # submission pack + checklist
└── dentex_data/            # dataset (validation โหลดแล้ว, training ยัง)
```

## วิธีเดินต่อ (เมื่อพร้อม train จริง)
```powershell
.\setup_phase1.ps1                      # ลง torch cu128 + deps + โหลด data 10GB
python scripts/prepare_labels.py        # แตก + ทำ label CSV
python scripts/train_multilabel.py      # train (~30-60 min)
python scripts/inference_gradcam.py <xray.png>   # ทดสอบ + heatmap
```

## Disease classes (4)
| EN | TH | category_id_3 |
|---|---|---|
| Caries | ฟันผุ | 1 |
| Deep Caries | ฟันผุลึก | 3 |
| Periapical Lesion | รอยโรคปลายราก | 2 |
| Impacted | ฟันคุด | 0 |

Normal = ทุก prob < 0.5 (ไม่ใช่ class แยก — DENTEX fully-labeled ทุกภาพมีโรค)

## Dataset
**DENTEX Challenge 2023** (MICCAI) · `ibrahimhamamci/DENTEX` · license CC-BY-NC-SA-4.0
- 1,005 disease-labeled panoramic + 1,571 unlabeled (pre-train)
- ⚠️ NC license = ใช้วิจัย/แข่งได้ ห้ามเชิงพาณิชย์ — ต้องเครดิตในเล่ม + poster

## Stack
PyTorch (cu128) · EfficientNet-B0 (timm) · focal loss · pytorch-grad-cam · albumentations
