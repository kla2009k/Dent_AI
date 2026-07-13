# DentScan AI — ROADMAP
> สร้าง: 2026-06-10 · งาน: PCSHS Symposium 2026

## Lock แล้ว

| หัวข้อ | ค่า |
|---|---|
| งาน | PCSHS Symposium 2026 · จภ.ตรัง |
| วันโชว์ | 31 ส.ค. – 3 ก.ย. 2569 |
| Track | **Oral Thai · สาขานวัตกรรม** |
| Team | ≤ 2 คน + ครูที่ปรึกษา |
| Format | Oral Thai (onsite) + คลิป online ภาษาไทย |
| Dataset | **DENTEX Challenge 2023** (HuggingFace `ibrahimhamamci/DENTEX`) |
| AI | EfficientNet-B0 multi-label (5 คลาส) + **Grad-CAM บน panoramic X-ray** |
| Input | X-ray upload + Symptom Form (Hybrid fusion) |
| Output | 5 โรค + prob + heatmap + LLM รายงานภาษาไทย |

## Disease Classes (5)
| # | คลาส | ไทย |
|---|---|---|
| 0 | Normal | ปกติ |
| 1 | Caries | ฟันผุ |
| 2 | Deep Caries | ฟันผุลึก |
| 3 | Periapical Lesion | รอยโรครากฟัน |
| 4 | Impacted Tooth | ฟันคุด |

## Timeline

```
Phase 0  10–16 มิ.ย.    Setup + DENTEX EDA
Phase 1  17 มิ.ย.–7 ก.ค.  Train EfficientNet + Grad-CAM
Phase 2   8–21 ก.ค.    Web dashboard + HF Spaces API
Phase 3  22 ก.ค.–4 ส.ค.  Hybrid fusion + LLM Thai report
Phase 4   5–18 ส.ค.    เล่มรายงาน + สไลด์ + คลิป online
Phase 5  19–30 ส.ค.    ซ้อม + buffer
🏆        31 ส.ค.      PCSHS Symposium
```

---

## Phase 0 — Setup + Data (10–16 มิ.ย.) ✅ DONE
- [x] โหลด DENTEX validation set (142MB) จาก HF — EDA local
- [x] รัน `notebooks/00_EDA_DENTEX.py` (รันบนเครื่องตรงๆ ไม่ใช้ Colab)
- [x] class distribution + ตัวอย่างภาพ + annotated bbox example
- [x] แปลง annotation → multi-label CSV
- [x] เช็คเครื่อง: **RTX 5060 Laptop 8GB** train ได้, Python 3.14
- **ผล EDA (val 50 ภาพ):** Caries 74% / Deep Caries 40% / Impacted 32% / Periapical 14% / Normal 8%
- **ตัวเลขจริง DENTEX:** 1,005 disease-labeled + 1,571 unlabeled (pre-train)
- **Decision:** ทาง 1 (DENTEX 1,005 + augmentation) ก่อน, ถ้าไม่พอค่อยรวม dataset อื่น (ทาง 2)
- **Disease design:** 4 heads (Caries/Deep Caries/Periapical/Impacted), Normal = ทุก prob < threshold

## Phase 1 — Train + Grad-CAM ✅ DONE (เร็วกว่าแผน — 11 มิ.ย.)
- [x] train EfficientNet-B0 multi-label 3 versions (RTX 5060, cu128)
- [x] **v1** (4-class, 512², focal) → **mean AUC 0.692** ← BEST/production
- [x] v2 (4-class, 640×320, anti-overfit) → 0.669
- [x] v3 (3-class merge Caries+DeepCaries) → 0.674
- [x] Grad-CAM heatmap ต่อโรค (ชี้ตำแหน่งฟันจริง — verified บน 3 samples)
- [x] pre-resize cache แก้ dataloader bottleneck
- **Per-class v1:** Caries 0.77 / Impacted 0.78 ดี · Deep Caries 0.60 / Periapical 0.62 อ่อน (data น้อย+val 50 noisy)
- **บทเรียน:** CoarseDropout API เปลี่ยนใน albumentations 2.x · timm ดึง torch CPU ทับ cu128 ถ้าลง parallel · v1 img_size=512 ≠ predictor default → ต้อง derive จาก cfg

### Phase 1 (เดิม)
- [ ] Preprocessing: resize 512², augmentation
- [ ] Focal loss แก้ class imbalance
- [ ] EfficientNet-B0, sigmoid 5 หัว (multi-label)
- [ ] Eval: per-class AUC, F1, threshold tuning
- [ ] **Grad-CAM** overlay บน X-ray ชี้ตำแหน่งโรค
- [ ] Export `.pth` + inference script
- **Go/No-Go 7 ก.ค.:** AUC mean < 0.70 → ลดเหลือ 3 โรค (Caries/Periapical/Impacted)

## Phase 1.5 — Intraoral Photo Track 🆕 (4 ก.ค.) — ขยาย scope
> ลูกค้าขอ "สแกนฟันปกติ" (ภาพถ่ายมือถือ) เพิ่มจาก panoramic X-ray → แข่งได้กว้างขึ้น
- **Dataset:** Zenodo 14827784 "Annotated intraoral image dataset for dental caries detection"
  - 6,313 ภาพ intraoral (กล้อง/มือถือ) · **CC BY 4.0** (เสรีกว่า DENTEX) · Benchmarking split train 5011/val 627/test 628
  - YOLO bbox (caries 'd' primary + 'D' permanent) → แปลงเป็น **image-level Caries** (มี bbox=1, no-txt=normal)
  - **Verified:** สุ่มดู 5 ภาพ no-txt → ไม่มีโพรงผุจริง (normal ถูกต้อง ไม่ใช่ unlabeled) · balance ~35% caries
- **Model:** EfficientNet-B0 classification (สถาปัตย์เดียวกับ X-ray) + Grad-CAM · img 384 · cache 448 pre-resize
  - `scripts/prep_intraoral.py` (CSV) → `cache_intraoral.py` (resize) → `train_intraoral.py` (generic: อ่าน class จาก CSV header → ขยาย 3 class ไม่ต้องแก้โค้ด)
  - **⚠️ ตรวจ leakage:** AUC สูง (~0.96) ต้อง verify Grad-CAM ว่ามองฟันผุจริง + วัด test set (`eval_intraoral.py`)
- **Serving:** `server/predictor_intraoral.py` แยกไฟล์ (X-ray track ไม่แตะ = zero risk) · `app.py` route ด้วย `modality` param
- **Web:** modality switch (🩻 X-ray / 📷 Photo) หน้า analyze · result render data-driven ตาม modality · photo samples
- **ค้าง:** ขยาย 3 class (Gingivitis+Calculus) ต้อง Kaggle `salmansajid05/oral-diseases` (ต้อง kaggle.json token)

## Phase 2 — Web + API ✅ DONE (11 มิ.ย.)
- [x] FastAPI backend (`server/app.py` + `predictor.py`) → `/api/predict` + `/api/health`
- [x] SPA dashboard (`web/`) — ธีม dental teal+coral (ต่างจาก VisionCare)
- [x] หน้า: Home / Analyze / Result / History / DiseaseInfo
- [x] Symptom Form 5 อาการ + ระยะเวลา
- [x] `apply_fusion()` rule-based ปรับ prob ตามอาการ (backend) — verified boost ทำงาน
- [x] Grad-CAM heatmap แสดงในเว็บ (b64 จาก API)
- [x] i18n ไทย/EN + theme light/dark + history (localStorage)
- [x] mock fallback (demo ได้แม้ไม่มี model)
- [x] **end-to-end verified:** upload → predict → fusion → heatmap → report ครบ
- **เปิด:** `cd server && python -m uvicorn app:app --port 8000` → http://localhost:8000
- **เหลือ:** image-quality guard, export/print report, deploy HF Spaces หรือ local package สำหรับวันส่ง

### Phase 2 (เดิม)

## Phase 3 — LLM Thai + Polish (22 ก.ค.–4 ส.ค.) 🟡 PARTIAL
- [x] LLM/report endpoint ภาษาไทย (`server/llm_report.py`, `/api/report`)
- [x] Dental chat endpoint (`/api/chat`)
- [x] LLM provider fallback: Gemini/Claude ถ้ามี key, rule-based ถ้าไม่มี key/API ล่ม
- [x] API error handling: base64/ไฟล์ใหญ่/ภาพเสีย/model dependency missing → fallback หรือ error ชัดเจน
- [x] Ethics disclaimer ใน UI: "สนับสนุนการตัดสินใจ ไม่แทนทันตแพทย์"
- [ ] Image-quality guard: เตือนเมื่อภาพไม่ใช่ X-ray/photo ฟันจริงหรือคุณภาพต่ำ
- [ ] UI polish รอบสุดท้าย: mobile, empty/error states, screenshot สำหรับกรรมการ
- [ ] Export report เป็น PDF/print view

## Phase 4 — เอกสาร (5–18 ส.ค.)
- [x] Draft submission pack สำหรับ TICTA (`TICTA2026/submission_pack.md`)
- [x] เล่มรายงาน draft (`report/DentScan_Report.docx`, `.pdf`)
- [x] ใส่ตัวเลขจริงจากไฟล์ metrics หลัก
- [ ] สไลด์ Oral Thai ~10-12 slides
- [ ] คลิป online ภาษาไทย ~5 นาทีสำหรับ TICTA และ 5-10 นาทีสำหรับ PCSHS
- [ ] One-page poster / executive summary สำหรับส่งหลายรายการ
- [ ] ภาพ asset: ใช้ DENTEX licensed เท่านั้น

## Phase 5 — ซ้อม (19–30 ส.ค.)
- [ ] ซ้อมพูด + Q&A
- [ ] Demo hardening (offline fallback)
- [ ] ตัวอย่าง X-ray test หลายเคส

---

## Critical Path
1. Phase 1 model ทำงาน (ทุกอย่างต่อจากนี้)
2. deadline สมัคร onsite ของโรงเรียน (รอประกาศ — เช็คด่วน)

## Risk
| เสี่ยง | แก้ |
|---|---|
| DENTEX ลงทะเบียนยาก | ใช้ Kaggle mirror หรือ Roboflow Universe |
| multi-label AUC ต่ำ | ลดเหลือ 3 โรค + single-label |
| GPU โควต้าหมด | สลับ Colab↔Kaggle |
| LLM/API key ล่มวันเดโม | ใช้ rule-based fallback ที่มีอยู่แล้ว |

## Stack
- PyTorch + EfficientNet (timm) + Grad-CAM (`pytorch-grad-cam`)
- HuggingFace Spaces (FastAPI/Gradio)
- SPA Vanilla JS (clone VisionCare dashboard)
- Claude API หรือ invokeLLM (bilingual report)
