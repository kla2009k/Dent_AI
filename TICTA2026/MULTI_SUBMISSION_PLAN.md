# DentScan AI — Multi-Submission Plan

อัปเดต: 2026-07-09

เป้าหมาย: ทำชุดส่งกลาง 1 ชุด แล้วปรับภาษา/ความยาวให้เหมาะกับแต่ละเวที โดยไม่แต่ง metric ใหม่และไม่กล่าวเกินขอบเขตของ AI คัดกรอง

## สถานะส่งได้ตอนนี้

| ชิ้นงาน | สถานะ | ไฟล์/หลักฐาน |
|---|---|---|
| Web demo | พร้อมเดโมแบบ local + mock fallback | `web/`, `server/app.py` |
| X-ray model | พร้อมอธิบายผล | `models/metrics.json`, `models/serving_config.json` |
| Photo model | พร้อมเป็นจุดขยายผลงาน | `models/intraoral/test_metrics.json` |
| Thai report/chat | พร้อมเดโม | `server/llm_report.py`, `/api/report`, `/api/chat` |
| Tests | ผ่าน 16 รายการ | `pytest tests/ -v` |
| Report draft | มีไฟล์แล้ว ต้องตรวจรูปแบบตามเวที | `report/DentScan_Report.docx`, `.pdf` |
| TICTA pack | มี draft ข้อมูลฟอร์ม+บทวิดีโอ | `TICTA2026/submission_pack.md` |

## Core Claim ที่ใช้ร่วมทุกเวที

DentScan AI เป็นระบบ AI ช่วยคัดกรองโรคช่องปากจากภาพ 2 ประเภท:

1. ภาพเอกซเรย์พาโนรามิก: ตรวจ 4 โรค คือ ฟันผุ ฟันผุลึก รอยโรคปลายราก และฟันคุด พร้อม Grad-CAM heatmap
2. ภาพถ่ายฟันสี: ตรวจฟันผุ เหงือกอักเสบ และหินปูน พร้อม heatmap
3. ผสานแบบประเมินอาการเพื่อช่วยจัดลำดับความเร่งด่วน
4. สร้างรายงานภาษาไทยและแชตตอบคำถามเบื้องต้น
5. เป็น decision-support เท่านั้น ไม่แทนการวินิจฉัยของทันตแพทย์

## ตัวเลขที่ใช้ได้

| ส่วน | Metric | หมายเหตุ |
|---|---:|---|
| X-ray single v1 | mean AUC 0.6922 | จาก `models/metrics.json` |
| X-ray ensemble + TTA | val mean AUC 0.7229 | จาก `models/serving_config.json`; val เล็ก ต้อง disclose |
| Photo 3-class test | mean AUC 0.9776 | จาก `models/intraoral/test_metrics.json`, n=991 |
| API/unit tests | 16 passed | หลังแก้ fallback วันที่ 2026-07-09 |

ห้ามพูดว่า "วินิจฉัยได้แม่นยำระดับแพทย์" หรือ "ใช้แทนทันตแพทย์ได้" ให้พูดว่า "ช่วยคัดกรอง", "ช่วยชี้จุดสงสัย", "ช่วยจัดลำดับส่งต่อ"

## ชุดส่งกลางที่ควรทำต่อ

1. Demo video master 4:45 นาที
   - ใช้บทใน `submission_pack.md`
   - ต้องมี screen-record: upload → result → heatmap → report → chat → history
   - ใส่ disclaimer บนจอช่วงท้าย: "AI screening only, dentist confirmation required"

2. Slide deck กลาง 12 หน้า
   - Problem
   - Target users
   - Solution overview
   - Workflow
   - X-ray model
   - Photo model
   - Explainability
   - Symptom fusion/report
   - Engineering quality/tests
   - Ethics/PDPA/local processing
   - Demo screenshots
   - Roadmap

3. One-page executive summary
   - สำหรับแนบฟอร์ม/กรรมการอ่านเร็ว
   - มี QR ไปวิดีโอและ demo
   - มี metric จริง 3 บรรทัด

4. Technical appendix
   - dataset/license
   - model architecture
   - train/eval scripts
   - limitations
   - test commands

## ปรับตามเวที

| เวที | เน้น | ความยาว/รูปแบบ |
|---|---|---|
| TICTA 2026 | innovation + function + quality standard | video <= 5:00, summary <= 200 chars |
| PCSHS Symposium | งานวิจัยนักเรียน + oral Thai | slide 10-12 หน้า, Q&A เทคนิคได้ |
| โครงงานวิทย์/นวัตกรรมโรงเรียน | กระบวนการทดลองและผลลัพธ์ | รายงานเต็ม + poster |
| Portfolio/รางวัลเยาวชน | impact + engineering maturity | one-page + video link |

## งานค้างก่อนส่งจริง

- [ ] อัดวิดีโอเดโมจริงและจับเวลาไม่เกิน 5 นาที
- [ ] เติมสถิติทันตแพทย์/การเข้าถึงบริการจากแหล่งราชการก่อนอัดเสียง
- [ ] ตรวจชื่อโรงเรียน ทีม และสมาชิกให้ตรงทุกฟอร์ม
- [ ] สร้างสไลด์ 12 หน้า
- [ ] สร้าง one-page summary พร้อม QR
- [ ] ตรวจ license/credit: DENTEX = non-commercial, Zenodo = CC BY 4.0, Kaggle dataset ต้องให้เครดิต
- [ ] ทดสอบเดโม offline: ไม่มี internet, ไม่มี LLM key, ยังต้องใช้ได้
- [ ] ถ้าจะโชว์โมเดลจริง ให้ติดตั้ง `torch`, `timm`, `grad-cam` ให้ครบก่อนวันอัด

## Command ตรวจความพร้อม

```powershell
cd "C:\Users\LENOVO LEGION5\Desktop\claude work space\Projects\Project_DentScanAI"
pytest tests/ -v
cd server
python -m uvicorn app:app --port 8000
```

เปิดเว็บที่ `http://localhost:8000`
