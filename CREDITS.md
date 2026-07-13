# Credits & Data License — DentScan AI

## Dataset: DENTEX Challenge 2023
- **Source:** Hamamci et al., *DENTEX: Dental Enumeration and Diagnosis on Panoramic X-rays Challenge* (MICCAI 2023)
- **HuggingFace:** `ibrahimhamamci/DENTEX`
- **Paper:** https://huggingface.co/papers/2305.19112
- **Project:** https://dentex.grand-challenge.org
- **License:** **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike)

### ⚠️ เงื่อนไขการใช้งาน (สำคัญสำหรับการส่งประกวด)
- **NC = NonCommercial** — ใช้เพื่อการศึกษา/วิจัย/ประกวดได้ **ห้ามใช้เชิงพาณิชย์**
- **BY = ต้องให้เครดิต** — ต้องอ้างอิงแหล่งที่มาในเล่มรายงาน + poster + สไลด์
- **SA = ShareAlike** — งานต่อยอดต้องใช้ license เดียวกัน
- ภาพ X-ray ในเว็บ (`web/samples/`) มาจาก DENTEX validation set — ใช้แสดง demo เท่านั้น

### Citation (ใส่ในเล่มรายงาน)
```
Hamamci, I.E., et al. (2023). DENTEX: An Abnormal Tooth Detection with Dental
Enumeration and Diagnosis Benchmark for Panoramic X-rays. MICCAI 2023.
```

## Dataset (intraoral photo track): Annotated Intraoral Image Dataset for Dental Caries Detection
- **Source:** *Annotated intraoral image dataset for dental caries detection*, Scientific Data (2025)
- **Paper (DOI):** https://doi.org/10.1038/s41597-025-05647-9
- **Data (Zenodo):** https://zenodo.org/records/14827784
- **License:** **CC BY 4.0** (Attribution — ใช้ได้เสรีรวมเชิงพาณิชย์ แค่ต้องอ้างอิง)
- **ขนาด:** 6,313 ภาพ intraoral (ถ่ายด้วยกล้อง) จากผู้มีอายุ 10–24 ปี · annotate caries (YOLO/COCO/VOC)
- ใช้ split Benchmarking Dataset (train 5,011 / valid 627 / test 628) → แปลงเป็น image-level Caries label
- ภาพถ่ายในเว็บ (`web/samples/photo*.jpg`) มาจาก dataset นี้ — ใช้แสดง demo

### Citation (ใส่ในเล่มรายงาน)
```
Annotated intraoral image dataset for dental caries detection.
Scientific Data (2025). https://doi.org/10.1038/s41597-025-05647-9
Data: Zenodo, https://zenodo.org/records/14827784 (CC BY 4.0)
```

## Dataset (intraoral photo — gingivitis + calculus): Kaggle "Oral Diseases"
- **Source:** Salman Sajid, *Oral Diseases* dataset, Kaggle
- **URL:** https://www.kaggle.com/datasets/salmansajid05/oral-diseases
- **ขนาด:** 12,653 ภาพ intraoral สี · 6 class (Calculus, Caries, Gingivitis, Mouth Ulcer, Tooth Discoloration, Hypodontia)
- **ใช้:** Gingivitis (2,349) + Calculus (1,296) + Caries original (219) → รวมกับ Zenodo (caries+ปกติ) เป็น multi-label 3-class
- ภาพถ่ายในเว็บ (`web/samples/photo7-10.jpg`) มาจาก dataset นี้ — ใช้แสดง demo

### Citation
```
Sajid, S. Oral Diseases dataset. Kaggle.
https://www.kaggle.com/datasets/salmansajid05/oral-diseases
```

> **หมายเหตุ license (3 แหล่ง):** DENTEX (**CC BY-NC-SA** — ห้ามเชิงพาณิชย์) · Zenodo intraoral (**CC BY**) ·
> Kaggle Oral Diseases (ตรวจ license บนหน้า Kaggle ก่อนใช้เชิงพาณิชย์). สำหรับงานประกวด/การศึกษา (non-commercial) ใช้ได้ · ต้อง cite ครบทั้ง 3 แหล่ง

## Software / Libraries
- **PyTorch** (BSD) · **timm** EfficientNet (Apache-2.0) · **pytorch-grad-cam** (MIT)
- **albumentations** (MIT) · **FastAPI** (MIT) · **Chart.js** (MIT)
- ฟอนต์ **Prompt / Sarabun** (SIL Open Font License) — Google Fonts

## Model
- EfficientNet-B0 pretrained บน ImageNet → fine-tune บน DENTEX (transfer learning)
- โมเดล/น้ำหนักที่เทรนเอง อยู่ภายใต้เงื่อนไข NC ตามต้นทางข้อมูล

## Disclaimer
เครื่องมือนี้เป็น **decision-support สำหรับคัดกรองเบื้องต้น** ไม่ใช่การวินิจฉัยทางการแพทย์
ผลลัพธ์ต้องได้รับการยืนยันจากทันตแพทย์ผู้เชี่ยวชาญเสมอ
