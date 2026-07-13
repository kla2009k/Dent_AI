# บทที่ 2 — งานวิจัยที่เกี่ยวข้อง: แหล่งอ้างอิง (verified)

> รวบรวมจาก deep-research 2026-06-14 · ทุกฉบับตรวจสอบ author/ปี/วารสารจากเว็บจริงแล้ว
> ใช้คู่กับเล่ม `DentScan_Report.docx` หัวข้อ 2.6

## บรรณานุกรมที่ใช้ในบท 2 (ตรงกับเลขในเล่ม)

| # | อ้างอิง | หัวข้อ | ลิงก์ |
|---|---------|--------|-------|
| 1 | Hamamci, I. E., et al. (2023). DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-ray. *arXiv:2305.19112* / MICCAI 2023 | ชุดข้อมูลหลัก | https://arxiv.org/abs/2305.19112 |
| 2 | Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019* | สถาปัตยกรรม | https://arxiv.org/abs/1905.11946 |
| 3 | Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017* | XAI | https://arxiv.org/abs/1610.02391 |
| 4 | Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017* | class imbalance | https://arxiv.org/abs/1708.02002 |
| 5 | Oztekin, F., et al. (2023). An Explainable Deep Learning Model to Prediction Dental Caries Using Panoramic Radiograph Images. *Diagnostics, 13(2), 226* | ฟันผุ + XAI | https://pmc.ncbi.nlm.nih.gov/articles/PMC9858273/ |
| 6 | Vinayahalingam, S., et al. (2021). Classification of caries in third molars on panoramic radiographs using deep learning. *Scientific Reports, 11, 12609* | ฟันผุ (ฟันกราม 3) | https://www.nature.com/articles/s41598-021-92121-2 |
| 7 | Çelik, B., et al. (2023). The role of deep learning for periapical lesion detection on panoramic radiographs. *Dentomaxillofacial Radiology, 52(8), 20230118* | รอยโรคปลายราก | https://pubmed.ncbi.nlm.nih.gov/37641964/ |
| 8 | Çelik, M. E. (2022). Deep Learning Based Detection Tool for Impacted Mandibular Third Molar Teeth. *Diagnostics, 12(4), 942* | ฟันคุด | https://pmc.ncbi.nlm.nih.gov/articles/PMC9025752/ |
| 9 | Golkarieh, A., et al. (2025). Advanced Deep Learning Techniques for Classifying Dental Conditions Using Panoramic X-Ray Images. *arXiv:2508.21088* | multi-disease | https://arxiv.org/abs/2508.21088 |

## ตัวเลขสำคัญที่อ้างในเล่ม (มาจาก source จริง)
- **[5] Oztekin 2023:** ResNet-50, ความแม่นยำ ~92%, Grad-CAM heatmap
- **[6] Vinayahalingam 2021:** AUC ~0.91 จำแนกฟันผุในฟันกรามซี่ที่ 3
- **[7] Çelik B. 2023:** mAP 0.83–0.95, accuracy 0.67–0.81, RetinaNet ดีสุด (F1 ~0.90); Grad-CAM ตรงจุดคลินิก >84% ของเคส
- **[9] Golkarieh 2025:** เทียบ EfficientNet/DenseNet/ResNet; ชี้ปัญหา multi-label + class imbalance

## หมายเหตุ
- บท 2.6 แบ่งเป็น 2.6.1 ฟันผุ / 2.6.2 รอยโรคปลายราก / 2.6.3 ฟันคุด / 2.6.4 multi-label+DENTEX / 2.6.5 XAI / 2.6.6 gap
- citation ในเล่มเป็นแบบ numbered [n] ตรงกับตารางนี้ทุกตัว ไม่มี placeholder
