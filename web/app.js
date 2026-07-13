/* ════ DentScan AI — frontend logic ════ */
const API = location.origin.includes("file") ? "http://localhost:8000" : "";

const DISEASES = [
  { en: "Caries", th: "ฟันผุ", ico: "🦷", color: "#2196F3",
    desc: "ฟันผุระยะเริ่ม-กลาง เกิดจากแบคทีเรียย่อยน้ำตาลเป็นกรดกัดเคลือบฟัน",
    cause: "คราบจุลินทรีย์ + น้ำตาล + เวลา", symptom: "เห็นจุดดำ/รู เสียวฟันเล็กน้อย",
    care: "อุดฟัน · แปรงฟันฟลูออไรด์ · ลดของหวาน" },
  { en: "Deep Caries", th: "ฟันผุลึก", ico: "🔴", color: "#FF9800",
    desc: "ฟันผุลึกเกือบ/ถึงโพรงประสาทฟัน เสี่ยงติดเชื้อในโพรงประสาท",
    cause: "ฟันผุที่ไม่ได้รักษาลุกลาม", symptom: "ปวดเวลาเคี้ยว เสียวร้อนเย็นมาก ปวดเอง",
    care: "อุดลึก/ครอบฟัน · อาจต้องรักษารากฟัน" },
  { en: "Periapical Lesion", th: "รอยโรคปลายราก", ico: "🟣", color: "#E91E63",
    desc: "การอักเสบ/ฝีที่ปลายรากฟัน จากเชื้อในโพรงประสาทที่ตายแล้ว",
    cause: "โพรงประสาทติดเชื้อ/ตาย", symptom: "ปวดตุบๆ เหงือกบวม มีตุ่มหนอง กดเจ็บ",
    care: "รักษารากฟัน (root canal) · บางกรณีถอน" },
  { en: "Impacted", th: "ฟันคุด", ico: "🦿", color: "#8b5cf6",
    desc: "ฟันที่ขึ้นไม่เต็มซี่/ฝังในกระดูก พบบ่อยที่ฟันกรามซี่สุดท้าย",
    cause: "พื้นที่ในขากรรไกรไม่พอ", symptom: "ปวดด้านใน เหงือกอักเสบ ดันฟันข้างเคียง",
    care: "ผ่าฟันคุด/ถอน · X-ray ติดตาม" },
];

// intraoral-photo track (visible-surface diseases a phone photo can show)
const DISEASES_PHOTO = [
  { en: "Caries", th: "ฟันผุ", ico: "🦷", color: "#2196F3",
    desc: "ฟันผุที่เห็นบนผิวฟัน — จุดดำ/รู/รอยสีน้ำตาลจากกรดกัดเคลือบฟัน",
    cause: "คราบจุลินทรีย์ + น้ำตาล + เวลา", symptom: "เห็นจุดดำ/รู เสียวฟัน",
    care: "อุดฟัน · แปรงฟันฟลูออไรด์ · ลดของหวาน" },
  { en: "Gingivitis", th: "เหงือกอักเสบ", ico: "🩸", color: "#EF4444",
    desc: "เหงือกแดง บวม เลือดออกง่าย จากคราบจุลินทรีย์สะสมริมเหงือก",
    cause: "คราบพลัค/หินปูนริมเหงือก", symptom: "เหงือกแดง บวม เลือดออกตอนแปรง",
    care: "ขูดหินปูน · แปรง+ไหมขัดฟัน · พบทันตแพทย์" },
  { en: "Calculus", th: "หินปูน", ico: "🪥", color: "#F59E0B",
    desc: "คราบพลัคแข็งตัวเป็นหินปูนเกาะคอฟัน/ริมเหงือก สีเหลือง-น้ำตาล",
    cause: "คราบพลัคไม่ถูกกำจัด แข็งตัว", symptom: "คราบแข็งสีเหลือง/น้ำตาลริมเหงือก",
    care: "ขูดหินปูนโดยทันตแพทย์ · ดูแลสุขอนามัยช่องปาก" },
];

// lookup by class key, spanning both modalities (result render is data-driven)
const DISEASE_META = Object.fromEntries(
  [...DISEASES, ...DISEASES_PHOTO].map(d => [d.en, d]));

// diseases present in a prediction response, with metadata, in stable order
function resultDiseases(data) {
  return Object.keys(data.predictions || {})
    .map(k => DISEASE_META[k]).filter(Boolean);
}

const I18N = {
  th: {
    brandSub: "คัดกรองโรคฟัน", navHome: "หน้าหลัก", navAnalyze: "วิเคราะห์ภาพ",
    navResult: "ผลวิเคราะห์", navHistory: "ประวัติ", navInfo: "ความรู้โรคฟัน",
    modelChecking: "กำลังเชื่อมต่อ…",
    heroTitle: "วิเคราะห์สุขภาพฟันด้วย AI — X-ray หรือ ถ่ายสด",
    heroSub: "เลือกได้ 2 แบบ: ภาพ X-ray พาโนรามิก (ตรวจ 4 โรค) หรือ ถ่ายฟันสีด้วยมือถือสดๆ (ผุ/เหงือกอักเสบ/หินปูน) → AI ชี้ตำแหน่งด้วย heatmap + รายงานภาษาไทย",
    heroCta: "เริ่มวิเคราะห์ →",
    heroDisc: "⚠️ เครื่องมือช่วยคัดกรองเบื้องต้น — ไม่ใช่การวินิจฉัยทางการแพทย์ ควรพบทันตแพทย์เพื่อยืนยัน",
    statDiseases: "โรคที่ตรวจ", statXai: "อธิบายได้ (Grad-CAM)", statHybrid: "ข้อมูล: ภาพ+อาการ", statReport: "รายงานภาษาไทย",
    diseaseListTitle: "โรคที่ระบบตรวจหา",
    analyzeTitle: "วิเคราะห์ภาพ X-ray", step1: "1. อัปโหลดภาพ X-ray พาโนรามิก",
    modXray: "ภาพ X-ray พาโนรามิก", modXraySub: "ฟิล์มรังสีทั้งปาก · 4 โรค",
    modPhoto: "ภาพถ่ายฟัน (สี)", modPhotoSub: "ถ่ายสด · ผุ/เหงือก/หินปูน",
    camBtn: "ถ่ายด้วยกล้อง (สแกนสด)", camShot: "ถ่าย", camCancel: "ยกเลิก",
    dropText: "ลากภาพมาวาง หรือคลิกเพื่อเลือก", orSample: "หรือลองภาพตัวอย่าง:",
    step2: "2. อาการ (เพิ่มความแม่นยำ)", symptomHint: "ระบบรวมข้อมูลภาพ + อาการ (hybrid) เพื่อปรับผลให้แม่นขึ้น",
    symHole: "เห็นรู/จุดดำบนฟัน", symSens: "เสียวฟันเวลาร้อน/เย็น", symChew: "ปวดเวลาเคี้ยว/กัด",
    symSpont: "ปวดเองโดยไม่กระตุ้น", symGum: "เหงือกบวม/มีหนอง", symDur: "ระยะเวลาที่มีอาการ:",
    durNone: "ไม่มีอาการ", durFew: "2-3 วัน", durWeek: "~1 สัปดาห์", durLong: "มากกว่า 2 สัปดาห์",
    analyzeBtn: "วิเคราะห์ด้วย AI", procText: "กำลังประมวลผล…",
    resultTitle: "ผลการวิเคราะห์", historyTitle: "ประวัติการวิเคราะห์", infoTitle: "ความรู้เกี่ยวกับโรคฟัน",
    navAsk: "ปรึกษา AI", navAbout: "เกี่ยวกับโปรเจค",
    askTitle: "ปรึกษาทันตกรรมกับ AI",
    askSub: "ถามได้ทุกเรื่องสุขภาพฟัน — อาการปวดฟัน ฟันคุด การดูแล การเตรียมตัวก่อนพบทันตแพทย์",
    askGreet: "สวัสดี! ผมคือผู้ช่วยทันตกรรม DentScan AI ถามเรื่องฟันได้เลย เช่น “ปวดฟันกลางคืนทำไงดี”",
    askDisc: "⚠️ คำแนะนำเบื้องต้นด้วย AI — ไม่ใช่การวินิจฉัย ควรพบทันตแพทย์เพื่อตรวจจริง",
  },
  en: {
    brandSub: "Dental Screening", navHome: "Home", navAnalyze: "Analyze",
    navResult: "Result", navHistory: "History", navInfo: "Dental Info",
    modelChecking: "Connecting…",
    heroTitle: "AI Dental Screening — X-ray or Live Photo",
    heroSub: "Two ways: panoramic X-ray (4 diseases) or a live color phone photo (caries) → AI localizes with a heatmap + a readable Thai report.",
    heroCta: "Start Analysis →",
    heroDisc: "⚠️ Preliminary screening tool — not a medical diagnosis. Consult a dentist to confirm.",
    statDiseases: "Diseases", statXai: "Explainable (Grad-CAM)", statHybrid: "Image + Symptoms", statReport: "Thai Report",
    diseaseListTitle: "Diseases Detected",
    analyzeTitle: "Analyze X-ray", step1: "1. Upload panoramic X-ray",
    modXray: "Panoramic X-ray", modXraySub: "Full-mouth radiograph · 4 diseases",
    modPhoto: "Dental Photo (color)", modPhotoSub: "Live · caries/gingivitis/calculus",
    camBtn: "Capture with camera (live)", camShot: "Capture", camCancel: "Cancel",
    dropText: "Drag image here or click to select", orSample: "Or try a sample:",
    step2: "2. Symptoms (boost accuracy)", symptomHint: "Hybrid fusion of image + symptoms for better accuracy",
    symHole: "Visible hole/dark spot", symSens: "Hot/cold sensitivity", symChew: "Pain when chewing",
    symSpont: "Spontaneous pain", symGum: "Gum swelling/pus", symDur: "Symptom duration:",
    durNone: "None", durFew: "2-3 days", durWeek: "~1 week", durLong: "More than 2 weeks",
    analyzeBtn: "Analyze with AI", procText: "Processing…",
    resultTitle: "Analysis Result", historyTitle: "History", infoTitle: "Dental Disease Info",
    navAsk: "Ask AI", navAbout: "About",
    askTitle: "Ask the AI Dental Assistant",
    askSub: "Ask anything about oral health — toothache, wisdom teeth, care, preparing for a dentist visit",
    askGreet: "Hi! I'm the DentScan AI dental assistant. Ask me anything about teeth, e.g. “What helps a toothache at night?”",
    askDisc: "⚠️ Preliminary AI advice — not a diagnosis. Please see a dentist for confirmation.",
  }
};

let lang = localStorage.getItem("ds_lang") || "th";
let currentImage = null;
let lastResult = null;
let currentModality = "photo";         // "xray" | "photo" — real photo model is the public default
let heatmapEnabled = true;

// escape dynamic text ก่อนใส่ innerHTML (defense-in-depth)
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ── i18n ──
function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const k = el.dataset.i18n;
    if (I18N[lang][k]) el.textContent = I18N[lang][k];
  });
  document.documentElement.lang = lang;
  document.getElementById("langToggle").textContent = lang === "th" ? "EN" : "TH";
}

// ── routing ──
function goto(page) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.getElementById("page-" + page).classList.add("active");
  document.querySelectorAll(".nav-item").forEach(n =>
    n.classList.toggle("active", n.dataset.page === page));
  document.getElementById("sidebar").classList.remove("open");
  if (page === "history") renderHistory();
  window.scrollTo(0, 0);
}

// ── model status ──
async function checkModel() {
  const badge = document.getElementById("modelBadge");
  const status = document.getElementById("modelStatus");
  try {
    const r = await fetch(API + "/api/health");
    const d = await r.json();
    const track = currentModality === "photo" ? d.photo : d.xray;
    const model = track?.model || d.model;
    heatmapEnabled = model.heatmap_enabled !== false;
    if (model.mock) {
      badge.className = "model-badge mock";
      status.textContent = lang === "th" ? "โหมด Demo (ยังไม่มี model)" : "Demo mode (no model)";
    } else {
      badge.className = "model-badge live";
      const score = model.val_mean_auc ?? model.mean_auc;
      const auc = score ? ` · AUC ${Number(score).toFixed(2)}` : "";
      status.textContent = (lang === "th" ? "AI พร้อม" : "AI ready") + auc;
    }
  } catch {
    badge.className = "model-badge mock";
    status.textContent = lang === "th" ? "ออฟไลน์ (เปิด server)" : "Offline (start server)";
  }
}

// ── render home diseases (X-ray 4 + photo-only extras) ──
function renderHomeDiseases() {
  const xrayEn = new Set(DISEASES.map(d => d.en));
  const photoOnly = DISEASES_PHOTO.filter(d => !xrayEn.has(d.en));
  const all = [...DISEASES, ...photoOnly];
  document.getElementById("homeDiseaseGrid").innerHTML = all.map(d => `
    <div class="disease-card" style="border-top-color:${d.color}" onclick="goto('info')">
      <div class="dc-ico">${d.ico}</div>
      <h4>${lang === "th" ? d.th : d.en}</h4>
      <p>${d.desc}</p>
    </div>`).join("");
}

// ── render info ──
function renderInfo() {
  const T = lang === "th";
  const card = d => `
    <div class="info-card" style="border-top-color:${d.color}">
      <div class="ic-head"><span class="ic-ico">${d.ico}</span><h4>${d.th} · ${d.en}</h4></div>
      <p>${d.desc}</p>
      <div class="ic-section"><strong>${T?"สาเหตุ":"Cause"}</strong><p>${d.cause}</p></div>
      <div class="ic-section"><strong>${T?"อาการ":"Symptoms"}</strong><p>${d.symptom}</p></div>
      <div class="ic-section"><strong>${T?"การดูแล/รักษา":"Care"}</strong><p>${d.care}</p></div>
    </div>`;
  // photo track diseases not already covered by the X-ray list (Caries shared)
  const xrayEn = new Set(DISEASES.map(d => d.en));
  const photoOnly = DISEASES_PHOTO.filter(d => !xrayEn.has(d.en));
  document.getElementById("infoContent").innerHTML = `
    <p class="info-group-label"><span class="eyebrow">📷 ${T?"ตรวจจากภาพถ่ายฟัน (สี)":"FROM DENTAL PHOTOS"}</span></p>
    <div class="info-grid">${[DISEASES[0], ...photoOnly].map(card).join("")}</div>
    <p class="info-group-label" style="margin-top:28px"><span class="eyebrow">🩻 ${T?"ตรวจจากภาพ X-ray พาโนรามิก":"FROM PANORAMIC X-RAY"}</span></p>
    <div class="info-grid">${DISEASES.map(card).join("")}</div>`;
}

// ── render About (project dashboard) ──
const PIPELINE = [
  { ico: "📤", th: "รับภาพ 2 แบบ", en: "Dual input", d_th: "อัปโหลด/ถ่ายสด — ภาพ X-ray พาโนรามิก หรือ ภาพถ่ายฟันสี (มือถือ)", d_en: "Upload or live-capture — panoramic X-ray or color intraoral photo" },
  { ico: "🧭", th: "เลือก modality", en: "Route by modality", d_th: "ระบบส่งเข้าโมเดลที่เหมาะกับชนิดภาพ (X-ray ↔ photo)", d_en: "Routes to the model matching the image type" },
  { ico: "🧠", th: "ตรวจด้วย AI", en: "AI inference", d_th: "EfficientNet-B0 — X-ray 4 โรค (multi-label) · photo ตรวจฟันผุ", d_en: "EfficientNet-B0 — X-ray 4 diseases · photo caries" },
  { ico: "🔥", th: "อธิบายผล", en: "Explain", d_th: "Grad-CAM heatmap ซ้อนบนภาพ ชี้ตำแหน่งที่ AI สนใจ (ปรับความเข้มได้)", d_en: "Grad-CAM overlay on the image, adjustable opacity" },
  { ico: "🔗", th: "รวมอาการ", en: "Fuse symptoms", d_th: "ปรับผลตามอาการที่ผู้ใช้กรอก (hybrid fusion)", d_en: "Adjust by reported symptoms" },
  { ico: "📝", th: "สร้างรายงาน", en: "Report", d_th: "LLM เขียนรายงานไทย 2 ระดับ (คนไข้/ทันตแพทย์) + แชทถาม-ตอบ", d_en: "LLM Thai report (patient/clinician) + chat" },
];
const TECH = ["PyTorch", "EfficientNet-B0 (timm)", "Grad-CAM", "Focal Loss", "FastAPI", "Chart.js", "DENTEX 2023", "Zenodo intraoral (CC BY)"];
// intraoral photo track — per-disease AUC on held-out TEST set (3-class)
const METRICS = [
  { dis: "Caries", th: "ฟันผุ", auc: 0.974, color: "#2196F3" },
  { dis: "Gingivitis", th: "เหงือกอักเสบ", auc: 0.983, color: "#EF4444" },
  { dis: "Calculus", th: "หินปูน", auc: 0.975, color: "#F59E0B" },
];
function renderAbout() {
  const T = lang === "th";
  const card = (cls, inner) => `<div class="about-card ${cls} reveal">${inner}</div>`;
  const metricRows = METRICS.map(m => `
    <div class="metric-row">
      <span class="m-name">${T ? m.th : m.dis}</span>
      <div class="m-bar"><div class="m-fill" style="width:${m.auc*100}%;background:${m.color}"></div></div>
      <span class="m-val">${m.auc.toFixed(2)}</span>
    </div>`).join("");
  const pipeSteps = PIPELINE.map((p, i) => `
    <div class="pipe-step">
      <div class="pipe-ico">${p.ico}</div>
      <div class="pipe-body"><b>${i+1}. ${T ? p.th : p.en}</b><span>${T ? p.d_th : p.d_en}</span></div>
    </div>`).join("");

  document.getElementById("aboutContent").innerHTML = `
    <div class="about-hero reveal">
      <div class="ah-badge">🦷 DentScan AI</div>
      <h1>${T ? "คัดกรองโรคฟันด้วย AI — จาก X-ray และภาพถ่ายฟันสี" : "AI Dental Screening — from X-rays and color photos"}</h1>
      <p>${T ? "รองรับ 2 แบบภาพ: X-ray พาโนรามิก (4 โรค) และ ภาพถ่ายฟันสีจากมือถือ ถ่ายสดตรวจ ฟันผุ/เหงือกอักเสบ/หินปูน + ฟันปกติ ผสานอาการ อธิบายด้วย Grad-CAM และรายงานไทย — เครื่องมือสนับสนุนการตัดสินใจ ไม่ใช่การวินิจฉัยแทนทันตแพทย์" : "Two input types: panoramic X-ray (4 diseases) and live color phone photos (caries / gingivitis / calculus + healthy). Grad-CAM explainability, Thai report — decision-support, not a diagnosis."}</p>
      <div class="ah-stats">
        <div><b>2</b><span>${T ? "โหมดภาพ (X-ray+ถ่ายสด)" : "input modes"}</span></div>
        <div><b>0.97</b><span>${T ? "AUC เฉลี่ยภาพถ่าย (test)" : "photo mean AUC (test)"}</span></div>
        <div><b>3+1</b><span>${T ? "โรค + ฟันปกติ" : "diseases + healthy"}</span></div>
        <div><b>10K+</b><span>${T ? "ภาพถ่ายฟันฝึก/ตรวจ" : "intraoral images"}</span></div>
      </div>
    </div>

    <div class="about-grid">
      ${card("", `<h3>🎯 ${T ? "ปัญหาที่แก้" : "Problem"}</h3><p>${T ? "โรคฟัน (ผุ/รอยโรคปลายราก/ฟันคุด) มักไม่แสดงอาการชัดในระยะแรก และการเข้าถึงทันตแพทย์ในพื้นที่ห่างไกลยังจำกัด ทำให้หลายคนตรวจพบช้าจนต้องรักษาที่ซับซ้อนขึ้น" : "Dental diseases are often silent early on, and dentist access is limited in remote areas — leading to late detection."}</p>`)}
      ${card("", `<h3>💡 ${T ? "แนวทางของเรา" : "Our solution"}</h3><p>${T ? "ให้ AI ช่วยคัดกรองเบื้องต้นจากภาพ X-ray ที่ถ่ายอยู่แล้ว ชี้จุดที่ควรสังเกตด้วย heatmap และสรุปเป็นภาษาที่คนทั่วไปเข้าใจ เพื่อช่วยให้ตัดสินใจไปพบทันตแพทย์ได้เร็วและตรงจุดขึ้น" : "AI pre-screens existing X-rays, highlights regions via heatmap, and summarizes in plain language to prompt earlier dentist visits."}</p>`)}
    </div>

    ${card("wide", `<h3>⚙️ ${T ? "ขั้นตอนการทำงาน (Pipeline)" : "How it works"}</h3><div class="pipe-grid">${pipeSteps}</div>`)}

    <div class="about-grid">
      ${card("", `<h3>📊 ${T ? "ผลแบบจำลอง — ภาพถ่ายฟัน (AUC ต่อโรค, ชุด test)" : "Photo model — per-disease AUC (TEST set)"}</h3>
        <div class="metric-list">${metricRows}</div>
        <p class="about-note">${T ? "ชุด test ~1,000 ภาพ (โมเดลไม่เคยเห็น) · mean AUC 0.978 · เสริมภาพผุด้านหน้า (Kaggle close-up) → ผุหน้าที่เคยพลาดตอนนี้ตรวจเจอ · caries threshold 0.40 (เน้น recall) · heatmap (HiResCAM@conv_head) ชี้ตรงกล่องผุ 88% · Grad-CAM เหงือก→ขอบเหงือก, หินปูน→คอฟัน (verify) · X-ray track: val AUC ~0.72" : "~1,000 held-out images · mean AUC 0.978 · added frontal caries close-ups → previously-missed anterior caries now detected · caries threshold 0.40 (recall-favoured) · caries heatmap hit 88% · Grad-CAM verified on gums/cervical · X-ray val AUC ~0.72"}</p>`)}
      ${card("", `<h3>🗂️ ${T ? "ชุดข้อมูล (ภาพถ่าย = 2 แหล่งรวมกัน)" : "Datasets"}</h3>
        <p><b>${T ? "ภาพถ่ายฟัน:" : "Photos:"}</b> ${T ? "Zenodo (ฟันผุ+ปกติ, CC BY 4.0) + Kaggle salmansajid05 (เหงือกอักเสบ/หินปูน) รวม ~10,000 ภาพ multi-label 3 โรค+ปกติ" : "Zenodo (caries+healthy, CC BY 4.0) + Kaggle salmansajid05 (gingivitis/calculus) — ~10,000 images, multi-label"}</p>
        <p><b>X-ray:</b> ${T ? "DENTEX 2023 (MICCAI) — พาโนรามิก 1,005 ภาพ" : "DENTEX 2023 — 1,005 panoramic X-rays"} <span class="about-note" style="border:0;padding:0;margin:0">CC BY-NC-SA</span></p>
        <p class="about-note">📜 ${T ? "อ้างอิงครบใน CREDITS.md" : "Full citations in CREDITS.md"}</p>`)}
    </div>

    ${(() => {
      const xrayEn = new Set(DISEASES.map(d => d.en));
      const photoOnly = DISEASES_PHOTO.filter(d => !xrayEn.has(d.en));
      const adis = d => `<div class="adis" style="border-left:4px solid ${d.color}"><b>${d.ico} ${T ? d.th : d.en}</b><span>${d.desc}</span></div>`;
      return card("wide", `<h3>🦷 ${T ? "โรคที่ระบบตรวจหา" : "Diseases detected"}</h3>
        <p class="adis-cat">📷 ${T ? "ภาพถ่ายฟัน (สี)" : "Dental photos"}</p>
        <div class="about-dis-grid">${[DISEASES[0], ...photoOnly].map(adis).join("")}</div>
        <p class="adis-cat" style="margin-top:16px">🩻 ${T ? "ภาพ X-ray พาโนรามิก" : "Panoramic X-ray"}</p>
        <div class="about-dis-grid">${DISEASES.map(adis).join("")}</div>`);
    })()}

    <div class="about-grid">
      ${card("", `<h3>🛠️ ${T ? "เทคโนโลยีที่ใช้" : "Tech stack"}</h3><div class="tech-tags">${TECH.map(t => `<span class="tech-tag">${t}</span>`).join("")}</div>`)}
      ${card("", `<h3>⚖️ ${T ? "จริยธรรม & ข้อจำกัด" : "Ethics & limits"}</h3><ul class="about-ul"><li>${T ? "เป็นเครื่องมือสนับสนุนการตัดสินใจ ไม่ใช่การวินิจฉัยทางการแพทย์" : "Decision-support, not a diagnosis"}</li><li>${T ? "ผลทุกครั้งต้องให้ทันตแพทย์ยืนยันด้วยการตรวจจริง" : "Always confirm with a dentist"}</li><li>${T ? "ความไว 77% — อาจพลาดฟันผุระยะเริ่ม/ในซอก (แนะนำถ่ายหลายมุม) · heatmap ชี้ตำแหน่งแม่น 96.5% แต่ยังเป็นตัวช่วย ไม่แทนการตรวจจริง" : "Sensitivity 77% — may miss early/interproximal caries (use multiple angles) · heatmap localizes at 96.5% but still assistive, not a diagnosis"}</li></ul>`)}
    </div>

    ${card("wide team", `<h3>👥 ${T ? "คณะผู้จัดทำ" : "Team"}</h3><div class="team-grid">
      <div class="team-mem"><span class="tm-av">🧑‍🔬</span><b>นายปัญญากร อาจหาญ</b><span>${T ? "ผู้พัฒนา" : "Developer"}</span></div>
      <div class="team-mem"><span class="tm-av">🧑‍🔬</span><b>นายปรเมษฐ์ มารศรี</b><span>${T ? "ผู้พัฒนา" : "Developer"}</span></div>
      <div class="team-mem"><span class="tm-av">🧑‍🏫</span><b>นายศตวรรษ ทัดมาลา</b><span>${T ? "ครูที่ปรึกษา" : "Advisor"}</span></div>
    </div><p class="about-note">${T ? "โรงเรียนวิทยาศาสตร์จุฬาภรณราชวิทยาลัย เลย · PCSHS Symposium 2026" : "PCSHS Loei · Symposium 2026"}</p>`)}

    <div class="about-cta reveal">
      <button class="btn-primary" onclick="goto('analyze')">${T ? "ลองวิเคราะห์ (X-ray / ภาพถ่าย) →" : "Try analyzing (X-ray / photo) →"}</button>
      <button class="btn-ghost" onclick="goto('ask')">${T ? "💬 ปรึกษา AI" : "💬 Ask AI"}</button>
    </div>`;
}

// ── upload handling ──
function setupUpload() {
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("fileInput");
  const preview = document.getElementById("preview");
  const empty = document.getElementById("dzEmpty");

  dz.onclick = () => input.click();
  dz.ondragover = e => { e.preventDefault(); dz.classList.add("drag"); };
  dz.ondragleave = () => dz.classList.remove("drag");
  dz.ondrop = e => { e.preventDefault(); dz.classList.remove("drag"); if (e.dataTransfer.files[0]) loadImage(e.dataTransfer.files[0]); };
  input.onchange = () => { if (input.files[0]) loadImage(input.files[0]); };

  function loadImage(file) {
    const reader = new FileReader();
    reader.onload = e => {
      currentImage = e.target.result;
      preview.src = currentImage; preview.hidden = false; empty.hidden = true;
      document.getElementById("analyzeBtn").disabled = false;
    };
    reader.readAsDataURL(file);
  }
  window._loadSample = src => {
    fetch(src).then(r => r.blob()).then(b => {
      const reader = new FileReader();
      reader.onload = e => {
        currentImage = e.target.result;
        preview.src = currentImage; preview.hidden = false; empty.hidden = true;
        document.getElementById("analyzeBtn").disabled = false;
      };
      reader.readAsDataURL(b);
    });
  };
}

// ── sample thumbnails (per modality) ──
// photo set covers all 4 states so the demo shows caries/gingivitis/calculus/normal
const PHOTO_SAMPLES = [
  { src: "samples/photo1.jpg", label: "ผุ", cls: "bad" },
  { src: "samples/photo5.jpg", label: "ผุ", cls: "bad" },
  { src: "samples/photo7.jpg", label: "เหงือกอักเสบ", cls: "bad" },
  { src: "samples/photo10.jpg", label: "เหงือกอักเสบ", cls: "bad" },
  { src: "samples/photo8.jpg", label: "หินปูน", cls: "bad" },
  { src: "samples/photo9.jpg", label: "หินปูน", cls: "bad" },
  { src: "samples/photo11.jpg", label: "เหงือก+หินปูน", cls: "bad" },
  { src: "samples/photo12.jpg", label: "เหงือก+หินปูน", cls: "bad" },
  { src: "samples/photo13.jpg", label: "เหงือก+หินปูน", cls: "bad" },
  { src: "samples/photo2.jpg", label: "ปกติ", cls: "ok" },
  { src: "samples/photo6.jpg", label: "ปกติ", cls: "ok" },
];
function renderSamples() {
  const box = document.getElementById("sampleThumbs");
  if (currentModality === "photo") {
    box.innerHTML = PHOTO_SAMPLES.map(s =>
      `<figure class="samp"><img src="${s.src}" onclick="_loadSample('${s.src}')" onerror="this.parentElement.style.display='none'" alt="sample ${s.label}"><figcaption class="samp-cap ${s.cls}">${s.label}</figcaption></figure>`).join("");
  } else {
    box.innerHTML = ["samples/sample1.png", "samples/sample2.png", "samples/sample3.png"].map(s =>
      `<img src="${s}" onclick="_loadSample('${s}')" onerror="this.style.display='none'" alt="sample">`).join("");
  }
}

// ── analyze ──
function getSymptoms() {
  return {
    visible_hole: document.getElementById("sym_visible_hole").checked,
    sensitive_hot_cold: document.getElementById("sym_sensitive_hot_cold").checked,
    pain_chew: document.getElementById("sym_pain_chew").checked,
    spontaneous_pain: document.getElementById("sym_spontaneous_pain").checked,
    gum_swelling: document.getElementById("sym_gum_swelling").checked,
    duration_days: parseInt(document.getElementById("sym_duration").value) || 0,
  };
}

const PROC_STEPS = [
  "โหลดภาพ X-ray", "ปรับคุณภาพภาพ (preprocess)", "ตรวจหาโรคด้วย AI (EfficientNet)",
  "สร้าง heatmap (Grad-CAM)", "รวมข้อมูลอาการ (fusion)", "สร้างรายงาน"
];

async function runAnalysis() {
  if (!currentImage) return;
  const proc = document.getElementById("processing");
  const stepsEl = document.getElementById("procSteps");
  document.querySelector(".analyze-layout").style.display = "none";
  proc.hidden = false;
  stepsEl.innerHTML = PROC_STEPS.map((s, i) =>
    `<div class="proc-step" id="ps${i}"><span class="ps-ico">○</span>${s}</div>`).join("");

  let stepTimers = PROC_STEPS.map((_, i) => setTimeout(() => {
    const el = document.getElementById("ps" + i);
    if (el) { el.classList.add("done"); el.querySelector(".ps-ico").textContent = "✓"; }
  }, 350 * i));

  try {
    const r = await fetch(API + "/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: currentImage, symptoms: getSymptoms(), heatmap: heatmapEnabled, modality: currentModality }),
    });
    const data = await r.json();
    lastResult = data;
    stepTimers.forEach(clearTimeout);
    PROC_STEPS.forEach((_, i) => { const el = document.getElementById("ps" + i); if (el) { el.classList.add("done"); el.querySelector(".ps-ico").textContent = "✓"; } });
    await new Promise(res => setTimeout(res, 400));
    proc.hidden = true;
    document.querySelector(".analyze-layout").style.display = "";
    renderResult(data);
    saveHistory(data);
    document.getElementById("navResult").disabled = false;
    goto("result");
  } catch (e) {
    proc.hidden = true;
    document.querySelector(".analyze-layout").style.display = "";
    alert((lang === "th" ? "เชื่อมต่อ server ไม่ได้ — เปิด server ก่อน\n" : "Cannot reach server\n") + e);
  }
}

// fade Grad-CAM overlay opacity (heatmap toolbar slider)
window.setHeatmapOpacity = v =>
  document.querySelectorAll(".hm-over").forEach(o => o.style.opacity = v / 100);

// ── render result ──
function renderResult(data) {
  const preds = data.predictions;
  const detected = data.detected;
  const isNormal = data.is_normal;
  const DS = resultDiseases(data);                 // diseases for THIS modality
  const dmap = Object.fromEntries(DS.map(d => [d.en, d]));

  let verdictClass = isNormal ? "normal" : (detected.length >= 2 ? "alert" : "warn");
  let verdictIco = isNormal ? "✅" : (detected.length >= 2 ? "⚠️" : "🔍");
  let verdictText = isNormal
    ? (lang === "th" ? "ไม่พบความผิดปกติชัดเจน" : "No clear abnormality")
    : (lang === "th" ? `พบความเสี่ยง ${detected.length} รายการ` : `${detected.length} finding(s)`);

  const probRows = DS.map(d => {
    const v = preds[d.en] || 0;
    return `<div class="prob-row">
      <div class="prob-head"><span>${d.ico} ${lang==="th"?d.th:d.en}</span><span class="prob-val" style="color:${d.color}">${(v*100).toFixed(0)}%</span></div>
      <div class="prob-bar"><div class="prob-fill" style="width:${v*100}%;background:${d.color}"></div></div>
    </div>`;
  }).join("");

  const heatmaps = Object.keys(data.heatmaps || {});
  const scanImg = data.image || currentImage;   // the analyzed image (kept for normal case)
  let heatmapHtml = "";
  if (heatmaps.length) {
    heatmapHtml = `
    <div class="heatmap-section">
      <h2 class="section-title"><span class="eyebrow">EXPLAINABLE AI</span> ${lang==="th"?"ตำแหน่งที่ AI สนใจ · Grad-CAM":"AI Attention · Grad-CAM"}</h2>
      <div class="heatmap-toolbar">
        <div class="hm-legend"><span>${lang==="th"?"ต่ำ":"LOW"}</span><span class="hm-scale"></span><span>${lang==="th"?"สูง":"HIGH"}</span>&nbsp;·&nbsp;${lang==="th"?"ความสนใจ AI":"AI ATTENTION"}</div>
        <label class="hm-slider">${lang==="th"?"ความเข้ม overlay":"Overlay"}
          <input type="range" min="0" max="100" value="70" oninput="setHeatmapOpacity(this.value)"></label>
      </div>
      <div class="heatmap-grid">
        ${heatmaps.map(d => { const m = dmap[d] || {color:"#1e6fd9", ico:"🦷", th:d};
          return `<div class="heatmap-item">
            <div class="hm-canvas">
              <img class="hm-base" src="${scanImg || data.heatmaps[d]}" alt="scan">
              <img class="hm-over" src="${data.heatmaps[d]}" style="opacity:.7" alt="Grad-CAM ${d}">
            </div>
            <div class="hm-cap" style="border-left:4px solid ${m.color}">${m.ico} ${lang==="th"?m.th:d}<span class="hm-pct">${(preds[d]*100).toFixed(0)}%</span></div>
          </div>`;
        }).join("")}
      </div>
    </div>`;
  } else if (scanImg) {
    // normal result: no heatmap, but still show the analyzed image
    heatmapHtml = `
    <div class="heatmap-section">
      <h2 class="section-title"><span class="eyebrow">SCAN</span> ${lang==="th"?"ภาพที่วิเคราะห์":"Analyzed image"}</h2>
      <div class="heatmap-grid">
        <div class="heatmap-item">
          <div class="hm-canvas"><img class="hm-base" src="${scanImg}" alt="scan"></div>
          <div class="hm-cap" style="border-left:4px solid var(--mint)">✅ ${lang==="th"?"ไม่พบความผิดปกติชัดเจน":"No clear finding"}</div>
        </div>
      </div>
    </div>`;
  }

  const fusionHtml = (data.fusion_notes && data.fusion_notes.length) ? `
    <div style="margin-top:14px">
      <strong style="font-size:14px">🔗 ${lang==="th"?"การปรับผลจากอาการ (hybrid fusion)":"Symptom-based adjustment"}</strong>
      ${data.fusion_notes.map(n => `<div class="fusion-note">${esc(n)}</div>`).join("")}
    </div>` : "";

  const report = buildReport(data, detected, isNormal, dmap);
  const risk = riskInfo(preds, detected, isNormal);
  let modelTag;
  if (data.model.mock) {
    modelTag = `<div class="fusion-note" style="border-left-color:var(--amber)">${lang==="th"?"⚠️ โหมด Demo — ผลจำลอง (ยังไม่ได้เสียบ trained model)":"⚠️ Demo mode — simulated"}</div>`;
  } else {
    const mode = data.model.mode === "ensemble"
      ? (lang==="th"?"Ensemble 2 โมเดล + TTA":"Ensemble (2 models) + TTA")
      : (lang==="th"?"โมเดลเดี่ยว":"single model");
    const auc = data.model.val_mean_auc ? ` · val AUC ${data.model.val_mean_auc.toFixed(3)}` : "";
    modelTag = `<div class="fusion-note">🤖 ${esc(mode)}${esc(auc)}</div>`;
  }

  document.getElementById("resultContent").innerHTML = `
    <div class="result-summary reveal">
      <div class="card verdict-card">
        <div class="verdict-big">${verdictIco}</div>
        <div class="verdict-badge ${verdictClass}">${verdictText}</div>
        <div class="gauge-wrap">
          <canvas id="riskGauge" width="180" height="110"></canvas>
          <div class="gauge-center">
            <span class="gauge-pct" data-countup="${(risk.score*100).toFixed(0)}">0</span><span class="gauge-unit">%</span>
            <span class="gauge-lbl" style="color:${risk.color}">${lang==="th"?risk.th:risk.en}</span>
          </div>
        </div>
        ${modelTag}
      </div>
      <div class="card xai-card">
        <h3>📊 ${lang==="th"?"ความน่าจะเป็นของแต่ละโรค (Explainable AI)":"Disease Probability (Explainable AI)"}</h3>
        <div class="xai-chart-box"><canvas id="confChart"></canvas></div>
        <div class="prob-list compact">${probRows}</div>${fusionHtml}
      </div>
    </div>
    ${heatmapHtml}
    <div class="report-card reveal">
      <div class="report-head">
        <h3>📝 ${lang==="th"?"รายงานสรุป (ภาษาไทย)":"Summary Report"} <span id="reportSrc" class="report-src"></span></h3>
        <button class="btn-ghost" id="dlReport" onclick="downloadReport()">⬇ ${lang==="th"?"ดาวน์โหลด PDF":"Download PDF"}</button>
      </div>
      <div id="reportBody">${report}</div>
    </div>
    <div class="chat-card reveal">
      <h3>💬 ${lang==="th"?"ถาม AI เกี่ยวกับผล":"Ask AI about this result"}</h3>
      <div class="chat-log" id="chatLog">
        <div class="chat-msg ai"><span class="chat-av">🤖</span><div class="chat-bub">${lang==="th"?"ถามได้เลย เช่น “ต้องรักษายังไง” “อันตรายไหม” “ดูแลฟันยังไง”":"Ask me e.g. “How is this treated?” “Is it urgent?”"}</div></div>
      </div>
      <div class="chat-quick" id="chatQuick"></div>
      <form class="chat-form" id="chatForm">
        <input type="text" id="chatInput" autocomplete="off" placeholder="${lang==="th"?"พิมพ์คำถาม…":"Type a question…"}">
        <button type="submit" class="btn-primary chat-send">➤</button>
      </form>
    </div>
    <div class="result-actions">
      <button class="btn-primary" data-page="analyze" onclick="goto('analyze')">${lang==="th"?"วิเคราะห์ภาพใหม่":"Analyze another"}</button>
    </div>`;

  renderCharts(preds, risk, DS);
  animateCountUps();
  setupChat(data);
  enhanceReport(data, report);   // async LLM report (Phase 3)
}

// ── risk scoring (low/medium/high) ──
function riskInfo(preds, detected, isNormal) {
  const maxP = Math.max(0, ...Object.values(preds || {}));
  // X-ray-specific escalators; absent (0) for photo modality
  const serious = (preds["Deep Caries"] || 0) > 0.5 || (preds["Periapical Lesion"] || 0) > 0.5;
  let level, th, en, color;
  if (isNormal || detected.length === 0) { level = "low"; th = "ความเสี่ยงต่ำ"; en = "Low risk"; color = "var(--teal)"; }
  else if (detected.length >= 3 || serious) { level = "high"; th = "ความเสี่ยงสูง"; en = "High risk"; color = "var(--coral)"; }
  else { level = "medium"; th = "ความเสี่ยงปานกลาง"; en = "Medium risk"; color = "var(--amber)"; }
  const score = isNormal ? (1 - maxP) * 0.4 : maxP;   // gauge fill
  return { level, th, en, color, score };
}

// ── Chart.js: risk gauge + confidence bar ──
let _charts = { gauge: null, conf: null };
function _cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function _hex(v) {  // resolve var(--x) → actual color
  const m = v.match(/var\((--[\w-]+)\)/); return m ? _cssVar(m[1]) : v;
}
function renderCharts(preds, risk, DS = DISEASES) {
  if (typeof Chart === "undefined") return;
  const txt = _cssVar("--text-2") || "#5b7170";
  const grid = _cssVar("--border") || "#dce8e8";
  Object.values(_charts).forEach(c => c && c.destroy());

  // risk gauge (semicircle doughnut)
  const gc = document.getElementById("riskGauge");
  if (gc) {
    const col = _hex(risk.color);
    _charts.gauge = new Chart(gc, {
      type: "doughnut",
      data: { datasets: [{ data: [risk.score, 1 - risk.score],
        backgroundColor: [col, grid], borderWidth: 0 }] },
      options: { rotation: -90, circumference: 180, cutout: "72%",
        responsive: false, plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { animateRotate: true, duration: 900 } },
    });
  }
  // confidence bar (all 4)
  const cc = document.getElementById("confChart");
  if (cc) {
    _charts.conf = new Chart(cc, {
      type: "bar",
      data: {
        labels: DS.map(d => (lang === "th" ? d.th : d.en)),
        datasets: [{ data: DS.map(d => +((preds[d.en] || 0) * 100).toFixed(0)),
          backgroundColor: DS.map(d => d.color), borderRadius: 8, maxBarThickness: 38 }],
      },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false,
        scales: {
          x: { min: 0, max: 100, ticks: { color: txt, callback: v => v + "%" }, grid: { color: grid } },
          y: { ticks: { color: txt, font: { size: 13 } }, grid: { display: false } },
        },
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: c => " " + c.parsed.x + "%" } } },
        animation: { duration: 800 },
      },
    });
  }
}

// ── count-up animation ──
function animateCountUps() {
  document.querySelectorAll("[data-countup]").forEach(el => {
    const target = parseFloat(el.dataset.countup) || 0;
    const t0 = performance.now(), dur = 900;
    function tick(now) {
      const p = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);  // easeOutCubic
      el.textContent = Math.round(target * e);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

// ── AI chat (Phase 3) — generic, reusable for result + standalone ──
const CHAT_QUICK = {
  result: {
    th: ["ต้องรักษายังไง", "อันตราย/เร่งด่วนไหม", "ดูแลฟันยังไง", "ผลแม่นแค่ไหน"],
    en: ["How is this treated?", "Is it urgent?", "How to care?", "How accurate?"],
  },
  general: {
    th: ["ฟันผุเป็นยังไง", "เหงือกอักเสบดูยังไง", "หินปูนอันตรายไหม", "ปวดฟันมากทำไงดี", "ฟันคุดต้องผ่าไหม", "แปรงฟันยังไงให้ถูก"],
    en: ["What is caries?", "Signs of gingivitis?", "Is calculus harmful?", "Severe toothache?", "Remove wisdom tooth?", "How to brush right?"],
  },
};
function mountChat(opts) {
  const { logId, formId, inputId, quickId, getData, quickSet = "result" } = opts;
  const form = document.getElementById(formId);
  const input = document.getElementById(inputId);
  const log = document.getElementById(logId);
  const quick = document.getElementById(quickId);
  if (!form) return;
  const chatHistory = [];
  const qs = CHAT_QUICK[quickSet][lang] || CHAT_QUICK[quickSet].th;
  quick.innerHTML = qs.map(q => `<button class="quick-chip" type="button">${esc(q)}</button>`).join("");
  quick.querySelectorAll(".quick-chip").forEach(b =>
    b.onclick = () => { input.value = b.textContent; form.requestSubmit(); });

  async function ask(q) {
    const u = document.createElement("div");
    u.className = "chat-msg user";
    u.innerHTML = `<div class="chat-bub">${esc(q)}</div><span class="chat-av">🙂</span>`;
    log.appendChild(u);
    const typing = document.createElement("div");
    typing.className = "chat-msg ai";
    typing.innerHTML = `<span class="chat-av">🤖</span><div class="chat-bub typing"><span></span><span></span><span></span></div>`;
    log.appendChild(typing);
    log.scrollTop = log.scrollHeight;
    const data = getData() || {};
    try {
      const r = await fetch(API + "/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, predictions: data.predictions || {},
          detected: data.detected || [], symptoms: data.symptoms || null, history: chatHistory,
          modality: (data.modality || currentModality) }),
      });
      const res = await r.json();
      const bub = typing.querySelector(".chat-bub");
      bub.classList.remove("typing");
      bub.innerHTML = fmtChat(res.answer || "—") +
        (res.source && res.source !== "llm" ? `<span class="chat-src">📋</span>` : `<span class="chat-src">🤖</span>`);
      chatHistory.push({ role: "user", content: q }, { role: "assistant", content: res.answer || "" });
    } catch (e) {
      const bub = typing.querySelector(".chat-bub");
      bub.classList.remove("typing");
      bub.textContent = lang === "th" ? "เชื่อมต่อไม่ได้" : "Connection error";
    }
    log.scrollTop = log.scrollHeight;
  }
  form.onsubmit = e => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    ask(q);
  };
}
// แปลงข้อความ AI → HTML ปลอดภัย (รองรับ bullet • / - และตัวหนา **..**)
function fmtChat(t) {
  const lines = String(t).split(/\n+/).map(l => l.trim()).filter(Boolean);
  return lines.map(l => {
    let h = esc(l).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    if (/^[-•*]\s/.test(l)) return `<div class="cl-li">• ${h.replace(/^[-•*]\s/, "")}</div>`;
    return `<div>${h}</div>`;
  }).join("");
}
// result-page chat
function setupChat(data) {
  mountChat({ logId: "chatLog", formId: "chatForm", inputId: "chatInput", quickId: "chatQuick",
    quickSet: "result", getData: () => ({ ...data, symptoms: getSymptoms() }) });
}

// ── download report as PDF (print) ──
function downloadReport() {
  document.body.classList.add("printing");
  window.print();
  setTimeout(() => document.body.classList.remove("printing"), 500);
}

// ── Phase 3: LLM bilingual report (fallback rule-based ถ้าไม่มี API key) ──
async function enhanceReport(data, fallbackHtml) {
  const body = document.getElementById("reportBody");
  const src = document.getElementById("reportSrc");
  if (!body) return;
  body.innerHTML = `<div class="report-loading">⏳ ${lang==="th"?"กำลังสร้างรายงาน…":"Generating report…"}</div>` + fallbackHtml;
  try {
    const r = await fetch(API + "/api/report", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ predictions: data.predictions, detected: data.detected, symptoms: getSymptoms(), modality: (data.modality || currentModality) }),
    });
    const rep = await r.json();
    const urgencyClass = rep.urgency && rep.urgency.includes("เร็ว") ? "alert"
      : (rep.urgency && rep.urgency.includes("พบ") ? "warn" : "normal");
    const recs = (rep.recommendations || []).map(x => `<li>${esc(x)}</li>`).join("");
    src.textContent = rep.source === "llm" ? "🤖 AI" : "📋 rule-based";
    body.innerHTML = `
      <div class="report-urgency ${urgencyClass}">${lang==="th"?"ความเร่งด่วน":"Urgency"}: <b>${esc(rep.urgency||"-")}</b></div>
      <div class="report-block"><strong>${lang==="th"?"สรุปสำหรับผู้ป่วย":"For the patient"}</strong><p>${esc(rep.summary_layperson||"")}</p></div>
      <details class="report-block"><summary><strong>${lang==="th"?"สรุปเชิงคลินิก (สำหรับทันตแพทย์)":"Clinical summary"}</strong></summary><p>${esc(rep.summary_clinical||"")}</p></details>
      <div class="report-block"><strong>${lang==="th"?"คำแนะนำ":"Recommendations"}</strong><ul>${recs}</ul></div>
      <p class="report-disc">⚠️ ${lang==="th"?"คัดกรองเบื้องต้นด้วย AI — ไม่ใช่การวินิจฉัย ต้องพบทันตแพทย์เพื่อยืนยัน":"AI screening — not a diagnosis. Consult a dentist."}</p>`;
  } catch (e) {
    // server ไม่ตอบ → คง fallback HTML เดิม
    body.innerHTML = fallbackHtml;
    if (src) src.textContent = "";
  }
}

// ── build readable Thai report (rule-based; LLM swap-in later) ──
function buildReport(data, detected, isNormal, dmap) {
  if (isNormal) {
    return `<p>จากการวิเคราะห์ภาพ X-ray ระบบ<b>ไม่พบความผิดปกติที่ชัดเจน</b>ของฟันผุ ฟันผุลึก รอยโรคปลายราก หรือฟันคุด อย่างไรก็ตามการคัดกรองด้วย AI มีข้อจำกัด แนะนำให้ตรวจสุขภาพฟันกับทันตแพทย์เป็นประจำทุก 6 เดือน</p>`;
  }
  const items = detected.map(en => {
    const d = dmap[en];
    const p = (data.predictions[en] * 100).toFixed(0);
    return `<li><b>${d.th} (${d.en})</b> — ความน่าจะเป็น ${p}%<br><span style="color:var(--text-2);font-size:13px">${d.symptom} · แนะนำ: ${d.care}</span></li>`;
  }).join("");
  return `<p>ระบบตรวจพบสัญญาณที่ควรเฝ้าระวัง <b>${detected.length} รายการ</b> ดังนี้:</p>
    <ul style="margin:12px 0 12px 20px;line-height:2">${items}</ul>
    <p style="color:var(--text-2);font-size:14px">⚠️ ผลนี้เป็นการคัดกรองเบื้องต้นจาก AI ไม่ใช่การวินิจฉัยขั้นสุดท้าย กรุณาพบทันตแพทย์เพื่อตรวจยืนยันและวางแผนการรักษา</p>`;
}

// ── history (localStorage) ──
function saveHistory(data) {
  const hist = JSON.parse(localStorage.getItem("ds_history") || "[]");
  hist.unshift({
    date: new Date().toISOString(),
    thumb: currentImage,
    detected: data.detected,
    predictions: data.predictions,
  });
  localStorage.setItem("ds_history", JSON.stringify(hist.slice(0, 30)));
}
function renderHistory() {
  const hist = JSON.parse(localStorage.getItem("ds_history") || "[]");
  const el = document.getElementById("historyContent");
  if (!hist.length) {
    el.innerHTML = `<div class="empty-state"><span class="es-ico">🕑</span><p>${lang==="th"?"ยังไม่มีประวัติการวิเคราะห์":"No history yet"}</p></div>`;
    return;
  }
  el.innerHTML = hist.map(h => {
    const dt = new Date(h.date).toLocaleString(lang === "th" ? "th-TH" : "en-US");
    const tags = h.detected.length
      ? h.detected.map(en => `<span class="tag hit">${DISEASE_META[en] ? (lang==="th"?DISEASE_META[en].th:en) : en}</span>`).join("")
      : `<span class="tag">${lang==="th"?"ปกติ":"normal"}</span>`;
    return `<div class="history-card"><img src="${h.thumb}"><div class="hc-body"><div class="hc-date">${dt}</div><div class="hc-tags">${tags}</div></div></div>`;
  }).join("");
}

// ── theme ──
function setupTheme() {
  const saved = localStorage.getItem("ds_theme") || "light";
  document.documentElement.dataset.theme = saved;
  document.getElementById("themeToggle").textContent = saved === "dark" ? "☀️" : "🌙";
  document.getElementById("themeToggle").onclick = () => {
    const cur = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = cur;
    localStorage.setItem("ds_theme", cur);
    document.getElementById("themeToggle").textContent = cur === "dark" ? "☀️" : "🌙";
    if (lastResult && _charts.conf) renderCharts(lastResult.predictions, riskInfo(
      lastResult.predictions, lastResult.detected, lastResult.is_normal));  // recolor charts
  };
}

// ── modality switch (X-ray ↔ photo) ──
function setModality(m) {
  currentModality = (m === "photo") ? "photo" : "xray";
  document.querySelectorAll("#modalitySwitch .mod-btn").forEach(b => {
    const on = b.dataset.modality === currentModality;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  // update analyze title + dropzone hint per modality
  const isPhoto = currentModality === "photo";
  const T = lang === "th";
  const title = document.querySelector("#page-analyze .page-title");
  if (title) title.textContent = isPhoto
    ? (T ? "วิเคราะห์ภาพถ่ายฟัน" : "Analyze Dental Photo")
    : (T ? "วิเคราะห์ภาพ X-ray" : "Analyze X-ray");
  const step1 = document.querySelector("#page-analyze .upload-card h3");
  if (step1) step1.textContent = isPhoto
    ? (T ? "1. อัปโหลดภาพถ่ายฟัน (มือถือ)" : "1. Upload a dental photo")
    : (T ? "1. อัปโหลดภาพ X-ray พาโนรามิก" : "1. Upload panoramic X-ray");
  const hint = document.querySelector("#page-analyze .dz-hint");
  if (hint) hint.textContent = isPhoto ? "PNG / JPG · intraoral photo" : "PNG / JPG · panoramic X-ray";
  // reset current image — a photo can't be analyzed as an X-ray and vice-versa
  currentImage = null;
  stopCam();
  const prev = document.getElementById("preview");
  const empty = document.getElementById("dzEmpty");
  if (prev) { prev.hidden = true; prev.src = ""; }
  if (empty) empty.hidden = false;
  document.getElementById("analyzeBtn").disabled = true;
  // live camera only makes sense for intraoral photos (not X-ray film)
  const camRow = document.getElementById("camRow");
  if (camRow) camRow.hidden = !isPhoto;
  renderSamples();
  checkModel();
}

function setupModality() {
  document.querySelectorAll("#modalitySwitch .mod-btn").forEach(b =>
    b.addEventListener("click", () => setModality(b.dataset.modality)));
}

// ── live camera capture (getUserMedia) — intraoral color photos ──
let camStream = null;
function stopCam() {
  if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
  const stage = document.getElementById("camStage"), dz = document.getElementById("dropzone");
  if (stage) stage.hidden = true;
  if (dz) dz.style.display = "";
}
function setupCamera() {
  const btn = document.getElementById("camBtn");
  if (!btn) return;
  const stage = document.getElementById("camStage");
  const video = document.getElementById("camVideo");
  const dz = document.getElementById("dropzone");

  btn.onclick = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert(lang === "th" ? "เบราว์เซอร์นี้ไม่รองรับกล้อง" : "Camera not supported"); return;
    }
    try {
      camStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false });
      video.srcObject = camStream;
      stage.hidden = false; dz.style.display = "none";
    } catch (e) {
      alert((lang === "th" ? "เปิดกล้องไม่ได้: " : "Camera error: ") + e.message);
    }
  };
  document.getElementById("camShot").onclick = () => {
    const c = document.getElementById("camCanvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    currentImage = c.toDataURL("image/jpeg", 0.92);
    const prev = document.getElementById("preview"), empty = document.getElementById("dzEmpty");
    prev.src = currentImage; prev.hidden = false; empty.hidden = true;
    document.getElementById("analyzeBtn").disabled = false;
    stopCam();
  };
  document.getElementById("camCancel").onclick = stopCam;
}

// ── init ──
function init() {
  applyI18n();
  renderHomeDiseases();
  renderInfo();
  renderAbout();
  renderSamples();
  setupUpload();
  setupModality();
  setModality(currentModality);
  setupCamera();
  setupTheme();
  mountChat({ logId: "askLog", formId: "askForm", inputId: "askInput", quickId: "askQuick",
    quickSet: "general", getData: () => ({ predictions: {}, detected: [] }) });

  document.querySelectorAll("[data-page]").forEach(el =>
    el.addEventListener("click", () => goto(el.dataset.page)));
  document.getElementById("analyzeBtn").onclick = runAnalysis;
  document.getElementById("menuFab").onclick = () =>
    document.getElementById("sidebar").classList.toggle("open");
  document.getElementById("langToggle").onclick = () => {
    lang = lang === "th" ? "en" : "th";
    localStorage.setItem("ds_lang", lang);
    applyI18n(); renderHomeDiseases(); renderInfo(); renderAbout(); checkModel();
    mountChat({ logId: "askLog", formId: "askForm", inputId: "askInput", quickId: "askQuick",
      quickSet: "general", getData: () => ({ predictions: {}, detected: [] }) });
    if (lastResult && document.getElementById("page-result").classList.contains("active"))
      renderResult(lastResult);   // re-render charts/chat with new lang
  };

  // deep-link: open a page from the URL hash (shareable)
  const hp = location.hash.replace("#", "");
  if (hp && document.getElementById("page-" + hp)) goto(hp);
  if (hp === "photo" || hp === "xray") { setModality(hp); goto("analyze"); }
}
init();
