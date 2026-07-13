"""
Phase 3 — LLM bilingual report generation (Claude API)
สร้างรายงานทันตกรรมภาษาไทย 2 ระดับ (คนทั่วไป + ทันตแพทย์) จากผล AI + อาการ

- ใช้ Anthropic SDK (claude-opus-4-8) + structured output (messages.parse)
- ถ้าไม่มี ANTHROPIC_API_KEY จริง → fallback rule-based (โปรเจกต์ทำงานได้เสมอ)
- decision-support เท่านั้น — system prompt กำกับให้ไม่วินิจฉัยเด็ดขาด/ไม่ alarmist
"""
import os
import pathlib
from typing import Optional

try:
    from pydantic import BaseModel, Field
    _HAS_PYDANTIC = True
except Exception:
    _HAS_PYDANTIC = False


def _load_dotenv():
    """โหลด .env จาก project root โดยไม่พึ่ง python-dotenv (กัน dependency เกิน)"""
    env = pathlib.Path(__file__).parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

DISEASES_TH = {
    "Caries": "ฟันผุ", "Deep Caries": "ฟันผุลึก",
    "Periapical Lesion": "รอยโรคปลายราก", "Impacted": "ฟันคุด",
    "Gingivitis": "เหงือกอักเสบ", "Calculus": "หินปูน",
}
SYMPTOM_TH = {
    "visible_hole": "เห็นรู/จุดดำบนฟัน", "sensitive_hot_cold": "เสียวฟันร้อน/เย็น",
    "pain_chew": "ปวดเวลาเคี้ยว", "spontaneous_pain": "ปวดเอง",
    "gum_swelling": "เหงือกบวม/มีหนอง",
}

MODEL = "claude-opus-4-8"                                    # Claude (ถ้าใช้)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_client = None          # Claude client cache
_gclient = None         # Gemini client cache


if _HAS_PYDANTIC:
    class DentalReport(BaseModel):
        summary_layperson: str = Field(
            description="สรุปผลภาษาไทยสำหรับคนทั่วไป 2-4 ประโยค เข้าใจง่าย ไม่ทำให้ตื่นตระหนก")
        summary_clinical: str = Field(
            description="สรุปเชิงคลินิกภาษาไทยสำหรับทันตแพทย์ ใช้ศัพท์เทคนิคได้ ระบุ finding ตาม prob")
        recommendations: list[str] = Field(
            description="ขั้นตอนที่ผู้ป่วยควรทำต่อ ภาษาไทย 2-4 ข้อ")
        urgency: str = Field(
            description="ระดับความเร่งด่วน: หนึ่งใน 'ไม่เร่งด่วน' | 'ควรพบทันตแพทย์' | 'ควรพบโดยเร็ว'")


def _is_placeholder(key: str) -> bool:
    return (not key or len(key) < 20
            or key.startswith(("your_", "sk-ant-placeholder", "xxx", "<")))


def _gemini_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return None if _is_placeholder(key) else key


def _claude_key() -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return None if _is_placeholder(key) else key


def _provider() -> Optional[str]:
    """เลือก LLM provider: gemini > claude > None (=rule-based)."""
    pref = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if pref == "gemini" and _gemini_key():
        return "gemini"
    if pref == "claude" and _claude_key() and _HAS_PYDANTIC:
        return "claude"
    if _gemini_key():
        return "gemini"
    if _claude_key() and _HAS_PYDANTIC:
        return "claude"
    return None


def _get_gemini():
    global _gclient
    if not _gemini_key():
        return None
    if _gclient is None:
        from google import genai
        _gclient = genai.Client(api_key=_gemini_key())
    return _gclient


def _get_client():
    global _client
    if not _claude_key() or not _HAS_PYDANTIC:
        return None
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _img_term(result: dict) -> str:
    """image wording per modality (photo vs panoramic X-ray)"""
    return "ภาพถ่ายฟัน (ในช่องปาก)" if (result or {}).get("modality") == "photo" \
        else "ภาพ X-ray พาโนรามิก"


def _build_prompt(result: dict, symptoms: dict) -> str:
    preds = result.get("predictions", {})
    detected = result.get("detected", [])
    lines = [f"ผลการวิเคราะห์{_img_term(result)}จากโมเดล AI (EfficientNet):"]
    for en, p in preds.items():
        th = DISEASES_TH.get(en, en)
        mark = " ← ตรวจพบ (เกิน threshold)" if en in detected else ""
        lines.append(f"  - {th} ({en}): ความน่าจะเป็น {p*100:.0f}%{mark}")
    s = symptoms or {}
    active = [SYMPTOM_TH[k] for k in SYMPTOM_TH if s.get(k)]
    dur = s.get("duration_days", 0) or 0
    lines.append("\nอาการที่ผู้ป่วยรายงาน: " + (", ".join(active) if active else "ไม่ระบุ"))
    if dur:
        lines.append(f"ระยะเวลามีอาการ: ~{dur} วัน")
    lines.append("\nเขียนรายงานตาม schema ที่กำหนด เป็นภาษาไทยทั้งหมด "
                 "อ้างอิงเฉพาะโรคที่ตรวจพบเป็นหลัก ระบุชัดว่าเป็นการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัย")
    return "\n".join(lines)


SYSTEM = (
    "คุณคือผู้ช่วยเขียนรายงานคัดกรองสุขภาพฟันจากภาพในช่องปาก (X-ray หรือภาพถ่าย) สำหรับระบบ decision-support "
    "เขียนภาษาไทยที่ถูกต้อง สุภาพ ใจเย็น ไม่ทำให้ผู้ป่วยตื่นตระหนก "
    "ย้ำเสมอว่าเป็นการคัดกรองเบื้องต้นด้วย AI ไม่ใช่การวินิจฉัยทางการแพทย์ "
    "ต้องให้ทันตแพทย์ยืนยัน ห้ามสั่งยาหรือระบุการรักษาที่ชัดเจนแทนทันตแพทย์ "
    "อ้างอิงค่าความน่าจะเป็นตามที่ได้รับ ไม่กล่าวเกินจริง"
)


def _fallback(result: dict, symptoms: dict) -> dict:
    """rule-based เมื่อไม่มี API key — โปรเจกต์ยังใช้งานได้"""
    detected = result.get("detected", [])
    preds = result.get("predictions", {})
    if not detected:
        classes = ", ".join(preds.keys()) or "ทุกคลาส"
        return {
            "source": "rule-based",
            "summary_layperson": f"ระบบไม่พบความผิดปกติที่ชัดเจนจาก{_img_term(result)} "
                                 "แต่การคัดกรองด้วย AI มีข้อจำกัด แนะนำตรวจสุขภาพฟันทุก 6 เดือน",
            "summary_clinical": f"ไม่พบ finding เกิน threshold ({classes})",
            "recommendations": ["ตรวจสุขภาพฟันกับทันตแพทย์ทุก 6 เดือน",
                                "แปรงฟันด้วยยาสีฟันฟลูออไรด์ วันละ 2 ครั้ง"],
            "urgency": "ไม่เร่งด่วน",
        }
    items = ", ".join(f"{DISEASES_TH.get(d,d)} ({preds.get(d,0)*100:.0f}%)" for d in detected)
    return {
        "source": "rule-based",
        "summary_layperson": f"ระบบพบสัญญาณที่ควรเฝ้าระวัง {len(detected)} รายการ ได้แก่ {items} "
                             "ควรพบทันตแพทย์เพื่อตรวจยืนยัน",
        "summary_clinical": f"Findings เกิน threshold: {items}. "
                            "แนะนำ clinical+radiographic correlation เพื่อยืนยัน",
        "recommendations": ["นัดพบทันตแพทย์เพื่อตรวจยืนยัน",
                            "หลีกเลี่ยงของหวาน/เครื่องดื่มกรดจัด", "ดูแลความสะอาดช่องปาก"],
        "urgency": "ควรพบทันตแพทย์" if len(detected) < 2 else "ควรพบโดยเร็ว",
    }


def _report_gemini(result: dict, symptoms: dict) -> dict:
    from google.genai import types
    client = _get_gemini()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(result, symptoms),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=DentalReport,
            max_output_tokens=2048,
            temperature=0.4,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # กัน JSON โดนตัด
        ),
    )
    r = resp.parsed                      # pydantic instance
    if r is None:                        # เผื่อ parse ไม่ได้ → อ่าน JSON ดิบ
        import json
        r = DentalReport(**json.loads(resp.text))
    return {
        "source": "llm", "provider": "gemini", "model": GEMINI_MODEL,
        "summary_layperson": r.summary_layperson,
        "summary_clinical": r.summary_clinical,
        "recommendations": list(r.recommendations),
        "urgency": r.urgency,
    }


def _report_claude(result: dict, symptoms: dict) -> dict:
    client = _get_client()
    resp = client.messages.parse(
        model=MODEL, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(result, symptoms)}],
        output_format=DentalReport,
    )
    r = resp.parsed_output
    return {
        "source": "llm", "provider": "claude", "model": MODEL,
        "summary_layperson": r.summary_layperson,
        "summary_clinical": r.summary_clinical,
        "recommendations": list(r.recommendations),
        "urgency": r.urgency,
    }


def generate_report(result: dict, symptoms: dict = None) -> dict:
    prov = _provider()
    if prov is None or not _HAS_PYDANTIC:
        return _fallback(result, symptoms)
    try:
        return _report_gemini(result, symptoms) if prov == "gemini" \
            else _report_claude(result, symptoms)
    except Exception as e:
        # API ล่ม/quota/timeout → fallback (ไม่ให้ผู้ใช้เห็น error)
        fb = _fallback(result, symptoms)
        fb["source"] = "rule-based-fallback"
        fb["_error"] = f"{prov}: {str(e)[:180]}"
        return fb


# ════════════════════ AI Chat — ถาม-ตอบเกี่ยวกับผล ════════════════════
CHAT_SYSTEM = (
    "คุณคือ 'ผู้ช่วยทันตกรรม DentScan AI' ตอบคำถามสุขภาพช่องปาก/ฟันได้ทุกเรื่อง "
    "เช่น อาการปวดฟัน ฟันผุ ฟันคุด เหงือก การดูแลฟัน การแปรงฟัน อาหาร และการเตรียมตัวก่อนพบทันตแพทย์ "
    "ตอบเป็นภาษาไทย กระชับ อบอุ่น เข้าใจง่าย ใช้ bullet ได้ถ้าช่วยให้อ่านง่าย ความยาวพอเหมาะ (ไม่ยาวเกินไป) "
    "ถ้ามีบริบทผลการคัดกรองให้ใช้ประกอบ แต่ถ้าผู้ใช้ถามทั่วไปก็ตอบความรู้ทันตกรรมทั่วไปได้เลย "
    "ย้ำเมื่อเหมาะสมว่าเป็นคำแนะนำเบื้องต้น ไม่ใช่การวินิจฉัย ควรพบทันตแพทย์เพื่อตรวจจริง "
    "ห้ามสั่งยาเฉพาะเจาะจงหรือยืนยันการรักษาแทนทันตแพทย์ "
    "ถ้าถามนอกเรื่องสุขภาพช่องปากโดยสิ้นเชิง ให้ดึงกลับเข้าเรื่องฟันอย่างสุภาพ"
)

# คำตอบสำเร็จรูปเมื่อไม่มี LLM key — ครอบคลุมคำถามที่พบบ่อยตอน demo
_CHAT_KB = [
    (("รักษา", "ทำไง", "ทำยังไง", "แก้", "หาย"),
     "แนวทางขึ้นกับโรคที่พบ: ฟันผุ→อุดฟัน, ฟันผุลึก→อุดลึก/รักษาราก, รอยโรคปลายราก→รักษารากฟัน (root canal), "
     "ฟันคุด→ผ่า/ถอน ทันตแพทย์จะเลือกวิธีที่เหมาะหลังตรวจยืนยัน"),
    (("เร่งด่วน", "อันตราย", "รีบ", "ด่วน", "หนัก"),
     "ถ้ามีอาการปวดเอง บวม มีหนอง หรือพบหลายโรคพร้อมกัน ควรพบทันตแพทย์โดยเร็ว "
     "หากยังไม่มีอาการชัด นัดตรวจตามปกติได้ แต่ไม่ควรปล่อยไว้นาน"),
    (("ฟันคุด", "impacted", "คุด"),
     "ฟันคุดคือฟันที่ขึ้นไม่เต็มซี่/ฝังในกระดูก มักเป็นฟันกรามซี่สุดท้าย อาจดันฟันข้างเคียงและอักเสบได้ "
     "ทันตแพทย์มักแนะนำเอกซเรย์ติดตามหรือผ่าออกถ้ามีปัญหา"),
    (("ปวด", "เสียว", "อาการ"),
     "อาการปวด/เสียวสัมพันธ์กับระดับความลึกของโรค ระหว่างนี้เลี่ยงของหวานจัด ร้อน/เย็นจัด "
     "และรักษาความสะอาดช่องปาก แล้วรีบพบทันตแพทย์เพื่อตรวจยืนยัน"),
    (("ป้องกัน", "ดูแล", "แปรง", "ทำความสะอาด"),
     "แปรงฟันด้วยยาสีฟันฟลูออไรด์วันละ 2 ครั้ง ใช้ไหมขัดฟัน ลดน้ำตาล "
     "และตรวจสุขภาพฟันทุก 6 เดือน ช่วยลดความเสี่ยงได้มาก"),
    (("แม่น", "ความแม่น", "เชื่อ", "ผิดพลาด", "accuracy", "auc"),
     "ระบบนี้เป็นเครื่องมือคัดกรองด้วย AI (ensemble EfficientNet) ใช้ช่วยชี้จุดที่ควรสังเกต "
     "ไม่ใช่การวินิจฉัยขั้นสุดท้าย ผลทุกครั้งควรให้ทันตแพทย์ยืนยันด้วยการตรวจจริง"),
    (("ปวดฟัน", "ปวด", "บรรเทา", "แก้ปวด"),
     "ระหว่างรอพบทันตแพทย์: บ้วนน้ำเกลืออุ่น ประคบเย็นด้านนอกแก้ม เลี่ยงอาหารร้อน/เย็น/หวานจัด "
     "และเคี้ยวข้างที่ไม่ปวด ถ้าปวดมาก/บวม/มีไข้ ควรรีบพบทันตแพทย์ ไม่ควรปล่อยไว้นาน"),
    (("เลือดออก", "เหงือก", "ไรฟัน", "เหงือกอักเสบ"),
     "เลือดออกตามไรฟันมักมาจากเหงือกอักเสบเพราะคราบจุลินทรีย์ ลองแปรงฟันให้ทั่วและนุ่มขึ้น "
     "ใช้ไหมขัดฟันทุกวัน ถ้ายังเป็นเกิน 1-2 สัปดาห์ ควรพบทันตแพทย์เพื่อขูดหินปูน"),
    (("กลิ่นปาก", "ปากเหม็น", "เหม็น"),
     "กลิ่นปากมักจากคราบบนลิ้นและซอกฟัน แปรงลิ้น ใช้ไหมขัดฟัน ดื่มน้ำให้พอ "
     "ถ้ามีฟันผุ/เหงือกอักเสบร่วมด้วยควรให้ทันตแพทย์ตรวจ"),
    (("ก่อนพบ", "เตรียมตัว", "ไปหาหมอ", "พบหมอ"),
     "ก่อนพบทันตแพทย์ จดอาการ/ระยะเวลา/ตำแหน่งที่ปวด นำผลคัดกรองและภาพไปด้วย "
     "และแจ้งโรคประจำตัว/ยาที่ใช้ จะช่วยให้วินิจฉัยได้ตรงขึ้น"),
]


def _chat_context(result: dict, symptoms: dict) -> str:
    preds = (result or {}).get("predictions", {})
    detected = (result or {}).get("detected", [])
    parts = []
    if not preds and not detected:
        parts.append("ไม่มีผลคัดกรองแนบมา (ผู้ใช้ถามทั่วไป)")
    elif detected:
        parts.append("ผลคัดกรองพบ: " + ", ".join(
            f"{DISEASES_TH.get(d,d)} {preds.get(d,0)*100:.0f}%" for d in detected))
    else:
        parts.append("ผลคัดกรอง: ไม่พบความผิดปกติชัดเจน")
    s = symptoms or {}
    active = [SYMPTOM_TH[k] for k in SYMPTOM_TH if s.get(k)]
    if active:
        parts.append("อาการที่รายงาน: " + ", ".join(active))
    return " | ".join(parts)


def _chat_fallback(question: str, result: dict, symptoms: dict) -> dict:
    q = (question or "").lower()
    for keys, ans in _CHAT_KB:
        if any(k.lower() in q for k in keys):
            return {"source": "rule-based", "answer": ans}
    # default — สรุป + ดึงกลับเข้าเรื่อง
    preds = (result or {}).get("predictions", {})
    detected = (result or {}).get("detected", [])
    if detected:
        ctx = _chat_context(result, symptoms)
        ans = f"จาก{ctx} แนะนำพบทันตแพทย์เพื่อตรวจยืนยันและวางแผนการรักษา "
    elif preds:
        ans = "จากภาพ ระบบไม่พบความผิดปกติชัดเจน แต่ AI มีข้อจำกัด ควรตรวจสุขภาพฟันเป็นประจำ "
    else:
        ans = "ยินดีช่วยเรื่องสุขภาพฟัน เช่น อาการปวดฟัน การดูแล หรือการเตรียมตัวพบทันตแพทย์ "
    ans += "(ลองถามเรื่องปวดฟัน การรักษา หรือวิธีดูแลฟันได้)"
    return {"source": "rule-based", "answer": ans}


def _chat_gemini(question, ctx, history):
    from google.genai import types
    client = _get_gemini()
    contents = []
    for h in (history or [])[-6:]:
        role = h.get("role")
        if role in ("user", "assistant") and h.get("content"):
            grole = "model" if role == "assistant" else "user"   # Gemini ใช้ 'model'
            contents.append(types.Content(role=grole,
                parts=[types.Part(text=str(h["content"])[:1500])]))
    contents.append(types.Content(role="user", parts=[types.Part(
        text=f"[บริบทผลคัดกรอง: {ctx}]\n\nคำถามผู้ใช้: {question}")]))
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=CHAT_SYSTEM, max_output_tokens=1200, temperature=0.6,
            thinking_config=types.ThinkingConfig(thinking_budget=0)),  # ปิด thinking → ตอบครบ ไม่โดนตัด
    )
    return {"source": "llm", "provider": "gemini", "model": GEMINI_MODEL,
            "answer": (resp.text or "").strip()}


def _chat_claude(question, ctx, history):
    client = _get_client()
    msgs = []
    for h in (history or [])[-6:]:
        role = h.get("role")
        if role in ("user", "assistant") and h.get("content"):
            msgs.append({"role": role, "content": str(h["content"])[:1500]})
    msgs.append({"role": "user",
                 "content": f"[บริบทผลคัดกรอง: {ctx}]\n\nคำถามผู้ใช้: {question}"})
    resp = client.messages.create(model=MODEL, max_tokens=600, system=CHAT_SYSTEM, messages=msgs)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return {"source": "llm", "provider": "claude", "model": MODEL, "answer": text.strip()}


def chat(question: str, result: dict, symptoms: dict = None, history: list = None) -> dict:
    """ตอบคำถามเกี่ยวกับผลการคัดกรอง — LLM ถ้ามี key, ไม่งั้น rule-based KB"""
    prov = _provider()
    if prov is None:
        return _chat_fallback(question, result, symptoms)
    try:
        ctx = _chat_context(result, symptoms)
        out = _chat_gemini(question, ctx, history) if prov == "gemini" \
            else _chat_claude(question, ctx, history)
        if not out.get("answer"):           # คำตอบว่าง → fallback
            raise ValueError("empty answer")
        return out
    except Exception as e:
        fb = _chat_fallback(question, result, symptoms)
        fb["source"] = "rule-based-fallback"
        fb["_error"] = f"{prov}: {str(e)[:180]}"
        return fb


if __name__ == "__main__":
    import json
    demo = {"predictions": {"Caries": 0.69, "Deep Caries": 0.57,
                            "Periapical Lesion": 0.50, "Impacted": 0.34},
            "detected": ["Caries", "Deep Caries", "Periapical Lesion"]}
    out = generate_report(demo, {"pain_chew": True, "sensitive_hot_cold": True})
    print(json.dumps(out, indent=2, ensure_ascii=False))
