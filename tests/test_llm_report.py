import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import llm_report  # noqa: E402


def test_gemini_is_selected_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-with-more-than-twenty-characters")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert llm_report._provider() == "gemini"


def test_missing_gemini_key_falls_back_without_exposing_secret(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert llm_report._provider() is None


def test_health_metadata_reports_provider_without_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-with-more-than-twenty-characters")

    assert llm_report.status() == {
        "configured": True,
        "provider": "gemini",
        "model": llm_report.GEMINI_MODEL,
    }


def test_chat_prompt_does_not_treat_probability_as_disease_severity():
    assert "ไม่ใช่ระดับความรุนแรง" in llm_report.CHAT_SYSTEM
