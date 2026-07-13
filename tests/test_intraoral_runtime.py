import pathlib
import sys


ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

import predictor_intraoral


def test_render_disables_expensive_features_by_default(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("ENABLE_HEATMAP", raising=False)
    monkeypatch.delenv("ENABLE_TTA", raising=False)

    assert predictor_intraoral._heatmap_enabled() is False
    assert predictor_intraoral._tta_enabled() is False


def test_feature_flags_can_be_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("ENABLE_HEATMAP", "true")
    monkeypatch.setenv("ENABLE_TTA", "1")

    assert predictor_intraoral._heatmap_enabled() is True
    assert predictor_intraoral._tta_enabled() is True
