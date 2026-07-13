import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

import model_artifacts


def test_existing_model_does_not_require_remote_config(tmp_path, monkeypatch):
    target = tmp_path / "best_model.pth"
    target.write_bytes(b"local-checkpoint")
    monkeypatch.delenv("HF_MODEL_REPO_ID", raising=False)

    assert model_artifacts.ensure_model_file(target) == target


def test_missing_model_without_remote_config_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODEL_REPO_ID", raising=False)

    assert model_artifacts.ensure_model_file(tmp_path / "missing.pth") is None


def test_invalid_private_model_repository_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_MODEL_REPO_ID", "https://example.com/model")

    with pytest.raises(ValueError, match="owner/repository"):
        model_artifacts.ensure_model_file(tmp_path / "best_model.pth")


def test_private_model_is_downloaded_to_expected_path(tmp_path, monkeypatch):
    cached = tmp_path / "hub-cache" / "checkpoint.pth"
    cached.parent.mkdir()
    cached.write_bytes(b"private-checkpoint")
    calls = []

    def fake_download(repo_id, filename, token, revision):
        calls.append((repo_id, filename, token, revision))
        return cached

    monkeypatch.setattr(model_artifacts, "_download", fake_download)
    monkeypatch.setenv("HF_MODEL_REPO_ID", "kla2009k/dentscan-private")
    monkeypatch.setenv("HF_MODEL_FILENAME", "best_model.pth")
    monkeypatch.setenv("HF_REVISION", "main")
    monkeypatch.setenv("HF_TOKEN", "test-read-token")

    target = tmp_path / "models" / "intraoral" / "best_model.pth"
    assert model_artifacts.ensure_model_file(target) == target
    assert target.read_bytes() == b"private-checkpoint"
    assert calls == [("kla2009k/dentscan-private", "best_model.pth",
                      "test-read-token", "main")]
