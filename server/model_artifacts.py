"""Resolve private model artifacts without committing weights to Git."""

import os
import pathlib
import re
import shutil


_REPO_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_FILENAME = re.compile(r"^[A-Za-z0-9._/-]+$")
_REVISION = re.compile(r"^[A-Za-z0-9._/-]+$")


def _download(repo_id: str, filename: str, token: str, revision: str | None):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
        token=token,
    )


def ensure_model_file(target: pathlib.Path) -> pathlib.Path | None:
    """Return a local model file, downloading from a private Hub repo if configured.

    Configuration:
      HF_MODEL_REPO_ID   owner/repository
      HF_MODEL_FILENAME  file in the repository (default: best_model.pth)
      HF_REVISION        optional immutable commit SHA or tag
      HF_TOKEN           read-only/fine-grained token stored as a platform secret
    """
    target = pathlib.Path(target)
    if target.is_file():
        return target

    repo_id = os.environ.get("HF_MODEL_REPO_ID", "").strip()
    if not repo_id:
        return None
    if not _REPO_ID.fullmatch(repo_id):
        raise ValueError("HF_MODEL_REPO_ID must be in owner/repository format")

    filename = os.environ.get("HF_MODEL_FILENAME", "best_model.pth").strip()
    if (not _FILENAME.fullmatch(filename) or ".." in pathlib.PurePosixPath(filename).parts
            or not filename.endswith(".pth")):
        raise ValueError("HF_MODEL_FILENAME must be a safe .pth path")

    revision = os.environ.get("HF_REVISION", "").strip() or None
    if revision and not _REVISION.fullmatch(revision):
        raise ValueError("HF_REVISION contains unsupported characters")

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for the configured private model repository")

    downloaded = pathlib.Path(_download(repo_id, filename, token, revision))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(downloaded, temporary)
    temporary.replace(target)
    return target
