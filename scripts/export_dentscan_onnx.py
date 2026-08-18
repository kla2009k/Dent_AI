"""Export a DentScan timm checkpoint with class activation map outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F


class TimmCAMExport(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(inputs)
        pooled = self.model.global_pool(features)
        pooled = F.dropout(
            pooled, p=float(self.model.drop_rate), training=self.model.training
        )
        logits = self.model.classifier(pooled)
        weights = self.model.classifier.weight[:, :, None, None]
        cams = F.conv2d(features, weights)
        return logits, cams


def checkpoint_dimensions(config: dict) -> tuple[int, int]:
    if "img_w" in config and "img_h" in config:
        return int(config["img_w"]), int(config["img_h"])
    size = int(config.get("img_size", 384))
    return size, size


def export_checkpoint(checkpoint_path: Path, output_path: Path, metadata_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("cfg", {})
    diseases = list(checkpoint["diseases"])
    backbone = config.get("backbone", "efficientnet_b0")
    width, height = checkpoint_dimensions(config)

    model = timm.create_model(backbone, pretrained=False, num_classes=len(diseases))
    model.load_state_dict(checkpoint["model"])
    wrapper = TimmCAMExport(model.eval()).eval()
    example = torch.zeros((1, 3, height, width), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        output_path,
        input_names=["image"],
        output_names=["logits", "cams"],
        opset_version=18,
        dynamo=False,
    )

    metadata = {
        "backbone": backbone,
        "diseases": diseases,
        "width": width,
        "height": height,
        "val_mean_auc": checkpoint.get("metrics", {}).get("mean_auc"),
        "target_layer": "conv_head",
        "attribution_method": "class_activation_mapping",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    verify_export(wrapper, output_path, width, height)


def verify_export(wrapper: nn.Module, output_path: Path, width: int, height: int) -> None:
    import onnxruntime as ort

    horizontal = torch.linspace(0.0, 1.0, width)[None, :].expand(height, width)
    vertical = torch.linspace(0.0, 1.0, height)[:, None].expand(height, width)
    sample = torch.stack((horizontal, vertical, (horizontal + vertical) / 2.0))[None]
    wrapper.eval()
    with torch.inference_mode():
        expected_logits, expected_cams = wrapper(sample)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual_logits, actual_cams = session.run(None, {"image": sample.numpy()})
    logits_error = float(np.max(np.abs(actual_logits - expected_logits.numpy())))
    cams_error = float(np.max(np.abs(actual_cams - expected_cams.numpy())))
    if not np.allclose(actual_logits, expected_logits.numpy(), rtol=1e-4, atol=1e-3):
        raise RuntimeError(f"ONNX logits parity failed: max error={logits_error:.3g}")
    if not np.allclose(actual_cams, expected_cams.numpy(), rtol=1e-4, atol=1e-3):
        raise RuntimeError(f"ONNX CAM parity failed: max error={cams_error:.3g}")
    print(
        f"ONNX parity passed for {output_path.name}: "
        f"logits={logits_error:.3g}, CAM={cams_error:.3g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, args.metadata)


if __name__ == "__main__":
    main()
