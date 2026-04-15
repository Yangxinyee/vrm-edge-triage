"""
Edge student model for VRM distillation.

Default behavior follows encoder-transfer with an EVA-family backbone and
optional initialization from an EVA-X checkpoint.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn


DEFAULT_EVA_BACKBONE = "eva02_tiny_patch14_224"
DEFAULT_EVA_CKPT = "checkpoints/eva_x_tiny_patch16_merged520k_mim.pt"
EVA_BACKBONE_CANDIDATES = (
    "eva02_tiny_patch14_224",
    "eva02_tiny_patch14_224.mim_in22k",
    "eva02_tiny_patch14_336.mim_in22k_ft_in1k",
)


def _unwrap_state_dict(checkpoint: dict) -> dict:
    """Extract the actual state-dict from common checkpoint wrappers."""
    for key in ("state_dict", "model_state_dict", "model", "module"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return checkpoint


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Remove a key prefix if all keys share it."""
    keys = list(state_dict.keys())
    if keys and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _load_backbone_weights(backbone: nn.Module, checkpoint_path: str) -> Dict[str, int]:
    """
    Load matching checkpoint tensors into the backbone only.

    Mismatched or task-head keys are skipped so training can proceed safely.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {"loaded": 0, "skipped": 0, "missing": len(backbone.state_dict())}

    state_dict = _unwrap_state_dict(ckpt)
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "backbone.")

    current = backbone.state_dict()
    filtered = {}
    skipped = 0
    for key, value in state_dict.items():
        if key.startswith(("head.", "classifier.", "fc_norm.", "norm.")):
            skipped += 1
            continue
        if key in current and current[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped += 1

    load_result = backbone.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "skipped": skipped,
        "missing": len(load_result.missing_keys),
    }


class EdgeStudent(nn.Module):
    """Image-only student model used in VRM distillation."""

    def __init__(
        self,
        backbone: str = DEFAULT_EVA_BACKBONE,
        pretrained: bool = False,
        num_classes: int = 2,
        dropout: float = 0.1,
        backbone_checkpoint: Optional[str] = DEFAULT_EVA_CKPT,
    ):
        super().__init__()
        self.backbone_name, self.backbone = self._create_backbone(backbone, pretrained)
        self.feature_dim = self._get_feature_dim()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, num_classes),
        )
        self._init_classifier()

        self.backbone_load_info = {
            "loaded": 0,
            "skipped": 0,
            "missing": 0,
            "used_checkpoint": False,
        }
        if backbone_checkpoint:
            ckpt_path = Path(backbone_checkpoint)
            if ckpt_path.exists():
                self.backbone_load_info = _load_backbone_weights(self.backbone, str(ckpt_path))
                self.backbone_load_info["used_checkpoint"] = True
            else:
                self.backbone_load_info["warning"] = (
                    f"Checkpoint not found: {backbone_checkpoint}. Using random init."
                )

    def _create_backbone(self, requested_backbone: str, pretrained: bool) -> Tuple[str, nn.Module]:
        candidates = [requested_backbone]
        for name in EVA_BACKBONE_CANDIDATES:
            if name not in candidates:
                candidates.append(name)

        last_error = None
        for name in candidates:
            try:
                model = timm.create_model(
                    name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="",
                )
                return name, model
            except Exception as exc:  # pragma: no cover - fallback path
                last_error = exc
        raise ValueError(f"Unable to create EVA backbone from candidates={candidates}: {last_error}")

    def _get_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            if features.dim() == 4:
                return features.shape[1]
            if features.dim() in (2, 3):
                return features.shape[-1]
            raise ValueError(f"Unexpected feature shape: {features.shape}")

    def _init_classifier(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        if features.dim() == 4:
            features = self.pool(features).flatten(1)
        elif features.dim() == 3:
            features = features[:, 0]

        logits = self.classifier(features)
        output = {"logits": logits}
        if return_features:
            output["features"] = features
        return output

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def count_parameters(self) -> Dict[str, int]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "backbone": sum(
                param.numel() for name, param in self.named_parameters() if "backbone" in name
            ),
            "classifier": sum(
                param.numel() for name, param in self.named_parameters() if "classifier" in name
            ),
        }


def create_student_model(
    backbone: str = DEFAULT_EVA_BACKBONE,
    pretrained: bool = False,
    num_classes: int = 2,
    dropout: float = 0.1,
    backbone_checkpoint: Optional[str] = DEFAULT_EVA_CKPT,
    device: str = "cuda",
) -> EdgeStudent:
    model = EdgeStudent(
        backbone=backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        backbone_checkpoint=backbone_checkpoint,
    )
    return model.to(device)


if __name__ == "__main__":
    model = create_student_model(device="cpu")
    stats = model.count_parameters()
    print(f"Student backbone: {model.backbone_name}")
    print(f"Total params: {stats['total']:,}")
    print(f"Feature dim: {model.feature_dim}")
    print(f"Backbone load info: {model.backbone_load_info}")

    x = torch.randn(2, 3, 224, 224)
    out = model(x, return_features=True)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Features shape: {out['features'].shape}")
