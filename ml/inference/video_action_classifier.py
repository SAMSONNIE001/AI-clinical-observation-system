from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from ml.training.video_action_dataset import load_video_clip


@dataclass(frozen=True)
class VideoActionPrediction:
    label: str
    confidence: float
    model_version: str
    label_mode: str


class TorchVisionVideoActionClassifier:
    model_version = "torchvision-r3d18-action-v1"

    def __init__(
        self,
        checkpoint_path: Path,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.labels = list(checkpoint["labels"])
        self.label_mode = str(checkpoint.get("label_mode", "detailed"))
        self.clip_frames = int(checkpoint.get("clip_frames", 16))
        self.frame_size = int(checkpoint.get("frame_size", 112))
        self.model = build_r3d18_model(num_classes=len(self.labels), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, video_path: Path) -> VideoActionPrediction:
        video = load_video_clip(
            video_path=video_path,
            clip_frames=self.clip_frames,
            frame_size=self.frame_size,
        )
        video = video.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(video)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, label_index = torch.max(probabilities, dim=0)

        return VideoActionPrediction(
            label=self.labels[int(label_index)],
            confidence=float(confidence),
            model_version=self.model_version,
            label_mode=self.label_mode,
        )


def build_r3d18_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights = R3D_18_Weights.DEFAULT if pretrained else None
    model = r3d_18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
