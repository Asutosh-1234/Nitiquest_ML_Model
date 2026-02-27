"""
model.py
--------
Two model options:
  1. YOLOv8 (recommended) - uses Ultralytics for detection + classification
  2. CustomCNN            - lightweight PyTorch CNN for pure classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────
# Option 1: YOLOv8 wrapper (detection + class)
# ─────────────────────────────────────────────
def get_yolo_model(model_size: str = "n", num_classes: int = 2):
    """
    Returns a YOLOv8 model ready for training.
    model_size: 'n' (nano), 's', 'm', 'l', 'x'
    """
    from ultralytics import YOLO
    model = YOLO(f"yolov8{model_size}.pt")   # downloads pretrained weights
    return model


# ─────────────────────────────────────────────
# Option 2: Custom CNN (classification only)
# ─────────────────────────────────────────────
class CustomCNN(nn.Module):
    """
    ResNet-50 backbone fine-tuned for binary classification (bird vs drone).
    Input: 640x640 JPEG images (resized to 224x224 internally)
    Output: 2-class logits
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
        super(CustomCNN, self).__init__()

        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Optionally freeze backbone layers (useful for small datasets)
        if freeze_backbone:
            for param in list(backbone.parameters())[:-10]:
                param.requires_grad = False

        # Replace final FC layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def get_cnn_model(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
    return CustomCNN(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
