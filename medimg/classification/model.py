import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from medimg.config import CLASSES


def build_model(arch: str = "densenet121", num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    if arch == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}")
    m.arch = arch
    return m


def backbone_params(model):
    head = model.classifier
    head_ids = {id(p) for p in head.parameters()}
    return [p for p in model.parameters() if id(p) not in head_ids]


def feature_map_and_logits(model, x):
    """Return (last conv feature map, logits). Used by Grad-CAM without hooks (avoids in-place ReLU issues)."""
    if model.arch == "densenet121":
        feats = F.relu(model.features(x))
        logits = model.classifier(torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1))
    else:
        feats = model.features(x)
        logits = model.classifier(torch.flatten(model.avgpool(feats), 1))
    return feats, logits


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval(), ck
