"""Grad-CAM implemented with autograd.grad (no hooks)."""
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from medimg.classification.model import feature_map_and_logits


def gradcam(model, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """x: [1,3,H,W] normalized. Returns cam in [0,1] with shape [H,W]."""
    model.eval()
    with torch.enable_grad():
        x = x.clone().requires_grad_(False)
        feats, logits = feature_map_and_logits(model, x)
        grads = torch.autograd.grad(logits[0, class_idx], feats)[0]
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feats).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam -= cam.min()
    return cam / (cam.max() + 1e-8)


def _jet(v: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def cam_to_rgba_png(cam: np.ndarray, size: tuple, max_alpha: float = 0.75) -> bytes:
    """Colourised transparent heatmap at the original image size; the viewer controls final opacity."""
    rgb = (_jet(cam) * 255).astype(np.uint8)
    alpha = (np.clip(cam * 1.5, 0, 1) * max_alpha * 255).astype(np.uint8)
    img = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
