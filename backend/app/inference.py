"""Model engines. Imports torch/MONAI lazily so the API and tests run without them installed."""
import io
import json
import logging

from PIL import Image

log = logging.getLogger("inference")


class ClassifierEngine:
    def __init__(self):
        import torch
        from medimg.classification.model import build_model, load_checkpoint
        from medimg.config import CLASSES, CLS_THRESHOLDS, CLS_WEIGHTS
        from medimg.preprocessing.transforms import eval_transform

        self.torch, self.classes, self.tf = torch, CLASSES, eval_transform()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if CLS_WEIGHTS.exists():
            self.model, ck = load_checkpoint(CLS_WEIGHTS, self.device)
            self.demo = False
            self.version = f"{ck['arch']}-ep{ck['epoch']}-valAUC{ck['val_mean_auc']:.3f}"
        else:
            log.warning("No weights at %s -> DEMO MODE (untrained head, predictions are meaningless)", CLS_WEIGHTS)
            torch.manual_seed(0)
            self.model = build_model("densenet121", len(CLASSES), pretrained=False).to(self.device).eval()
            self.demo, self.version = True, "DEMO-untrained"
        self.thresholds = json.loads(CLS_THRESHOLDS.read_text()) if CLS_THRESHOLDS.exists() else {}

    def predict(self, img: Image.Image) -> dict:
        from medimg.explain.gradcam import cam_to_rgba_png, gradcam

        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            probs = self.torch.sigmoid(self.model(x))[0].cpu().numpy()
        top = int(probs.argmax())
        cam = gradcam(self.model, x, top)
        thr = {c: self.thresholds.get(c, 0.5) for c in self.classes}
        return {
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "top_class": self.classes[top], "confidence": float(probs[top]),
            "flagged": [c for c, p in zip(self.classes, probs) if p >= thr[c]],
            "thresholds": thr, "gradcam_target": self.classes[top],
            "heatmap_png": cam_to_rgba_png(cam, img.size),
        }


def load_segmentation_engine():
    from medimg.config import SEG_WEIGHTS
    if not SEG_WEIGHTS.exists():
        log.warning("No segmentation weights at %s -> CT segmentation disabled", SEG_WEIGHTS)
        return None
    from medimg.segmentation.infer import SegmentationEngine
    return SegmentationEngine(SEG_WEIGHTS)


def rgba_overlay_png(mask2d, color=(255, 64, 64), alpha=120) -> bytes:
    import numpy as np
    rgba = np.zeros((*mask2d.shape, 4), dtype=np.uint8)
    rgba[mask2d > 0] = (*color, alpha)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, "PNG")
    return buf.getvalue()
