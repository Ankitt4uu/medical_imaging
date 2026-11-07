"""Shared constants for training and serving. Keep train/serve preprocessing identical."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
CLS_WEIGHTS = MODELS_DIR / "cls_best.pt"
CLS_THRESHOLDS = MODELS_DIR / "cls_thresholds.json"
SEG_WEIGHTS = MODELS_DIR / "seg_best.pt"

# NIH ChestX-ray14 pathology labels (multi-label). Order matters: it is baked into checkpoints.
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Segmentation (Medical Segmentation Decathlon Task09_Spleen, CT). Change together with training.
SEG_SPACING = (1.5, 1.5, 2.0)
SEG_INTENSITY = (-57.0, 164.0)   # CT HU window used for spleen
SEG_ROI = (96, 96, 96)
SEG_LABEL = "Spleen"

MODEL_VERSION = os.getenv("MODEL_VERSION", "densenet121-v1")
EDU_NOTICE = "For education only. Not for clinical use."
