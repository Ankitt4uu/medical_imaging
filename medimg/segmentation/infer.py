"""Segmentation inference. Input NIfTI (any orientation/spacing) -> binary mask in the INPUT's voxel grid (Invertd)."""
import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscreted, Compose, Invertd

from medimg.config import SEG_ROI
from medimg.segmentation.model import load_unet
from medimg.segmentation.transforms import infer_transforms


class SegmentationEngine:
    def __init__(self, weights, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_unet(weights, self.device)
        self.pre = infer_transforms()
        self.post = Compose([
            Invertd(keys="pred", transform=self.pre, orig_keys="image", nearest_interp=False, to_tensor=True),
            AsDiscreted(keys="pred", argmax=True),
        ])

    @torch.no_grad()
    def predict(self, nifti_path: str) -> np.ndarray:
        loader = DataLoader(Dataset([{"image": nifti_path}], transform=self.pre), batch_size=1)
        for batch in loader:
            batch["pred"] = sliding_window_inference(batch["image"].to(self.device), SEG_ROI, 4, self.model)
            out = [self.post(i) for i in decollate_batch(batch)][0]["pred"]
            return out[0].cpu().numpy().astype(np.uint8)  # [X,Y,Z] of the input NIfTI
