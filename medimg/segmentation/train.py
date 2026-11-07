"""3D U-Net on Medical Segmentation Decathlon Task09_Spleen.

Download: http://medicaldecathlon.com  ->  data/Task09_Spleen/{imagesTr,labelsTr}
python -m medimg.segmentation.train --root data/Task09_Spleen --epochs 150
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from medimg.config import MODELS_DIR, SEG_ROI
from medimg.segmentation.model import build_unet
from medimg.segmentation.transforms import train_transforms, val_transforms


def files(root: Path):
    imgs = sorted((root / "imagesTr").glob("*.nii.gz"))
    imgs = [p for p in imgs if not p.name.startswith("._")]
    return [{"image": str(p), "label": str(root / "labelsTr" / p.name)} for p in imgs]


def main(a):
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = files(Path(a.root))
    n_val = max(2, int(len(data) * 0.2))
    tr, va = data[:-n_val], data[-n_val:]
    tl = DataLoader(CacheDataset(tr, train_transforms(), cache_rate=a.cache_rate, num_workers=2), batch_size=2, shuffle=True)
    vl = DataLoader(CacheDataset(va, val_transforms(), cache_rate=a.cache_rate, num_workers=2), batch_size=1)

    model = build_unet().to(dev)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    metric = DiceMetric(include_background=False, reduction="mean")
    post_p, post_l = AsDiscrete(argmax=True, to_onehot=2), AsDiscrete(to_onehot=2)

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for b in tl:
            x, y = b["image"].to(dev), b["label"].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        row = {"epoch": ep + 1, "train_loss": tot / len(tl)}
        if (ep + 1) % a.val_every == 0:
            model.eval()
            with torch.no_grad():
                for b in vl:
                    out = sliding_window_inference(b["image"].to(dev), SEG_ROI, 4, model)
                    metric([post_p(i) for i in decollate_batch(out)], [post_l(i) for i in decollate_batch(b["label"].to(dev))])
            dice = float(metric.aggregate().item())
            metric.reset()
            row["val_dice"] = dice
            if dice > best:
                best = dice
                MODELS_DIR.mkdir(exist_ok=True, parents=True)
                torch.save({"model_state": model.state_dict(), "val_dice": dice, "epoch": ep + 1}, MODELS_DIR / "seg_best.pt")
        hist.append(row)
        print(row)
    Path("results").mkdir(exist_ok=True)
    Path("results/seg_train_history.json").write_text(json.dumps(hist, indent=2))
    print("best val dice", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Task09_Spleen")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--cache-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
