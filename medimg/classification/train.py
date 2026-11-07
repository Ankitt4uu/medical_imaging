"""Transfer-learning trainer: phase 1 trains the head, phase 2 fine-tunes everything.

python -m medimg.classification.train --data data/processed --arch densenet121 --epochs 12
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from medimg.classification.dataset import CXRDataset
from medimg.classification.model import backbone_params, build_model
from medimg.config import CLASSES, MODELS_DIR
from medimg.preprocessing.transforms import eval_transform, train_transform


def per_class_auc(y, p):
    out = {}
    for i, c in enumerate(CLASSES):
        out[c] = float(roc_auc_score(y[:, i], p[:, i])) if 0 < y[:, i].sum() < len(y) else float("nan")
    return out


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(a.data)
    tr = CXRDataset(data / "train.csv", train_transform())
    va = CXRDataset(data / "val.csv", eval_transform())
    tl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=dev == "cuda")
    vl = DataLoader(va, a.batch_size, num_workers=a.workers)

    model = build_model(a.arch).to(dev)
    pos = np.array([[float(r[c]) for c in CLASSES] for r in tr.rows]).sum(0)
    pos_weight = torch.tensor(np.clip((len(tr) - pos) / np.maximum(pos, 1), 1, 50), dtype=torch.float32).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    best, history = -1.0, []
    for ep in range(a.epochs):
        frozen = ep < a.freeze_epochs
        for p in backbone_params(model):
            p.requires_grad = not frozen
        lr = a.lr if frozen else a.lr / 10
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
        yv, pv = predict(model, vl, dev)
        aucs = per_class_auc(yv, pv)
        mean_auc = float(np.nanmean(list(aucs.values())))
        history.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_mean_auc": mean_auc, "frozen": frozen})
        print(history[-1])
        if mean_auc > best:
            best = mean_auc
            MODELS_DIR.mkdir(exist_ok=True, parents=True)
            torch.save({"model_state": model.state_dict(), "arch": a.arch, "classes": CLASSES,
                        "val_mean_auc": best, "epoch": ep + 1}, MODELS_DIR / "cls_best.pt")
    Path("results").mkdir(exist_ok=True)
    Path("results/cls_train_history.json").write_text(json.dumps(history, indent=2))
    print("best val mean AUC", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--arch", default="densenet121", choices=["densenet121", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
