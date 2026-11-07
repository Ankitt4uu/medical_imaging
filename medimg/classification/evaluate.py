"""Pick per-class thresholds on VAL (Youden J), report TEST AUC/sensitivity/specificity/ECE.

python -m medimg.classification.evaluate --data data/processed
Writes models/cls_thresholds.json and results/cls_test_metrics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from medimg.classification.dataset import CXRDataset
from medimg.classification.model import load_checkpoint
from medimg.classification.train import per_class_auc, predict
from medimg.config import CLASSES, CLS_THRESHOLDS, CLS_WEIGHTS
from medimg.preprocessing.transforms import eval_transform


def youden_threshold(y, p):
    if not (0 < y.sum() < len(y)):
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint(CLS_WEIGHTS, dev)
    data = Path(a.data)
    load = lambda n: DataLoader(CXRDataset(data / f"{n}.csv", eval_transform()), a.batch_size, num_workers=2)
    yv, pv = predict(model, load("val"), dev)
    yt, pt = predict(model, load("test"), dev)
    thr = {c: youden_threshold(yv[:, i], pv[:, i]) for i, c in enumerate(CLASSES)}
    CLS_THRESHOLDS.write_text(json.dumps(thr, indent=2))
    aucs = per_class_auc(yt, pt)
    metrics = {"checkpoint_epoch": ck["epoch"], "test_images": int(len(yt)), "per_class": {}}
    for i, c in enumerate(CLASSES):
        pred = pt[:, i] >= thr[c]
        tp, fn = int((pred & (yt[:, i] == 1)).sum()), int((~pred & (yt[:, i] == 1)).sum())
        tn, fp = int((~pred & (yt[:, i] == 0)).sum()), int((pred & (yt[:, i] == 0)).sum())
        metrics["per_class"][c] = {
            "auc": aucs[c], "threshold": thr[c], "positives": int(yt[:, i].sum()),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}, "ece": ece(yt[:, i], pt[:, i]),
        }
    metrics["mean_auc"] = float(np.nanmean(list(aucs.values())))
    Path("results").mkdir(exist_ok=True)
    Path("results/cls_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("test mean AUC", round(metrics["mean_auc"], 4))
    for c, m in metrics["per_class"].items():
        print(f"{c:20s} AUC {m['auc']:.3f}  sens {m['sensitivity']:.2f}  spec {m['specificity']:.2f}  n+={m['positives']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--batch-size", type=int, default=64)
    main(ap.parse_args())
