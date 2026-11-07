"""Build patient-level train/val/test CSVs from the NIH ChestX-ray14 sample (Kaggle: nih-chest-xrays/sample).

python -m medimg.preprocessing.prepare_nih --csv data/raw/sample_labels.csv --images data/raw/images --out data/processed
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from medimg.config import CLASSES


def build_index(images_dir: Path) -> dict:
    return {p.name: p for p in images_dir.rglob("*.png")}


def split_patients(patients, seed=42, frac=(0.7, 0.1, 0.2)):
    """Split by patient id so no patient appears in two splits (prevents leakage)."""
    pats = sorted(set(patients))
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
    return {"train": set(pats[:a]), "val": set(pats[a:b]), "test": set(pats[b:])}


def main(csv_path, images, out, seed=42):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(Path(images))
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            name = r["Image Index"]
            if name not in idx:
                continue
            labels = set(r["Finding Labels"].split("|"))
            rows.append({
                "path": str(idx[name]), "patient_id": r["Patient ID"],
                **{c: int(c in labels) for c in CLASSES},
            })
    splits = split_patients([r["patient_id"] for r in rows], seed)
    counts = {}
    for name, pats in splits.items():
        sub = [r for r in rows if r["patient_id"] in pats]
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sub)
        counts[name] = {"images": len(sub), "patients": len(pats),
                        "positives": {c: int(sum(r[c] for r in sub)) for c in CLASSES}}
    (out / "split_summary.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps({k: (v["images"], v["patients"]) for k, v in counts.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.csv, a.images, a.out, a.seed)
