"""Turn results/cls_test_metrics.json (written by evaluate.py) into a markdown table for the README / model card."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "results/cls_test_metrics.json")
m = json.loads(p.read_text())
print(f"Test images: {m['test_images']} | mean AUC: {m['mean_auc']:.3f}\n")
print("| Finding | n+ | AUC | Sens | Spec | ECE |\n|---|---:|---:|---:|---:|---:|")
for c, v in m["per_class"].items():
    print(f"| {c.replace('_', ' ')} | {v['positives']} | {v['auc']:.3f} | {v['sensitivity']:.2f} | {v['specificity']:.2f} | {v['ece']:.3f} |")
