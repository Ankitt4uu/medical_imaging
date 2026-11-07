# Model Card

**Intended use:** education and demonstration of an AI-assisted review workflow. **Not** for diagnosis, triage or any clinical decision.

| | Classifier | Segmenter |
|---|---|---|
| Architecture | DenseNet121 (ImageNet init; EfficientNet-B0 option) | MONAI 3D U-Net, residual units, channels 16-256 |
| Task | Multi-label, 14 chest findings | Spleen vs background (CT) |
| Data | NIH ChestX-ray14 sample, patient-level 70/10/20 split | Medical Segmentation Decathlon Task09 |
| Input | 224x224, ImageNet normalisation | 1.5x1.5x2.0 mm, HU window [-57,164], 96^3 patches |
| Loss | BCE with positive-class weights | Dice |
| Training | 2 epochs head only, then full fine-tune at lr/10, AdamW, seed 42 | AdamW 1e-3, sliding-window validation |
| Selection | Best validation mean AUC | Best validation Dice |
| Explanation | Grad-CAM on final feature map, top class | n/a |

## Evaluation
Run `python -m medimg.classification.evaluate` then `python scripts/make_results_table.py` and paste the table here.
Thresholds are chosen per class on the validation set (Youden J) and applied once to the test set.
Segmentation: report mean validation Dice from `results/seg_train_history.json`.

_Results not yet filled in: no training was run when this scaffold was created._

## Limits and bias
See [LIMITS_AND_BIAS.md](LIMITS_AND_BIAS.md).

## Reproducibility
Seeds fixed (42), versions pinned by the DVC pipeline (`dvc.yaml`, `params.yaml`), data versioned with DVC, model version string stored with every prediction.
