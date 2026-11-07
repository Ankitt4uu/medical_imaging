# AI-Assisted Medical Imaging and Diagnosis System

> **For education only. Not for clinical use.** Research prototype; outputs must never inform patient care.

X-ray / CT scans are pushed to an **Orthanc** DICOM server. A worker pulls new studies, runs deep-learning inference
(**DenseNet121** classifier + **MONAI 3D U-Net** segmenter), computes **Grad-CAM** explanations, and stores results.
A **React** viewer shows the scan, the AI overlay and confidence, and lets a radiologist **Approve / Reject**, with every
action written to a **tamper-evident audit log**.

```
Upload / C-STORE -> Orthanc -> worker (poll /changes) -> model + Grad-CAM -> SQLite + artifacts -> FastAPI -> React viewer
                                                                                                   ^ audit log (append-only, hash-chained)
```
Diagram and data flow: [docs/architecture.md](docs/architecture.md) · API: [docs/API.md](docs/API.md) ·
Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · Limits and bias: [docs/LIMITS_AND_BIAS.md](docs/LIMITS_AND_BIAS.md)

## Quickstart (no GPU, no dataset): demo mode

```bash
docker compose up --build
python scripts/upload_to_orthanc.py --synthetic 3     # fake studies to test the plumbing
# open http://localhost:8080   login: radiologist / radiologist
```
Without trained weights in `models/` the backend runs in **demo mode**: the classifier head is untrained, so predictions and
heatmaps are meaningless. The UI shows a warning banner on every such result. This proves the pipeline, not the model.

## Full pipeline (real model)

Train on a GPU (Kaggle/Colab free tier is enough), then copy the weights into `models/`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/ml.txt

# 1. Data: NIH ChestX-ray14 *sample* (Kaggle: nih-chest-xrays/sample) -> data/raw/{sample_labels.csv,images/}
dvc init && dvc add data/raw && dvc remote add -d storage <path-or-gdrive-url>

# 2. Preprocess + patient-level split, then train + evaluate (DVC pipeline)
dvc repro            # prepare -> train_cls -> eval_cls ; writes models/cls_best.pt, models/cls_thresholds.json, results/*.json

# 3. Segmentation: download Task09_Spleen from medicaldecathlon.com -> data/Task09_Spleen
python -m medimg.segmentation.train --root data/Task09_Spleen --epochs 150     # -> models/seg_best.pt

# 4. Restart the backend so it loads the weights
docker compose restart backend
python scripts/make_results_table.py        # paste into the Results section below
```
To use EfficientNet instead, set `arch: efficientnet_b0` in `params.yaml`.

## Using it
1. Sign in at http://localhost:8080 (change `USERS`, `JWT_SECRET` and the Orthanc password before any shared deployment).
2. Upload a DICOM/PNG/JPG from the header, push via DICOM C-STORE to port 4242, or use `scripts/upload_to_orthanc.py`.
3. Open a study from the queue, toggle the heatmap, adjust opacity, then Approve or Reject (with an optional comment).
4. **Audit log** tab: every login, view, upload, AI result and decision, with chain verification.

## Tests
```bash
pip install -r requirements/dev.txt && pytest -q     # API, audit log, worker flow, DICOM utils (ML tests auto-skip without torch)
pip install torch torchvision && pytest -q           # also runs the Grad-CAM / model tests
```

## Results
_Fill in after training. Do not report numbers you did not measure._

| Finding | n+ | AUC | Sens | Spec | ECE |
|---|---:|---:|---:|---:|---:|
| _run `scripts/make_results_table.py`_ | | | | | |

Segmentation (Spleen, validation Dice): _from `results/seg_train_history.json`_

## Repo layout
```
medimg/            ML package: preprocessing, classification, segmentation (MONAI), explain (Grad-CAM)
backend/app/       FastAPI, Orthanc client, polling worker, SQLite + hash-chained audit log
frontend/          React (Vite) viewer
scripts/           upload_to_orthanc.py, make_results_table.py
tests/  docs/  dvc.yaml  params.yaml  docker-compose.yml  orthanc.json
```

## Known gaps (read before the demo)
- Training and the Docker images were **not run** where this scaffold was written; the backend, worker, DICOM utilities,
  audit log and frontend build are tested, the model code is syntax-checked only. Expect to fix small issues on first run.
- The CT segmentation path (series -> NIfTI -> MONAI -> per-slice overlays) is unverified on real DICOM series;
  check mask orientation on one case before presenting it.
- Auth is demo-grade (env-configured users). Orthanc is localhost-only by default; do not expose it.
