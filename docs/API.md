# API

All routes except `/api/health` and `/api/login` need `Authorization: Bearer <token>`. Interactive docs: `/docs` on port 8000 (when run outside nginx).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness + whether Orthanc is reachable |
| POST | `/api/login` `{username,password}` | Returns JWT |
| GET | `/api/results?status=` | Queue (`pending_review`, `approved`, `rejected`, `failed`) |
| GET | `/api/results/{id}` | Result detail (audited as `view_result`) |
| GET | `/api/results/{id}/image?slice=` | Scan PNG proxied from Orthanc (`slice` for CT) |
| GET | `/api/results/{id}/overlay?slice=` | Grad-CAM heatmap (RGBA PNG) or segmentation overlay |
| POST | `/api/results/{id}/decision` `{decision: approve\|reject, comment?}` | One decision per result; 409 if already decided |
| GET | `/api/audit?result_id=&limit=` | Entries + chain integrity check |
| POST | `/api/upload` (multipart `file`, `patient_id`) | DICOM, PNG or JPG -> Orthanc |

Classification `prediction`: `{probabilities{14}, top_class, confidence, flagged[], thresholds, gradcam_target}`.
Segmentation `prediction`: `{label, volume_ml, slices_with_mask, n_slices}`.
