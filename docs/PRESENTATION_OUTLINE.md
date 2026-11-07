# 10-minute presentation outline

| Min | Slide | Content |
|---|---|---|
| 0-1 | Problem | Radiologist workload; AI as second reader needs explainability and accountability. Education-only notice. |
| 1-3 | Design | Architecture diagram: Orthanc -> worker -> models -> API -> viewer; why DICOM/Orthanc, why human-in-the-loop. |
| 3-5 | Models | DenseNet121 transfer learning (2-phase), MONAI 3D U-Net, Grad-CAM; data, patient-level split, DVC. |
| 5-7 | Live demo | Upload -> auto-analysis -> heatmap + confidence -> Approve/Reject -> audit log with chain verification. (Record a backup video.) |
| 7-8 | Results | Per-class AUC/sens/spec with n+, Dice, confusion matrix, one good and one failing Grad-CAM example. |
| 8-9 | Challenges | Label noise, imbalance, shortcut learning, DICOM quirks (MONOCHROME1, windowing), 3D orientation. |
| 9-10 | Limits and next steps | Bias audit, external validation, calibration, DICOM-SR/Secondary Capture write-back, OHIF, real auth. |
