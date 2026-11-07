# Limits and Bias

**For education only. Not for clinical use.** Not validated, not cleared, not a medical device.

## Data limits
- NIH ChestX-ray14 labels were mined from radiology reports by NLP, not read from images. They are noisy; the NIH authors estimate >90% accuracy for the labels, and independent audits found worse on some classes (e.g. pneumothorax, effusion with chest tubes).
- Single institution (NIH Clinical Center), frontal views, one population. No external validation done here.
- Severe imbalance: rare classes (Hernia, Pneumonia, Fibrosis) have few positives in the sample, so per-class metrics are unstable. Report n+ beside every metric.
- The sample subset is about 5.6k images; full-dataset results will differ.

## Known failure modes
- **Shortcut learning:** models can key on text markers, chest tubes, portable-vs-fixed equipment or hospital artefacts instead of pathology. Check Grad-CAM for activations outside the lungs.
- **Distribution shift:** different scanners, exposure, patient positioning, paediatric patients, or post-surgical anatomy.
- **Calibration:** confidences are not probabilities of disease; check the ECE column and reliability on your own test split.
- **Multi-label ambiguity:** "Infiltration" and "Consolidation" overlap heavily; do not over-interpret class names.
- **Grad-CAM:** a coarse, class-conditional saliency map. It shows where the model looked, not that the finding is there.
- **Segmentation:** trained on one task (spleen, portal-venous CT, from one centre); do not apply to other organs, phases or MRI.

## Bias
- Demographics are not controlled or audited here. Published work on ChestX-ray14-like data shows underdiagnosis disparities by sex, age, race and insurance status.
- Do a subgroup check if metadata is available (sex, age band: NIH provides both) and report any gaps rather than hiding them.
- Splits are by patient, but the sample may not reflect any real deployment population.

## Safeguards in this system
- Banner and README notice on every screen.
- Human-in-the-loop: nothing is final until a radiologist approves or rejects; all actions are audited.
- Demo-mode results are flagged in the UI.
- Model version and prediction are recorded with every audit entry.
