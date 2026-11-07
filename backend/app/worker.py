"""Polls Orthanc /changes, runs inference on new studies, stores results + audit entries."""
import io
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
import pydicom

from medimg.config import MODEL_VERSION, SEG_LABEL
from medimg.preprocessing.dicom_utils import dicom_series_to_volume, dicom_to_pil

from .config import settings
from .inference import ClassifierEngine, load_segmentation_engine, rgba_overlay_png

log = logging.getLogger("worker")


class Worker(threading.Thread):
    def __init__(self, db, orthanc):
        super().__init__(daemon=True, name="orthanc-worker")
        self.db, self.orthanc, self.stop_event = db, orthanc, threading.Event()
        self.cls = self.seg = None
        self.seg_loaded = False

    # engines are created lazily so the API starts fast and works without ML deps until needed
    def _cls(self):
        if self.cls is None:
            self.cls = ClassifierEngine()
        return self.cls

    def _seg(self):
        if not self.seg_loaded:
            self.seg, self.seg_loaded = load_segmentation_engine(), True
        return self.seg

    def run(self):
        log.info("worker started, polling %s every %ss", settings.orthanc_url, settings.poll_seconds)
        while not self.stop_event.is_set():
            try:
                self.poll_once()
            except Exception as e:  # keep the loop alive if Orthanc is down
                log.warning("poll failed: %s", e)
            self.stop_event.wait(settings.poll_seconds)

    def poll_once(self):
        since = int(self.db.get_state("last_change", "0"))
        resp = self.orthanc.changes(since)
        for ch in resp["Changes"]:
            try:
                if ch["ChangeType"] == "NewInstance":
                    self.on_instance(ch["ID"])
                elif ch["ChangeType"] == "StableSeries":
                    self.on_series(ch["ID"])
            except Exception as e:
                log.exception("processing %s failed", ch)
                self._fail(ch["ID"], str(e))
            self.db.set_state("last_change", str(ch["Seq"]))
        return len(resp["Changes"])

    def _fail(self, oid, err):
        if not self.db.has_orthanc_id(oid):
            rid = self.db.create_result(orthanc_id=oid, kind="classification", status="failed", error=err[:500])
            self.db.audit("system", "inference_failed", rid, detail=err[:200])

    # ---- 2D classification (DX/CR) ------------------------------------------------
    def on_instance(self, iid):
        if self.db.has_orthanc_id(iid):
            return
        tags = self.orthanc.instance_tags(iid)
        if tags.get("Modality") not in settings.cls_modalities:
            return
        ds = pydicom.dcmread(io.BytesIO(self.orthanc.instance_file(iid)))
        img = dicom_to_pil(ds)
        out = self._cls().predict(img)
        art_dir = settings.results_dir / iid
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "heatmap.png").write_bytes(out.pop("heatmap_png"))
        eng = self._cls()
        rid = self.db.create_result(
            orthanc_id=iid, kind="classification", modality=tags.get("Modality"), patient_id=tags.get("PatientID"),
            study_uid=tags.get("StudyInstanceUID"), status="pending_review", prediction=out,
            artifacts={"heatmap": "heatmap.png"}, model_version=eng.version, demo_mode=int(eng.demo))
        self.db.audit("system", "ai_result_created", rid, eng.version, {"top": out["top_class"], "p": round(out["confidence"], 4)})
        log.info("classified %s -> %s (%.2f)", iid, out["top_class"], out["confidence"])

    # ---- 3D segmentation (CT series) -----------------------------------------------
    def on_series(self, sid):
        if self.db.has_orthanc_id(sid):
            return
        ids = self.orthanc.series_instances(sid)
        if len(ids) < settings.min_series_slices:
            return
        first = self.orthanc.instance_tags(ids[0])
        if first.get("Modality") not in settings.seg_modalities:
            return
        seg = self._seg()
        if seg is None:
            log.warning("CT series %s skipped: no segmentation weights", sid)
            return
        import nibabel as nib
        datasets = [pydicom.dcmread(io.BytesIO(self.orthanc.instance_file(i))) for i in ids]
        vol, aff, order = dicom_series_to_volume(datasets)
        ordered_ids = [ids[i] for i in order]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "series.nii.gz")
            nib.save(nib.Nifti1Image(vol.transpose(2, 1, 0), aff), path)  # array order (cols, rows, D)
            mask_xyz = seg.predict(path)
        mask = mask_xyz.transpose(2, 1, 0)  # back to [D, rows, cols]
        art_dir = settings.results_dir / sid
        art_dir.mkdir(parents=True, exist_ok=True)
        for k in range(mask.shape[0]):
            (art_dir / f"overlay_{k:04d}.png").write_bytes(rgba_overlay_png(mask[k]))
        vox_ml = abs(np.linalg.det(aff[:3, :3])) / 1000.0
        pred = {"label": SEG_LABEL, "volume_ml": round(float(mask.sum() * vox_ml), 1),
                "slices_with_mask": int((mask.reshape(mask.shape[0], -1).sum(1) > 0).sum()), "n_slices": int(mask.shape[0])}
        rid = self.db.create_result(
            orthanc_id=sid, kind="segmentation", modality=first.get("Modality"), patient_id=first.get("PatientID"),
            study_uid=first.get("StudyInstanceUID"), status="pending_review", prediction=pred,
            artifacts={"instances": ordered_ids}, model_version=f"unet3d-{MODEL_VERSION}", demo_mode=0)
        self.db.audit("system", "ai_result_created", rid, "unet3d", pred)
