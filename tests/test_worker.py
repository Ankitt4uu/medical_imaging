"""Worker flow with a fake Orthanc and a stub model (no torch needed)."""
import io

from PIL import Image

from backend.app.config import settings
from backend.app.db import DB
from backend.app.worker import Worker
from medimg.preprocessing.dicom_utils import make_dicom_from_image


class FakeEngine:
    demo, version = True, "stub-v0"
    def predict(self, img):
        return {"probabilities": {"Effusion": 0.9, "Mass": 0.1}, "top_class": "Effusion", "confidence": 0.9,
                "flagged": ["Effusion"], "thresholds": {}, "gradcam_target": "Effusion", "heatmap_png": b"png"}


class FakeOrthanc:
    def __init__(self):
        buf = io.BytesIO()
        make_dicom_from_image(Image.new("L", (32, 32), 90), "PX").save_as(buf, write_like_original=False)
        self.file = buf.getvalue()
        self.sent = False
    def changes(self, since, limit=100):
        ch = [] if since >= 1 else [{"Seq": 1, "ChangeType": "NewInstance", "ID": "inst-A"}]
        return {"Changes": ch, "Done": True, "Last": 1}
    def instance_tags(self, iid): return {"Modality": "DX", "PatientID": "PX", "StudyInstanceUID": "1.2.3"}
    def instance_file(self, iid): return self.file


def test_new_instance_is_analysed_once(tmp_path):
    settings.results_dir = tmp_path
    db = DB(":memory:")
    w = Worker(db, FakeOrthanc())
    w.cls = FakeEngine()
    assert w.poll_once() == 1
    assert w.poll_once() == 0                       # cursor advanced, no reprocessing
    [r] = db.list_results()
    assert r["status"] == "pending_review" and r["prediction"]["top_class"] == "Effusion" and r["demo_mode"] == 1
    assert (tmp_path / "inst-A" / "heatmap.png").read_bytes() == b"png"
    assert [e["action"] for e in db.audit_entries()] == ["ai_result_created"]


def test_failure_is_recorded_not_fatal(tmp_path):
    settings.results_dir = tmp_path
    db = DB(":memory:")
    o = FakeOrthanc()
    o.instance_file = lambda iid: b"not dicom"
    w = Worker(db, o)
    w.cls = FakeEngine()
    w.poll_once()
    [r] = db.list_results()
    assert r["status"] == "failed" and r["error"]
