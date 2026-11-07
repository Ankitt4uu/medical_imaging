import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import create_app


class FakeOrthanc:
    def preview(self, iid): return b"\x89PNG fake"
    def ping(self): return True
    def upload(self, data): return {"ID": "abc"}


@pytest.fixture()
def client(tmp_path):
    settings.db_path = str(tmp_path / "t.db")
    settings.results_dir = tmp_path / "art"
    settings.worker_enabled = False
    with TestClient(create_app()) as c:
        c.app.state.orthanc = FakeOrthanc()
        yield c


def auth(client, user="radiologist", pw="radiologist"):
    t = client.post("/api/login", json={"username": user, "password": pw}).json()["token"]
    return {"Authorization": f"Bearer {t}"}


def seed(client, **kw):
    return client.app.state.db.create_result(
        orthanc_id=kw.pop("oid", "inst1"), kind="classification", modality="DX", patient_id="P1",
        status="pending_review", prediction={"top_class": "Effusion", "confidence": 0.8},
        artifacts={"heatmap": "heatmap.png"}, model_version="test", demo_mode=1)


def test_requires_auth(client):
    assert client.get("/api/results").status_code == 401
    assert client.post("/api/login", json={"username": "x", "password": "y"}).status_code == 401


def test_review_flow_and_audit(client):
    h = auth(client)
    rid = seed(client)
    assert client.get("/api/results", headers=h).json()[0]["id"] == rid
    assert client.get(f"/api/results/{rid}/image", headers=h).status_code == 200
    r = client.post(f"/api/results/{rid}/decision", json={"decision": "approve", "comment": "agree"}, headers=h)
    assert r.status_code == 200 and r.json()["status"] == "approved" and r.json()["decided_by"] == "radiologist"
    # second decision is rejected
    r2 = client.post(f"/api/results/{rid}/decision", json={"decision": "reject"}, headers=h)
    assert r2.status_code == 409
    a = client.get("/api/audit", headers=h).json()
    actions = [e["action"] for e in a["entries"]]
    assert a["integrity"]["valid"] and "decision_approve" in actions and "login" in actions


def test_bad_decision_value(client):
    h = auth(client)
    rid = seed(client)
    assert client.post(f"/api/results/{rid}/decision", json={"decision": "maybe"}, headers=h).status_code == 422


def test_overlay_served(client, tmp_path):
    h = auth(client)
    rid = seed(client)
    d = settings.results_dir / "inst1"
    d.mkdir(parents=True)
    (d / "heatmap.png").write_bytes(b"png")
    assert client.get(f"/api/results/{rid}/overlay", headers=h).content == b"png"


def test_upload_png_is_wrapped_as_dicom(client):
    import io
    from PIL import Image
    h = auth(client)
    buf = io.BytesIO()
    Image.new("L", (64, 64), 100).save(buf, "PNG")
    r = client.post("/api/upload", files={"file": ("x.png", buf.getvalue(), "image/png")}, headers=h)
    assert r.status_code == 200 and r.json()["ID"] == "abc"
    assert client.post("/api/upload", files={"file": ("x.txt", b"hello", "text/plain")}, headers=h).status_code == 400
