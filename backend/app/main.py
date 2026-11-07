import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import httpx
import pydicom
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from medimg.config import EDU_NOTICE
from medimg.preprocessing.dicom_utils import make_dicom_from_image

from .auth import authenticate, current_user, make_token
from .config import settings
from .db import DB
from .orthanc import Orthanc

logging.basicConfig(level=logging.INFO)


class Login(BaseModel):
    username: str
    password: str


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    comment: Optional[str] = None


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.db = DB(settings.db_path)
        app.state.orthanc = Orthanc()
        app.state.worker = None
        if settings.worker_enabled:
            from .worker import Worker
            app.state.worker = Worker(app.state.db, Orthanc())
            app.state.worker.start()
        yield
        if app.state.worker:
            app.state.worker.stop_event.set()

    app = FastAPI(title="AI-Assisted Medical Imaging (education only)", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
    db = lambda r: r.app.state.db  # noqa: E731

    def result_or_404(request: Request, rid: int):
        r = db(request).get_result(rid)
        if not r:
            raise HTTPException(404, "Result not found")
        return r

    @app.get("/api/health")
    def health(request: Request):
        return {"ok": True, "orthanc": request.app.state.orthanc.ping(), "notice": EDU_NOTICE}

    @app.post("/api/login")
    def login(body: Login, request: Request):
        if not authenticate(body.username, body.password):
            db(request).audit(body.username, "login_failed")
            raise HTTPException(401, "Wrong username or password")
        db(request).audit(body.username, "login")
        return {"token": make_token(body.username), "user": body.username, "notice": EDU_NOTICE}

    @app.get("/api/results")
    def list_results(request: Request, status: Optional[str] = None, user: str = Depends(current_user)):
        return db(request).list_results(status)

    @app.get("/api/results/{rid}")
    def get_result(rid: int, request: Request, user: str = Depends(current_user)):
        r = result_or_404(request, rid)
        db(request).audit(user, "view_result", rid, r["model_version"], r["prediction"])
        return r

    @app.get("/api/results/{rid}/image")
    def scan_image(rid: int, request: Request, slice: int = 0, user: str = Depends(current_user)):
        r = result_or_404(request, rid)
        if r["kind"] == "classification":
            iid = r["orthanc_id"]
        else:
            ids = (r["artifacts"] or {}).get("instances", [])
            iid = ids[slice] if 0 <= slice < len(ids) else None
        if not iid:
            raise HTTPException(404, "Slice not found")
        try:
            return Response(request.app.state.orthanc.preview(iid), media_type="image/png")
        except httpx.HTTPError:
            raise HTTPException(502, "Orthanc unavailable")

    @app.get("/api/results/{rid}/overlay")
    def overlay(rid: int, request: Request, slice: int = 0, user: str = Depends(current_user)):
        r = result_or_404(request, rid)
        d = settings.results_dir / r["orthanc_id"]
        f = d / ("heatmap.png" if r["kind"] == "classification" else f"overlay_{slice:04d}.png")
        if not f.is_file():
            raise HTTPException(404, "Overlay not found")
        return Response(f.read_bytes(), media_type="image/png")

    @app.post("/api/results/{rid}/decision")
    def decide(rid: int, body: Decision, request: Request, user: str = Depends(current_user)):
        r = result_or_404(request, rid)
        if not db(request).decide(rid, body.decision, user, body.comment):
            raise HTTPException(409, f"Result already {r['status']}")
        db(request).audit(user, f"decision_{body.decision}", rid, r["model_version"], r["prediction"], body.comment)
        return db(request).get_result(rid)

    @app.get("/api/audit")
    def audit(request: Request, result_id: Optional[int] = None, limit: int = 200, user: str = Depends(current_user)):
        return {"integrity": db(request).verify_audit_chain(), "entries": db(request).audit_entries(limit, result_id)}

    @app.post("/api/upload")
    async def upload(request: Request, file: UploadFile = File(...), patient_id: str = Form("DEMO-001"),
                     user: str = Depends(current_user)):
        data = await file.read()
        if len(data) > 200 * 1024 * 1024:
            raise HTTPException(413, "File too large")
        try:
            pydicom.dcmread(io.BytesIO(data))  # already DICOM
        except Exception:
            try:
                buf = io.BytesIO()
                make_dicom_from_image(Image.open(io.BytesIO(data)), patient_id).save_as(buf, write_like_original=False)
                data = buf.getvalue()
            except Exception:
                raise HTTPException(400, "Upload a DICOM, PNG or JPG file")
        try:
            res = request.app.state.orthanc.upload(data)
        except httpx.HTTPError:
            raise HTTPException(502, "Orthanc unavailable")
        db(request).audit(user, "upload", detail=f"{file.filename} -> orthanc {res.get('ID')}")
        return res

    return app


app = create_app()
