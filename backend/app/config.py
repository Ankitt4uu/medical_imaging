import os
from pathlib import Path


class Settings:
    def __init__(self):
        self.db_path = os.getenv("DB_PATH", "data/app.db")
        self.results_dir = Path(os.getenv("RESULTS_DIR", "results/artifacts"))
        self.orthanc_url = os.getenv("ORTHANC_URL", "http://localhost:8042").rstrip("/")
        self.orthanc_user = os.getenv("ORTHANC_USER", "orthanc")
        self.orthanc_password = os.getenv("ORTHANC_PASSWORD", "change-me")
        self.jwt_secret = os.getenv("JWT_SECRET", "dev-secret-change-me")
        self.jwt_hours = int(os.getenv("JWT_HOURS", "8"))
        # "user:password,user2:password2" -- demo auth, replace with a real IdP for anything beyond a demo
        self.users = dict(p.split(":", 1) for p in os.getenv("USERS", "radiologist:radiologist,admin:admin").split(","))
        self.worker_enabled = os.getenv("WORKER_ENABLED", "1") == "1"
        self.poll_seconds = float(os.getenv("POLL_SECONDS", "3"))
        self.cls_modalities = set(os.getenv("CLS_MODALITIES", "DX,CR,DR").split(","))
        self.seg_modalities = set(os.getenv("SEG_MODALITIES", "CT").split(","))
        self.min_series_slices = int(os.getenv("MIN_SERIES_SLICES", "20"))


settings = Settings()
