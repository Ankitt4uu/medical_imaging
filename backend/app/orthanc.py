import httpx

from .config import settings


class Orthanc:
    def __init__(self, url=None, user=None, password=None):
        self.c = httpx.Client(base_url=url or settings.orthanc_url, timeout=60,
                              auth=(user or settings.orthanc_user, password or settings.orthanc_password))

    def changes(self, since: int, limit: int = 100):
        r = self.c.get("/changes", params={"since": since, "limit": limit})
        r.raise_for_status()
        return r.json()

    def instance_file(self, iid: str) -> bytes:
        r = self.c.get(f"/instances/{iid}/file")
        r.raise_for_status()
        return r.content

    def instance_tags(self, iid: str) -> dict:
        r = self.c.get(f"/instances/{iid}/simplified-tags")
        r.raise_for_status()
        return r.json()

    def series_instances(self, sid: str) -> list:
        r = self.c.get(f"/series/{sid}/instances")
        r.raise_for_status()
        return [i["ID"] for i in r.json()]

    def preview(self, iid: str) -> bytes:
        r = self.c.get(f"/instances/{iid}/preview", headers={"Accept": "image/png"})
        r.raise_for_status()
        return r.content

    def upload(self, data: bytes) -> dict:
        r = self.c.post("/instances", content=data, headers={"Content-Type": "application/dicom"})
        r.raise_for_status()
        return r.json()

    def ping(self) -> bool:
        try:
            return self.c.get("/system").status_code == 200
        except httpx.HTTPError:
            return False
