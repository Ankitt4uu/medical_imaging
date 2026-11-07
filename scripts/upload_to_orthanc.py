"""Push studies to Orthanc. PNG/JPG are wrapped as DICOM first.

python scripts/upload_to_orthanc.py --dir data/raw/images --limit 20
python scripts/upload_to_orthanc.py --synthetic 3        # smoke test without any dataset
"""
import argparse
import io
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medimg.preprocessing.dicom_utils import make_dicom_from_image  # noqa: E402


def synthetic_cxr(seed: int, size=512) -> Image.Image:
    """Crude fake 'chest' (two lung ellipses + noise). For plumbing tests only, not medically meaningful."""
    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), 70)
    d = ImageDraw.Draw(img)
    d.ellipse([size * .12, size * .15, size * .46, size * .85], fill=25)
    d.ellipse([size * .54, size * .15, size * .88, size * .85], fill=25)
    d.rectangle([size * .46, size * .1, size * .54, size * .9], fill=130)
    arr = np.array(img.filter(ImageFilter.GaussianBlur(8)), dtype=np.float32) + rng.normal(0, 6, (size, size))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def to_dicom_bytes(img, pid):
    buf = io.BytesIO()
    make_dicom_from_image(img, pid).save_as(buf, write_like_original=False)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--url", default="http://localhost:8042")
    ap.add_argument("--user", default="orthanc")
    ap.add_argument("--password", default="change-me")
    ap.add_argument("--dry-run", action="store_true", help="convert only, don't upload")
    a = ap.parse_args()

    items = []
    for i in range(a.synthetic):
        items.append((f"SYN-{i:03d}", to_dicom_bytes(synthetic_cxr(i), f"SYN-{i:03d}")))
    if a.dir:
        files = sorted(p for p in Path(a.dir).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".dcm"})[:a.limit]
        for p in files:
            items.append((p.stem, p.read_bytes() if p.suffix.lower() == ".dcm" else to_dicom_bytes(Image.open(p), p.stem.split("_")[0])))
    if not items:
        sys.exit("Nothing to upload: pass --dir or --synthetic N")
    with httpx.Client(base_url=a.url, auth=(a.user, a.password), timeout=60) as c:
        for name, data in items:
            if a.dry_run:
                print(f"{name}: {len(data)} bytes (dry run)")
                continue
            r = c.post("/instances", content=data, headers={"Content-Type": "application/dicom"})
            r.raise_for_status()
            print(f"{name}: {r.json().get('Status')} {r.json().get('ID')}")


if __name__ == "__main__":
    main()
