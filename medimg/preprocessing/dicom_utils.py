"""DICOM helpers: pixel data -> 8-bit PIL image, PNG -> DICOM, CT series -> volume + affine."""
from __future__ import annotations

import datetime as dt
from typing import List, Tuple

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

DX_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.1.1"  # Digital X-Ray Image Storage - For Presentation


def _first(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
        return float(v[0]) if len(v) else None
    return float(v)


def dicom_to_pil(ds: pydicom.Dataset) -> Image.Image:
    """Apply rescale + VOI window, fix MONOCHROME1, return 8-bit grayscale image.
    Compressed transfer syntaxes need `pylibjpeg` or `gdcm` installed."""
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # multi-frame or RGB: take first frame, or average colour channels
        arr = arr.mean(-1) if arr.shape[-1] in (3, 4) else arr[0]
    arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    wc, ww = _first(getattr(ds, "WindowCenter", None)), _first(getattr(ds, "WindowWidth", None))
    if wc is not None and ww and ww > 1:
        lo, hi = wc - ww / 2, wc + ww / 2
    else:
        lo, hi = np.percentile(arr, (0.5, 99.5))
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = 255 - arr
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def make_dicom_from_image(img: Image.Image, patient_id: str, study_desc: str = "Chest X-ray",
                          modality: str = "DX", study_uid: str | None = None) -> FileDataset:
    """Wrap a PNG/JPG as a minimal valid DICOM so it can be pushed to Orthanc (datasets ship as PNG)."""
    arr = np.array(img.convert("L"), dtype=np.uint8)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DX_SOP_CLASS
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.PatientID, ds.PatientName = str(patient_id), f"PATIENT^{patient_id}"
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality, ds.StudyDescription = modality, study_desc
    now = dt.datetime.now()
    ds.StudyDate = ds.SeriesDate = ds.ContentDate = now.strftime("%Y%m%d")
    ds.StudyTime = ds.ContentTime = now.strftime("%H%M%S")
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.PixelData = arr.tobytes()
    return ds


def dicom_series_to_volume(datasets: List[pydicom.Dataset]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Sort a CT series by position; return (volume[D,rows,cols] in HU, RAS affine for array order (cols,rows,D), order).

    `order[k]` is the index in the input list of the k-th slice, so callers can map slice k -> instance id.
    NOTE: verified on synthetic data only; check orientation on a real series before trusting masks.
    """
    ds0 = datasets[0]
    iop = np.array(ds0.ImageOrientationPatient, dtype=float)
    r, c = iop[:3], iop[3:]
    s = np.cross(r, c)
    pos = np.array([np.array(d.ImagePositionPatient, dtype=float) for d in datasets])
    order = [int(i) for i in np.argsort(pos @ s)]
    pos = pos[order]
    dz = float(np.median(np.diff(pos @ s))) if len(order) > 1 else float(getattr(ds0, "SliceThickness", 1.0))
    dy, dx = (float(v) for v in ds0.PixelSpacing)  # PixelSpacing = [row spacing, col spacing]
    vol = np.stack([
        datasets[i].pixel_array.astype(np.float32) * float(getattr(datasets[i], "RescaleSlope", 1))
        + float(getattr(datasets[i], "RescaleIntercept", 0)) for i in order
    ])
    aff = np.eye(4)
    aff[:3, 0], aff[:3, 1], aff[:3, 2], aff[:3, 3] = r * dx, c * dy, s * dz, pos[0]
    aff = np.diag([-1, -1, 1, 1]) @ aff  # LPS -> RAS
    return vol, aff, order
