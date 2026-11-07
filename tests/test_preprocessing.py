import csv
import io

import numpy as np
import pydicom
import pytest
from PIL import Image

from medimg.preprocessing.dicom_utils import dicom_series_to_volume, dicom_to_pil, make_dicom_from_image
from medimg.preprocessing.prepare_nih import split_patients


def test_png_to_dicom_roundtrip():
    img = Image.fromarray((np.random.rand(80, 60) * 255).astype(np.uint8))
    buf = io.BytesIO()
    make_dicom_from_image(img, "P42").save_as(buf, write_like_original=False)
    ds = pydicom.dcmread(io.BytesIO(buf.getvalue()))
    assert ds.PatientID == "P42" and ds.Modality == "DX"
    out = dicom_to_pil(ds)
    assert out.size == (60, 80) and out.mode == "L"


def test_monochrome1_is_inverted():
    arr = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    ds = make_dicom_from_image(Image.fromarray(arr), "P1")
    a = np.array(dicom_to_pil(ds))
    ds.PhotometricInterpretation = "MONOCHROME1"
    b = np.array(dicom_to_pil(ds))
    assert a[0, 0] < a[0, -1] and b[0, 0] > b[0, -1]


def test_patient_split_has_no_leakage():
    pats = [f"p{i}" for i in range(200)] * 3
    s = split_patients(pats)
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"])
    assert len(s["train"]) == 140 and len(s["val"]) == 20 and len(s["test"]) == 40


def _ct_slice(z, val):
    ds = make_dicom_from_image(Image.fromarray(np.full((16, 16), val, np.uint8)), "P")
    ds.Modality = "CT"
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, z]
    ds.PixelSpacing = [0.8, 0.9]
    ds.SliceThickness = 2.0
    return ds


def test_series_sorted_by_position_and_affine_spacing():
    zs = [6.0, 0.0, 4.0, 2.0]               # shuffled on purpose
    ds = [_ct_slice(z, int(z * 10)) for z in zs]
    vol, aff, order = dicom_series_to_volume(ds)
    assert order == [1, 3, 2, 0]
    assert [int(vol[k, 0, 0]) for k in range(4)] == [0, 20, 40, 60]
    # voxel sizes recovered from affine: cols=0.9 (dx), rows=0.8 (dy), slice=2.0
    assert np.allclose(np.linalg.norm(aff[:3, :3], axis=0), [0.9, 0.8, 2.0])
