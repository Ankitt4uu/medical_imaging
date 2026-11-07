from monai.transforms import (Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              Orientationd, RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd)

from medimg.config import SEG_INTENSITY, SEG_ROI, SEG_SPACING


def _base(keys, with_label):
    lo, hi = SEG_INTENSITY
    modes = ("bilinear", "nearest") if with_label else ("bilinear",)
    return [
        LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys), Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=SEG_SPACING, mode=modes),
        ScaleIntensityRanged(keys="image", a_min=lo, a_max=hi, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
    ]


def train_transforms():
    keys = ["image", "label"]
    return Compose(_base(keys, True) + [
        RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=SEG_ROI, pos=1, neg=1,
                               num_samples=4, image_key="image", image_threshold=0),
        EnsureTyped(keys=keys),
    ])


def val_transforms():
    return Compose(_base(["image", "label"], True) + [EnsureTyped(keys=["image", "label"])])


def infer_transforms():
    return Compose(_base(["image"], False) + [EnsureTyped(keys=["image"])])
