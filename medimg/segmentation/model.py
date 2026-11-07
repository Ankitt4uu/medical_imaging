import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet


def build_unet() -> UNet:
    """3D U-Net (MONAI). 2 output channels: background / spleen."""
    return UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH)


def load_unet(path, device="cpu"):
    m = build_unet().to(device)
    ck = torch.load(path, map_location=device)
    m.load_state_dict(ck["model_state"] if "model_state" in ck else ck)
    return m.eval()
