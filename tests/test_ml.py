"""Needs torch/torchvision; skipped where they aren't installed (run in the Docker/Colab env)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_gradcam_shape_and_range():
    from medimg.classification.model import build_model
    from medimg.explain.gradcam import cam_to_rgba_png, gradcam
    m = build_model("densenet121", 14, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    cam = gradcam(m, x, 3)
    assert cam.shape == (224, 224) and 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
    assert cam_to_rgba_png(cam, (300, 200)).startswith(b"\x89PNG")


@pytest.mark.parametrize("arch", ["densenet121", "efficientnet_b0"])
def test_models_output_14_logits(arch):
    from medimg.classification.model import build_model
    m = build_model(arch, 14, pretrained=False).eval()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 14)
