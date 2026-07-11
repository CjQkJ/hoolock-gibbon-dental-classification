import torch
from PIL import Image

from src.transforms import build_transforms


def test_transforms_return_expected_shape():
    train_transform, test_transform = build_transforms(
        224,
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )
    image = Image.new("RGB", (600, 600), (128, 128, 128))
    assert train_transform(image).shape == torch.Size([3, 224, 224])
    assert test_transform(image).shape == torch.Size([3, 224, 224])

