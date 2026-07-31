from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_transforms(image_size: int, mean: tuple[float, ...], std: tuple[float, ...]):
    """Return the fixed SAM-31 online augmentation and deterministic test preprocessing.

    The frozen e73b33b protocol uses a direct square resize during training,
    not RandomResizedCrop. Test inference is deterministic: one resize, tensor
    conversion, and normalization.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.RandomRotation(
                        degrees=5,
                        interpolation=InterpolationMode.BICUBIC,
                    )
                ],
                p=0.5,
            ),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, test_transform
