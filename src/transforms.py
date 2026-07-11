from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_transforms(image_size: int, mean: tuple[float, ...], std: tuple[float, ...]):
    """返回正式实验使用的 safe 在线增强和测试预处理。"""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.85, 1.0),
                ratio=(1.0, 1.0),
                interpolation=InterpolationMode.BICUBIC,
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
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, test_transform

