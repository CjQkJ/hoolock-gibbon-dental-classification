from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


AUGMENTATIONS = (
    "bright_saturated",
    "dark_contrasted",
    "warm_tone",
    "cool_tone",
    "high_contrast",
    "soft",
    "vivid",
    "muted",
)


def hue_shift(image: Image.Image, rng: random.Random) -> Image.Image:
    hsv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + rng.randint(-15, 15)) % 180
    return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))


def apply_augmentation(image: Image.Image, name: str, rng: random.Random) -> tuple[Image.Image, str]:
    if name == "bright_saturated":
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(1.1, 1.3))
        return ImageEnhance.Color(image).enhance(rng.uniform(1.2, 1.4)), "bright_sat"
    if name == "dark_contrasted":
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.7, 0.9))
        return ImageEnhance.Contrast(image).enhance(rng.uniform(1.2, 1.4)), "dark_con"
    if name == "warm_tone":
        return ImageEnhance.Color(hue_shift(image, rng)).enhance(rng.uniform(1.1, 1.3)), "warm"
    if name == "cool_tone":
        return ImageEnhance.Brightness(hue_shift(image, rng)).enhance(rng.uniform(0.9, 1.1)), "cool"
    if name == "high_contrast":
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(1.3, 1.5))
        return ImageEnhance.Sharpness(image).enhance(rng.uniform(1.2, 1.4)), "hicon"
    if name == "soft":
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 0.95))
        return ImageEnhance.Color(image).enhance(rng.uniform(0.85, 0.95)), "soft"
    if name == "vivid":
        image = ImageEnhance.Color(image).enhance(rng.uniform(1.3, 1.5))
        return ImageEnhance.Contrast(image).enhance(rng.uniform(1.1, 1.2)), "vivid"
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.6, 0.8))
    return ImageEnhance.Brightness(image).enhance(rng.uniform(0.95, 1.05)), "muted"


def main() -> None:
    parser = argparse.ArgumentParser(description="为每张训练原图随机选择 5/8 类离线颜色增强。")
    parser.add_argument("--source-root", required=True, help="只含原图的 train/test 数据集。")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--augmentations-per-image", type=int, default=5)
    args = parser.parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    rng = random.Random(args.seed)

    image_paths = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    for source in tqdm(sorted(image_paths)):
        relative = source.relative_to(source_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if relative.parts[0] != "train":
            continue
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        selected = rng.sample(AUGMENTATIONS, args.augmentations_per_image)
        for augmentation_name in selected:
            augmented, suffix = apply_augmentation(image, augmentation_name, rng)
            augmented.save(
                destination.with_name(f"{destination.stem}_aug_{suffix}{destination.suffix}"),
                quality=95,
            )


if __name__ == "__main__":
    main()

