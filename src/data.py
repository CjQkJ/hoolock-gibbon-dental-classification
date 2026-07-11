from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        split: str,
        transform: Any,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform
        with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"manifest 中没有 split={split} 的样本")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.data_root / Path(row["relative_path"])
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), row

    def class_counts(self) -> Counter[int]:
        return Counter(int(row["label"]) for row in self.rows)


def collate_batch(batch):
    import torch

    images, labels, rows = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(rows)

