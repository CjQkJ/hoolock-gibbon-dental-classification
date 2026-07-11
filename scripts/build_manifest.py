from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def source_type(path: Path) -> str:
    name = path.stem.lower()
    if "_aug_" in name:
        return "offline_augmented"
    if "_style" in name or any(part.lower().endswith("-style") for part in path.parts):
        return "museum_style"
    return "original"


def parse_filename(path: Path) -> dict[str, str]:
    parts = path.stem.split("#")
    if len(parts) < 8:
        raise ValueError(f"文件名不符合 # 分隔规范: {path.name}")
    individual_id = parts[2]
    museum_match = re.match(r"[A-Za-z]+", individual_id)
    # 两张 AMNH112721 历史 style 文件缺失 layer 字段；正式清单将 layer 留空。
    missing_layer = parts[6].upper() in {"M1", "M2", "M3"}
    layer = "" if missing_layer else parts[6]
    tooth_index = 6 if missing_layer else 7
    direction_index = 7 if missing_layer else 8
    direction = parts[direction_index].split("_", 1)[0]
    return {
        "genus": parts[0],
        "filename_species": parts[1],
        "individual_id": individual_id,
        "museum": museum_match.group(0).upper() if museum_match else "",
        "position": parts[5],
        "layer": layer,
        "tooth_type": parts[tooth_index],
        "direction": direction,
    }


def build_manifest(data_root: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for split in ("train", "test"):
        split_root = data_root / split
        for class_name, label in (("hoolock", 0), ("leuconedys", 1)):
            class_root = split_root / class_name
            for path in sorted(class_root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                metadata = parse_filename(path)
                rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "label": label,
                        "source_type": source_type(path),
                        **metadata,
                        "relative_path": path.relative_to(data_root).as_posix(),
                    }
                )
    return rows


def validate(rows: list[dict[str, str | int]], strict_paper_counts: bool) -> None:
    counts = Counter((row["split"], row["class_name"]) for row in rows)
    sources = Counter((row["split"], row["source_type"]) for row in rows)
    train_individuals = {row["individual_id"] for row in rows if row["split"] == "train"}
    test_individuals = {row["individual_id"] for row in rows if row["split"] == "test"}
    overlap = train_individuals & test_individuals
    if overlap:
        raise ValueError(f"发现个体级数据泄漏: {sorted(overlap)}")
    if strict_paper_counts:
        expected_counts = {
            ("train", "hoolock"): 966,
            ("train", "leuconedys"): 1018,
            ("test", "hoolock"): 28,
            ("test", "leuconedys"): 54,
        }
        expected_sources = {
            ("train", "original"): 258,
            ("train", "museum_style"): 436,
            ("train", "offline_augmented"): 1290,
            ("test", "original"): 82,
        }
        if counts != Counter(expected_counts) or sources != Counter(expected_sources):
            raise ValueError(f"数据量与论文口径不一致: classes={counts}, sources={sources}")
    print(f"样本数: {len(rows)}")
    print(f"类别统计: {dict(counts)}")
    print(f"来源统计: {dict(sources)}")
    print(f"个体数: train={len(train_individuals)}, test={len(test_individuals)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="metadata/dataset_manifest.csv")
    parser.add_argument("--strict-paper-counts", action="store_true")
    args = parser.parse_args()
    rows = build_manifest(Path(args.data_root))
    validate(rows, args.strict_paper_counts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
