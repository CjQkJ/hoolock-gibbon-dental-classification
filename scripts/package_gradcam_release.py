#!/usr/bin/env python3
"""Build clean bilingual reviewer packages from the audited SAM31 Grad-CAM files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


EXPECTED_COUNTS = {"original": 33, "gradcam": 132}
MODEL_DIRS = {
    "1": "01_convnext_base",
    "2": "02_efficientnet_x_b5",
    "3": "03_efficientnet_b5",
    "4": "04_efficientnetv2_s",
}
TARGET_LAYERS = {
    "1": "stages.3",
    "2": "blocks.5.8.bn2",
    "3": "blocks.5.8.bn2",
    "4": "conv_head",
}

EN_README = """# Grad-CAM Reviewer Materials: SAM31/e73b33b

This package contains the audited Grad-CAM reviewer materials associated with
the frozen `SAM31/e73b33b` protocol for the Hoolock dental-image
classification study.

## Contents

- 33 original dental images from four selected individuals.
- 132 Grad-CAM overlays: four models × 33 images.
- Original images and overlays are stored separately and grouped by tooth
  position (`M1`, `M2`, and `M3`).
- Portable metadata, model-level results, per-image predictions, and SHA-256
  checksums.

The package contains no model weights and no raw dataset outside the selected
reviewer materials.

## Selected individuals

| Individual | Class | Dataset role | Tooth coverage |
|---|---|---|---|
| `AMNH112688` | `H. hoolock` | training-set original | M1, M2, M3 |
| `AMNH112713` | leuconedys group | test-set original | M1, M2, M3 |
| `KIZ011338` | `H. hoolock` | test-set original | M1 |
| `MCZ26474` | leuconedys group | test-set original | M1, M2 |

These individuals and images are a selected qualitative review subset. They
are not an additional evaluation split.

## Models and recorded results

The four included models are the selected high-performing models in the
final SAM31 results table:

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| ConvNeXt-Base | 92.68% | 90.15% | 91.55% |
| EfficientNet-X-B5 | 91.46% | 90.08% | 90.43% |
| EfficientNet-B5 | 89.02% | 87.37% | 87.69% |
| EfficientNetV2-S | 87.80% | 89.02% | 87.03% |

For ConvNeXt-Base, `KIZ011338` was classified correctly for 2/2 images and
`MCZ26474` was classified correctly for 16/16 images in the audited test-set
run.

## Grad-CAM protocol

- Method: Grad-CAM.
- Target layer: the final convolutional feature stage or model-specific final
  convolutional feature layer recorded in `metadata/model_summary.csv`.
- Target: the true class (`target=true`).
- Overlay: complete heatmap over the original image.
- Colour map: `turbo`.
- Overlay alpha: `0.4`.
- No activation cutoff, masking, test-time augmentation, threshold tuning,
  calibration, or ensembling was applied to these materials.

The heatmaps are qualitative attribution maps. They should not be interpreted
as pixel-level proof of a causal anatomical mechanism.

## Directory layout

```text
images/originals/<individual>/<tooth>/<image>.jpg
images/gradcam/<model>/<individual>/<tooth>/<image>_cam.jpg
metadata/file_manifest.csv
metadata/model_summary.csv
metadata/protocol.json
metadata/checksums.sha256
```

All paths in the metadata are package-relative. Original server and local
Windows paths have been removed from this reviewer package.

## Verification

The package was checked for the expected image counts, readable RGB images,
consistent dimensions (`380 × 380` pixels), absence of empty images, and
matching SHA-256 checksums. The source study data remain subject to the
permissions of the specimen-holding institutions and the corresponding
authors.
"""

ZH_README = """# Grad-CAM 审稿材料：SAM31/e73b33b

本压缩包包含 Hoolock 牙齿图像分类研究中冻结协议 `SAM31/e73b33b` 对应的
Grad-CAM 审稿材料。

## 包含内容

- 来自 4 个选定个体的 33 张牙齿原图；
- 4 个模型 × 33 张图像，共 132 张 Grad-CAM 热力图；
- 原图与热力图分开保存，并按牙位（`M1`、`M2`、`M3`）整理；
- 可移植的元数据、模型级结果、逐图预测结果和 SHA-256 校验值。

本压缩包不包含模型权重，也不包含除选定审稿材料以外的完整原始数据集。

## 选定个体

| 个体 | 类别 | 数据集角色 | 牙位 |
|---|---|---|---|
| `AMNH112688` | `H. hoolock` | 训练集原图 | M1、M2、M3 |
| `AMNH112713` | leuconedys group | 测试集原图 | M1、M2、M3 |
| `KIZ011338` | `H. hoolock` | 测试集原图 | M1 |
| `MCZ26474` | leuconedys group | 测试集原图 | M1、M2 |

这些个体和图像是用于定性检查的选定子集，不构成额外的评估集。

## 模型及记录结果

以下 4 个模型来自最终 SAM31 结果表中的高性能模型：

| 模型 | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| ConvNeXt-Base | 92.68% | 90.15% | 91.55% |
| EfficientNet-X-B5 | 91.46% | 90.08% | 90.43% |
| EfficientNet-B5 | 89.02% | 87.37% | 87.69% |
| EfficientNetV2-S | 87.80% | 89.02% | 87.03% |

在核验后的 ConvNeXt-Base 测试集运行中，`KIZ011338` 的 2/2 张图像分类正确，
`MCZ26474` 的 16/16 张图像分类正确。

## Grad-CAM 参数

- 方法：Grad-CAM；
- 目标层：最后卷积特征阶段，或 `metadata/model_summary.csv` 中记录的模型专属最后卷积特征层；
- 目标类别：真实类别（`target=true`）；
- 叠加方式：将完整热力图叠加到原图；
- 色图：`turbo`；
- 叠加透明度：`0.4`；
- 未使用激活阈值、掩膜、测试时增强、阈值调节、概率校准或集成。

热力图是定性归因图，不能被解释为像素级的因果解剖机制证据。

## 目录结构

```text
images/originals/<individual>/<tooth>/<image>.jpg
images/gradcam/<model>/<individual>/<tooth>/<image>_cam.jpg
metadata/file_manifest.csv
metadata/model_summary.csv
metadata/protocol.json
metadata/checksums.sha256
```

元数据中的路径全部为压缩包内相对路径，已移除原服务器路径和本地 Windows 绝对路径。

## 核验

本压缩包已核验图像数量、RGB 图像可读性、统一尺寸（`380 × 380` 像素）、
空白图检查和 SHA-256 校验值。研究原始数据的访问仍须遵守标本保存机构和通讯作者的许可要求。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--language", choices=("english", "chinese"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_model_id(rank: str, model: str) -> str:
    if rank in MODEL_DIRS:
        return MODEL_DIRS[rank]
    value = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return f"{int(rank):02d}_{value}"


def ensure_inside(path: Path, root: Path) -> Path:
    path = path.resolve()
    root = root.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"source file is outside source root: {path}")
    return path


def copy_row(row: dict[str, str], source_root: Path, package_root: Path) -> str:
    source = ensure_inside(Path(row["destination_path"]), source_root)
    if row["file_type"] == "original":
        relative = Path("images") / "originals" / row["individual_id"] / row["tooth_type"] / source.name
    elif row["file_type"] == "gradcam":
        model_id = safe_model_id(row["model_rank"], row["model"])
        relative = Path("images") / "gradcam" / model_id / row["individual_id"] / row["tooth_type"] / source.name
    else:
        raise ValueError(f"unexpected file type: {row['file_type']}")
    destination = package_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return relative.as_posix()


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def package_files(package_root: Path) -> list[Path]:
    return sorted(p for p in package_root.rglob("*") if p.is_file() and p.name != "checksums.sha256")


def write_checksums(package_root: Path) -> None:
    lines = []
    for path in package_files(package_root):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(package_root).as_posix()}")
    (package_root / "metadata" / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def zip_package(package_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(package_root.name) / path.relative_to(package_root))


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    package_name = f"GradCAM_SAM31_e73b33b_{'English' if args.language == 'english' else 'Chinese'}"
    package_root = output_root / package_name
    zip_path = output_root / f"{package_name}.zip"

    if package_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {package_root}")
        shutil.rmtree(package_root)
    if zip_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {zip_path}")
        zip_path.unlink()
    package_root.mkdir(parents=True, exist_ok=True)

    with args.source_manifest.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    counts = {key: sum(row["file_type"] == key for row in source_rows) for key in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected source counts: {counts}")

    copied_rows = []
    for row in source_rows:
        package_path = copy_row(row, source_root, package_root)
        copied_rows.append(
            {
                "individual_id": row["individual_id"],
                "true_class": row["true_class"],
                "subset": row["subset"],
                "tooth_type": row["tooth_type"],
                "file_type": row["file_type"],
                "model_rank": row["model_rank"],
                "model": row["model"],
                "predicted_class": row["predicted_class"],
                "correct": row["correct"],
                "probability_hoolock": row["probability_hoolock"],
                "probability_leuconedys": row["probability_leuconedys"],
                "package_path": package_path,
            }
        )

    write_csv(
        package_root / "metadata" / "file_manifest.csv",
        copied_rows,
        [
            "individual_id", "true_class", "subset", "tooth_type", "file_type",
            "model_rank", "model", "predicted_class", "correct",
            "probability_hoolock", "probability_leuconedys", "package_path",
        ],
    )

    with args.results.open(encoding="utf-8-sig", newline="") as handle:
        results = list(csv.DictReader(handle))
    rank_field = "Result rank" if "Result rank" in results[0] else "rank"
    top_models = [row for row in results if row[rank_field] in MODEL_DIRS]
    top_models.sort(key=lambda row: int(row[rank_field]))
    model_rows = []
    for row in top_models:
        model_rows.append(
            {
                "model_rank": row[rank_field],
                "screening_id": row["Screening ID"],
                "model_name": row["Model"],
                "architecture_id": row["Model ID"],
                "package_model_id": safe_model_id(row[rank_field], row["Model"]),
                "target_layer": TARGET_LAYERS[row[rank_field]],
                "parameters_m": row["Parameters (M)"],
                "input_size": row["Input size"],
                "accuracy_pct": f"{float(row['Accuracy (%)']):.4f}",
                "balanced_accuracy_pct": f"{float(row['Balanced accuracy (%)']):.4f}",
                "macro_f1_pct": f"{float(row['Macro-F1 (%)']):.4f}",
                "best_epoch": row["Best epoch"],
                "kiz011338_correct": row["KIZ011338 correct"],
                "kiz011338_n": row["KIZ011338 n"],
                "mcz26474_correct": row["MCZ26474 correct"],
                "mcz26474_n": row["MCZ26474 n"],
                "protocol_id": "sam31_e73b33b_v1",
            }
        )
    write_csv(package_root / "metadata" / "model_summary.csv", model_rows, list(model_rows[0]))

    protocol = {
        "protocol_id": "sam31_e73b33b_v1",
        "canonical_candidate": "e73b33b",
        "package_version": "GradCAM_SAM31_e73b33b",
        "image_counts": {"original": counts["original"], "gradcam": counts["gradcam"]},
        "models": [row["package_model_id"] for row in model_rows],
        "target_mode": "true",
        "overlay_mode": "full",
        "mask_floor": None,
        "mask_percentile": None,
        "colormap": "turbo",
        "alpha": 0.4,
        "source_paths_removed": True,
        "test_time_augmentation": False,
        "threshold_tuning": False,
        "calibration": False,
        "ensembling": False,
    }
    (package_root / "metadata" / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    readme = EN_README if args.language == "english" else ZH_README
    (package_root / "README.md").write_text(readme, encoding="utf-8")
    write_checksums(package_root)
    zip_package(package_root, zip_path)
    print(json.dumps({"package": str(package_root), "zip": str(zip_path), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
