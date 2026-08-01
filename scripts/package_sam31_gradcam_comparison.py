#!/usr/bin/env python3
"""Build bilingual Grad-CAM comparison packages using the legacy gallery layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


MODEL_ORDER = [
    "convnext_base_with_style",
    "efficientnet_x_b5_with_style",
    "efficientnet_b5_with_style",
    "efficientnetv2_s_with_style",
    "convnext_base_no_style",
]

MODEL_DIR = {
    "convnext_base_with_style": "convnext_style",
    "efficientnet_x_b5_with_style": "ex_b5_style",
    "efficientnet_b5_with_style": "b5_style",
    "efficientnetv2_s_with_style": "v2s_style",
    "convnext_base_no_style": "convnext_no_style",
}

MODEL_TEXT = {
    "convnext_base_with_style": {
        "en": "ConvNeXt-Base (with style augmentation)",
        "zh": "ConvNeXt-Base（含风格化增强）",
    },
    "efficientnet_x_b5_with_style": {
        "en": "EfficientNet-X-B5 (with style augmentation)",
        "zh": "EfficientNet-X-B5（含风格化增强）",
    },
    "efficientnet_b5_with_style": {
        "en": "EfficientNet-B5 (with style augmentation)",
        "zh": "EfficientNet-B5（含风格化增强）",
    },
    "efficientnetv2_s_with_style": {
        "en": "EfficientNetV2-S (with style augmentation)",
        "zh": "EfficientNetV2-S（含风格化增强）",
    },
    "convnext_base_no_style": {
        "en": "ConvNeXt-Base (no-style control)",
        "zh": "ConvNeXt-Base（无风格化对照）",
    },
}

MANIFEST_FIELDS = [
    "index",
    "subset",
    "source_split",
    "class_name",
    "label",
    "individual_id",
    "individual_dir",
    "museum",
    "tooth_type",
    "source_image_name",
    "gradcam_image_path",
    "pred_class",
    "pred_prob",
    "prob_hoolock",
    "prob_leuconedys",
    "correct",
    "cam_class",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "NA"


def copy_images_and_manifest(source_model: Path, package_model: Path) -> list[dict[str, object]]:
    rows = read_csv(source_model / "gradcam_manifest.csv")
    shutil.copytree(source_model / "images", package_model / "images")
    image_paths_by_name: dict[str, Path] = {}
    for path in (source_model / "images").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_paths_by_name[path.name] = path.relative_to(source_model)
    clean_rows: list[dict[str, object]] = []
    for row in rows:
        image_name = Path(row["output_path"]).name
        if image_name not in image_paths_by_name:
            raise FileNotFoundError(f"Grad-CAM image listed in manifest was not found: {image_name}")
        rel = image_paths_by_name[image_name]
        clean_rows.append(
            {
                "index": row["index"],
                "subset": row["subset"],
                "source_split": row["source_split"],
                "class_name": row["class_name"],
                "label": row["label"],
                "individual_id": row["individual_id"],
                "individual_dir": row["individual_dir"],
                "museum": row["museum"],
                "tooth_type": row["tooth_type"],
                "source_image_name": Path(row["source_image_path"]).name,
                "gradcam_image_path": rel.as_posix(),
                "pred_class": row["pred_class"],
                "pred_prob": row["pred_prob"],
                "prob_hoolock": row["prob_hoolock"],
                "prob_leuconedys": row["prob_leuconedys"],
                "correct": row["correct"],
                "cam_class": row["target_class"],
            }
        )
    return clean_rows


def clean_summary(source_model: Path, model_key: str, lang: str) -> dict[str, object]:
    raw = json.loads((source_model / "summary.json").read_text(encoding="utf-8"))
    model = raw.get("model", {}) if isinstance(raw.get("model"), dict) else {}
    cam = raw.get("cam", {}) if isinstance(raw.get("cam"), dict) else {}
    selection = raw.get("selection", {}) if isinstance(raw.get("selection"), dict) else {}
    return {
        "model_key": model_key,
        "directory": MODEL_DIR[model_key],
        "display_name": MODEL_TEXT[model_key][lang],
        "model_id": model.get("model_name"),
        "image_size": model.get("image_size"),
        "preprocess_variant": model.get("preprocess_variant"),
        "training_group": model.get("training_group"),
        "target_layer": cam.get("target_layer"),
        "run_metrics": raw.get("run_metrics", {}),
        "selection": {
            "selected_rows": selection.get("selected_rows"),
            "selected_by_subset": selection.get("selected_by_subset"),
            "selected_by_subset_class": selection.get("selected_by_subset_class"),
            "selected_by_subset_class_tooth": selection.get("selected_by_subset_class_tooth"),
            "tooth_types": selection.get("tooth_types"),
        },
        "gradcam": {
            "method": cam.get("method", "gradcam"),
            "cam_class": "true_class",
            "overlay_mode": cam.get("overlay_mode"),
            "alpha": cam.get("alpha"),
            "colormap": cam.get("colormap"),
            "mask_floor": cam.get("mask_floor"),
            "mask_percentile": cam.get("mask_percentile"),
        },
        "success_count": raw.get("success_count"),
        "failed_count": raw.get("failed_count"),
    }


def html_doc(title: str, body: str, lang: str, extra_css: str = "") -> str:
    return f"""<!doctype html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --ink:#142632; --muted:#667783; --line:#dce9e7; --bg:#f5fbf9; --brand:#16a390; --bad:#b04747; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,"Microsoft YaHei","Noto Sans SC",Arial,sans-serif; color:var(--ink); background:var(--bg); }}
    p {{ margin:8px 0; color:var(--muted); line-height:1.7; }}
    a {{ color:#127c70; font-weight:800; text-decoration:none; }}
    code {{ background:#e9f5f2; padding:2px 6px; border-radius:5px; }}
    table {{ width:100%; border-collapse:collapse; margin:8px 0 18px; }}
    td,th {{ padding:8px 10px; border-bottom:1px solid var(--line); color:#36524c; text-align:left; }}
    {extra_css}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def root_page(summaries: list[dict[str, object]], lang: str) -> str:
    zh = lang == "zh"
    title = "SAM31 Macro-F1 前四模型与无风格化对照 Grad-CAM 图集" if zh else "SAM31 Macro-F1 Top Four and No-Style Control Grad-CAM Gallery"
    lead = (
        "五个页面使用同一份 340 张原始牙齿图像清单和一致的可视化设置：true-class Grad-CAM、完整热力图叠加、turbo 色图、alpha=0.4。"
        if zh
        else "The five galleries use the same 340 original dental-image records and consistent visualization settings: true-class Grad-CAM, full heatmap overlay, turbo colour map, and alpha=0.4."
    )
    labels = {
        "models": "模型页面" if zh else "Model galleries",
        "per": "每个模型热力图" if zh else "Heatmaps per model",
        "train": "训练集原图" if zh else "Training originals",
        "test": "测试集原图" if zh else "Test images",
        "open": "打开热力图" if zh else "Open gallery",
        "summary": "汇总文件" if zh else "Summary files",
        "source": "数据说明" if zh else "Data note",
    }
    css = """
    header,main { width:min(1160px,calc(100% - 48px)); margin:0 auto; }
    header { padding:38px 0 20px; }
    h1 { margin:0 0 10px; font-size:36px; }
    h2 { margin:0 0 10px; font-size:20px; }
    .metrics,.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
    .metric,article,.provenance { background:white; border:1px solid var(--line); border-radius:8px; padding:16px; }
    .metric span { display:block; color:var(--muted); font-size:13px; margin-bottom:8px; }
    .metric strong { font-size:28px; }
    .provenance { margin:14px 0; }
    main { padding-bottom:50px; }
    @media (max-width:760px) { .metrics,.grid { grid-template-columns:1fr; } h1 { font-size:30px; } }
    """
    cards = []
    for summary in summaries:
        key = str(summary["model_key"])
        directory = str(summary["directory"])
        metrics = summary.get("run_metrics", {}) if isinstance(summary.get("run_metrics"), dict) else {}
        cards.append(
            "<article>"
            f"<h2>{html.escape(str(summary['display_name']))}</h2>"
            f"<p>Model ID: <code>{html.escape(str(summary.get('model_id')))}</code></p>"
            f"<p>Accuracy: <b>{pct(metrics.get('accuracy'))}</b>; Macro-F1: <b>{pct(metrics.get('macro_f1'))}</b>; Balanced accuracy: <b>{pct(metrics.get('balanced_accuracy'))}</b></p>"
            f"<p>{'目标层' if zh else 'Target layer'}: <code>{html.escape(str(summary.get('target_layer')))}</code>; {'输入' if zh else 'input'}: {html.escape(str(summary.get('image_size')))}×{html.escape(str(summary.get('image_size')))}</p>"
            f"<p><a href=\"{html.escape(directory)}/index.html\">{labels['open']}</a> · <a href=\"{html.escape(directory)}/gradcam_manifest.csv\">CSV</a> · <a href=\"{html.escape(directory)}/summary.json\">Summary</a></p>"
            "</article>"
        )
    body = f"""
  <header>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(lead)}</p>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><span>{labels['models']}</span><strong>5</strong></div>
      <div class="metric"><span>{labels['per']}</span><strong>340</strong></div>
      <div class="metric"><span>{labels['train']}</span><strong>258</strong></div>
      <div class="metric"><span>{labels['test']}</span><strong>82</strong></div>
    </section>
    <section class="provenance">
      <h2>{labels['source']}</h2>
      <p>{html.escape('The image set contains 258 training-set originals and 82 test images. Augmented training images and style-transfer training images are not displayed in these Grad-CAM galleries.' if not zh else '图集展示 258 张训练集原图和 82 张测试图，不展示训练增强图或风格化训练图。')}</p>
      <p><a href="source_records_340_manifest.csv">source_records_340_manifest.csv</a> · <a href="model_comparison.csv">model_comparison.csv</a> · <a href="selection_summary.json">selection_summary.json</a></p>
    </section>
    <section class="grid">{''.join(cards)}</section>
  </main>
"""
    return html_doc(title, body, "zh-CN" if zh else "en", css)


def group_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["tooth_type"]), str(row["class_name"]))].append(row)
    return grouped


def model_page(summary: dict[str, object], rows: list[dict[str, object]], lang: str) -> str:
    zh = lang == "zh"
    title = f"{summary['display_name']} Grad-CAM"
    labels = {
        "back": "返回总索引" if zh else "Back to index",
        "records": "原始图片" if zh else "Original records",
        "train": "训练集原图" if zh else "Training originals",
        "test": "测试集原图" if zh else "Test images",
        "total_train": "最终训练集" if zh else "Final training set",
        "source": "数据来源说明" if zh else "Data source note",
        "search": "搜索个体、文件名或类别" if zh else "Search individual, file name, or class",
        "all_subset": "全部子集" if zh else "All subsets",
        "all_tooth": "全部牙齿" if zh else "All teeth",
        "all_class": "全部类别" if zh else "All classes",
        "sample": "样本" if zh else "Images",
        "correct": "正确" if zh else "Correct",
        "accuracy": "准确率" if zh else "Accuracy",
        "true": "真实类别" if zh else "True class",
        "pred": "预测" if zh else "Prediction",
        "cam": "CAM 类别" if zh else "CAM class",
    }
    metrics = summary.get("run_metrics", {}) if isinstance(summary.get("run_metrics"), dict) else {}
    css = """
    header { padding:38px 5vw 24px; background:linear-gradient(135deg,#e5fbf6,#fff); border-bottom:1px solid var(--line); }
    main { padding:24px 5vw 52px; }
    h1 { margin:0 0 10px; font-size:clamp(28px,4vw,44px); letter-spacing:0; }
    .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:20px 0; }
    .stats div,.panel { background:white; border:1px solid var(--line); border-radius:12px; padding:16px; box-shadow:0 10px 24px rgba(35,89,78,.06); }
    .stats span { display:block; color:var(--muted); font-size:13px; margin-bottom:8px; }
    .stats strong { font-size:27px; }
    .toolbar { position:sticky; top:0; z-index:2; display:flex; flex-wrap:wrap; gap:10px; padding:12px 0; background:rgba(245,251,249,.94); backdrop-filter:blur(10px); }
    input,select { height:38px; border:1px solid var(--line); border-radius:8px; padding:0 12px; background:white; color:var(--ink); }
    .group { margin:22px 0 34px; }
    .group h2 { margin:0 0 6px; font-size:24px; }
    .tag { display:inline-flex; margin:0 8px 8px 0; padding:5px 9px; border-radius:999px; background:#e9f5f2; color:#2e5f57; font-weight:800; font-size:12px; }
    .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:16px; }
    .card { background:white; border:1px solid var(--line); border-radius:12px; overflow:hidden; box-shadow:0 10px 24px rgba(35,89,78,.06); }
    .card img { display:block; width:100%; aspect-ratio:1.08; object-fit:cover; background:#eef6f3; }
    .meta { display:grid; gap:5px; padding:12px; font-size:13px; color:var(--muted); }
    .meta b { color:var(--ink); font-size:15px; }
    .ok { color:#0b856f; font-weight:800; }
    .bad { color:var(--bad); font-weight:800; }
    .path { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .small-note { color:var(--muted); font-size:14px; }
    """
    header_note = (
        "This page shows Grad-CAM images for the shared 340-record source set."
        if not zh
        else "本页展示同一 340 条来源记录对应的 Grad-CAM 图集。"
    )
    body = [
        "<header>",
        f'<p><a href="../index.html">{labels["back"]}</a></p>',
        f"<h1>{html.escape(str(title))}</h1>",
        f"<p>{html.escape(header_note)}</p>",
        f"<p>Model ID: <code>{html.escape(str(summary.get('model_id')))}</code>; {'目标层' if zh else 'target layer'}: <code>{html.escape(str(summary.get('target_layer')))}</code>; {'输入尺寸' if zh else 'input size'}: {html.escape(str(summary.get('image_size')))}×{html.escape(str(summary.get('image_size')))}; {'CAM 类别' if zh else 'CAM class'}: {'真实类别' if zh else 'true class'}.</p>",
        f"<p>{'测试集指标' if zh else 'Test-set metrics'}: Accuracy {pct(metrics.get('accuracy'))}; Macro-F1 {pct(metrics.get('macro_f1'))}; Balanced accuracy {pct(metrics.get('balanced_accuracy'))}</p>",
        "</header><main>",
        f'<div class="stats"><div><span>{labels["records"]}</span><strong>340</strong></div><div><span>{labels["train"]}</span><strong>258</strong></div><div><span>{labels["test"]}</span><strong>82</strong></div><div><span>{labels["total_train"]}</span><strong>1984</strong></div></div>',
        '<section class="panel">',
        f'<h2>{labels["source"]}</h2>',
        f"<p>{html.escape('The displayed image set contains training-set originals and test images only; augmented training images are excluded from the gallery.' if not zh else '当前展示图集只包含训练集原图和测试图；训练增强图不在本图集中展示。')}</p>",
        '<p><a href="../source_records_340_manifest.csv">source_records_340_manifest.csv</a> - <a href="summary.json">summary.json</a></p>',
        "</section>",
        '<div class="toolbar">',
        f'<input id="q" placeholder="{html.escape(labels["search"])}">',
        f'<select id="subset"><option value="">{labels["all_subset"]}</option><option value="train_original">{labels["train"]}</option><option value="test">{labels["test"]}</option></select>',
        f'<select id="tooth"><option value="">{labels["all_tooth"]}</option><option value="M1">M1</option><option value="M2">M2</option><option value="M3">M3</option></select>',
        f'<select id="cls"><option value="">{labels["all_class"]}</option><option value="hoolock">hoolock</option><option value="leuconedys">leuconedys</option></select>',
        "</div>",
        '<section id="groups">',
    ]
    for (tooth, cls), items in sorted(group_rows(rows).items()):
        correct = sum(str(item["correct"]).lower() == "true" for item in items)
        acc = correct / len(items) if items else 0
        body.append(f'<section class="group" data-tooth="{html.escape(tooth)}" data-class="{html.escape(cls)}">')
        body.append(f"<h2>{html.escape(tooth)} · {html.escape(cls)}</h2>")
        body.append(f'<p><span class="tag">{labels["sample"]} {len(items)}</span><span class="tag">{labels["correct"]} {correct}</span><span class="tag">{labels["accuracy"]} {acc:.2%}</span></p>')
        body.append('<div class="gallery">')
        for item in items:
            status = "ok" if str(item["correct"]).lower() == "true" else "bad"
            subset_label = labels["train"] if item["subset"] == "train_original" else labels["test"]
            body.append(f'<article class="card" data-subset="{html.escape(str(item["subset"]))}" data-class="{html.escape(str(item["class_name"]))}" data-tooth="{html.escape(str(item["tooth_type"]))}" data-individual="{html.escape(str(item["individual_id"]))}">')
            body.append(f'<img src="{html.escape(str(item["gradcam_image_path"]))}" loading="lazy" alt="{html.escape(str(item["individual_id"]))}">')
            body.append('<div class="meta">')
            body.append(f'<b>{html.escape(str(item["class_name"]))} / {html.escape(str(item["tooth_type"]))}</b>')
            body.append(f'<span>{html.escape(subset_label)} · {html.escape(str(item["individual_id"]))}</span>')
            body.append(f'<span>{labels["true"]}: {html.escape(str(item["class_name"]))}</span>')
            body.append(f'<span class="{status}">{labels["pred"]}: {html.escape(str(item["pred_class"]))} ({float(item["pred_prob"]):.3f})</span>')
            body.append(f'<span>{labels["cam"]}: {html.escape(str(item["cam_class"]))}</span>')
            body.append(f'<span class="path">{html.escape(str(item["source_image_name"]))}</span>')
            body.append("</div></article>")
        body.append("</div></section>")
    body.extend(
        [
            "</section></main>",
            """<script>
    const q = document.getElementById('q');
    const subset = document.getElementById('subset');
    const tooth = document.getElementById('tooth');
    const cls = document.getElementById('cls');
    const cards = [...document.querySelectorAll('.card')];
    const groups = [...document.querySelectorAll('.group')];
    function applyFilter() {
      const text = q.value.toLowerCase();
      const sub = subset.value;
      const t = tooth.value;
      const c = cls.value;
      cards.forEach(card => {
        const okText = card.innerText.toLowerCase().includes(text);
        const okSub = !sub || card.dataset.subset === sub;
        const okTooth = !t || card.dataset.tooth === t;
        const okCls = !c || card.dataset.class === c;
        card.style.display = okText && okSub && okTooth && okCls ? '' : 'none';
      });
      groups.forEach(group => {
        group.style.display = [...group.querySelectorAll('.card')].some(card => card.style.display !== 'none') ? '' : 'none';
      });
    }
    [q, subset, tooth, cls].forEach(el => el.addEventListener('input', applyFilter));
  </script>""",
        ]
    )
    return html_doc(str(title), "\n".join(body), "zh-CN" if zh else "en", css)


def package_readme(lang: str) -> str:
    if lang == "zh":
        return """# SAM31 Grad-CAM 对比材料

本包按既有 Grad-CAM 图集样式整理，包含 5 个模型页面：按 Macro-F1 排列的前 4 个 SAM31 模型和 1 个 ConvNeXt-Base 无风格化对照。每个模型页面展示同一 340 条来源记录对应的 Grad-CAM 图像。

包内元数据使用相对路径；服务器路径和本地 Windows 绝对路径已移除。本包不包含模型权重，也不包含完整原始数据集。
"""
    return """# SAM31 Grad-CAM Comparison Materials

This package follows the legacy Grad-CAM gallery layout and contains five model pages: the four highest-ranked SAM31 models by Macro-F1 and one ConvNeXt-Base no-style control. Each model page displays Grad-CAM images for the same 340 source records.

Package metadata uses relative paths only. Server paths and local Windows absolute paths have been removed. This package does not contain model weights or the complete raw dataset.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_package(source_root: Path, output_root: Path, lang: str, overwrite: bool) -> Path:
    package_root = output_root / ("SAM31_gradcam_zh" if lang == "zh" else "SAM31_gradcam_en")
    if package_root.exists():
        if not overwrite:
            raise FileExistsError(package_root)
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)

    summaries: list[dict[str, object]] = []
    source_rows_written = False
    for model_key in MODEL_ORDER:
        source_model = source_root / model_key
        package_model = package_root / MODEL_DIR[model_key]
        package_model.mkdir(parents=True)
        rows = copy_images_and_manifest(source_model, package_model)
        summary = clean_summary(source_model, model_key, lang)
        write_csv(package_model / "gradcam_manifest.csv", rows, MANIFEST_FIELDS)
        (package_model / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (package_model / "index.html").write_text(model_page(summary, rows, lang), encoding="utf-8")
        summaries.append(summary)
        if not source_rows_written:
            write_csv(package_root / "source_records_340_manifest.csv", rows, MANIFEST_FIELDS)
            source_rows_written = True

    comparison_rows = []
    for summary in summaries:
        metrics = summary.get("run_metrics", {}) if isinstance(summary.get("run_metrics"), dict) else {}
        comparison_rows.append(
            {
                "model_key": summary["model_key"],
                "directory": summary["directory"],
                "display_name": summary["display_name"],
                "model_id": summary["model_id"],
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "target_layer": summary["target_layer"],
                "images": summary["success_count"],
            }
        )
    write_csv(
        package_root / "model_comparison.csv",
        comparison_rows,
        ["model_key", "directory", "display_name", "model_id", "accuracy", "balanced_accuracy", "macro_f1", "target_layer", "images"],
    )
    package_info = {
        "package_name": package_root.name,
        "configuration": "SAM31 reference configuration",
        "language": lang,
        "model_count": len(MODEL_ORDER),
        "source_records_per_model": 340,
        "gradcam_images_total": 340 * len(MODEL_ORDER),
        "layout": "legacy_gallery_direct_model_directories",
        "paths": "package_relative",
    }
    (package_root / "selection_summary.json").write_text(json.dumps(package_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (package_root / "README.md").write_text(package_readme(lang), encoding="utf-8")
    (package_root / "index.html").write_text(root_page(summaries, lang), encoding="utf-8")

    checksum_path = package_root / "checksums.sha256"
    lines = []
    for path in sorted(p for p in package_root.rglob("*") if p.is_file() and p != checksum_path):
        lines.append(f"{sha256(path)}  {path.relative_to(package_root).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = output_root / f"sam31_gradcam_{lang}.zip"
    if zip_path.exists():
        if not overwrite:
            raise FileExistsError(zip_path)
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(p for p in package_root.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(output_root).as_posix())
    return zip_path


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    zips = [
        build_package(args.source_root, args.output_root, "en", args.overwrite),
        build_package(args.source_root, args.output_root, "zh", args.overwrite),
    ]
    print(json.dumps({"zip_paths": [str(path) for path in zips]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
