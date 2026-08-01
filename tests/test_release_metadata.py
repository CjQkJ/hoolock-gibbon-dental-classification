import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_matches_paper_counts_and_has_no_individual_leakage():
    with (ROOT / "metadata/dataset_manifest.csv").open("r", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2066
    assert Counter((row["split"], row["class_name"]) for row in rows) == Counter(
        {
            ("train", "hoolock"): 966,
            ("train", "leuconedys"): 1018,
            ("test", "hoolock"): 28,
            ("test", "leuconedys"): 54,
        }
    )
    assert Counter((row["split"], row["source_type"]) for row in rows) == Counter(
        {
            ("train", "original"): 258,
            ("train", "museum_style"): 436,
            ("train", "offline_augmented"): 1290,
            ("test", "original"): 82,
        }
    )
    train_ids = {row["individual_id"] for row in rows if row["split"] == "train"}
    test_ids = {row["individual_id"] for row in rows if row["split"] == "test"}
    assert len(train_ids) == 60
    assert len(test_ids) == 15
    assert not train_ids & test_ids


def test_model_release_contains_31_unique_models():
    models = json.loads((ROOT / "configs/models_31.json").read_text(encoding="utf-8"))["models"]
    assert len(models) == 31
    assert len({row["key"] for row in models}) == 31
    assert len({row["rank"] for row in models}) == 31


def test_public_release_metadata_is_portable():
    files = [
        ROOT / "configs/sam31_e73b33b.json",
        ROOT / "configs/sam31_e73b33b_models.lock.json",
        ROOT / "configs/models_31.json",
        ROOT / "results/table_s4_results.csv",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert not re.search(r"[A-Za-z]:[\\/]", text)

    protocol = json.loads(files[0].read_text(encoding="utf-8"))
    lock = json.loads(files[1].read_text(encoding="utf-8"))
    assert protocol["sources"]["weight_archive_id"] == "shortlist_50_20260502"
    assert protocol["dataset"]["dataset_id"] == "dataset_augmented"
    assert "weights_root" not in protocol["sources"]
    assert "weights_root" not in lock

    for row in lock["models"]:
        relative_path = Path(row["weight_relative_path"])
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts

