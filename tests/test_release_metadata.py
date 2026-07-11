import csv
import json
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

