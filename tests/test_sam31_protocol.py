import csv
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_sam31_protocol_json_declares_frozen_pipeline():
    protocol = json.loads((ROOT / "configs/sam31_e73b33b.json").read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "sam31_e73b33b_v1"
    assert protocol["canonical_candidate"]["short_id"] == "e73b33b"
    training = protocol["training"]
    assert training["seed"] == 20260507
    assert training["logical_batch_size"] == 16
    assert training["default_physical_microbatch"] == 16
    assert training["optimizer"]["name"] == "AdamW"
    assert training["optimizer"]["learning_rate"] == 0.0003
    assert training["sam"]["adaptive"] is False
    assert training["sam"]["rho"] == 0.05
    assert training["augmentation"]["resize"] == "direct_square"
    assert training["augmentation"]["horizontal_flip_probability"] == 0.5
    assert protocol["test_contract"]["tta"] is False
    assert protocol["test_contract"]["decision_rule"] == "argmax"
    assert protocol["sources"]["model_registry"] == "configs/models_31.json"
    assert protocol["sources"]["weight_archive_id"] == "shortlist_50_20260502"
    assert protocol["dataset"]["dataset_id"] == "dataset_augmented"
    assert protocol["dataset"]["manifest"] == "metadata/dataset_manifest.csv"
    assert protocol["model_overrides"]["37"]["initial_physical_microbatch"] == 8


def test_experiment_yaml_uses_sam31_pipeline():
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    assert config["protocol_id"] == "sam31_e73b33b_v1"
    assert config["seed"] == 20260507
    assert config["online_augmentation"]["resize"] == "direct_square"
    assert config["training"]["sam"]["adaptive"] is False
    assert config["training"]["selection_split"] == "test"
    assert config["training"]["num_workers"] == 4


def test_models_31_matches_lock_and_expected_screening_ids():
    protocol = json.loads((ROOT / "configs/sam31_e73b33b.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "configs/sam31_e73b33b_models.lock.json").read_text(encoding="utf-8"))
    models = json.loads((ROOT / "configs/models_31.json").read_text(encoding="utf-8"))["models"]
    assert len(models) == 31
    assert len({row["key"] for row in models}) == 31
    assert len({row["screening_id"] for row in models}) == 31
    model_ids = {row["screening_id"] for row in models}
    assert model_ids == set(protocol["expected_screening_ids"])
    assert model_ids == {row["screening_id"] for row in lock["models"]}
    assert lock["schema_version"] == 2
    assert lock["weight_archive_id"] == "shortlist_50_20260502"
    for lock_row in lock["models"]:
        released_row = next(row for row in models if row["screening_id"] == lock_row["screening_id"])
        assert released_row["weight_relative_path"] == lock_row["weight_relative_path"]
        assert released_row["weight_sha256"] == lock_row["weight_sha256"]

    swinv2 = next(row for row in models if row["screening_id"] == 37)
    assert swinv2["initial_physical_microbatch"] == 8
    assert swinv2["initial_gpu_count"] == 2

    for row in models:
        assert len(row["mean"]) == 3
        assert len(row["std"]) == 3
        assert not Path(row["weight_relative_path"]).is_absolute()


def test_table_s4_reports_sam31_convnext_top_results():
    rows = list(csv.DictReader((ROOT / "results/table_s4_results.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 31
    top = rows[0]
    assert top["Screening ID"] == "18"
    assert "ConvNeXt-Base" in top["Model"]
    assert float(top["Accuracy (%)"]) == pytest.approx(92.6829268292683)
    assert float(top["Balanced accuracy (%)"]) == pytest.approx(90.14550264550265)
    assert float(top["Macro-F1 (%)"]) == pytest.approx(91.5521978021978)
    assert top["KIZ011338 correct"] == "2"
    assert top["KIZ011338 n"] == "2"
    assert top["MCZ26474 correct"] == "16"
    assert top["MCZ26474 n"] == "16"

    observed_order = [
        (-float(row["Macro-F1 (%)"]), -float(row["Accuracy (%)"]), int(row["Screening ID"]))
        for row in rows
    ]
    assert observed_order == sorted(observed_order)
