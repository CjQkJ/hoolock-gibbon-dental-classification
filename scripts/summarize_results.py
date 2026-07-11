from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="results/model_summary.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.runs).glob("*/run_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["run_dir"] = path.parent.as_posix()
        rows.append(row)
    rows.sort(key=lambda row: float(row["best_test_macro_f1"]), reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

