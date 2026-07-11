from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="依次运行论文 Table S4 的 31 个模型。")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--models", default="configs/models_31.json")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    rows = json.loads(Path(args.models).read_text(encoding="utf-8"))["models"]
    selected = set(args.only or [row["key"] for row in rows])
    for row in rows:
        if row["key"] not in selected:
            continue
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            args.config,
            "--models",
            args.models,
            "--model-key",
            row["key"],
            "--data-root",
            args.data_root,
            "--output-root",
            args.output_root,
        ]
        if int(row.get("recommended_gpus", 1)) > 1:
            command.append("--data-parallel")
        print("RUN", " ".join(command), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0 and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
