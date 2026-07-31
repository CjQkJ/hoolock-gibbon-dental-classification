from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="按 SAM31 e73b33b 协议依次运行论文 Table S4 的 31 个模型。")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--models", default="configs/models_31.json")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--checkpoint-root", default=None, help="锁定权重根目录；按 models_31.json 的 weight_relative_path 定位模型权重。")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    rows = json.loads(Path(args.models).read_text(encoding="utf-8"))["models"]
    selected = set(args.only or [row["key"] for row in rows])
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else None
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
        if checkpoint_root is not None:
            relative_weight = row.get("weight_relative_path")
            if relative_weight:
                command.extend(["--checkpoint", str(checkpoint_root / relative_weight)])
        if int(row.get("initial_physical_microbatch") or 0) > 0:
            command.extend(["--physical-microbatch", str(row["initial_physical_microbatch"])])
        if int(row.get("initial_gpu_count") or 1) > 1:
            command.extend(["--gpu-count", str(row["initial_gpu_count"])])
        print("RUN", " ".join(command), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0 and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
