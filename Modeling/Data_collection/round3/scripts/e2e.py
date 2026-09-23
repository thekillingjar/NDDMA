#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COLLECT = SCRIPT_DIR / "collect.py"
FIT = SCRIPT_DIR / "fit.py"
DRAW = SCRIPT_DIR / "draw.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NDDMA2 Round3 2D/3D/4D/5D end-to-end workflow."
    )
    parser.add_argument("action", choices=("all", "collect", "fit", "draw"))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--model-output-dir", default="")
    parser.add_argument("--msprof-bin", default="")
    parser.add_argument("--factor-csv", default="")
    parser.add_argument("--run-output-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def invoke(script: Path, args: argparse.Namespace, data_dir: str = "",
           output_dir: str = "", collect_options: bool = False) -> int:
    command = [sys.executable, str(script)]
    if data_dir and script == FIT:
        command.append(f"--data-dir={data_dir}")
    elif data_dir and script == DRAW:
        command.append(f"--input-dir={data_dir}")
    if args.measurement_csv and script == FIT:
        command.append(f"--measurement-csv={args.measurement_csv}")
    if output_dir:
        command.append(f"--output-dir={output_dir}")
    if args.msprof_bin and collect_options:
        command.append(f"--msprof-bin={args.msprof_bin}")
    if args.factor_csv and collect_options:
        command.append(f"--factor-csv={args.factor_csv}")
    if args.dry_run:
        command.append("--dry-run")
    print("[INFO]", " ".join(command))
    return 0 if args.dry_run else subprocess.run(command).returncode


def main() -> int:
    args = parse_args()
    if args.action in ("all", "collect"):
        status = invoke(COLLECT, args, output_dir=args.run_output_dir,
                        collect_options=True)
        if status:
            return status
    if args.action in ("all", "fit"):
        status = invoke(FIT, args, data_dir=args.data_dir or args.run_output_dir,
                        output_dir=args.model_output_dir)
        if status:
            return status
    if args.action in ("all", "draw"):
        status = invoke(DRAW, args, data_dir=args.model_output_dir,
                        output_dir="")
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
