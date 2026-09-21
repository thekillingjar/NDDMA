#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_COLLECTION_DIR = SCRIPT_DIR.parent / "Data_collection"
HARNESS_DIR = DATA_COLLECTION_DIR / "common" / "executables" / "standalone_nddma"
ANALYSIS_SCRIPT = HARNESS_DIR / "analyze_profiling_with_params.py"
DEFAULT_ANA_DIR = SCRIPT_DIR.parent / "Ana" / "round1"
DEFAULT_OUTPUT_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_FACTOR_CSV = DEFAULT_ANA_DIR / "round1_1d_single_core_factor.csv"
ROUND_ID = "r1_1d_single_core"
GROUP_ID = "R1_1D"
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
BYTE_VALUES = (4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 49152, 57344, 61440)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and collect focused NDDMA2 Round1 1D data.")
    parser.add_argument("--factor-csv", default=str(DEFAULT_FACTOR_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--executable-dir", default=str(HARNESS_DIR))
    parser.add_argument("--msprof-bin", default=os.environ.get("MSPROF_BIN", ""))
    parser.add_argument("--analysis-pipes", default="mte2,mte3")
    parser.add_argument("--op-name-filter", default="nddma")
    parser.add_argument("--kernel-repeat", type=int, default=200)
    parser.add_argument("--execution-repeat-count", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--run-mode", default="")
    parser.add_argument("--npu-arch", default="dav-3101")
    parser.add_argument("--cmake-extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def join_dims(values: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in values)


def generate_factor(path: Path, kernel_repeat: int, execution_repeat_count: int) -> None:
    rows = []
    index = 1
    for dtype in DTYPES:
        dtype_size = DTYPE_SIZES[dtype]
        for requested_bytes in BYTE_VALUES:
            total_bytes = requested_bytes - requested_bytes % dtype_size
            output_dims = total_bytes // dtype_size
            token = f"{ROUND_ID}_{dtype}_b{total_bytes}_c1"
            rows.append({
                "experiment_idx": str(index),
                "token": token,
                "round_id": ROUND_ID,
                "sample_id": token,
                "execution_repeat_index": "",
                "execution_repeat_count": str(execution_repeat_count),
                "sensitivity_id": GROUP_ID,
                "sensitivity_key": "round1_1d_single_core",
                "sensitivity_name": "single_core_single_dimensional_contiguous",
                "stage_define": "1",
                "group_id": "A",
                "group_name": "round1_1d_single_core",
                "model_target": "cycles = alpha + bytes / T",
                "metric_target": "nddma_mte2_cycles_per_block",
                "scan_variable": "dtype,logical_total_bytes",
                "controlled_variables": (
                    "dim=1,block_dim=1,input_stride=1,output_stride=1,"
                    "GM/UB contiguous,src/dst aligned"
                ),
                "dtype": dtype,
                "dtype_size": str(dtype_size),
                "dim": "1",
                "block_dim": "1",
                "repeat": str(kernel_repeat),
                "enable_store": "0",
                "output_dims": str(output_dims),
                "output_stride": "1",
                "input_stride": "1",
                "input_stride_axis": "-1",
                "input_stride_multiplier": "1",
                "output_stride_axis": "-1",
                "output_stride_multiplier": "1",
                "src_offset_elem": "0",
                "dst_offset_elem": "0",
                "src_align_mod32": "0",
                "dst_align_mod32": "0",
                "inner_elems": str(output_dims),
                "inner_bytes": str(total_bytes),
                "total_elems": str(output_dims),
                "total_bytes": str(total_bytes),
                "bytes_per_core": str(total_bytes),
                "logical_total_bytes": str(total_bytes),
                "gm_span_elems": str(output_dims),
                "ub_span_elems": str(output_dims),
                "input_stride_pattern": "contiguous",
                "output_stride_pattern": "contiguous",
                "layout_pattern": "contiguous",
                "notes": "NDDMA2 Round1 single-core single-dimensional contiguous baseline.",
                "fit_role": "fit",
                "bytes_region": "single_core_1d",
                "shape_policy": "dim1",
                "model_family": "contiguous_baseline",
            })
            index += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote factor csv: {path} ({len(rows)} rows)")


def resolve_msprof_bin(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    parts = shlex.split(text)
    if len(parts) == 2 and parts[0] == "which":
        found = shutil.which(parts[1])
        if found:
            return found
        raise ValueError(f"unable to resolve msprof command: {text}")
    found = shutil.which(text)
    if found:
        return found
    path = Path(text).expanduser()
    if path.exists():
        return str(path.resolve())
    raise ValueError(f"msprof binary not found: {text}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_step(command: list[str], dry_run: bool) -> int:
    print("[INFO]", shlex.join(command))
    if dry_run:
        return 0
    return subprocess.run(command).returncode


def reset_output_dir(output_dir: Path) -> None:
    for name in ("profiling_raw", "experiment", "analysis"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    for name in ("command.log", "stdout.log", "stderr.log", "analysis.log"):
        (output_dir / name).unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def has_op_summary(path: Path) -> bool:
    return any(path.rglob("op_summary_*.csv"))


def collect_measurements(args: argparse.Namespace, factor_csv: Path,
                         output_dir: Path, executable: Path) -> int:
    reset_output_dir(output_dir)
    experiment_dir = output_dir / "experiment"
    raw_dir = output_dir / "profiling_raw"
    analysis_dir = output_dir / "analysis"
    app_command = [
        str(executable),
        f"--factor-csv={factor_csv}",
        f"--output-dir={experiment_dir}",
    ]
    command = (
        [args.msprof_bin, f"--output={raw_dir}", f"--application={shlex.join(app_command)}"]
        if args.msprof_bin else app_command
    )
    write_text(output_dir / "command.log", shlex.join(command) + "\n")
    status = run_step(command, args.dry_run)
    if status != 0:
        return status
    if args.dry_run:
        write_text(output_dir / "stdout.log", "dry-run\n")
        return 0
    if not args.msprof_bin:
        print("[WARN] msprof was not provided; profiling analysis was skipped.")
        return 0

    if not has_op_summary(raw_dir):
        print(f"[ERROR] no op_summary_*.csv found under {raw_dir}")
        return 1
    analysis_command = [
        sys.executable,
        str(ANALYSIS_SCRIPT),
        f"--profiling-dir={raw_dir}",
        f"--experiment-log={experiment_dir / 'experiment.log'}",
        f"--factor-csv={factor_csv}",
        f"--output-dir={analysis_dir}",
        f"--pipes={args.analysis_pipes}",
        f"--op-name-filter={args.op_name_filter}",
    ]
    print("[INFO]", shlex.join(analysis_command))
    completed = subprocess.run(analysis_command, text=True, capture_output=True)
    write_text(
        output_dir / "analysis.log",
        "COMMAND\n" + shlex.join(analysis_command)
        + "\n\nSTDOUT\n" + completed.stdout
        + "\n\nSTDERR\n" + completed.stderr,
    )
    if completed.returncode != 0:
        print(f"[ERROR] profiling analysis failed; see {output_dir / 'analysis.log'}")
    return completed.returncode


def main() -> int:
    args = parse_args()
    if args.msprof_bin:
        try:
            args.msprof_bin = resolve_msprof_bin(args.msprof_bin)
        except ValueError as error:
            print(f"[ERROR] {error}")
            return 1

    factor_csv = Path(args.factor_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    executable_dir = Path(args.executable_dir).resolve()
    generate_factor(factor_csv, args.kernel_repeat, args.execution_repeat_count)

    build_dir = output_dir / "build"
    executable = build_dir / "demo_nddma"
    configure = [
        "cmake", "-S", str(executable_dir), "-B", str(build_dir),
        "-DNDDMA_STAGE=1", f"-DNPU_ARCH={args.npu_arch}",
    ]
    if args.run_mode:
        configure.append(f"-DRUN_MODE={args.run_mode}")
    configure.extend(args.cmake_extra_arg)
    status = run_step(configure, args.dry_run)
    if status:
        return status
    status = run_step(
        ["cmake", "--build", str(build_dir), "--target", "demo_nddma", "-j", str(args.jobs)],
        args.dry_run,
    )
    if status:
        return status
    return collect_measurements(args, factor_csv, output_dir, executable)


if __name__ == "__main__":
    raise SystemExit(main())
