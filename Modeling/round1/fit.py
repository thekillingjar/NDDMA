#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANA_DIR = SCRIPT_DIR.parent / "Ana" / "round1"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_OUTPUT_DIR = DEFAULT_ANA_DIR
MODEL_FILENAME = "round1_1d_single_multi_core_model.json"
PREDICTIONS_FILENAME = "round1_1d_single_core_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
BLOCK_DIMS = tuple(range(1, 57))
SPLIT_BLOCK_DIM = 2
METRIC_FIELDS = (
    "actual_y",
    "nddma_mte2_cycles_per_block",
    "mte2_cycles_per_block",
    "mte2_cycles",
    "nddma_mte2_cycles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the NDDMA2 Round1 1D single/multi-core contiguous model."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def measurement_value(row: dict[str, str]) -> float:
    for field in METRIC_FIELDS:
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        value = float(raw)
        repeat = float(row.get("repeat") or "1")
        if field != "actual_y":
            value /= max(1.0, repeat)
        if math.isfinite(value) and value > 0:
            return value
    raise ValueError(f"no positive measurement found for token={row.get('token', '')}")


def find_measurement_files(data_dir: Path) -> list[Path]:
    direct = data_dir / "measurements.csv"
    if direct.exists():
        return [direct]
    files = sorted(data_dir.rglob("profiling_with_params_mean.csv"))
    if not files:
        files = sorted(data_dir.rglob("profiling_with_params.csv"))
    return files


def aggregate_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    valid_block_dims = {str(value) for value in BLOCK_DIMS}
    for row in rows:
        if row.get("dim") != "1":
            continue
        if row.get("block_dim") not in valid_block_dims:
            continue
        if row.get("input_stride") != "1" or row.get("output_stride") != "1":
            continue
        if row.get("input_stride_pattern") not in ("", "contiguous"):
            continue
        if row.get("output_stride_pattern") not in ("", "contiguous"):
            continue
        token = row.get("config_id") or row.get("token") or ""
        grouped[token].append((row, measurement_value(row)))

    result: list[dict[str, object]] = []
    for values in grouped.values():
        row = values[0][0]
        measurements = sorted(value for _, value in values)
        middle = len(measurements) // 2
        actual = measurements[middle] if len(measurements) % 2 else (
            measurements[middle - 1] + measurements[middle]
        ) / 2.0
        result.append({
            "token": row.get("token", ""),
            "dtype": row["dtype"],
            "block_dim": int(row["block_dim"]),
            "bytes_per_core": float(row.get("bytes_per_core") or row["total_bytes"]),
            "logical_total_bytes": float(
                row.get("logical_total_bytes") or row["total_bytes"]
            ),
            "actual": actual,
        })
    return result


def fit_line(points: list[dict[str, object]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("at least two measured points are required for each dtype/block_dim")
    xs = [float(point["bytes_per_core"]) for point in points]
    ys = [float(point["actual"]) for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError("measured points must contain at least two byte values")
    cycles_per_byte = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator
    alpha = mean_y - cycles_per_byte * mean_x
    if cycles_per_byte <= 0:
        raise ValueError(
            f"fitted cycles_per_byte must be positive, got {cycles_per_byte}"
        )
    return alpha, cycles_per_byte


def branch_for_block_dim(block_dim: int) -> str:
    return "le2" if block_dim <= SPLIT_BLOCK_DIM else "gt2"


def predict(branch: dict[str, object], bytes_per_core: float) -> float:
    return float(branch["alpha"]) + bytes_per_core / float(branch["T_bytes_per_cycle"])


def calculate_metrics(
    points: list[dict[str, object]], branch: dict[str, object]
) -> dict[str, float | int]:
    errors = [
        predict(branch, float(point["bytes_per_core"])) - float(point["actual"])
        for point in points
    ]
    absolute = [abs(value) for value in errors]
    return {
        "count": len(points),
        "rmse_cycles": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "mae_cycles": sum(absolute) / len(absolute),
        "max_absolute_error_cycles": max(absolute),
    }


def fit_branch(
    points: list[dict[str, object]], branch_name: str
) -> dict[str, object]:
    block_fits = []
    for block_dim in sorted({int(point["block_dim"]) for point in points}):
        block_points = [
            point for point in points if int(point["block_dim"]) == block_dim
        ]
        alpha, cycles_per_byte = fit_line(block_points)
        t_bytes_per_cycle = 1.0 / cycles_per_byte
        block_fits.append({
            "block_dim": block_dim,
            "alpha": alpha,
            "T_bytes_per_cycle": t_bytes_per_cycle,
            "cycles_per_byte": cycles_per_byte,
            "sample_count": len(block_points),
        })

    alpha = sum(float(row["alpha"]) for row in block_fits) / len(block_fits)
    t_bytes_per_cycle = sum(
        float(row["T_bytes_per_cycle"]) for row in block_fits
    ) / len(block_fits)
    branch = {
        "alpha": alpha,
        "T_bytes_per_cycle": t_bytes_per_cycle,
        "cycles_per_byte": 1.0 / t_bytes_per_cycle,
    }
    return branch


def write_predictions(
    path: Path,
    rows: list[dict[str, object]],
    parameters: dict[str, dict[str, object]],
) -> None:
    output_rows = []
    for point in rows:
        dtype = str(point["dtype"])
        branch_name = branch_for_block_dim(int(point["block_dim"]))
        branch = parameters[dtype][branch_name]
        predicted = predict(branch, float(point["bytes_per_core"]))
        output_rows.append({
            "token": point["token"],
            "dtype": dtype,
            "block_dim": point["block_dim"],
            "bytes_per_core": point["bytes_per_core"],
            "logical_total_bytes": point["logical_total_bytes"],
            "branch": branch_name,
            "actual_cycles": point["actual"],
            "predicted_cycles": predicted,
            "error_cycles": predicted - float(point["actual"]),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        fieldnames = list(output_rows[0])
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def fit_model(rows: list[dict[str, object]]) -> dict[str, object]:
    parameters: dict[str, dict[str, object]] = {}
    for dtype in DTYPES:
        dtype_rows = [row for row in rows if row["dtype"] == dtype]
        if not dtype_rows:
            raise ValueError(f"no fitting rows found for {dtype}")
        branch_points = {
            branch_name: [
                row for row in dtype_rows
                if branch_for_block_dim(int(row["block_dim"])) == branch_name
            ]
            for branch_name in ("le2", "gt2")
        }
        branches = {
            branch_name: fit_branch(points, branch_name)
            for branch_name, points in branch_points.items()
        }
        parameters[dtype] = branches

    return {
        "model": "NDDMA_ROUND1_1D_PIECEWISE_SINGLE_MULTI_CORE",
        "formula": {
            "name": "round2_c_group_piecewise_constant_t_alpha",
            "split_block_dim": SPLIT_BLOCK_DIM,
            "le2": "cycles = alpha_le2(dtype) + bytes_per_core / T_le2(dtype)",
            "gt2": "cycles = alpha_gt2(dtype) + bytes_per_core / T_gt2(dtype)",
            "bytes_definition": "bytes_per_core = logical_total_bytes / block_dim",
            "fitting_method": (
                "fit alpha and T independently for each dtype/block_dim, "
                "then average alpha and T within each block_dim branch"
            ),
        },
        "parameters": parameters,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    files = (
        [Path(args.measurement_csv).resolve()]
        if args.measurement_csv
        else find_measurement_files(Path(args.data_dir).resolve())
    )
    if not files:
        raise SystemExit(f"[ERROR] no profiling measurement csv found under {args.data_dir}")

    raw_rows = [row for path in files for row in read_rows(path)]
    rows = aggregate_rows(raw_rows)
    if not rows:
        raise SystemExit(
            "[ERROR] no 1D contiguous measurements with block_dim in 1..56 found"
        )
    model = fit_model(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_path.write_text(
        json.dumps(model, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_predictions(
        output_dir / PREDICTIONS_FILENAME,
        rows,
        model["parameters"],
    )
    print(f"[INFO] fitted {len(rows)} points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
