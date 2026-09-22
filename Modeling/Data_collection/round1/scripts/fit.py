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
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round1"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_OUTPUT_DIR = DEFAULT_ANA_DIR
MODEL_FILENAME = "round1_1d_single_multi_core_model.json"
PREDICTIONS_FILENAME = "round1_1d_single_core_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
BLOCK_DIMS = tuple(range(1, 57))
SPLIT_BLOCK_DIM = 2
TOTAL_METRIC_FIELDS = ("nddma_mte2_cycles", "mte2_cycles")
PER_BLOCK_METRIC_FIELDS = (
    "nddma_mte2_cycles_per_block",
    "mte2_cycles_per_block",
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
    """Return total cycles; older exports can be reconstructed from per-block cycles."""
    repeat = max(1.0, float(row.get("repeat") or "1"))
    block_dim = max(1.0, float(row.get("block_dim") or "1"))
    for field in TOTAL_METRIC_FIELDS:
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        value = float(raw) / repeat
        if math.isfinite(value) and value > 0:
            return value
    for field in PER_BLOCK_METRIC_FIELDS:
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        value = float(raw) / repeat * block_dim
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


def solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    if not matrix or len(matrix) < len(matrix[0]):
        raise ValueError("not enough samples for linear fit")
    width = len(matrix[0])
    scales = [
        max(abs(row[column]) for row in matrix) or 1.0
        for column in range(width)
    ]
    scaled = [
        [value / scales[column] for column, value in enumerate(row)]
        for row in matrix
    ]
    normal = [
        [sum(row[i] * row[j] for row in scaled) for j in range(width)]
        for i in range(width)
    ]
    rhs = [
        sum(row[i] * value for row, value in zip(scaled, target))
        for i in range(width)
    ]
    for i in range(width):
        normal[i][i] += 1e-12
    for column in range(width):
        pivot = max(
            range(column, width),
            key=lambda row: abs(normal[row][column]),
        )
        if abs(normal[pivot][column]) < 1e-14:
            raise ValueError("singular linear fit")
        normal[column], normal[pivot] = normal[pivot], normal[column]
        rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        pivot_value = normal[column][column]
        normal[column] = [value / pivot_value for value in normal[column]]
        rhs[column] /= pivot_value
        for row in range(width):
            if row == column:
                continue
            factor = normal[row][column]
            normal[row] = [
                a - factor * b
                for a, b in zip(normal[row], normal[column])
            ]
            rhs[row] -= factor * rhs[column]
    return [value / scale for value, scale in zip(rhs, scales)]


def branch_for_block_dim(block_dim: int) -> str:
    return "le2" if block_dim <= SPLIT_BLOCK_DIM else "gt2"


def predict(params: dict[str, object], block_dim: int, bytes_per_core: float) -> float:
    if block_dim <= SPLIT_BLOCK_DIM:
        suffix = "1"
    else:
        suffix = "2"
    base = (
        float(params[f"h_{suffix}"])
        + bytes_per_core / float(params[f"T_{suffix}"])
    )
    return block_dim * base + float(params[f"H_{suffix}"])


def summarize_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {
            "count": 0,
            "rmse_cycles": 0.0,
            "mae_cycles": 0.0,
            "mape_percent": 0.0,
            "max_absolute_error_cycles": 0.0,
        }
    errors = [float(row["predicted"]) - float(row["actual"]) for row in rows]
    absolute = [abs(value) for value in errors]
    ape = [
        abs(error) / max(abs(float(row["actual"])), 1e-12)
        for error, row in zip(errors, rows)
    ]
    return {
        "count": len(rows),
        "rmse_cycles": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "mae_cycles": sum(absolute) / len(absolute),
        "mape_percent": 100.0 * sum(ape) / len(ape),
        "max_absolute_error_cycles": max(absolute),
    }


def fit_branch(
    points: list[dict[str, object]], branch_name: str
) -> dict[str, object]:
    matrix = []
    target = []
    for point in points:
        block_dim = float(point["block_dim"])
        bytes_per_core = float(point["bytes_per_core"])
        matrix.append([block_dim, 1.0, block_dim * bytes_per_core])
        target.append(float(point["actual"]))
    h, fixed_cycles, cycles_per_byte = solve(matrix, target)
    if cycles_per_byte <= 0:
        raise ValueError(
            f"{branch_name} fitted cycles_per_byte must be positive, "
            f"got {cycles_per_byte}"
        )
    return {
        "h": h,
        "H": fixed_cycles,
        "T": 1.0 / cycles_per_byte,
    }


def write_predictions(
    path: Path,
    rows: list[dict[str, object]],
    parameters: dict[str, dict[str, float]],
) -> None:
    output_rows = []
    for point in rows:
        dtype = str(point["dtype"])
        block_dim = int(point["block_dim"])
        branch_name = branch_for_block_dim(block_dim)
        predicted = predict(parameters[dtype], block_dim, float(point["bytes_per_core"]))
        output_rows.append({
            "token": point["token"],
            "dtype": dtype,
            "block_dim": point["block_dim"],
            "bytes_per_core": point["bytes_per_core"],
            "logical_total_bytes": point["logical_total_bytes"],
            "branch": branch_name,
            "actual_cycles": point["actual"],
            "predicted_cycles": predicted,
            "actual_cycles_per_block": float(point["actual"]) / block_dim,
            "predicted_cycles_per_block": predicted / block_dim,
            "error_cycles": predicted - float(point["actual"]),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        fieldnames = list(output_rows[0])
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def fit_model(rows: list[dict[str, object]]) -> dict[str, object]:
    parameters: dict[str, dict[str, float]] = {}
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
        parameters[dtype] = {
            "T_1": float(branches["le2"]["T"]),
            "h_1": float(branches["le2"]["h"]),
            "H_1": float(branches["le2"]["H"]),
            "T_2": float(branches["gt2"]["T"]),
            "h_2": float(branches["gt2"]["h"]),
            "H_2": float(branches["gt2"]["H"]),
        }

    metric_rows = []
    for row in rows:
        dtype = str(row["dtype"])
        block_dim = int(row["block_dim"])
        branch_name = branch_for_block_dim(block_dim)
        predicted = predict(parameters[dtype], block_dim, float(row["bytes_per_core"]))
        metric_rows.append({**row, "branch": branch_name, "predicted": predicted})

    return {
        "model": "NDDMA_ROUND1_1D_PIECEWISE_SINGLE_MULTI_CORE",
        "formula": {
            "name": "arbitrary_dim_multicore_1d_contiguous_base",
            "split_block_dim": SPLIT_BLOCK_DIM,
            "le2": "cycles = (h_1(dtype) + bytes_per_core / T_1(dtype)) * block_dim + H_1(dtype)",
            "gt2": "cycles = (h_2(dtype) + bytes_per_core / T_2(dtype)) * block_dim + H_2(dtype)",
            "normalized": "cycles_per_block = h(dtype) + bytes_per_core / T(dtype) + H(dtype) / block_dim",
            "bytes_definition": "bytes_per_core = logical_total_bytes / block_dim",
            "fitting_method": (
                "jointly fit h, H and 1/T across block_dim and bytes_per_core "
                "within each dtype/branch"
            ),
        },
        "parameters": parameters,
        "metrics": {
            "all": summarize_metrics(metric_rows),
            "by_dtype": {
                dtype: summarize_metrics([
                    row for row in metric_rows if row["dtype"] == dtype
                ])
                for dtype in DTYPES
            },
            "by_branch": {
                branch_name: summarize_metrics([
                    row for row in metric_rows if row["branch"] == branch_name
                ])
                for branch_name in ("le2", "gt2")
            },
        },
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
