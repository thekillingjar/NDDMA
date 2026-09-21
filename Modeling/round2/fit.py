#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "results_analysis" / "round2_1d_single_core"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"
MODEL_FILENAME = "round2_1d_single_core_model.json"
PREDICTIONS_FILENAME = "round2_1d_single_core_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
METRIC_FIELDS = (
    "actual_y",
    "nddma_mte2_cycles_per_block",
    "mte2_cycles_per_block",
    "mte2_cycles",
    "nddma_mte2_cycles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the reduced Round2 1D single-core model.")
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
    grouped: dict[str, list[tuple[dict[str, str], float]]] = {}
    for row in rows:
        if row.get("dim") != "1" or row.get("block_dim") != "1":
            continue
        if row.get("input_stride") != "1" or row.get("output_stride") != "1":
            continue
        if row.get("input_stride_pattern") not in ("", "contiguous"):
            continue
        if row.get("output_stride_pattern") not in ("", "contiguous"):
            continue
        token = row.get("config_id") or row.get("token") or ""
        grouped.setdefault(token, []).append((row, measurement_value(row)))

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
            "bytes": float(row.get("logical_total_bytes") or row["total_bytes"]),
            "actual": actual,
        })
    return result


def fit_line(points: list[dict[str, object]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("at least two measured points are required for each dtype")
    xs = [float(point["bytes"]) for point in points]
    ys = [float(point["actual"]) for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError("measured points must contain at least two byte values")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    alpha = mean_y - slope * mean_x
    return alpha, slope


def metrics(points: list[dict[str, object]], alpha: float, t: float) -> dict[str, float | int]:
    errors = []
    absolute = []
    for point in points:
        predicted = alpha + float(point["bytes"]) * t
        error = predicted - float(point["actual"])
        errors.append(error)
        absolute.append(abs(error))
    rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    return {
        "count": len(points),
        "rmse_cycles": rmse,
        "mae_cycles": sum(absolute) / len(absolute),
        "max_absolute_error_cycles": max(absolute),
    }


def write_predictions(path: Path, points_by_dtype: dict[str, list[dict[str, object]]],
                      params: dict[str, dict[str, float]]) -> None:
    rows = []
    for dtype, points in points_by_dtype.items():
        alpha = params[dtype]["alpha"]
        cycles_per_byte = params[dtype]["cycles_per_byte"]
        for point in points:
            predicted = alpha + float(point["bytes"]) * cycles_per_byte
            rows.append({
                "token": point["token"],
                "dtype": dtype,
                "bytes": point["bytes"],
                "actual_cycles": point["actual"],
                "predicted_cycles": predicted,
                "error_cycles": predicted - float(point["actual"]),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_model(rows: list[dict[str, object]]) -> dict[str, object]:
    points_by_dtype = {dtype: [row for row in rows if row["dtype"] == dtype] for dtype in DTYPES}
    dtype_models: dict[str, dict[str, object]] = {}
    parameters: dict[str, dict[str, float]] = {}
    for dtype in DTYPES:
        points = points_by_dtype[dtype]
        alpha, cycles_per_byte = fit_line(points)
        if cycles_per_byte <= 0:
            raise ValueError(f"{dtype}: fitted cycles_per_byte must be positive, got {cycles_per_byte}")
        t = 1.0 / cycles_per_byte
        parameters[dtype] = {
            "alpha": alpha,
            "T_bytes_per_cycle": t,
            "cycles_per_byte": cycles_per_byte,
        }
        dtype_models[dtype] = {
            **parameters[dtype],
            "bytes_min": min(float(point["bytes"]) for point in points),
            "bytes_max": max(float(point["bytes"]) for point in points),
            "metrics": metrics(points, alpha, cycles_per_byte),
        }

    return {
        "model": "NDDMA_ROUND2_1D_SINGLE_CORE_PIECEWISE",
        "formula": {
            "name": "round4_inherited_piecewise",
            "split_block_dim": 2,
            "le2": "cycles = alpha(dtype) + bytes_per_core / T(dtype)",
            "gt2": None,
            "bytes_definition": "logical_total_bytes == bytes_per_core because block_dim=1",
        },
        "fit_scope": {
            "dim": 1,
            "block_dim": 1,
            "input_stride": 1,
            "output_stride": 1,
            "layout": "contiguous",
            "dtype_order": list(DTYPES),
            "sample_count": len(rows),
        },
        "dtype_models": dtype_models,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.measurement_csv:
        files = [Path(args.measurement_csv).resolve()]
    else:
        files = find_measurement_files(Path(args.data_dir).resolve())
    if not files:
        raise SystemExit(f"[ERROR] no profiling measurement csv found under {args.data_dir}")

    raw_rows = [row for path in files for row in read_rows(path)]
    rows = aggregate_rows(raw_rows)
    if not rows:
        raise SystemExit("[ERROR] no dim=1, block_dim=1, contiguous measurements found")
    model = fit_model(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_path.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    points_by_dtype = {dtype: [row for row in rows if row["dtype"] == dtype] for dtype in DTYPES}
    write_predictions(output_dir / PREDICTIONS_FILENAME, points_by_dtype, model["dtype_models"])
    print(f"[INFO] fitted {model['fit_scope']['sample_count']} points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
