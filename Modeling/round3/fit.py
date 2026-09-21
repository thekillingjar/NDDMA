#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANA_DIR = SCRIPT_DIR.parent / "Ana" / "round3"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_ROUND2_MODEL = SCRIPT_DIR.parent / "Ana" / "round2" / "round2_1d_noncontiguous_model.json"
MODEL_FILENAME = "round3_multidim_model.json"
PREDICTIONS_FILENAME = "round3_multidim_predictions.csv"
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
METRIC_FIELDS = (
    "actual_y", "nddma_mte2_cycles_per_block", "mte2_cycles_per_block",
    "mte2_cycles", "nddma_mte2_cycles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit NDDMA2 Round3 2D/3D/4D/5D extension model."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_ANA_DIR))
    parser.add_argument("--round2-model", default=str(DEFAULT_ROUND2_MODEL))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def find_files(data_dir: Path) -> list[Path]:
    direct = data_dir / "measurements.csv"
    if direct.exists():
        return [direct]
    files = sorted(data_dir.rglob("profiling_with_params_mean.csv"))
    return files or sorted(data_dir.rglob("profiling_with_params.csv"))


def actual_value(row: Mapping[str, str]) -> float:
    for field in METRIC_FIELDS:
        text = str(row.get(field, "")).strip()
        if not text or text.upper() == "N/A":
            continue
        value = float(text)
        if field != "actual_y":
            value /= max(1.0, float(row.get("repeat") or 1.0))
        if math.isfinite(value) and value > 0.0:
            return value
    raise ValueError(f"no positive measurement found for {row.get('token', '')}")


def aggregate(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("dim", "")).strip() not in {"2", "3", "4", "5"}:
            continue
        token = row.get("config_id") or row.get("token") or ""
        grouped[token].append(row)
    result = []
    for token, values in grouped.items():
        row = dict(values[0])
        row["config_id"] = token
        row["actual_y"] = str(median(actual_value(value) for value in values))
        result.append(row)
    return result


def parse_tuple(value: object, expected_len: int) -> tuple[int, ...]:
    parts = str(value).replace(";", "x").split("x")
    if len(parts) != expected_len:
        raise ValueError(f"expected {expected_len} values, got {value!r}")
    return tuple(int(part) for part in parts)


def product_int(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def base_params(round2: Mapping[str, object], dtype: str, block_dim: int) -> Mapping[str, object]:
    base = round2["dtype_models"][dtype]["base"]
    return base["le2" if block_dim <= 2 else "gt2"]


def one_d_base(round2: Mapping[str, object], dtype: str, bytes_value: float,
               block_dim: int) -> float:
    params = base_params(round2, dtype, block_dim)
    return float(params["alpha"]) + bytes_value / float(params["T_bytes_per_cycle"])


def one_d_correction(round2: Mapping[str, object], dtype: str, bytes_value: float,
                     input_stride: int, output_stride: int, block_dim: int) -> float:
    if input_stride <= 1 and output_stride <= 1:
        return 0.0
    model = round2["dtype_models"][dtype]
    s = min(float(input_stride) * DTYPE_SIZES[dtype], 128.0)
    gate = min(1.0, max(0.0, float(output_stride - 1)))
    ng = (float(model["N_G"]["a1"]) + float(model["N_G"]["a2"]) * bytes_value) * s
    ngu = ((float(model["N_GU"]["b1"]) + float(model["N_GU"]["b2"]) * s)
           + (float(model["N_GU"]["b3"]) + float(model["N_GU"]["b4"]) * s)
           * bytes_value) * gate
    if block_dim <= 2:
        return ng + ngu
    rho = model["rho"]
    multiplier = (
        float(rho["c1"]) + float(rho["c2"]) * s
        + gate * (float(rho["c3"]) + float(rho["c4"]) * s)
    )
    return multiplier * (ng + ngu)


def inherited_terms(round2: Mapping[str, object], row: Mapping[str, str]) -> tuple[float, list[dict[str, float | int]]]:
    dim = int(row["dim"])
    dtype = row["dtype"]
    block_dim = int(row.get("block_dim") or 1)
    dims = parse_tuple(row["output_dims"], dim)
    input_stride_api = parse_tuple(row["input_stride"], dim)
    output_stride_api = parse_tuple(row["output_stride"], dim)
    loop_sizes = tuple(reversed(dims))
    input_stride = tuple(reversed(input_stride_api))
    output_stride = tuple(reversed(output_stride_api))
    total_elems = product_int(loop_sizes)
    total_bytes = total_elems * DTYPE_SIZES[dtype]
    n_base = one_d_base(round2, dtype, total_bytes, block_dim)
    terms: list[dict[str, float | int]] = []
    inner_product = 1
    for axis, loop_size in enumerate(loop_sizes):
        term_elems = total_elems // inner_product
        if axis == 0:
            input_delta = int(input_stride[0])
            output_delta = int(output_stride[0])
        else:
            expected_input = sum(
                int(loop_sizes[lower]) * int(input_stride[lower])
                for lower in range(axis)
            )
            expected_output = sum(
                int(loop_sizes[lower]) * int(output_stride[lower])
                for lower in range(axis)
            )
            input_delta = abs(int(input_stride[axis]) - expected_input) + 1
            output_delta = abs(int(output_stride[axis]) - expected_output) + 1
        term_bytes = term_elems * DTYPE_SIZES[dtype]
        correction = one_d_correction(
            round2, dtype, term_bytes, input_delta, output_delta, block_dim
        )
        terms.append({
            "axis": axis,
            "loop_size": int(loop_size),
            "data_volume_bytes": int(term_bytes),
            "input_stride_delta": int(input_delta),
            "output_stride_delta": int(output_delta),
            "t1_cycles": float(correction),
        })
        inner_product *= int(loop_size)
    return n_base, terms


def metrics(points: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    errors = [float(point["predicted_cycles"]) - float(point["actual_cycles"])
              for point in points]
    absolute = [abs(error) for error in errors]
    ape = [
        abs(error) / max(abs(float(point["actual_cycles"])), 1e-12)
        for error, point in zip(errors, points)
    ]
    return {
        "count": len(points),
        "rmse_cycles": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "mae_cycles": sum(absolute) / len(absolute),
        "mape_percent": 100.0 * sum(ape) / len(ape),
        "max_ape_percent": 100.0 * max(ape),
    }


def fit_model(rows: list[dict[str, str]], round2: Mapping[str, object]) -> dict[str, object]:
    points: list[dict[str, object]] = []
    for row in rows:
        n_base, terms = inherited_terms(round2, row)
        t1_sum = sum(float(term["t1_cycles"]) for term in terms)
        actual = actual_value(row)
        points.append({
            "token": row.get("config_id") or row.get("token"),
            "dtype": row["dtype"],
            "dim": int(row["dim"]),
            "layout": row.get("layout_pattern") or row.get("sensitivity_key") or "",
            "source_round": row.get("source_round", ""),
            "output_dims": row["output_dims"],
            "input_stride": row["input_stride"],
            "output_stride": row["output_stride"],
            "data_volume_bytes": float(row.get("logical_total_bytes") or row["total_bytes"]),
            "actual_cycles": actual,
            "n_base_cycles": n_base,
            "t1_cycles": t1_sum,
            "n1_terms_json": json.dumps(terms, separators=(",", ":")),
        })

    for point in points:
        predicted = float(point["n_base_cycles"]) + float(point["t1_cycles"])
        point["predicted_cycles"] = predicted
        point["error_cycles"] = predicted - float(point["actual_cycles"])

    model_metrics = {
        dtype: {
            str(dim): metrics([
                point for point in points
                if point["dtype"] == dtype and int(point["dim"]) == dim
            ])
            for dim in sorted({int(point["dim"]) for point in points if point["dtype"] == dtype})
        }
        for dtype in sorted({str(point["dtype"]) for point in points})
    }
    return {
        "model": "NDDMA_ROUND3_MULTIDIM_EXTENSION_2D_3D_4D_5D",
        "formula": {
            "source": "inherits round2 1D non-contiguous model",
            "n_base": "N_base = round2.base(dtype,total_bytes,block_dim)",
            "per_axis": "T_axis = round2.N_1_correction(dtype,axis_bytes,input_delta,output_delta)",
            "prediction": "cycles = N_base + sum(T_axis)",
            "round3_fitted_parameters": "none",
            "dimensions": [2, 3, 4, 5],
        },
        "round2_model_source": str(DEFAULT_ROUND2_MODEL),
        "fit_scope": {
            "sample_count": len(points),
            "dimensions": sorted({int(point["dim"]) for point in points}),
            "source_rounds": sorted({str(point["source_round"]) for point in points}),
        },
        "metrics": model_metrics,
        "predictions": points,
    }


def write_predictions(path: Path, points: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "token", "source_round", "dtype", "dim", "layout", "output_dims",
        "input_stride", "output_stride", "data_volume_bytes",
        "actual_cycles", "n_base_cycles", "t1_cycles",
        "predicted_cycles", "error_cycles",
        "n1_terms_json",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for point in points:
            writer.writerow({field: point.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    round2_model_path = Path(args.round2_model).resolve()
    if not round2_model_path.exists():
        raise SystemExit(f"[ERROR] round2 model JSON not found: {round2_model_path}")
    round2 = json.loads(round2_model_path.read_text(encoding="utf-8"))
    files = ([Path(args.measurement_csv).resolve()] if args.measurement_csv
             else find_files(Path(args.data_dir).resolve()))
    if not files:
        raise SystemExit(f"[ERROR] no profiling measurement csv found under {args.data_dir}")
    rows = aggregate(row for path in files for row in read_rows(path))
    if not rows:
        raise SystemExit("[ERROR] no 2D/3D/4D/5D measurements found")
    model = fit_model(rows, round2)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_without_predictions = {
        key: value for key, value in model.items() if key != "predictions"
    }
    model_without_predictions["round2_model_source"] = str(round2_model_path)
    model_path.write_text(json.dumps(model_without_predictions, indent=2) + "\n", encoding="utf-8")
    write_predictions(output_dir / PREDICTIONS_FILENAME, model["predictions"])
    print(f"[INFO] fitted {len(rows)} multidimensional points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
