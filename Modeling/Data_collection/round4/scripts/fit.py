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
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round4"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
MODEL_FILENAME = "round4_2d_ub_contiguous_ng2_model.json"
PREDICTIONS_FILENAME = "round4_2d_ub_contiguous_ng2_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
LOW_BYTE_STRIDE_MAX = 128.0
METRIC_FIELDS = (
    "actual_y", "nddma_mte2_cycles_per_block", "mte2_cycles_per_block",
    "mte2_cycles", "nddma_mte2_cycles",
)

ONE_D = {
    "int8_t": {"T_1": 11.7626, "H_1": 194.421, "T_2": 6.05735, "H_2": 373.274,
               "a_1": 2.2901165, "a_2": 0.014561351, "b_1": -69.410121,
               "b_2": -2.0861213, "b_3": 6.9111296, "b_4": -0.014567499,
               "c_1": 2.2823982, "c_2": 0.0054381207, "c_3": -0.29869463,
               "c_4": -0.0054311976},
    "int16_t": {"T_1": 25.7579, "H_1": 204.604, "T_2": 13.259, "H_2": 399.909,
                "a_1": 1.9720487, "a_2": 0.0073653238, "b_1": -65.754578,
                "b_2": -1.7626747, "b_3": 3.4565828, "b_4": -0.0073719789,
                "c_1": 1.6501023, "c_2": 0.016002982, "c_3": 0.33029872,
                "c_4": -0.015981633},
    "int32_t": {"T_1": 57.2624, "H_1": 235.137, "T_2": 29.4096, "H_2": 453.859,
                "a_1": 1.2140303, "a_2": 0.0037227686, "b_1": 31.174096,
                "b_2": -0.83730451, "b_3": 0.21970194, "b_4": -0.0032801415,
                "c_1": 1.9774169, "c_2": 0.0056400644, "c_3": -0.61376163,
                "c_4": 0.021569482},
    "int64_t": {"T_1": 57.2346, "H_1": 243.205, "T_2": 29.3906, "H_2": 468.971,
                "a_1": 1.7169033, "a_2": 0.0015857085, "b_1": 63.566231,
                "b_2": -1.4367614, "b_3": 0.090304855, "b_4": -0.0012246069,
                "c_1": 0.76071525, "c_2": 0.02630908, "c_3": 0.89945693,
                "c_4": -0.015756802},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit NDDMA2 Round4 Round5 N_G2 2D UB-contiguous model."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_ANA_DIR))
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
        if math.isfinite(value) and value > 0:
            return value
    raise ValueError(f"no positive measurement found for {row.get('token', '')}")


def parse_dims(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.replace(";", "x").split("x") if part)


def aggregate(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("dim") != "2":
            continue
        dims = parse_dims(row.get("output_dims", ""))
        input_stride = parse_dims(row.get("input_stride", ""))
        output_stride = parse_dims(row.get("output_stride", ""))
        if len(dims) != 2 or len(input_stride) != 2 or len(output_stride) != 2:
            continue
        if not (input_stride[0] < input_stride[1]):
            continue
        if output_stride != (dims[1], 1):
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


def one_d_base(dtype: str, bytes_value: float, block_dim: int) -> float:
    params = ONE_D[dtype]
    if block_dim <= 2:
        return float(params["H_1"]) + bytes_value / float(params["T_1"])
    return float(params["H_2"]) + bytes_value / float(params["T_2"])


def one_d_gm_correction(dtype: str, bytes_value: float, input_stride: int,
                        block_dim: int) -> float:
    params = ONE_D[dtype]
    s = min(float(input_stride) * DTYPE_SIZES[dtype], LOW_BYTE_STRIDE_MAX)
    ng = (float(params["a_1"]) + float(params["a_2"]) * bytes_value) * s
    if block_dim <= 2:
        return ng
    multiplier = float(params["c_1"]) + float(params["c_2"]) * s
    return multiplier * ng


def predict_ng2(params: Mapping[str, object], m: int, is2: int) -> float:
    return (
        (float(params["g10"]) + float(params["g11_M"]) * float(m)) * float(is2)
        + float(params["g00"]) + float(params["g01_M"]) * float(m)
    )


def solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    if not matrix or len(matrix) < len(matrix[0]):
        raise ValueError("not enough samples for linear fit")
    width = len(matrix[0])
    scales = [max(abs(row[col]) for row in matrix) or 1.0 for col in range(width)]
    scaled = [[value / scales[col] for col, value in enumerate(row)] for row in matrix]
    normal = [[sum(row[i] * row[j] for row in scaled) for j in range(width)]
              for i in range(width)]
    rhs = [sum(row[i] * value for row, value in zip(scaled, target))
           for i in range(width)]
    for i in range(width):
        normal[i][i] += 1e-12
    for col in range(width):
        pivot = max(range(col, width), key=lambda row: abs(normal[row][col]))
        if abs(normal[pivot][col]) < 1e-14:
            raise ValueError("singular linear fit")
        normal[col], normal[pivot] = normal[pivot], normal[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        pivot_value = normal[col][col]
        normal[col] = [value / pivot_value for value in normal[col]]
        rhs[col] /= pivot_value
        for row in range(width):
            if row == col:
                continue
            factor = normal[row][col]
            if factor == 0:
                continue
            normal[row] = [a - factor * b for a, b in zip(normal[row], normal[col])]
            rhs[row] -= factor * rhs[col]
    return [value / scale for value, scale in zip(rhs, scales)]


def prepare(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    points = []
    for row in rows:
        dtype = row["dtype"]
        m, n = parse_dims(row["output_dims"])
        is2, is1 = parse_dims(row["input_stride"])
        block_dim = int(row["block_dim"])
        dtype_size = DTYPE_SIZES[dtype]
        total_bytes = float(m * n * dtype_size)
        inner_bytes = float(n * dtype_size)
        n_base = one_d_base(dtype, total_bytes, block_dim)
        n_g1 = one_d_gm_correction(dtype, inner_bytes, is1, block_dim)
        actual = actual_value(row)
        if abs(n_g1) <= 1e-12:
            raise ValueError(f"{row.get('token', '')}: inherited N_G1 is zero")
        points.append({
            "token": row.get("config_id") or row.get("token"),
            "dtype": dtype,
            "block_dim": block_dim,
            "m": m,
            "n": n,
            "is2": is2,
            "is1": is1,
            "bytes_per_core": float(row.get("bytes_per_core") or total_bytes),
            "logical_total_bytes": float(row.get("logical_total_bytes") or total_bytes),
            "n_base_cycles": n_base,
            "n_g1_cycles": n_g1,
            "actual_cycles": actual,
            "observed_n_g2": (actual - n_base) / n_g1,
        })
    return points


def calc_metrics(points: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    if not points:
        return {
            "count": 0,
            "rmse_cycles": 0.0,
            "mae_cycles": 0.0,
            "mape_percent": 0.0,
            "max_absolute_error_cycles": 0.0,
        }
    errors = [
        float(point["predicted_cycles"]) - float(point["actual_cycles"])
        for point in points
    ]
    absolute = [abs(value) for value in errors]
    ape = [
        abs(error) / max(abs(float(point["actual_cycles"])), 1e-12)
        for error, point in zip(errors, points)
    ]
    return {
        "count": len(points),
        "rmse_cycles": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "mae_cycles": sum(absolute) / len(absolute),
        "mape_percent": 100.0 * sum(ape) / len(ape),
        "max_absolute_error_cycles": max(absolute),
    }


def fit_model(rows: list[dict[str, str]]) -> dict[str, object]:
    points = prepare(rows)
    parameters: dict[str, dict[str, float]] = {}
    for dtype in DTYPES:
        selected = [point for point in points if point["dtype"] == dtype]
        if len(selected) < 4:
            raise ValueError(f"{dtype}: need at least four Round5 N_G2 samples")
        matrix = [
            [float(point["is2"]), float(point["m"]) * float(point["is2"]),
             1.0, float(point["m"])]
            for point in selected
        ]
        target = [float(point["observed_n_g2"]) for point in selected]
        g10, g11_m, g00, g01_m = solve(matrix, target)
        parameters[dtype] = {
            "g10": g10,
            "g11_M": g11_m,
            "g00": g00,
            "g01_M": g01_m,
        }

    for point in points:
        dtype = str(point["dtype"])
        n_g2 = predict_ng2(parameters[dtype], int(point["m"]), int(point["is2"]))
        predicted = float(point["n_base_cycles"]) + float(point["n_g1_cycles"]) * n_g2
        point["predicted_n_g2"] = n_g2
        point["predicted_cycles"] = predicted
        point["error_cycles"] = predicted - float(point["actual_cycles"])

    return {
        "model": "NDDMA_ROUND4_ROUND5_NG2_2D_UB_CONTIGUOUS",
        "formula": {
            "scope": "[M,N]/[is2,is1]/[N,1], is2<is1",
            "N_base": "N_base(B,k), B=M*N*dtype_size",
            "N_G1": "N_G1=N_1'(B1,is1,1,k), B1=N*dtype_size",
            "N_G2": "N_G2=(g10+g11_M*M)*is2+g00+g01_M*M",
            "prediction": "N_2=N_base+N_G1*N_G2",
        },
        "parameters": {
            "N_G2": parameters,
            "one_dimensional": ONE_D,
        },
        "metrics": {
            "all": calc_metrics(points),
            "by_dtype": {
                dtype: calc_metrics([
                    point for point in points if point["dtype"] == dtype
                ])
                for dtype in DTYPES
            },
            "by_block_dim": {
                str(block_dim): calc_metrics([
                    point for point in points
                    if int(point["block_dim"]) == block_dim
                ])
                for block_dim in sorted({int(point["block_dim"]) for point in points})
            },
        },
        "predictions": points,
    }


def write_predictions(path: Path, points: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "token", "dtype", "block_dim", "m", "n", "is2", "is1",
        "bytes_per_core", "logical_total_bytes", "n_base_cycles", "n_g1_cycles",
        "observed_n_g2", "predicted_n_g2", "actual_cycles", "predicted_cycles",
        "error_cycles",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for point in points:
            writer.writerow({field: point.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    files = ([Path(args.measurement_csv).resolve()] if args.measurement_csv
             else find_files(Path(args.data_dir).resolve()))
    if not files:
        raise SystemExit(f"[ERROR] no profiling measurement csv found under {args.data_dir}")
    rows = aggregate(row for path in files for row in read_rows(path))
    if not rows:
        raise SystemExit("[ERROR] no Round5 N_G2 2D UB-contiguous measurements found")
    model = fit_model(rows)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_without_predictions = {
        key: value for key, value in model.items() if key != "predictions"
    }
    model_path.write_text(
        json.dumps(model_without_predictions, indent=2) + "\n",
        encoding="utf-8",
    )
    write_predictions(output_dir / PREDICTIONS_FILENAME, model["predictions"])
    print(f"[INFO] fitted {len(rows)} Round5 N_G2 points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
