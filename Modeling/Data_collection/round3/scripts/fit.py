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
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round3"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
MODEL_FILENAME = "round3_multidim_model.json"
PREDICTIONS_FILENAME = "round3_multidim_predictions.csv"
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
METRIC_FIELDS = (
    "actual_y", "nddma_mte2_cycles_per_block", "mte2_cycles_per_block",
    "mte2_cycles", "nddma_mte2_cycles",
)
PARAMETER_SOURCE = "NDDMA/Modeling/DOC/任意维度多核模型.md section 6"
UNIFIED_PARAMETERS = {
    "int8_t": {
        "T_1": 11.7626, "H_1": 194.421, "T_2": 6.05735, "H_2": 373.274,
        "a_1": 2.2901165, "a_2": 0.014561351,
        "b_1": -69.410121, "b_2": -2.0861213,
        "b_3": 6.9111296, "b_4": -0.014567499,
        "c_1": 2.2823982, "c_2": 0.0054381207,
        "c_3": -0.29869463, "c_4": -0.0054311976,
    },
    "int16_t": {
        "T_1": 25.7579, "H_1": 204.604, "T_2": 13.259, "H_2": 399.909,
        "a_1": 1.9720487, "a_2": 0.0073653238,
        "b_1": -65.754578, "b_2": -1.7626747,
        "b_3": 3.4565828, "b_4": -0.0073719789,
        "c_1": 1.6501023, "c_2": 0.016002982,
        "c_3": 0.33029872, "c_4": -0.015981633,
    },
    "int32_t": {
        "T_1": 57.2624, "H_1": 235.137, "T_2": 29.4096, "H_2": 453.859,
        "a_1": 1.2140303, "a_2": 0.0037227686,
        "b_1": 31.174096, "b_2": -0.83730451,
        "b_3": 0.21970194, "b_4": -0.0032801415,
        "c_1": 1.9774169, "c_2": 0.0056400644,
        "c_3": -0.61376163, "c_4": 0.021569482,
    },
    "int64_t": {
        "T_1": 57.2346, "H_1": 243.205, "T_2": 29.3906, "H_2": 468.971,
        "a_1": 1.7169033, "a_2": 0.0015857085,
        "b_1": 63.566231, "b_2": -1.4367614,
        "b_3": 0.090304855, "b_4": -0.0012246069,
        "c_1": 0.76071525, "c_2": 0.02630908,
        "c_3": 0.89945693, "c_4": -0.015756802,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit NDDMA2 Round3 2D/3D/4D/5D extension model."
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


def one_d_base(dtype: str, bytes_value: float, block_dim: int) -> float:
    params = UNIFIED_PARAMETERS[dtype]
    if block_dim <= 2:
        return bytes_value / params["T_1"] + params["H_1"]
    return bytes_value / params["T_2"] + params["H_2"]


def one_d_correction(dtype: str, bytes_value: float, input_stride: int,
                     output_stride: int, block_dim: int) -> float:
    params = UNIFIED_PARAMETERS[dtype]
    s = min(float(input_stride) * DTYPE_SIZES[dtype], 128.0)
    gate = min(1.0, max(0.0, float(output_stride - 1)))
    ng = (params["a_1"] + params["a_2"] * bytes_value) * s
    ngu = (
        (params["b_1"] + params["b_2"] * s)
        + (params["b_3"] + params["b_4"] * s) * bytes_value
    ) * gate
    if block_dim <= 2:
        return ng + ngu
    multiplier = (
        params["c_1"] + params["c_2"] * s
        + gate * (params["c_3"] + params["c_4"] * s)
    )
    return multiplier * (ng + ngu)


def unified_terms(row: Mapping[str, str]) -> tuple[float, list[dict[str, float | int]]]:
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
    n_base = one_d_base(dtype, total_bytes, block_dim)
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
            dtype, term_bytes, input_delta, output_delta, block_dim)
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


def fit_model(rows: list[dict[str, str]]) -> dict[str, object]:
    points: list[dict[str, object]] = []
    for row in rows:
        n_base, terms = unified_terms(row)
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
        "model": "NDDMA2_ROUND3_ARBITRARY_DIMENSION_MULTICORE_UNIFIED",
        "formula": {
            "source": "NDDMA arbitrary-dimensional multicore unified formula",
            "n_base": "N_base=B/T_1+H_1 for k<=2; B/T_2+H_2 for k>2",
            "axis_bytes": "B_j = B / prod_{t=0}^{j-1}(ls_t)",
            "effective_input_stride": "is_hat_j = abs(is_j - sum_{t=0}^{j-1}(ls_t*is_t)) + 1, j>=1; is_hat_0=is_0",
            "effective_output_stride": "os_hat_j = abs(os_j - sum_{t=0}^{j-1}(ls_t*os_t)) + 1, j>=1; os_hat_0=os_0",
            "per_axis": "N_1'=N_G+N_GU for k<=2; N_1'=(N_G+N_GU)*rho for k>2",
            "N_G": "N_G=(a_1+a_2*B_j)*s",
            "N_GU": "N_GU=((b_1+b_2*s)+(b_3+b_4*s)*B_j)*min(1,os_hat_j-1)",
            "rho": "rho=(c_1+c_2*s)+min(1,os_hat_j-1)*(c_3+c_4*s)",
            "prediction": "N_D = N_base + sum_j(N_1'(B_j,is_hat_j,os_hat_j,k))",
            "round3_fitted_parameters": "none; parameters are fixed from the document",
            "dimensions": [2, 3, 4, 5],
        },
        "parameter_source": PARAMETER_SOURCE,
        "parameters": UNIFIED_PARAMETERS,
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
    files = ([Path(args.measurement_csv).resolve()] if args.measurement_csv
             else find_files(Path(args.data_dir).resolve()))
    if not files:
        raise SystemExit(f"[ERROR] no profiling measurement csv found under {args.data_dir}")
    rows = aggregate(row for path in files for row in read_rows(path))
    if not rows:
        raise SystemExit("[ERROR] no 2D/3D/4D/5D measurements found")
    model = fit_model(rows)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_without_predictions = {
        key: value for key, value in model.items() if key != "predictions"
    }
    model_path.write_text(json.dumps(model_without_predictions, indent=2) + "\n", encoding="utf-8")
    write_predictions(output_dir / PREDICTIONS_FILENAME, model["predictions"])
    print(f"[INFO] fitted {len(rows)} multidimensional points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
