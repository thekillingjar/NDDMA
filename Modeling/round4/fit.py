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
DEFAULT_ANA_DIR = SCRIPT_DIR.parent / "Ana" / "round4"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
MODEL_FILENAME = "round4_2d_transpose_multicore_model.json"
PREDICTIONS_FILENAME = "round4_2d_transpose_multicore_predictions.csv"
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

TWO_D_RESIDUAL = {
    "int8_t": {
        "byte_stride_lt128": {"a0": 0.09008404383, "a1": -0.01026269497, "b": -226.0427898, "d": -55.20923231},
        "byte_stride_ge128": {"a0": -1.2895225, "a1": -0.00005673311835, "b": -326.977851, "d": 1680.30515},
    },
    "int16_t": {
        "byte_stride_lt128": {"a0": 0.1147179878, "a1": -0.0111467759, "b": -204.2465443, "d": -9.679316828},
        "byte_stride_ge128": {"a0": -1.355039786, "a1": -0.00004883132643, "b": -309.400175, "d": 1362.591254},
    },
    "int32_t": {
        "byte_stride_lt128": {"a0": 0.09418001368, "a1": -0.01088005879, "b": -207.4124997, "d": 93.04782625},
        "byte_stride_ge128": {"a0": -1.55680738, "a1": -0.00001164850262, "b": -294.0304232, "d": 1399.193913},
    },
    "int64_t": {
        "byte_stride_lt128": {"a0": 0.1959931613, "a1": -0.01092369082, "b": -229.3678605, "d": 173.7016138},
        "byte_stride_ge128": {"a0": -1.737825597, "a1": 0.000004718964916, "b": -255.3441558, "d": 700.5878799},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit NDDMA2 Round4 standalone 2D transpose multicore model."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_ANA_DIR))
    parser.add_argument("--fit-byte-stride-max", type=float, default=LOW_BYTE_STRIDE_MAX)
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


def aggregate(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("dim") == "2" and int(row.get("block_dim") or 0) > 1:
            grouped[row.get("config_id") or row.get("token") or ""].append(row)
    result = []
    for token, values in grouped.items():
        row = dict(values[0])
        row["config_id"] = token
        row["actual_y"] = str(median(actual_value(value) for value in values))
        result.append(row)
    return result


def parse_dims(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.replace(";", "x").split("x") if part)


def one_d_multicore(dtype: str, n: int, is1: int, block_dim: int) -> tuple[float, float, float]:
    params = ONE_D[dtype]
    size = DTYPE_SIZES[dtype]
    b = float(n * size)
    s = min(float(is1 * size), LOW_BYTE_STRIDE_MAX)
    if block_dim <= 2:
        base = float(params["H_1"]) + b / float(params["T_1"])
    else:
        base = float(params["H_2"]) + b / float(params["T_2"])
    n_g = (float(params["a_1"]) + float(params["a_2"]) * b) * s
    correction = n_g
    if block_dim > 2:
        correction *= float(params["c_1"]) + float(params["c_2"]) * s
    return base + correction, base, correction


def single_core_residual(dtype: str, m: int, n: int, is1: int) -> tuple[float, str, float]:
    raw_s = float(is1 * DTYPE_SIZES[dtype])
    region = "byte_stride_lt128" if raw_s < LOW_BYTE_STRIDE_MAX else "byte_stride_ge128"
    params = TWO_D_RESIDUAL[dtype][region]
    residual = (
        (float(params["a0"]) + float(params["a1"]) * raw_s) * float(n * (m - 1))
        + float(params["b"]) * float(m)
        + float(params["d"])
    )
    return residual, region, raw_s


def prepare(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    points = []
    for row in rows:
        dtype = row["dtype"]
        m, n = parse_dims(row["output_dims"])
        is1 = parse_dims(row["input_stride"])[1]
        block_dim = int(row["block_dim"])
        actual = actual_value(row)
        t1d, t1d_base, t1d_correction = one_d_multicore(dtype, n, is1, block_dim)
        residual, region, raw_s = single_core_residual(dtype, m, n, is1)
        baseline = float(m) * t1d
        points.append({
            "token": row.get("config_id") or row.get("token"),
            "dtype": dtype,
            "block_dim": block_dim,
            "m": m,
            "n": n,
            "is1": is1,
            "s": min(raw_s, LOW_BYTE_STRIDE_MAX),
            "raw_byte_stride": raw_s,
            "region": region,
            "bytes_per_core": float(row["bytes_per_core"]),
            "logical_total_bytes": float(row["logical_total_bytes"]),
            "t1d_multicore_cycles": t1d,
            "t1d_base_cycles": t1d_base,
            "t1d_correction_cycles": t1d_correction,
            "baseline_cycles": baseline,
            "single_core_residual_cycles": residual,
            "actual_cycles": actual,
        })
    return points


def fit_line(xs: list[float], ys: list[float], weights: list[float]) -> tuple[float, float]:
    wx = [x * w for x, w in zip(xs, weights)]
    wy = [y * w for y, w in zip(ys, weights)]
    ww = [w * w for w in weights]
    s00 = sum(ww)
    s01 = sum(x * w for x, w in zip(wx, weights))
    s11 = sum(x * x for x in wx)
    t0 = sum(y * w for y, w in zip(wy, weights))
    t1 = sum(x * y for x, y in zip(wx, wy))
    det = s00 * s11 - s01 * s01
    if abs(det) < 1e-12:
        raise ValueError("rank deficient rho fit")
    return (t0 * s11 - t1 * s01) / det, (s00 * t1 - s01 * t0) / det


def calc_metrics(points: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    errors = [float(p["predicted_cycles"]) - float(p["actual_cycles"]) for p in points]
    ape = [abs(e) / max(abs(float(p["actual_cycles"])), 1e-12) for e, p in zip(errors, points)]
    return {
        "count": len(points),
        "rmse_cycles": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "mae_cycles": sum(abs(e) for e in errors) / len(errors),
        "mape_percent": 100.0 * sum(ape) / len(ape),
        "max_ape_percent": 100.0 * max(ape),
    }


def fit_model(rows: list[dict[str, str]], fit_byte_stride_max: float) -> dict[str, object]:
    points = prepare(rows)
    models: dict[str, dict[str, float | int | str]] = {}
    for dtype in DTYPES:
        selected = [
            p for p in points
            if p["dtype"] == dtype
            and float(p["raw_byte_stride"]) <= fit_byte_stride_max
            and abs(float(p["single_core_residual_cycles"])) > 1e-12
        ]
        if len(selected) < 2:
            raise ValueError(f"{dtype}: need at least two identifiable rho samples")
        xs = [float(p["s"]) for p in selected]
        ys = [
            (float(p["actual_cycles"]) - float(p["baseline_cycles"]))
            / float(p["single_core_residual_cycles"])
            for p in selected
        ]
        weights = [abs(float(p["single_core_residual_cycles"])) for p in selected]
        c1, c2 = fit_line(xs, ys, weights)
        models[dtype] = {
            "formula": "rho_2d=c1+c2*s; c3=c4=0 because output_stride=[N,1]",
            "c1": c1,
            "c2": c2,
            "c3": 0.0,
            "c4": 0.0,
            "fit_sample_count": len(selected),
            "fit_byte_stride_max": fit_byte_stride_max,
        }

    for point in points:
        dtype = str(point["dtype"])
        model = models[dtype]
        rho = float(model["c1"]) + float(model["c2"]) * float(point["s"])
        predicted_no_rho = (
            float(point["baseline_cycles"])
            + float(point["single_core_residual_cycles"])
        )
        predicted = (
            float(point["baseline_cycles"])
            + rho * float(point["single_core_residual_cycles"])
        )
        point["rho_observed"] = (
            (float(point["actual_cycles"]) - float(point["baseline_cycles"]))
            / float(point["single_core_residual_cycles"])
        )
        point["rho_predicted"] = rho
        point["predicted_no_rho"] = predicted_no_rho
        point["predicted_cycles"] = predicted
        point["error_cycles"] = predicted - float(point["actual_cycles"])

    return {
        "model": "NDDMA_ROUND4_2D_TRANSPOSE_MULTICORE_RHO",
        "formula": {
            "target": "T_2d = M*T_1d_multicore + rho_2d*r_singlecore",
            "rho_2d": "rho_2d=(c1+c2*s)+min(1,os-1)*(c3+c4*s)",
            "current_data": "output_stride=[N,1], so os=1 and c3/c4 are fixed to 0",
            "s": "min(is1*dtype_size,128)",
            "single_core_residual": "r=(a0+a1*S)*N*(M-1)+b*M+d",
        },
        "fit_scope": {
            "dim": 2,
            "layout": "[M,N]/[1,is1]/[N,1]",
            "block_dims": sorted({int(p["block_dim"]) for p in points}),
            "sample_count": len(points),
            "fit_byte_stride_max": fit_byte_stride_max,
        },
        "inherited_parameters": {
            "one_d_multicore_source": "NDDMA DOC 任意维度多核模型",
            "single_core_2d_residual_source": "NDDMA DOC 二维转置residual模型",
            "one_d_multicore": ONE_D,
            "single_core_2d_residual": TWO_D_RESIDUAL,
        },
        "dtype_models": models,
        "metrics": {
            "all": calc_metrics(points),
            "by_dtype": {
                dtype: calc_metrics([p for p in points if p["dtype"] == dtype])
                for dtype in DTYPES
            },
            "by_block_dim": {
                str(k): calc_metrics([p for p in points if int(p["block_dim"]) == k])
                for k in sorted({int(p["block_dim"]) for p in points})
            },
        },
        "predictions": points,
    }


def write_predictions(path: Path, points: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "token", "dtype", "block_dim", "m", "n", "is1", "s",
        "raw_byte_stride", "region", "bytes_per_core", "logical_total_bytes",
        "t1d_multicore_cycles", "baseline_cycles",
        "single_core_residual_cycles", "rho_observed", "rho_predicted",
        "actual_cycles", "predicted_no_rho", "predicted_cycles", "error_cycles",
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
        raise SystemExit("[ERROR] no 2D transpose multicore measurements found")
    model = fit_model(rows, args.fit_byte_stride_max)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_without_predictions = {
        key: value for key, value in model.items() if key != "predictions"
    }
    model_path.write_text(json.dumps(model_without_predictions, indent=2) + "\n", encoding="utf-8")
    write_predictions(output_dir / PREDICTIONS_FILENAME, model["predictions"])
    print(f"[INFO] fitted {len(rows)} 2D transpose multicore points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
