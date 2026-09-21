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
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round2"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
MODEL_FILENAME = "round2_1d_noncontiguous_model.json"
PREDICTIONS_FILENAME = "round2_1d_noncontiguous_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
METRIC_FIELDS = (
    "actual_y", "nddma_mte2_cycles_per_block", "mte2_cycles_per_block",
    "mte2_cycles", "nddma_mte2_cycles",
)
LOWER_BRANCH = "le2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the NDDMA2 Round2 1D non-contiguous staged model."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_ANA_DIR))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def actual_value(row: dict[str, str]) -> float:
    for field in METRIC_FIELDS:
        text = str(row.get(field, "")).strip()
        if not text:
            continue
        value = float(text)
        if field != "actual_y":
            value /= max(1.0, float(row.get("repeat") or 1.0))
        if math.isfinite(value) and value > 0:
            return value
    raise ValueError(f"no positive measurement found for {row.get('token', '')}")


def find_files(data_dir: Path) -> list[Path]:
    direct = data_dir / "measurements.csv"
    if direct.exists():
        return [direct]
    files = sorted(data_dir.rglob("profiling_with_params_mean.csv"))
    return files or sorted(data_dir.rglob("profiling_with_params.csv"))


def aggregate(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for row in rows:
        if row.get("dim") != "1":
            continue
        token = row.get("config_id") or row.get("token") or ""
        grouped[token].append((row, actual_value(row)))
    result = []
    for values in grouped.values():
        row = values[0][0]
        measurements = sorted(value for _, value in values)
        middle = len(measurements) // 2
        actual = measurements[middle] if len(measurements) % 2 else (
            measurements[middle - 1] + measurements[middle]
        ) / 2.0
        result.append({
            "token": row.get("token", ""),
            "group": row.get("group_id", ""),
            "dtype": row["dtype"],
            "block_dim": int(row["block_dim"]),
            "bytes_per_core": float(row["bytes_per_core"]),
            "logical_total_bytes": float(row["logical_total_bytes"]),
            "input_stride": float(row["input_stride"]),
            "output_stride": float(row["output_stride"]),
            "actual": actual,
        })
    return result


def branch(block_dim: int) -> str:
    return "le2" if block_dim <= 2 else "gt2"


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


def predict_base(params: dict[str, object], dtype: str, k: int, b: float) -> float:
    values = params[dtype][branch(k)] if dtype in params else params[branch(k)]
    return float(values["alpha"]) + b / float(values["T_bytes_per_cycle"])


def s_value(dtype: str, input_stride: float) -> float:
    return min(128.0, input_stride * DTYPE_SIZES[dtype])


def predict_correction(params: dict[str, object], dtype: str, b: float,
                       input_stride: float, output_stride: float) -> tuple[float, float, float]:
    s = s_value(dtype, input_stride)
    gate = min(1.0, max(0.0, output_stride - 1.0))
    ng = (float(params[dtype]["N_G"]["a1"])
          + float(params[dtype]["N_G"]["a2"]) * b) * s
    ngu = ((float(params[dtype]["N_GU"]["b1"])
            + float(params[dtype]["N_GU"]["b2"]) * s)
           + (float(params[dtype]["N_GU"]["b3"])
              + float(params[dtype]["N_GU"]["b4"]) * s) * b) * gate
    return ng, ngu, s


def rho(params: dict[str, object], dtype: str, s: float, gate: float) -> float:
    values = params[dtype]["rho"]
    return ((float(values["c1"]) + float(values["c2"]) * s)
            + gate * (float(values["c3"]) + float(values["c4"]) * s))


def fit_model(rows: list[dict[str, object]]) -> dict[str, object]:
    params: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, object] = {}
    for dtype in DTYPES:
        dtype_rows = [row for row in rows if row["dtype"] == dtype]
        base_rows = [
            row for row in dtype_rows
            if row["group"] == "F" and row["input_stride"] == 1
            and row["output_stride"] == 1
        ]
        base: dict[str, object] = {}
        for segment in ("le2", "gt2"):
            selected = [row for row in base_rows if branch(int(row["block_dim"])) == segment]
            coefficients = solve(
                [[1.0, float(row["bytes_per_core"])] for row in selected],
                [float(row["actual"]) for row in selected],
            )
            base[segment] = {
                "alpha": coefficients[0],
                "T_bytes_per_cycle": 1.0 / coefficients[1],
                "sample_count": len(selected),
            }

        ng_rows = [
            row for row in dtype_rows
            if row["group"] == "A" and row["block_dim"] == 1
            and row["output_stride"] == 1
        ]
        ng_coefficients = solve(
            [[s_value(dtype, row["input_stride"]),
              s_value(dtype, row["input_stride"]) * row["bytes_per_core"]]
             for row in ng_rows],
            [row["actual"] - predict_base(base, dtype, 1, row["bytes_per_core"])
             for row in ng_rows],
        )
        ngu_rows = [
            row for row in dtype_rows
            if row["group"] in {"B", "C"} and row["block_dim"] == 1
            and row["output_stride"] > 1
        ]
        ngu_coefficients = solve(
            [[gate, gate * s, gate * row["bytes_per_core"],
              gate * s * row["bytes_per_core"]]
             for row in ngu_rows
             for s, gate in [(s_value(dtype, row["input_stride"]),
                              min(1.0, row["output_stride"] - 1.0))]],
            [row["actual"]
             - predict_base(base, dtype, 1, row["bytes_per_core"])
             - (ng_coefficients[0] + ng_coefficients[1] * row["bytes_per_core"])
             * s_value(dtype, row["input_stride"])
             for row in ngu_rows],
        )
        params[dtype] = {
            "base": base,
            "N_G": {"a1": ng_coefficients[0], "a2": ng_coefficients[1]},
            "N_GU": {
                "b1": ngu_coefficients[0], "b2": ngu_coefficients[1],
                "b3": ngu_coefficients[2], "b4": ngu_coefficients[3],
            },
        }

        multi_rows = [
            row for row in dtype_rows
            if row["group"] in {"G", "H"} and row["block_dim"] > 2
        ]
        matrix = []
        target = []
        for row in multi_rows:
            ng, ngu, s = predict_correction(params, dtype, row["bytes_per_core"],
                                             row["input_stride"], row["output_stride"])
            correction = ng + ngu
            gate = min(1.0, row["output_stride"] - 1.0)
            matrix.append([correction, s * correction, gate * correction,
                           gate * s * correction])
            target.append(row["actual"] - predict_base(base, dtype, row["block_dim"],
                                                       row["bytes_per_core"]))
        rho_coefficients = solve(matrix, target)
        params[dtype]["rho"] = {
            "c1": rho_coefficients[0], "c2": rho_coefficients[1],
            "c3": rho_coefficients[2], "c4": rho_coefficients[3],
            "sample_count": len(multi_rows),
        }
        diagnostics[dtype] = {
            "base_fit_samples": len(base_rows),
            "N_G_fit_samples": len(ng_rows),
            "N_GU_fit_samples": len(ngu_rows),
            "multicore_rho_fit_samples": len(multi_rows),
        }

    output_rows = []
    for row in rows:
        dtype = str(row["dtype"])
        k = int(row["block_dim"])
        b = float(row["bytes_per_core"])
        base_prediction = predict_base(
            {dtype: params[dtype]["base"]}, dtype, k, b
        )
        ng, ngu, s = predict_correction(
            params, dtype, b, float(row["input_stride"]),
            float(row["output_stride"]),
        )
        gate = min(1.0, max(0.0, float(row["output_stride"]) - 1.0))
        multiplier = 1.0 if k <= 2 else rho(params, dtype, s, gate)
        predicted = base_prediction + (ng + ngu) * multiplier
        output_rows.append({
            **row,
            "branch": branch(k),
            "base_cycles": base_prediction,
            "N_G": ng,
            "N_GU": ngu,
            "rho": multiplier,
            "predicted": predicted,
            "error": predicted - float(row["actual"]),
        })

    return {
        "model": "NDDMA_ROUND2_1D_NONCONTIGUOUS_NG_NGU_MULTICORE",
        "formula": {
            "base": "N_base = B/T_le2 + alpha_le2 (k<=2), else B/T_gt2 + alpha_gt2",
            "N_G": "N_G=(a1+a2*B)*s",
            "N_GU": "N_GU=((b1+b2*s)+(b3+b4*s)*B)*min(1,os-1)",
            "rho": "rho=(c1+c2*s)+min(1,os-1)*(c3+c4*s)",
            "final": "N_1=N_base+N_G+N_GU (k<=2); N_base+rho*(N_G+N_GU) (k>2)",
            "definitions": {
                "B": "bytes_per_core = logical_total_bytes / block_dim",
                "s": "min(input_stride*dtype_size,128)",
                "os_gate": "min(1,output_stride-1)",
            },
            "fit_order": ["base", "N_G", "N_GU", "multicore_rho"],
        },
        "fit_scope": {
            "dimension": 1,
            "fit_groups": ["A", "B", "C", "F", "G", "H"],
            "validation_groups": ["I", "J"],
            "sample_count": len(rows),
            "sample_count_by_group": {
                group: sum(row["group"] == group for row in rows)
                for group in ("A", "B", "C", "F", "G", "H", "I", "J")
            },
        },
        "dtype_models": params,
        "diagnostics": diagnostics,
        "predictions": output_rows,
    }


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("token", "group", "dtype", "block_dim", "bytes_per_core",
              "logical_total_bytes", "input_stride", "output_stride", "branch",
              "base_cycles", "N_G", "N_GU", "rho", "actual_cycles",
              "predicted_cycles", "error_cycles")
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "token": row["token"], "group": row["group"], "dtype": row["dtype"],
                "block_dim": row["block_dim"], "bytes_per_core": row["bytes_per_core"],
                "logical_total_bytes": row["logical_total_bytes"],
                "input_stride": row["input_stride"], "output_stride": row["output_stride"],
                "branch": row["branch"], "base_cycles": row["base_cycles"],
                "N_G": row["N_G"], "N_GU": row["N_GU"], "rho": row["rho"],
                "actual_cycles": row["actual"], "predicted_cycles": row["predicted"],
                "error_cycles": row["error"],
            })


def main() -> int:
    args = parse_args()
    files = ([Path(args.measurement_csv).resolve()] if args.measurement_csv
             else find_files(Path(args.data_dir).resolve()))
    if not files:
        raise SystemExit(f"[ERROR] no profiling measurement csv found under {args.data_dir}")
    rows = aggregate(row for path in files for row in read_rows(path))
    if not rows:
        raise SystemExit("[ERROR] no Round2 1D measurements found")
    model = fit_model(rows)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_without_rows = {key: value for key, value in model.items() if key != "predictions"}
    model_path.write_text(json.dumps(model_without_rows, indent=2) + "\n", encoding="utf-8")
    write_predictions(output_dir / PREDICTIONS_FILENAME, model["predictions"])
    print(f"[INFO] fitted {len(rows)} points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
