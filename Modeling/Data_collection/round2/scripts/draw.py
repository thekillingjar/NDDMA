#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_DIR = MODELING_DIR / "Ana" / "round2"
DEFAULT_OUTPUT_DIR = MODELING_DIR / "Ana" / "round2" / "figures"
PREDICTIONS_FILENAME = "round2_1d_noncontiguous_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
INPUT_STRIDE_LABEL_MIN = {"int8_t": 128, "int16_t": 64, "int32_t": 256, "int64_t": 128}
INPUT_STRIDE_EXTRA_LABELS = {"int32_t": {32}, "int64_t": {16}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Round2 1D model diagnostics.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def svg(dtype: str, rows: list[dict[str, str]], output: Path, residual: bool) -> None:
    width, height = 920, 580
    left, top, right, bottom = 90, 35, 30, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(row["logical_total_bytes"]) for row in rows]
    if residual:
        ys = [float(row["error_cycles"]) for row in rows]
        title = f"Round2 1D {dtype}: prediction error"
        y_label = "predicted - actual cycles"
    else:
        ys = [float(row["actual_cycles"]) for row in rows]
        predicted = [float(row["predicted_cycles"]) for row in rows]
        ys += predicted
        title = f"Round2 1D {dtype}: actual vs predicted"
        y_label = "cycles"
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.08, 1.0)
    y_min -= pad
    y_max += pad

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#1f2937}"
        ".axis{stroke:#374151}.actual{fill:#111827}.predicted{fill:#fff;stroke:#2563eb;"
        "stroke-width:2}.error{fill:#b91c1c}</style>",
        f'<text x="{width/2}" y="22" text-anchor="middle">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle">logical total bytes</text>',
        f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 18 {top+plot_h/2})">{y_label}</text>',
        f'<text x="{left}" y="{top+plot_h+22}">{x_min:.0f}</text>',
        f'<text x="{left+plot_w}" y="{top+plot_h+22}" text-anchor="end">{x_max:.0f}</text>',
        f'<text x="{left-8}" y="{py(y_min)+4}" text-anchor="end">{y_min:.0f}</text>',
        f'<text x="{left-8}" y="{py(y_max)+4}" text-anchor="end">{y_max:.0f}</text>',
    ]
    if residual and y_min < 0 < y_max:
        parts.append(f'<line x1="{left}" y1="{py(0)}" x2="{left+plot_w}" y2="{py(0)}" stroke="#9ca3af"/>')
    for row in rows:
        x = px(float(row["logical_total_bytes"]))
        actual = float(row["actual_cycles"])
        predicted = float(row["predicted_cycles"])
        value = float(row["error_cycles"]) if residual else actual
        parts.append(f'<circle class="{"error" if residual else "actual"}" cx="{x:.2f}" cy="{py(value):.2f}" r="3"/>')
        if not residual:
            parts.append(f'<circle class="predicted" cx="{x:.2f}" cy="{py(predicted):.2f}" r="3"/>')
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def output_dim(dtype: str, row: dict[str, str]) -> int:
    return int(round(float(row["bytes_per_core"]) / DTYPE_SIZES[dtype]))


def input_stride_vs_cycles_svg(
    dtype: str,
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    selected = [
        row for row in rows
        if row["dtype"] == dtype
        and row["group"] == "A"
        and int(float(row["block_dim"])) == 1
        and int(float(row["output_stride"])) == 1
    ]
    if not selected:
        return

    curves: dict[int, list[dict[str, str]]] = {}
    for row in selected:
        curves.setdefault(output_dim(dtype, row), []).append(row)
    for curve_rows in curves.values():
        curve_rows.sort(key=lambda row: float(row["input_stride"]))

    width, height = 1220, 700
    left, top, right, bottom = 105, 50, 310, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    strides = sorted({float(row["input_stride"]) for row in selected})
    cycles = [float(row["actual_cycles"]) for row in selected]
    x_min, x_max = min(strides), max(strides)
    y_min, y_max = min(cycles), max(cycles)
    y_pad = max((y_max - y_min) * 0.05, 1.0)
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad
    palette = (
        "#b91c1c", "#0369a1", "#15803d", "#7c3aed", "#c2410c", "#0f766e",
        "#be185d", "#4338ca", "#4d7c0f", "#a16207", "#0e7490", "#6d28d9",
        "#374151",
    )

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<defs><clipPath id="sweep-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w/2}" y="29" text-anchor="middle" font-family="sans-serif" font-size="20">{dtype}: input stride vs actual cycles</text>',
        f'<text x="24" y="{top + plot_h/2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h/2})" font-family="sans-serif" font-size="18" font-weight="600">actual cycles</text>',
        f'<text x="{left + plot_w/2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">input stride</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>',
    ]
    min_labeled_stride = INPUT_STRIDE_LABEL_MIN[dtype]
    extra_labeled_strides = INPUT_STRIDE_EXTRA_LABELS.get(dtype, set())
    for stride in strides:
        x = px(stride)
        parts.extend([
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
            f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#555"/>',
        ])
        if stride >= min_labeled_stride or int(stride) in extra_labeled_strides:
            parts.append(
                f'<text class="x-stride-label" x="{x:.2f}" y="{top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="17">{stride:.0f}</text>'
            )
    for index in range(6):
        fraction = index / 5
        y_value = y_min + fraction * (y_max - y_min)
        y = py(y_value)
        parts.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
            f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555"/>',
            f'<text x="{left - 12}" y="{y + 6:.2f}" text-anchor="end" font-family="sans-serif" font-size="17">{y_value:.5g}</text>',
        ])

    parts.append('<g clip-path="url(#sweep-clip)">')
    for curve_index, shape in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        curve_rows = curves[shape]
        points = " ".join(
            f'{px(float(row["input_stride"])):.2f},{py(float(row["actual_cycles"])):.2f}'
            for row in curve_rows
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        )
        for row in curve_rows:
            parts.append(
                f'<circle cx="{px(float(row["input_stride"])):.2f}" cy="{py(float(row["actual_cycles"])):.2f}" r="3" fill="{color}"><title>output_dim={shape}; input_stride={float(row["input_stride"]):.0f}; actual={float(row["actual_cycles"]):.4g}</title></circle>'
            )
    parts.append("</g>")

    legend_x = left + plot_w + 35
    legend_y = top + 15
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="16" font-weight="bold">output_dim</text>'
    )
    for curve_index, shape in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        column, row_index = divmod(curve_index, 12)
        x = legend_x + column * 112
        y = legend_y + 28 + row_index * 27
        parts.extend([
            f'<line x1="{x}" y1="{y - 5}" x2="{x + 24}" y2="{y - 5}" stroke="{color}" stroke-width="2.5"/>',
            f'<text x="{x + 32}" y="{y}" font-family="sans-serif" font-size="14">{shape}</text>',
        ])
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    with (input_dir / PREDICTIONS_FILENAME).open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    for dtype in DTYPES:
        selected = [row for row in rows if row["dtype"] == dtype]
        svg(dtype, selected, output_dir / f"round2_1d_noncontiguous_actual_vs_predicted_{dtype}.svg", False)
        svg(dtype, selected, output_dir / f"round2_1d_noncontiguous_residual_{dtype}.svg", True)
        input_stride_vs_cycles_svg(
            dtype,
            rows,
            output_dir / f"round2_1d_input_stride_vs_cycles_{dtype}.svg",
        )
    print(f"[INFO] wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
