#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_DIR = MODELING_DIR / "Ana" / "round1"
DEFAULT_OUTPUT_DIR = MODELING_DIR / "Ana" / "round1" / "figures"
MODEL_FILENAME = "round1_1d_single_multi_core_model.json"
PREDICTIONS_FILENAME = "round1_1d_single_core_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
SHAPE_COLORS = {"min": "#0369a1", "max": "#b91c1c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw NDDMA2 Round1 1D model diagnostics.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def svg_for_dtype(dtype: str, rows: list[dict[str, str]], output: Path) -> None:
    width, height = 900, 560
    left, top, right, bottom = 85, 35, 30, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    if not rows:
        return
    bytes_values = sorted({float(row["bytes_per_core"]) for row in rows})
    selected = [bytes_values[0]]
    if bytes_values[-1] != bytes_values[0]:
        selected.append(bytes_values[-1])
    series = []
    for index, bytes_value in enumerate(selected):
        label = "min" if index == 0 else "max"
        series_rows = sorted(
            [
                row for row in rows
                if float(row["bytes_per_core"]) == bytes_value
            ],
            key=lambda row: int(row["block_dim"]),
        )
        series.append((label, bytes_value, series_rows))

    xs = [float(row["block_dim"]) for _, _, series_rows in series for row in series_rows]
    actual = [float(row["actual_cycles"]) for _, _, series_rows in series for row in series_rows]
    predicted = [float(row["predicted_cycles"]) for _, _, series_rows in series for row in series_rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(actual + predicted), max(actual + predicted)
    y_pad = max((y_max - y_min) * 0.08, 1.0)
    y_min -= y_pad
    y_max += y_pad

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#1f2937}.axis{stroke:#374151}.actual{fill:#111827}.predicted{fill:#fff;stroke-width:2}.line{fill:none;stroke-width:1.5;opacity:.85}</style>",
        f'<text x="{width/2}" y="22" text-anchor="middle">Round1 1D single-core cycles vs block_dim {dtype}: min/max shape</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/><line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle">block_dim</text>',
        f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 18 {top+plot_h/2})">single-core cycles</text>',
        f'<text x="{left}" y="{top+plot_h+22}">{x_min:.0f}</text>',
        f'<text x="{left+plot_w}" y="{top+plot_h+22}" text-anchor="end">{x_max:.0f}</text>',
        f'<text x="{left-8}" y="{py(y_min)+4}" text-anchor="end">{y_min:.0f}</text>',
        f'<text x="{left-8}" y="{py(y_max)+4}" text-anchor="end">{y_max:.0f}</text>',
    ]
    for label, bytes_value, series_rows in series:
        color = SHAPE_COLORS[label]
        actual_points = [
            f'{px(float(row["block_dim"])):.2f},{py(float(row["actual_cycles"])):.2f}'
            for row in series_rows
        ]
        predicted_points = [
            f'{px(float(row["block_dim"])):.2f},{py(float(row["predicted_cycles"])):.2f}'
            for row in series_rows
        ]
        parts.append(
            f'<polyline class="line" stroke="{color}" points="{" ".join(actual_points)}"/>'
        )
        parts.append(
            f'<polyline class="line" stroke="{color}" stroke-dasharray="5 4" points="{" ".join(predicted_points)}"/>'
        )
        for row in series_rows:
            x_value = float(row["block_dim"])
            actual_value = float(row["actual_cycles"])
            predicted_value = float(row["predicted_cycles"])
            title = f'{label} bytes_per_core={bytes_value:.0f}; block_dim={x_value:.0f}'
            parts.append(
                f'<circle class="actual" cx="{px(x_value)}" cy="{py(actual_value)}" r="3" fill="{color}"><title>{title}; actual={actual_value:.4g}</title></circle>'
            )
            parts.append(
                f'<circle class="predicted" cx="{px(x_value)}" cy="{py(predicted_value)}" r="3" stroke="{color}"><title>{title}; predicted={predicted_value:.4g}</title></circle>'
            )
    parts.extend([
        f'<line class="line" x1="{left+plot_w-170}" y1="{top+24}" x2="{left+plot_w-140}" y2="{top+24}" stroke="{SHAPE_COLORS["min"]}"/><text x="{left+plot_w-132}" y="{top+29}">min bytes/core</text>',
        f'<line class="line" x1="{left+plot_w-170}" y1="{top+47}" x2="{left+plot_w-140}" y2="{top+47}" stroke="{SHAPE_COLORS["max"]}"/><text x="{left+plot_w-132}" y="{top+52}">max bytes/core</text>',
        f'<circle class="actual" cx="{left+plot_w-170}" cy="{top+70}" r="3"/><text x="{left+plot_w-160}" y="{top+75}">actual</text>',
        f'<circle class="predicted" cx="{left+plot_w-95}" cy="{top+70}" r="3" stroke="#374151"/><text x="{left+plot_w-85}" y="{top+75}">predicted</text>',
        "</svg>",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    prediction_path = input_dir / PREDICTIONS_FILENAME
    with prediction_path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    for dtype in DTYPES:
        svg_for_dtype(dtype, [row for row in rows if row["dtype"] == dtype],
                      output_dir / f"round1_1d_single_core_{dtype}.svg")
    print(f"[INFO] wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
