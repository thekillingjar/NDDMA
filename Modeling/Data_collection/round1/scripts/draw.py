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


def triangle_points(x: float, y: float, size: float) -> str:
    return (
        f"{x:.2f},{y - size:.2f} "
        f"{x - size:.2f},{y + size:.2f} "
        f"{x + size:.2f},{y + size:.2f}"
    )


def svg_for_shape(
    dtype: str,
    shape_label: str,
    bytes_per_core: float,
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    width, height = 900, 560
    left, top, right, bottom = 95, 24, 34, 86
    plot_w, plot_h = width - left - right, height - top - bottom
    if not rows:
        return
    series_rows = sorted(rows, key=lambda row: int(row["block_dim"]))
    xs = [float(row["block_dim"]) for row in series_rows]
    actual = [float(row["actual_cycles"]) for row in series_rows]
    predicted = [float(row["predicted_cycles"]) for row in series_rows]
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
        "<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151}.tick{font-size:15px}.axis-label{font-size:19px;font-weight:600}.legend{font-size:16px}.actual{stroke:#111827;stroke-width:1.2}.predicted{fill:#fff;stroke-width:2.4}.line{fill:none;stroke-width:2;opacity:.9}</style>",
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/><line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<text class="axis-label" x="{left+plot_w/2}" y="{height-24}" text-anchor="middle">block_dim</text>',
        f'<text class="axis-label" x="24" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 24 {top+plot_h/2})">single-core cycles</text>',
        f'<text class="tick" x="{left}" y="{top+plot_h+24}">{x_min:.0f}</text>',
        f'<text class="tick" x="{left+plot_w}" y="{top+plot_h+24}" text-anchor="end">{x_max:.0f}</text>',
        f'<text class="tick" x="{left-10}" y="{py(y_min)+5}" text-anchor="end">{y_min:.0f}</text>',
        f'<text class="tick" x="{left-10}" y="{py(y_max)+5}" text-anchor="end">{y_max:.0f}</text>',
    ]
    color = SHAPE_COLORS[shape_label]
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
        f'<polyline class="line" stroke="{color}" stroke-dasharray="6 5" points="{" ".join(predicted_points)}"/>'
    )
    for row in series_rows:
        x_value = float(row["block_dim"])
        actual_value = float(row["actual_cycles"])
        predicted_value = float(row["predicted_cycles"])
        title = (
            f'{shape_label} bytes_per_core={bytes_per_core:.0f}; '
            f'block_dim={x_value:.0f}'
        )
        parts.append(
            f'<polygon class="actual" points="{triangle_points(px(x_value), py(actual_value), 5.2)}" fill="{color}"><title>{title}; actual={actual_value:.4g}</title></polygon>'
        )
        parts.append(
            f'<circle class="predicted" cx="{px(x_value)}" cy="{py(predicted_value)}" r="3.8" stroke="{color}"><title>{title}; predicted={predicted_value:.4g}</title></circle>'
        )

    legend_x = left + plot_w - 190
    legend_y = top + plot_h - 58
    parts.extend([
        f'<text class="legend" x="{legend_x}" y="{legend_y}">{shape_label} bytes/core={bytes_per_core:.0f}</text>',
        f'<polygon class="actual" points="{triangle_points(legend_x + 7, legend_y + 22, 5.2)}" fill="{color}"/><text class="legend" x="{legend_x + 20}" y="{legend_y + 27}">actual</text>',
        f'<circle class="predicted" cx="{legend_x + 7}" cy="{legend_y + 48}" r="3.8" stroke="{color}"/><text class="legend" x="{legend_x + 20}" y="{legend_y + 53}">predicted</text>',
        "</svg>",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def selected_shape_rows(rows: list[dict[str, str]]) -> list[tuple[str, float, list[dict[str, str]]]]:
    if not rows:
        return []
    bytes_values = sorted({float(row["bytes_per_core"]) for row in rows})
    selected = [("min", bytes_values[0])]
    if bytes_values[-1] != bytes_values[0]:
        selected.append(("max", bytes_values[-1]))
    return [
        (
            label,
            bytes_value,
            [row for row in rows if float(row["bytes_per_core"]) == bytes_value],
        )
        for label, bytes_value in selected
    ]


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    prediction_path = input_dir / PREDICTIONS_FILENAME
    with prediction_path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    for dtype in DTYPES:
        dtype_rows = [row for row in rows if row["dtype"] == dtype]
        for label, bytes_value, shape_rows in selected_shape_rows(dtype_rows):
            svg_for_shape(
                dtype,
                label,
                bytes_value,
                shape_rows,
                output_dir / f"round1_1d_single_core_{dtype}_{label}.svg",
            )
    print(f"[INFO] wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
