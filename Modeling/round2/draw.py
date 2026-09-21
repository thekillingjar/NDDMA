#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "Ana" / "round2"
PREDICTIONS_FILENAME = "round2_1d_noncontiguous_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")


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


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    with (input_dir / PREDICTIONS_FILENAME).open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    for dtype in DTYPES:
        selected = [row for row in rows if row["dtype"] == dtype]
        svg(dtype, selected, output_dir / f"round2_1d_noncontiguous_actual_vs_predicted_{dtype}.svg", False)
        svg(dtype, selected, output_dir / f"round2_1d_noncontiguous_residual_{dtype}.svg", True)
    print(f"[INFO] wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
