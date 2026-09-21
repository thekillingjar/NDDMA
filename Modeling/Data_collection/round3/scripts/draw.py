#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_DIR = MODELING_DIR / "Ana" / "round3"
DEFAULT_OUTPUT_DIR = MODELING_DIR / "Ana" / "round3" / "figures"
PREDICTIONS_FILENAME = "round3_multidim_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Round3 multidimensional diagnostics.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def draw_svg(rows: list[dict[str, str]], output: Path, title: str, residual: bool) -> None:
    width, height = 920, 580
    left, top, right, bottom = 90, 36, 30, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(row["data_volume_bytes"]) for row in rows]
    if residual:
        ys = [float(row["error_cycles"]) for row in rows]
        ylabel = "predicted - actual cycles"
    else:
        ys = [float(row["actual_cycles"]) for row in rows] + [
            float(row["predicted_cycles"]) for row in rows
        ]
        ylabel = "cycles"
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.08, 1.0)
    y_min -= pad
    y_max += pad

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    colors = {
        "continuous": "#2563eb",
        "transpose_outer": "#dc2626",
        "transpose_outer_ub_gap": "#ea580c",
        "transpose_dim2": "#16a34a",
        "transpose_dim1_dim3": "#9333ea",
        "transpose_3d": "#0891b2",
        "gm_noncontiguous": "#ca8a04",
        "transpose_dim1_dim4": "#be123c",
        "transpose_dim3_dim4": "#15803d",
        "transpose_dim1_dim5": "#7c3aed",
        "transpose_dim1_dim2": "#0f766e",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#1f2937}"
        ".axis{stroke:#374151}.actual{fill:#111827}.predicted{fill:#fff;stroke:#2563eb;"
        "stroke-width:2}.error{fill-opacity:.75}</style>",
        f'<text x="{width/2}" y="23" text-anchor="middle">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle">data volume bytes</text>',
        f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 18 {top+plot_h/2})">{ylabel}</text>',
        f'<text x="{left}" y="{top+plot_h+22}">{x_min:.0f}</text>',
        f'<text x="{left+plot_w}" y="{top+plot_h+22}" text-anchor="end">{x_max:.0f}</text>',
        f'<text x="{left-8}" y="{py(y_min)+4}" text-anchor="end">{y_min:.0f}</text>',
        f'<text x="{left-8}" y="{py(y_max)+4}" text-anchor="end">{y_max:.0f}</text>',
    ]
    if residual and y_min < 0 < y_max:
        parts.append(f'<line x1="{left}" y1="{py(0):.2f}" x2="{left+plot_w}" y2="{py(0):.2f}" stroke="#9ca3af"/>')
    for row in rows:
        color = colors.get(row["layout"], "#4b5563")
        x = px(float(row["data_volume_bytes"]))
        if residual:
            parts.append(
                f'<circle class="error" cx="{x:.2f}" cy="{py(float(row["error_cycles"])):.2f}" '
                f'r="3" fill="{color}"><title>{row["layout"]}; error={row["error_cycles"]}</title></circle>'
            )
        else:
            parts.append(f'<circle class="actual" cx="{x:.2f}" cy="{py(float(row["actual_cycles"])):.2f}" r="3"/>')
            parts.append(f'<circle class="predicted" cx="{x:.2f}" cy="{py(float(row["predicted_cycles"])):.2f}" r="3"/>')
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    with (input_dir / PREDICTIONS_FILENAME).open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    for dtype in sorted({row["dtype"] for row in rows}):
        for dim in sorted({row["dim"] for row in rows if row["dtype"] == dtype}, key=int):
            selected = [row for row in rows if row["dtype"] == dtype and row["dim"] == dim]
            stem = f"round3_d{dim}_{dtype}"
            draw_svg(selected, output_dir / f"{stem}_actual_vs_predicted.svg",
                     f"Round3 {dim}D {dtype}: actual vs predicted", False)
            draw_svg(selected, output_dir / f"{stem}_residual.svg",
                     f"Round3 {dim}D {dtype}: residual", True)
    print(f"[INFO] wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
