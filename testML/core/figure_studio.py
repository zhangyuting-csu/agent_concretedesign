import base64
import io
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core.plotting_utils import apply_nature_style, convert_svg_bytes_to_png_bytes, fig_to_base64


def _decode_data_url(data_url):
    if not data_url or "," not in data_url:
        raise ValueError("Invalid image payload.")
    header, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    if "image/svg+xml" in header:
        raw = convert_svg_bytes_to_png_bytes(raw)
    image = Image.open(io.BytesIO(raw)).convert("RGBA")
    return np.asarray(image)


def _label_text(idx, style):
    style = (style or "lower").lower()
    base = chr(ord("a") + idx % 26)
    if style == "upper":
        return base.upper()
    return base


def _auto_grid_boxes(count, columns, gap, margin):
    cols = max(1, int(columns or 1))
    rows = max(1, math.ceil(count / cols))
    total_w = 1 - 2 * margin - gap * (cols - 1)
    total_h = 1 - 2 * margin - gap * (rows - 1)
    cell_w = total_w / cols
    cell_h = total_h / rows
    boxes = []
    for idx in range(count):
        row = idx // cols
        col = idx % cols
        left = margin + col * (cell_w + gap)
        bottom = 1 - margin - (row + 1) * cell_h - row * gap
        boxes.append((left, bottom, cell_w, cell_h))
    return boxes


def _preset_boxes(count, preset, gap, margin):
    preset = (preset or "Auto Grid").lower()
    if preset == "1+2" and count >= 3:
        width = 1 - 2 * margin - gap
        left_w = width * 0.52
        right_w = width - left_w
        height = 1 - 2 * margin - gap
        top_h = height / 2
        return [
            (margin, margin, left_w, 1 - 2 * margin),
            (margin + left_w + gap, margin + top_h + gap, right_w, top_h),
            (margin + left_w + gap, margin, right_w, top_h),
        ]
    if preset == "2+1" and count >= 3:
        width = 1 - 2 * margin - gap
        right_w = width * 0.52
        left_w = width - right_w
        height = 1 - 2 * margin - gap
        top_h = height / 2
        return [
            (margin, margin + top_h + gap, left_w, top_h),
            (margin, margin, left_w, top_h),
            (margin + left_w + gap, margin, right_w, 1 - 2 * margin),
        ]
    if preset == "1+3" and count >= 4:
        width = 1 - 2 * margin
        top_h = (1 - 2 * margin - gap) * 0.42
        bottom_h = 1 - 2 * margin - gap - top_h
        cell_w = (width - 2 * gap) / 3
        return [
            (margin, margin + bottom_h + gap, width, top_h),
            (margin, margin, cell_w, bottom_h),
            (margin + cell_w + gap, margin, cell_w, bottom_h),
            (margin + 2 * (cell_w + gap), margin, cell_w, bottom_h),
        ]
    if preset == "2x2":
        return _auto_grid_boxes(count, 2, gap, margin)
    if preset == "3x2":
        return _auto_grid_boxes(count, 3, gap, margin)
    return _auto_grid_boxes(count, max(1, math.ceil(math.sqrt(count))), gap, margin)


def compose_figure(panels, canvas_width=7.2, canvas_height=5.4, label_size=16, label_style="lower", gap=0.025, margin=0.05, columns=2, preset="Auto Grid", fmt="png", dpi=300):
    apply_nature_style()
    valid_panels = [panel for panel in panels if panel.get("data_url")]
    if not valid_panels:
        raise ValueError("At least one panel image is required.")

    fig = plt.figure(figsize=(float(canvas_width), float(canvas_height)), facecolor="white")
    boxes = _preset_boxes(len(valid_panels), preset, float(gap), float(margin))
    if len(boxes) < len(valid_panels):
        boxes = _auto_grid_boxes(len(valid_panels), columns, float(gap), float(margin))

    for idx, (panel, box) in enumerate(zip(valid_panels, boxes)):
        image = _decode_data_url(panel["data_url"])
        left, bottom, width, height = box
        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        label = panel.get("label") or _label_text(idx, label_style)
        if panel.get("show_label", True):
            fig.text(
                left + width * 0.01,
                bottom + height * 0.99,
                label,
                fontsize=float(label_size),
                fontweight="bold",
                ha="left",
                va="top",
                color="black",
            )

    return fig_to_base64(fig, fmt, dpi)
