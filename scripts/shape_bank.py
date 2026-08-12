#!/usr/bin/env python3
"""Self-contained drawing stack for the vision stimuli.

Vendored from binding_probe_repo/datasets/utils.py so that a clone of this repo
can re-render all 120 stimuli with no external checkout. Rendering is
byte-identical to the original: the same four 32x32 templates, the same colour
values, and draw_shape() copied verbatim except that templates resolve from the
local asset instead of that repo's imgs.npy.

Only what the benchmark uses is vendored -- 5 colours and 4 shapes (hexagon is
built at runtime by render_vision.py). The upstream bank holds 102 templates and
13 colours; assets/shape_templates.npz carries just indices 100, 101, 9 and 98
of it (circle, square, triangle, star), 1.2 KB instead of 835 KB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

ASSET = Path(__file__).resolve().parent.parent / "assets" / "shape_templates.npz"

# Verbatim from binding_probe_repo/datasets/utils.py (HIGH_ENTROPY_COLORS),
# restricted to the five the benchmark uses.
ALL_COLORS: dict[str, Tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "purple": (128, 0, 128),
    "yellow": (255, 255, 0),
}

_data = np.load(ASSET)
# White background (255), dark shape -- the imgs.npy convention.
SHAPE_TEMPLATES: np.ndarray = _data["templates"]
SHAPE_INDICES: dict[str, int] = {n: i for i, n in enumerate(_data["names"])}


def draw_shape(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    shape: str,
    color: Tuple[int, int, int],
    shape_size: int = 28,
) -> None:
    """Template -> LANCZOS resize -> colorize, then paste centred in bbox.

    Resolves SHAPE_TEMPLATES / SHAPE_INDICES as module globals on every call, so
    render_vision.py can rebind them to a bank extended with hexagon.

    Note the paste is OPAQUE white outside the shape: callers that need a
    transparent composite must render into a scratch tile and mask it, which is
    what render_vision.paste_object() does.
    """
    x1, y1, x2, y2 = bbox
    shape_idx = SHAPE_INDICES.get(shape)
    if shape_idx is None:
        raise KeyError(f"unknown shape {shape!r}; have {sorted(SHAPE_INDICES)}")
    template = SHAPE_TEMPLATES[shape_idx]
    box_w = x2 - x1
    box_h = y2 - y1
    if shape_size <= 0:
        raise ValueError(f"shape_size must be positive. Got: {shape_size}")
    target_size = min(shape_size, box_w, box_h)
    template_img = Image.fromarray(template.astype(np.uint8), mode="L")
    template_resized = template_img.resize((target_size, target_size),
                                           Image.Resampling.LANCZOS)
    template_array = np.array(template_resized)
    colored_array = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    mask = template_array < 128
    colored_array[mask] = color
    colored_shape = Image.fromarray(colored_array, mode="RGB")
    paste_x = x1 + (box_w - target_size) // 2
    paste_y = y1 + (box_h - target_size) // 2
    img.paste(colored_shape, (int(paste_x), int(paste_y)))
