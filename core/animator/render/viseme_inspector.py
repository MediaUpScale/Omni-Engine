"""Dock the nine mechanical visemes on the approved frontal chin and export one sheet."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.pipeline_paths import assets_root, outputs_root

from .visemes import MOUTH_DOCK_CHIN_RATIO, PALETTES

SHEET_ORDER: tuple[tuple[str, str], ...] = (
    ("X", "Rest"),
    ("A", "P/B/M"),
    ("B", "K/S/T"),
    ("C", "Eh/Ae"),
    ("D", "Ah/Wide"),
    ("E", "Oh/Round"),
    ("F", "Oo/Tight"),
    ("G", "F/V"),
    ("H", "L/Th"),
)
CELL_W = 640
CELL_H = 820
LABEL_H = 72


def _approved_head(character_id: str) -> Image.Image:
    root = assets_root() / "puppets" / character_id
    for relative in ("views/facing_front/head.png", "views/facing_left/head.png", "head.png"):
        path = root / relative
        if path.is_file():
            with Image.open(path) as opened:
                return opened.convert("RGBA")
    raise FileNotFoundError(f"no approved head for {character_id}")


def _chin_plate(head: Image.Image) -> tuple[int, int, int, int]:
    """Warm lower chin guard, ``(x0, y0, x1, y1)`` in head-image pixels."""
    rgba = np.asarray(head.convert("RGBA"))
    alpha = rgba[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    if xs.size == 0:
        raise RuntimeError("head has no opaque pixels")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    head_h = y1 - y0 + 1
    lower_cut = y0 + int(round(head_h * 0.55))
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    warm = (
        (hsv[..., 0] >= 4)
        & (hsv[..., 0] <= 40)
        & (hsv[..., 1] >= 40)
        & (hsv[..., 2] >= 70)
        & (alpha > 16)
    )
    warm[:lower_cut, :] = False
    count, _, stats, _centroids = cv2.connectedComponentsWithStats(warm.astype(np.uint8), connectivity=8)
    best_area = 0
    plate = (x0, y0 + int(head_h * 0.62), x1, y1)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            left = int(stats[index, cv2.CC_STAT_LEFT])
            top = int(stats[index, cv2.CC_STAT_TOP])
            plate = (
                left,
                top,
                left + int(stats[index, cv2.CC_STAT_WIDTH]),
                top + int(stats[index, cv2.CC_STAT_HEIGHT]),
            )
    return plate


def _chin_anchor(head: Image.Image) -> tuple[int, int]:
    """Mouth seat on the chin guard.

    Frontal chin plates run long, so the centroid sits on the beard. Llama's
    mouth lives in the upper third of its plate; the same fraction is used here.
    """
    left, top, right, bottom = _chin_plate(head)
    return (left + right) // 2, top + int(round((bottom - top) * 0.32))


def _fit_head(head: Image.Image) -> tuple[Image.Image, float]:
    """Scale the head to the cell, leaving room for the label."""
    budget_h = CELL_H - LABEL_H - 36
    budget_w = CELL_W - 36
    scale = min(budget_w / head.width, budget_h / head.height)
    resized = head.resize(
        (max(1, int(round(head.width * scale))), max(1, int(round(head.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def _dock_mouth(
    head: Image.Image,
    mouth: Image.Image,
    anchor: tuple[int, int],
    scale: float,
    chin_width: int,
) -> Image.Image:
    """Place the mouth sprite on the chin. The head pixels are not rewritten on disk."""
    target_w = max(32, int(round(chin_width * scale * MOUTH_DOCK_CHIN_RATIO)))
    mouth_scale = target_w / mouth.width
    resized = mouth.resize(
        (max(1, int(round(mouth.width * mouth_scale))), max(1, int(round(mouth.height * mouth_scale)))),
        Image.Resampling.LANCZOS,
    )
    plate = Image.new("RGBA", (CELL_W, CELL_H - LABEL_H), (22, 18, 14, 255))
    paste_x = (CELL_W - head.width) // 2
    paste_y = (CELL_H - LABEL_H - head.height) // 2
    plate.alpha_composite(head, (paste_x, paste_y))
    mouth_x = paste_x + int(round(anchor[0] * scale)) - resized.width // 2
    mouth_y = paste_y + int(round(anchor[1] * scale)) - resized.height // 2
    plate.alpha_composite(resized, (mouth_x, mouth_y))
    return plate


def export_inspection_sheet(
    character_ids: tuple[str, ...] = tuple(PALETTES),
    destination: Path | None = None,
) -> Path:
    """Write one sheet: ChatGPT on the left 3x3, Claude on the right 3x3."""
    columns = 3 * len(character_ids)
    sheet = Image.new("RGB", (CELL_W * columns, CELL_H * 3), (14, 12, 10))
    try:
        font = ImageFont.truetype("arialbd.ttf", 28)
        small = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        small = font
    for character_index, character_id in enumerate(character_ids):
        head = _approved_head(character_id)
        plate = _chin_plate(head)
        anchor = _chin_anchor(head)
        chin_width = plate[2] - plate[0]
        fitted, scale = _fit_head(head)
        mouths = assets_root() / "puppets" / character_id / "mouths"
        title = "CHATGPT" if character_id.startswith("chatgpt") else "CLAUDE"
        for slot, (code, label) in enumerate(SHEET_ORDER):
            row, column = divmod(slot, 3)
            mouth_path = mouths / f"mouth_{code}.png"
            if not mouth_path.is_file():
                raise FileNotFoundError(mouth_path)
            with Image.open(mouth_path) as opened:
                mouth = opened.convert("RGBA")
            cell = _dock_mouth(fitted, mouth, anchor, scale, chin_width)
            draw = ImageDraw.Draw(cell)
            origin_x = (character_index * 3 + column) * CELL_W
            origin_y = row * CELL_H
            banner = Image.new("RGB", (CELL_W, LABEL_H), (28, 22, 16))
            banner_draw = ImageDraw.Draw(banner)
            banner_draw.text((18, 8), f"{title}  {code}", fill=(244, 214, 150), font=font)
            banner_draw.text((18, 40), label, fill=(210, 186, 140), font=small)
            sheet.paste(banner, (origin_x, origin_y))
            sheet.paste(cell.convert("RGB"), (origin_x, origin_y + LABEL_H))
            del draw
    destination = destination or (
        outputs_root() / "aiwake" / "_test_harness" / "viseme_inspection_sheet.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", compress_level=1)
    return destination
