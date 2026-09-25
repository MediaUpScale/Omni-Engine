"""Split an auto-rigged bust into articulated head, body, and collar layers."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class SlicedLayers:
    body: Image.Image
    head: Image.Image
    collar: Image.Image
    collar_y: int
    overlap_px: int
    facing: str


def _masked_layer(rgba: np.ndarray, mask: np.ndarray) -> Image.Image:
    layer = rgba.copy()
    layer[..., 3] = cv2.bitwise_and(layer[..., 3], mask)
    return Image.fromarray(layer)


def slice_character_layers(
    character: Image.Image,
    *,
    neck_pivot: tuple[int, int],
    chin_boundary: int | None = None,
    facing: str = "right",
    facial_plate_bbox: tuple[int, int, int, int] | None = None,
    overlap_px: int = 45,
) -> SlicedLayers:
    """Create full-canvas layers with a body-owned neck articulation overlap.

    ``head`` ends at the detected jaw. ``body`` contains torso, collar and the
    neck extending behind the jaw; ``collar`` is composited last as a rim.
    """
    rgba = np.asarray(character.convert("RGBA"), dtype=np.uint8)
    height, width = rgba.shape[:2]
    normalized_facing = (facing or "right").strip().lower()
    if normalized_facing not in {"left", "right"}:
        raise ValueError("facing must be 'left' or 'right'")
    overlap = max(8, int(overlap_px))
    collar_y = int(np.clip(neck_pivot[1], height * 0.48, height * 0.84))
    chin_y = int(
        np.clip(
            chin_boundary if chin_boundary is not None else collar_y,
            0,
            collar_y,
        )
    )

    head_mask = np.zeros((height, width), dtype=np.uint8)
    body_mask = np.zeros((height, width), dtype=np.uint8)
    center_x = int(np.clip(neck_pivot[0], 0, width - 1))
    visible_y, _visible_x = np.nonzero(rgba[..., 3] > 8)
    crown_y = int(visible_y.min()) if visible_y.size else 0
    if facial_plate_bbox is not None:
        plate_x0, _plate_y0, plate_x1, _plate_y1 = facial_plate_bbox
        plate_width = max(1, plate_x1 - plate_x0)
        neck_half_width = max(24, int(round(plate_width * 0.45)))
        upper_head_alpha = rgba[
            crown_y : max(crown_y + 1, min(chin_y, facial_plate_bbox[1])),
            :,
            3,
        ]
        _upper_ys, upper_xs = np.nonzero(upper_head_alpha > 8)
        if upper_xs.size:
            helmet_left = int(upper_xs.min())
            helmet_right = int(upper_xs.max()) + 1
        else:
            helmet_half_width = int(round(plate_width * 1.70))
            helmet_left = max(0, center_x - helmet_half_width)
            helmet_right = min(width, center_x + helmet_half_width)
    else:
        sample_bottom = max(
            crown_y + 1,
            min(chin_y, int(round(height * 0.38))),
        )
        _sample_ys, sample_xs = np.nonzero(
            rgba[crown_y:sample_bottom, :, 3] > 8
        )
        horizontal_pad = int(round(width * 0.08))
        if sample_xs.size:
            helmet_left = max(0, int(sample_xs.min()) - horizontal_pad)
            helmet_right = min(
                width,
                int(sample_xs.max()) + 1 + horizontal_pad,
            )
        else:
            helmet_left, helmet_right = 0, width
        neck_half_width = max(24, int(round(width * 0.14)))

    # Decoupled anatomy: the head ends at the jaw. The neck belongs entirely
    # to the body and rises behind the jaw by ``overlap`` source pixels.
    head_mask[crown_y : min(height, chin_y + 1), helmet_left:helmet_right] = 255
    body_mask[collar_y:] = 255
    neck_top = max(crown_y, chin_y - overlap)
    body_mask[
        neck_top:collar_y,
        max(0, center_x - neck_half_width) : min(
            width, center_x + neck_half_width
        ),
    ] = 255

    # Preserve every authored torso/shoulder pixel outside the isolated head.
    # The central neck is then restored beneath the jaw for tilt-safe overlap.
    visible = rgba[..., 3]
    body_mask = cv2.bitwise_and(cv2.bitwise_not(head_mask), visible)
    neck_slice = (
        slice(neck_top, collar_y),
        slice(
            max(0, center_x - neck_half_width),
            min(width, center_x + neck_half_width),
        ),
    )
    body_mask[neck_slice] = visible[neck_slice]

    collar_mask = np.zeros((height, width), dtype=np.uint8)
    band_top = max(0, collar_y - max(8, overlap // 2))
    band_bottom = min(height, collar_y + overlap)
    center_x = int(np.clip(neck_pivot[0], 0, width - 1))
    if facial_plate_bbox is not None:
        plate_width = max(1, facial_plate_bbox[2] - facial_plate_bbox[0])
        neck_half_width = max(24, int(round(plate_width * 0.18)))
    else:
        neck_half_width = max(24, int(round(width * 0.07)))
    # Side armor remains in front. The lower third closes the collar ring
    # across the center without covering the articulated neck above it.
    collar_half_width = max(
        neck_half_width * 3,
        int(round(width * 0.34)),
    )
    collar_left = max(0, center_x - collar_half_width)
    collar_right = min(width, center_x + collar_half_width)
    if normalized_facing == "left":
        collar_right = width
    else:
        collar_left = 0
    collar_mask[band_top:band_bottom, collar_left:collar_right] = 255
    center_left = max(0, center_x - neck_half_width)
    center_right = min(width, center_x + neck_half_width)
    collar_mask[band_top:band_bottom, center_left:center_right] = 0
    lower_rim = band_top + int(round((band_bottom - band_top) * 0.50))
    collar_mask[lower_rim:band_bottom, center_left:center_right] = 255

    head_mask = cv2.bitwise_and(head_mask, visible)
    collar_mask = cv2.bitwise_and(collar_mask, visible)
    return SlicedLayers(
        body=_masked_layer(rgba, body_mask),
        head=_masked_layer(rgba, head_mask),
        collar=_masked_layer(rgba, collar_mask),
        collar_y=collar_y,
        overlap_px=overlap,
        facing=normalized_facing,
    )


def slice_jaw_contour(
    character: Image.Image,
    *,
    overlap_px: int = 25,
) -> tuple[Image.Image, Image.Image]:
    """Split a bust on the jaw curve. The neck stays on the body."""
    rgba = np.asarray(character.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    if not xs.size:
        raise RuntimeError("cannot slice an empty bust")
    top, bottom = int(ys.min()), int(ys.max())
    span = max(1, bottom - top)
    neck_y = top + int(span * 0.46)
    neck_width = 10**9
    for y in range(top + int(span * 0.30), top + int(span * 0.58)):
        width = int(np.count_nonzero(alpha[y] > 16))
        if 8 < width < neck_width:
            neck_width = width
            neck_y = y
    center = int(np.median(xs))
    half = max(12, neck_width // 2)
    chin_y = neck_y
    for y in range(neck_y, min(bottom, neck_y + int(span * 0.12))):
        band = alpha[y, max(0, center - half) : center + half]
        if int(np.count_nonzero(band > 16)) > half:
            chin_y = y
        else:
            break
    head_mask = np.zeros(alpha.shape, dtype=np.uint8)
    left = max(0, center - neck_width)
    right = min(alpha.shape[1] - 1, center + neck_width)
    for x in range(int(xs.min()), int(xs.max()) + 1):
        column = np.nonzero(alpha[:, x] > 16)[0]
        above = column[column <= neck_y]
        if not above.size:
            continue
        dist = abs(x - center) / max(1, neck_width)
        curve = chin_y - int(round((dist * dist) * (chin_y - neck_y + 18)))
        kept = column[column <= max(neck_y, curve)]
        if kept.size and int(kept.min()) < neck_y:
            head_mask[kept, x] = 255
    # Shoulders beside the neck are body, even when they rise beside the jaw.
    head_mask[:, :left] = 0
    head_mask[:, right:] = 0
    head_mask = cv2.bitwise_and(head_mask, np.where(alpha > 16, 255, 0).astype(np.uint8))
    body_mask = cv2.bitwise_and(
        cv2.bitwise_not(head_mask),
        np.where(alpha > 16, 255, 0).astype(np.uint8),
    )
    overlap = slice(max(0, chin_y - max(1, overlap_px)), chin_y + 1)
    body_mask[overlap, max(0, center - half) : center + half] = np.where(
        alpha[overlap, max(0, center - half) : center + half] > 16,
        255,
        0,
    )
    return _masked_layer(rgba, head_mask), _masked_layer(rgba, body_mask)


__all__ = ["SlicedLayers", "slice_character_layers", "slice_jaw_contour"]
