"""Dominant semantic palette extraction from transparent character art."""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from PIL import Image

from .landmark_detector import ImageInput


class PuppetPalette(TypedDict):
    ink_outline: str
    casing_color: str
    cavity_interior: str
    teeth_color: str


def _visible_rgb(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    else:
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("character image must be RGB or RGBA")
        if array.shape[2] == 3:
            return array.reshape(-1, 3)
        rgba = array
    pixels = rgba[..., :3][rgba[..., 3] > 16]
    if not pixels.size:
        raise ValueError("character image has no non-transparent facial pixels")
    return pixels


def _hex(rgb: np.ndarray) -> str:
    channels = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return "#%02X%02X%02X" % tuple(int(value) for value in channels)


def _luminance(colors: np.ndarray) -> np.ndarray:
    return colors @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def _darken(color: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(color.astype(np.float32) * factor, 0, 255)


def extract_palette(image: ImageInput, *, clusters: int = 6) -> PuppetPalette:
    """Quantize visible pixels and map clusters onto the puppet color schema."""
    pixels = _visible_rgb(image)
    # A deterministic stride bounds K-Means cost for multi-megapixel art while
    # retaining pixels from the entire face instead of a top-left crop.
    if len(pixels) > 40_000:
        step = int(np.ceil(len(pixels) / 40_000))
        pixels = pixels[::step]
    unique = np.unique(pixels, axis=0)
    cluster_count = max(1, min(int(clusters), len(unique)))
    samples = np.ascontiguousarray(pixels.astype(np.float32))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.8)
    cv2.setRNGSeed(1337)
    _compactness, labels, centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.ravel(), minlength=cluster_count)
    dominant = counts >= max(2, int(len(samples) * 0.008))
    if not dominant.any():
        dominant[:] = True
    indices = np.flatnonzero(dominant)
    colors = centers[indices]
    weights = counts[indices]
    luma = _luminance(colors)

    darkest_order = np.argsort(luma)
    ink = colors[darkest_order[0]]
    if len(colors) > 1:
        cavity = colors[darkest_order[1]]
        if _luminance(cavity[None, :])[0] > 105:
            cavity = _darken(ink, 0.68)
    else:
        cavity = _darken(ink, 0.68)

    light_order = np.argsort(luma)[::-1]
    teeth = colors[light_order[0]]
    if luma[light_order[0]] < 145:
        teeth = np.clip(teeth * 0.35 + 255 * 0.65, 0, 255)

    # The casing is the most frequent non-extreme surface cluster. Falling
    # back to the most frequent cluster handles monochrome line art.
    eligible = [
        index
        for index in range(len(colors))
        if 35 <= luma[index] <= 225 and index not in {darkest_order[0], light_order[0]}
    ]
    casing_index = (
        max(eligible, key=lambda index: int(weights[index]))
        if eligible
        else int(np.argmax(weights))
    )
    casing = colors[casing_index]

    # Preserve a useful semantic distinction on nearly monochrome inputs.
    if np.linalg.norm(cavity - ink) < 8:
        cavity = _darken(ink, 0.58)
    if np.linalg.norm(casing - ink) < 12:
        casing = np.clip(casing * 0.65 + 255 * 0.35, 0, 255)

    return {
        "ink_outline": _hex(ink),
        "casing_color": _hex(casing),
        "cavity_interior": _hex(cavity),
        "teeth_color": _hex(teeth),
    }


__all__ = ["PuppetPalette", "extract_palette"]
