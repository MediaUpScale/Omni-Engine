# -*- coding: utf-8 -*-
"""Independent computer-vision ground truth for the render QA harness.

**This module must never import anything from the code under test.** Its
whole purpose is to define what a compliant frame looks like *from first
principles* — synthetic reference images drawn with raw ``numpy``/``cv2``
primitives — so the thresholds applied to a real render are not derived
from that render's own output. Calibrating a check against the thing it is
supposed to police is how a broken layout gets to approve itself, which is
exactly the failure mode this module exists to prevent.

It ships three kinds of artefact:

* **Positive references** — an idealised compliant frame for each check
  (single centred character, HUD nameplate, karaoke subtitle line).
* **A negative control** — the *old, rejected* split-screen layout. The
  harness asserts the checks FAIL on it, proving they have teeth.
* **Measurement primitives** — the identical functions are run over both
  the synthetic references and the real decoded MP4 frames, so the two are
  directly comparable.

:func:`assert_no_engine_imports` enforces the isolation rule at runtime by
inspecting this file's own source.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Canvas + zone contract (the spec's coordinates, restated independently)
# --------------------------------------------------------------------------- #
FRAME_W = 1080
FRAME_H = 1920

HUD_BAND = (50, 270)          # former banner region, now atmospheric headroom
BANNER_SIDE_X = ((110, 180), (900, 970))
HERO_BAND = (400, 1300)       # y range sampled for the hero character
SUBTITLE_BAND = (1360, 1530)  # elevated social-safe karaoke band around y=1440
TORSO_FOOT_BAND = (1840, 1910)  # torso must continue through the lower frame

HERO_X_WINDOW = (200, 880)    # x window the hero's mass must sit inside

# --------------------------------------------------------------------------- #
# Colour classification (OpenCV HSV: H is 0-179)
# --------------------------------------------------------------------------- #
CYAN_HUE = (80, 100)   # #00F0FF -> H ~ 90
AMBER_HUE = (10, 35)   # #FFB300 -> H ~ 21
SAT_MIN = 110
VAL_MIN = 110

CYAN_RGB = (0, 240, 255)
AMBER_RGB = (255, 179, 0)

# Near-white glyph classification for subtitle text density.
TEXT_VAL_MIN = 170
TEXT_SAT_MAX = 90


# --------------------------------------------------------------------------- #
# Measurement primitives
# --------------------------------------------------------------------------- #
def band(image: np.ndarray, y_range: tuple[int, int]) -> np.ndarray:
    return image[y_range[0] : y_range[1]]


def hue_mass(rgb: np.ndarray, hue_range: tuple[int, int]) -> int:
    """Count pixels whose hue falls in ``hue_range`` at real saturation."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= hue_range[0]) & (h <= hue_range[1]) & (s >= SAT_MIN) & (v >= VAL_MIN)
    return int(mask.sum())


def accent_mass(rgb: np.ndarray, colour: tuple[int, int, int], *, tolerance: int = 40) -> int:
    """Count H.264-tolerant pixels close to one exact caption accent."""
    delta = np.max(np.abs(rgb.astype(np.int16) - np.asarray(colour, dtype=np.int16)), axis=2)
    return int((delta <= tolerance).sum())


def banner_side_mass(rgb: np.ndarray, hue_range: tuple[int, int]) -> int:
    """Accent mass in the old banner's side rails, away from avatar crowns."""
    strip = band(rgb, HUD_BAND)
    return sum(hue_mass(strip[:, x0:x1], hue_range) for x0, x1 in BANNER_SIDE_X)


def foreground_mask(rgb: np.ndarray, *, floor_delta: int = 28) -> np.ndarray:
    """Boolean mask of "not background" pixels.

    The background floor is taken as the 5th percentile luminance of the
    band itself, so this works on both a synthetic pure-black reference and
    a real gradient-lit frame without either being special-cased.
    """
    luminance = rgb.max(axis=2).astype(np.int16)
    floor = float(np.percentile(luminance, 5))
    return luminance > (floor + floor_delta)


def x_concentration(mask: np.ndarray, x_window: tuple[int, int] = HERO_X_WINDOW) -> float:
    """Fraction of foreground mass sitting inside ``x_window``."""
    column_mass = mask.sum(axis=0)
    total = int(column_mass.sum())
    if total == 0:
        return 0.0
    return float(column_mass[x_window[0] : x_window[1]].sum()) / total


def blob_count(mask: np.ndarray, *, rel_threshold: float = 0.10, min_gap: int = 30) -> int:
    """Number of distinct horizontal clusters of foreground mass.

    A single centred character produces one cluster; the rejected
    side-by-side split screen produces two. Gaps narrower than ``min_gap``
    px are treated as part of the same subject (antialiasing, a thin arm
    gap), so this counts *characters*, not silhouette details.
    """
    column_mass = mask.sum(axis=0)
    peak = column_mass.max()
    if peak == 0:
        return 0
    occupied = column_mass > (peak * rel_threshold)

    clusters = 0
    gap_run = min_gap + 1
    for value in occupied:
        if value:
            if gap_run > min_gap:
                clusters += 1
            gap_run = 0
        else:
            gap_run += 1
    return clusters


def text_density(rgb: np.ndarray) -> float:
    """Fraction of pixels that read as bright, low-saturation glyph strokes."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    mask = (v >= TEXT_VAL_MIN) & (s <= TEXT_SAT_MAX)
    return float(mask.sum()) / float(mask.size)


# --------------------------------------------------------------------------- #
# Synthetic references — primitive drawing only, no engine code
# --------------------------------------------------------------------------- #
def _blank() -> np.ndarray:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def single_character_reference() -> np.ndarray:
    """One solid subject filling the hero window; pure black elsewhere."""
    img = _blank()
    cv2.rectangle(img, (HERO_X_WINDOW[0], HERO_BAND[0]), (HERO_X_WINDOW[1], HERO_BAND[1]), CYAN_RGB, -1)
    return img


def split_screen_negative_reference() -> np.ndarray:
    """NEGATIVE control: the rejected side-by-side layout (two subjects)."""
    img = _blank()
    cv2.rectangle(img, (60, HERO_BAND[0]), (470, HERO_BAND[1]), CYAN_RGB, -1)
    cv2.rectangle(img, (610, HERO_BAND[0]), (1020, HERO_BAND[1]), AMBER_RGB, -1)
    return img


def full_bleed_torso_reference() -> np.ndarray:
    """One centred torso continuing beyond the bottom frame edge."""
    img = _blank()
    cv2.rectangle(
        img,
        (HERO_X_WINDOW[0], HERO_BAND[0]),
        (HERO_X_WINDOW[1], FRAME_H - 1),
        CYAN_RGB,
        -1,
    )
    return img


def hud_reference(*, amber: bool = False) -> np.ndarray:
    """The rejected top banner, used as a negative reference."""
    img = _blank()
    colour = AMBER_RGB if amber else CYAN_RGB
    cv2.rectangle(img, (140, 76), (940, 236), colour, 5)
    cv2.rectangle(img, (154, 102), (172, 210), colour, -1)
    cv2.rectangle(img, (908, 102), (926, 210), colour, -1)
    cv2.rectangle(img, (120, 250), (960, 258), colour, -1)
    return img


def karaoke_reference(*, amber: bool = False) -> np.ndarray:
    """White word blocks plus one accent-coloured "highlighted word" block."""
    img = _blank()
    colour = AMBER_RGB if amber else CYAN_RGB
    block_w, block_h = 120, 70
    gap = 30
    n_blocks = 6
    total = n_blocks * block_w + (n_blocks - 1) * gap
    x = (FRAME_W - total) // 2
    y = (SUBTITLE_BAND[0] + SUBTITLE_BAND[1]) // 2 - block_h // 2
    for index in range(n_blocks):
        fill = colour if index == 2 else (255, 255, 255)
        left = x + index * (block_w + gap)
        cv2.rectangle(img, (left, y), (left + block_w, y + block_h), fill, -1)
    return img


# --------------------------------------------------------------------------- #
# Threshold derivation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Thresholds:
    """Pass/fail limits, every one derived from a synthetic reference.

    Each ``*_FACTOR`` is the documented tolerance applied to the reference
    measurement, and the reason it exists is recorded in
    :meth:`explain` so any later change has to justify itself.
    """

    banner_reference_mass: int
    banner_max_mass: float

    hero_reference_concentration: float
    hero_min_concentration: float
    hero_expected_blobs: int
    accent_reference_mass: int
    hero_opposite_accent_max: float
    torso_reference_mass: int
    torso_min_mass: float

    subtitle_reference_density: float
    subtitle_min_density: float
    subtitle_reference_highlight: int
    subtitle_min_highlight: float
    subtitle_opposite_highlight_max: float

    # -- Tolerance factors, stated once, applied everywhere ------------- #
    BANNER_MAX_FACTOR = 0.05
    HERO_CONCENTRATION_FACTOR = 0.80
    HERO_OPPOSITE_FACTOR = 0.006
    TORSO_MIN_FACTOR = 0.02
    SUBTITLE_DENSITY_FACTOR = 0.12
    SUBTITLE_HIGHLIGHT_FACTOR = 0.05
    SUBTITLE_OPPOSITE_FACTOR = 0.01

    def explain(self) -> list[str]:
        return [
            f"Former-banner side-rail accent mass <= {self.banner_max_mass:,.0f} px "
            f"({self.BANNER_MAX_FACTOR:.0%} of the {self.banner_reference_mass:,} px rejected-banner reference) — "
            "the top HUD must be absent while the center remains available for atmospheric headroom.",
            f"Hero x-concentration >= {self.hero_min_concentration:.2f} "
            f"({self.HERO_CONCENTRATION_FACTOR:.0%} of the {self.hero_reference_concentration:.2f} reference) — a real "
            "silhouette flares at the shoulders beyond the idealised box.",
            f"Hero cluster count == {self.hero_expected_blobs} — the single-subject reference yields 1, the "
            "split-screen negative control yields 2.",
            f"Hero opposite-accent mass <= {self.hero_opposite_accent_max:,.0f} px "
            f"({self.HERO_OPPOSITE_FACTOR:.1%} of the {self.accent_reference_mass:,} px solid-accent reference) — "
            "the absent character's accent colour "
            "must not appear at all.",
            f"Full-bleed torso foreground >= {self.torso_min_mass:,.0f} px in y{TORSO_FOOT_BAND} "
            f"({self.TORSO_MIN_FACTOR:.0%} of the {self.torso_reference_mass:,} px reference) — the body must exit "
            "through the bottom frame instead of ending on a horizontal crop.",
            f"Subtitle text density >= {self.subtitle_min_density:.4f} "
            f"({self.SUBTITLE_DENSITY_FACTOR:.0%} of the {self.subtitle_reference_density:.4f} solid-block reference) — "
            "glyph strokes cover roughly 15-30% of the block footprint they occupy.",
            f"Subtitle highlight mass >= {self.subtitle_min_highlight:,.0f} px "
            f"({self.SUBTITLE_HIGHLIGHT_FACTOR:.0%} of the {self.subtitle_reference_highlight:,} px reference block) — "
            "one highlighted word is far smaller than a solid reference block.",
            f"Subtitle opposite-highlight mass <= {self.subtitle_opposite_highlight_max:,.0f} px "
            f"({self.SUBTITLE_OPPOSITE_FACTOR:.0%} of reference) — the inactive speaker's colour must be absent.",
        ]


def derive_thresholds() -> Thresholds:
    """Measure the synthetic references and turn them into pass/fail limits."""
    banner_ref = banner_side_mass(hud_reference(), CYAN_HUE)
    accent_ref = FRAME_W * (HUD_BAND[1] - HUD_BAND[0])

    hero_ref = single_character_reference()
    hero_conc = x_concentration(foreground_mask(band(hero_ref, HERO_BAND)))
    torso_mass = int(
        foreground_mask(band(full_bleed_torso_reference(), TORSO_FOOT_BAND)).sum()
    )

    kara_ref = karaoke_reference()
    kara_band = band(kara_ref, SUBTITLE_BAND)
    kara_density = text_density(kara_band)
    kara_highlight = hue_mass(kara_band, CYAN_HUE)

    t = Thresholds
    return Thresholds(
        banner_reference_mass=banner_ref,
        banner_max_mass=banner_ref * t.BANNER_MAX_FACTOR,
        hero_reference_concentration=hero_conc,
        hero_min_concentration=hero_conc * t.HERO_CONCENTRATION_FACTOR,
        hero_expected_blobs=blob_count(foreground_mask(band(hero_ref, HERO_BAND))),
        accent_reference_mass=accent_ref,
        hero_opposite_accent_max=accent_ref * t.HERO_OPPOSITE_FACTOR,
        torso_reference_mass=torso_mass,
        torso_min_mass=torso_mass * t.TORSO_MIN_FACTOR,
        subtitle_reference_density=kara_density,
        subtitle_min_density=kara_density * t.SUBTITLE_DENSITY_FACTOR,
        subtitle_reference_highlight=kara_highlight,
        subtitle_min_highlight=kara_highlight * t.SUBTITLE_HIGHLIGHT_FACTOR,
        subtitle_opposite_highlight_max=kara_highlight * t.SUBTITLE_OPPOSITE_FACTOR,
    )


# --------------------------------------------------------------------------- #
# Isolation guard
# --------------------------------------------------------------------------- #
_FORBIDDEN_IMPORT_TOKENS = ("core.animator", "animator_bridge", "compositor", "puppet", "subtitles")


def assert_no_engine_imports() -> None:
    """Fail loudly if this module ever starts importing the code it judges."""
    source = Path(__file__).read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from ")) and "__future__" not in line
    ]
    for line in import_lines:
        for token in _FORBIDDEN_IMPORT_TOKENS:
            if token in line:
                raise AssertionError(
                    "cv_ground_truth must stay independent of the code under test, "
                    f"but it imports {token!r}: {line!r}"
                )


__all__ = [
    "AMBER_HUE",
    "CYAN_HUE",
    "HERO_BAND",
    "HERO_X_WINDOW",
    "HUD_BAND",
    "SUBTITLE_BAND",
    "TORSO_FOOT_BAND",
    "Thresholds",
    "assert_no_engine_imports",
    "accent_mass",
    "band",
    "banner_side_mass",
    "blob_count",
    "derive_thresholds",
    "foreground_mask",
    "full_bleed_torso_reference",
    "hud_reference",
    "hue_mass",
    "karaoke_reference",
    "single_character_reference",
    "split_screen_negative_reference",
    "text_density",
    "x_concentration",
]
