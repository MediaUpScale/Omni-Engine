"""One-click landmark analysis and inspection for isolated puppet heads.

The analyzer works entirely in head-local pixels.  Transparent alpha gives
the head bounds, circular contrast gives the optical apertures, and the warm
lower connected component gives the chin basin.  No per-character landmark
coordinates are embedded here.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from utils.pipeline_paths import assets_root, outputs_root

from ..render.facial_rig import (
    SCENE_GRAPH_BROW_ANGLES,
    compose_scene_graph_head,
)
from ..render.skia_mouths import (
    ARCHETYPES,
    VIEW_FOLDERS,
    archetype_for,
    generate_mouth_suite,
    generate_view_mouth_suites,
)

# Defaults for any later chassis, including a DeepSeek puppet.
FAR_EYE_FORESHORTEN_RATIO = 0.82
BROW_VERTICAL_CLEARANCE_PX = 6
# Screen-space jaw seat after the shared chin reanchor.
# dx is viewer's right-positive. dy is downward-positive.
CHASSIS_JAW_PROTRUSION_PX = {
    "cyber_capsule": {
        "facing_front": (0, 0),
        "facing_left": (4, 0),
        "facing_right": (-4, 0),
    },
    "organic_brass_cutout": {
        "facing_front": (0, 5),
        "facing_left": (-7, 5),
        "facing_right": (7, 5),
    },
    "industrial_louver": {
        "facing_front": (0, 5),
        "facing_left": (-7, 5),
        "facing_right": (7, 5),
    },
}

VIEW_ORDER = ("facing_front", "facing_left", "facing_right")
VIEW_LABELS = {
    "facing_front": "FRONT VIEW",
    "facing_left": "FACING LEFT",
    "facing_right": "FACING RIGHT",
}
MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "master_head_rig_inspection.png"
)
V3_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "master_head_rig_inspection_v3.png"
)
V4_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v4_universal_rig"
    / "master_head_rig_inspection_v4.png"
)
V5_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v5_final_calibration"
    / "master_head_rig_inspection_v5.png"
)
V6_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v6_master_approved"
    / "master_head_rig_inspection_v6.png"
)
V6_VIDEO = V6_MASTER_SHEET.parent / "chatgpt_vs_claude_v6_final_test.mp4"
V7_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v7_pixel_perfect"
    / "master_head_rig_inspection_v7.png"
)
V7_VIDEO = V7_MASTER_SHEET.parent / "chatgpt_vs_claude_v7_final.mp4"
V8_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v8_golden_master"
    / "master_head_rig_inspection_v8.png"
)
V8_VIDEO = V8_MASTER_SHEET.parent / "chatgpt_vs_claude_v8_golden.mp4"
V9_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v9_production_final"
    / "master_head_rig_inspection_v9.png"
)
V9_VIDEO = V9_MASTER_SHEET.parent / "chatgpt_vs_claude_v9_production.mp4"
V10_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v10_release_candidate"
    / "master_head_rig_inspection_v10.png"
)
V10_VIDEO = V10_MASTER_SHEET.parent / "chatgpt_vs_claude_v10_release.mp4"
V11_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v11_gold_master"
    / "master_head_rig_inspection_v11.png"
)
V11_VIDEO = V11_MASTER_SHEET.parent / "chatgpt_vs_claude_v11_master.mp4"
V12_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v12_definitive_release"
    / "master_head_rig_inspection_v12.png"
)
V12_VIDEO = V12_MASTER_SHEET.parent / "chatgpt_vs_claude_v12_definitive.mp4"
V13_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v13_golden_seal"
    / "master_head_rig_inspection_v13.png"
)
V13_VIDEO = V13_MASTER_SHEET.parent / "chatgpt_vs_claude_v13_final.mp4"
V14_MASTER_SHEET = (
    outputs_root()
    / "aiwake"
    / "_test_harness"
    / "v14_master_signoff"
    / "master_head_rig_inspection_v14.png"
)
V14_VIDEO = V14_MASTER_SHEET.parent / "chatgpt_vs_claude_v14_master.mp4"


@dataclass(frozen=True, slots=True)
class EyeLandmark:
    center: tuple[int, int]
    radius: int

    @property
    def top(self) -> int:
        return self.center[1] - self.radius


@dataclass(frozen=True, slots=True)
class HeadAnalysis:
    eye_left: EyeLandmark
    eye_right: EyeLandmark
    eye_radius: int
    nameplate_bottom: int
    mouth_center: tuple[int, int]
    mouth_rotation_deg: float
    mouth_target_width: int
    perspective_scale: float
    chin_bbox: tuple[int, int, int, int]
    eyebrow_y: int

    def scene_graph(
        self,
        blink_asset: str,
        half_blink_asset: str,
    ) -> dict:
        """Serializable parent-child graph in head-local coordinates."""
        brow_width = max(
            40,
            int(round(max(self.eye_left.radius, self.eye_right.radius) * 2.25)),
        )
        return {
            "root": "head",
            "head": {"parent": None, "space": "head_local"},
            "eyes_lids": {
                "parent": "head",
                "asset": blink_asset,
                "full_asset": blink_asset,
                "half_asset": half_blink_asset,
                "left": {
                    "center": list(self.eye_left.center),
                    "radius": self.eye_left.radius,
                },
                "right": {
                    "center": list(self.eye_right.center),
                    "radius": self.eye_right.radius,
                },
            },
            "eyebrows": {
                "parent": "head",
                "level": True,
                "angles": {
                    key: list(value)
                    for key, value in SCENE_GRAPH_BROW_ANGLES.items()
                },
                "left": {
                    "center": [self.eye_left.center[0], self.eyebrow_y],
                    "width": brow_width,
                },
                "right": {
                    "center": [self.eye_right.center[0], self.eyebrow_y],
                    "width": brow_width,
                },
            },
            "mouth": {
                "parent": "head",
                "center": list(self.mouth_center),
                "rot_deg": round(self.mouth_rotation_deg, 3),
                "target_width": self.mouth_target_width,
                "perspective_scale": round(self.perspective_scale, 4),
                "chin_bbox": list(self.chin_bbox),
            },
        }


@dataclass(frozen=True, slots=True)
class _Circle:
    x: float
    y: float
    radius: float
    rank: int
    lens_signal: float


def _opaque_bbox(rgba: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(rgba[..., 3] > 16)
    if xs.size == 0:
        raise RuntimeError("head sprite has no opaque pixels")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _lens_signal(
    hsv: np.ndarray,
    x_pos: float,
    y_pos: float,
    radius: float,
) -> float:
    """Brightness/chroma score from the central optical glass."""
    height, width = hsv.shape[:2]
    reach = max(5, int(round(radius * 0.38)))
    x0, x1 = max(0, int(x_pos) - reach), min(width, int(x_pos) + reach + 1)
    y0, y1 = max(0, int(y_pos) - reach), min(height, int(y_pos) + reach + 1)
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    cy, cx = (patch.shape[0] - 1) / 2.0, (patch.shape[1] - 1) / 2.0
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= max(1.0, reach * 0.85) ** 2
    values = patch[..., 2][mask]
    saturation = patch[..., 1][mask]
    if values.size == 0:
        return 0.0
    bright = float(np.percentile(values, 82)) / 255.0
    chroma = float(np.percentile(saturation, 65)) / 255.0
    hot = float(np.mean((values > 185) & (saturation > 45)))
    return bright + chroma * 0.55 + hot * 0.65


def _circle_candidates(rgba: np.ndarray) -> list[_Circle]:
    height, width = rgba.shape[:2]
    rgb = rgba[..., :3]
    gray = cv2.GaussianBlur(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
        (0, 0),
        2.0,
    )
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(52, int(round(width * 0.095))),
        param1=80,
        param2=28,
        minRadius=max(22, int(round(height * 0.032))),
        maxRadius=max(34, int(round(height * 0.115))),
    )
    if circles is None:
        return []
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidates: list[_Circle] = []
    for rank, (x_pos, y_pos, radius) in enumerate(circles[0]):
        # Isolated heads place optical apertures in the central face band.
        # A tighter band rejects forehead bolts and lower jaw/ear dials.
        if not (0.35 * height < y_pos < 0.58 * height):
            continue
        if not (0.025 * width < x_pos < 0.975 * width):
            continue
        candidates.append(
            _Circle(
                float(x_pos),
                float(y_pos),
                float(radius),
                rank,
                _lens_signal(hsv, float(x_pos), float(y_pos), float(radius)),
            )
        )
    return candidates


def _color_eye_pair(
    rgba: np.ndarray,
    view_name: str | None,
) -> tuple[EyeLandmark, EyeLandmark] | None:
    """Pair cyan cores or amber glass components before the Hough fallback."""
    rgb = rgba[..., :3]
    height, width = rgb.shape[:2]
    cyan = (
        (rgb[..., 1] > 150)
        & (rgb[..., 2] > 125)
        & (rgb[..., 0] < 205)
        & (rgb[..., 1].astype(np.int16) > rgb[..., 0].astype(np.int16) + 18)
    )
    amber = (
        (rgb[..., 0] > 205)
        & (rgb[..., 1] > 140)
        & (rgb[..., 2] < 155)
        & (rgb[..., 0].astype(np.int16) > rgb[..., 2].astype(np.int16) + 58)
    )
    mask = (cyan | amber) & (rgba[..., 3] > 16)
    mask[: int(height * 0.32), :] = False
    mask[int(height * 0.63) :, :] = False
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    blobs: list[tuple[int, float, float, int, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        blob_w = int(stats[index, cv2.CC_STAT_WIDTH])
        blob_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        x_pos, y_pos = centroids[index]
        if area < 100 or blob_w < 10 or blob_h < 10:
            continue
        if blob_w > width * 0.24 or blob_h > height * 0.22:
            continue
        blobs.append((area, float(x_pos), float(y_pos), blob_w, blob_h))
    midpoint_targets = {
        "facing_left": width * 0.30,
        "facing_right": width * 0.70,
    }
    midpoint_target = midpoint_targets.get(view_name, width * 0.50)
    best: tuple[
        float,
        tuple[int, float, float, int, int],
        tuple[int, float, float, int, int],
    ] | None = None
    for index, first in enumerate(blobs):
        for second in blobs[index + 1 :]:
            left, right = sorted((first, second), key=lambda blob: blob[1])
            separation = right[1] - left[1]
            vertical = abs(right[2] - left[2])
            if not 0.20 * width <= separation <= 0.50 * width:
                continue
            if vertical > 0.095 * height:
                continue
            midpoint = (left[1] + right[1]) * 0.5
            target_separation = width * (
                0.30
                if view_name in {"facing_left", "facing_right"}
                else 0.36
            )
            shape_delta = abs(
                max(left[3], left[4]) - max(right[3], right[4])
            ) / max(max(left[3], left[4]), max(right[3], right[4]), 1)
            score = (
                abs(separation - target_separation) / width * 3.0
                + vertical / height * 4.0
                + abs(midpoint - midpoint_target) / width * 5.0
                + shape_delta * 0.35
                - min(left[0] + right[0], 2500) / 2500.0 * 0.18
            )
            if best is None or score < best[0]:
                best = (score, left, right)
    if best is None:
        return None

    def landmark(blob: tuple[int, float, float, int, int]) -> EyeLandmark:
        _area, x_pos, y_pos, blob_w, blob_h = blob
        diameter = max(blob_w, blob_h)
        # ChatGPT exposes a small cyan emitter inside a larger bezel; Claude's
        # amber component fills almost the entire optical aperture.
        radius = (
            diameter * 2.05
            if diameter < width * 0.075
            else diameter * 0.60
        )
        radius = float(np.clip(radius, height * 0.055, height * 0.105))
        return EyeLandmark(
            (int(round(x_pos)), int(round(y_pos))),
            int(round(radius)),
        )

    _, left_blob, right_blob = best
    return landmark(left_blob), landmark(right_blob)


def _pair_eyes(
    rgba: np.ndarray,
    view_name: str | None = None,
) -> tuple[EyeLandmark, EyeLandmark]:
    """Choose the two optical apertures from circular contrast candidates."""
    color_pair = _color_eye_pair(rgba, view_name)
    if color_pair is not None:
        # Amber glass often contains a broad highlight that pulls a component
        # centroid sideways. Snap only amber centers to a nearby circular edge;
        # cyan emitters are already precise and remain untouched.
        circles = _circle_candidates(rgba)
        refined: list[EyeLandmark] = []
        for landmark in color_pair:
            x_pos, y_pos = landmark.center
            pixel = rgba[y_pos, x_pos, :3].astype(np.int16)
            is_amber = pixel[0] > pixel[2] + 48 and pixel[0] > 180
            if not is_amber:
                refined.append(landmark)
                continue
            nearby = sorted(
                circles,
                key=lambda circle: math.hypot(
                    circle.x - x_pos,
                    circle.y - y_pos,
                ),
            )
            snapped = landmark
            for circle in nearby:
                distance = math.hypot(circle.x - x_pos, circle.y - y_pos)
                radius_ratio = circle.radius / max(landmark.radius, 1)
                if (
                    distance <= landmark.radius * 0.95
                    and 0.70 <= radius_ratio <= 1.55
                ):
                    snapped = EyeLandmark(
                        (int(round(circle.x)), int(round(circle.y))),
                        landmark.radius,
                    )
                    break
            refined.append(snapped)
        return refined[0], refined[1]
    height, width = rgba.shape[:2]
    candidates = _circle_candidates(rgba)
    best: tuple[float, _Circle, _Circle] | None = None
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            left, right = sorted((first, second), key=lambda circle: circle.x)
            separation = right.x - left.x
            if not 0.22 * width <= separation <= 0.52 * width:
                continue
            vertical = abs(right.y - left.y)
            if vertical > 0.105 * height:
                continue
            target_separation = width * 0.35
            radius_delta = abs(right.radius - left.radius) / max(
                right.radius,
                left.radius,
                1.0,
            )
            # Lower is better. Hough rank is only a tie-breaker; glass signal
            # prevents ear dials and speaker grilles from winning.
            midpoint_targets = {
                "facing_left": width * 0.30,
                "facing_right": width * 0.70,
            }
            midpoint_target = midpoint_targets.get(view_name, width * 0.50)
            midpoint = (left.x + right.x) * 0.5
            score = (
                abs(separation - target_separation) / width * 3.2
                + vertical / height * 4.0
                + abs(midpoint - midpoint_target) / width * 4.2
                + radius_delta * 1.20
                + (left.rank + right.rank) * 0.040
                - (left.lens_signal + right.lens_signal) * 0.72
            )
            if best is None or score < best[0]:
                best = (score, left, right)
    if best is None:
        raise RuntimeError("could not pair two circular eye apertures")
    _, left, right = best
    return (
        EyeLandmark(
            (int(round(left.x)), int(round(left.y))),
            int(round(left.radius)),
        ),
        EyeLandmark(
            (int(round(right.x)), int(round(right.y))),
            int(round(right.radius)),
        ),
    )


def _nameplate_bottom(
    rgba: np.ndarray,
    left: EyeLandmark,
    right: EyeLandmark,
) -> int:
    """Lowest strong horizontal edge above the optical top rims."""
    height, width = rgba.shape[:2]
    eye_top = min(left.top, right.top)
    x0 = max(0, min(left.center[0], right.center[0]) - int(width * 0.08))
    x1 = min(width, max(left.center[0], right.center[0]) + int(width * 0.08))
    y0 = int(height * 0.10)
    y1 = max(y0 + 8, eye_top - 8)
    band = rgba[y0:y1, x0:x1, :3].astype(np.int16)
    if band.shape[0] < 4:
        return max(0, eye_top - 36)
    delta = np.abs(np.diff(band, axis=0)).mean(axis=(1, 2))
    smooth = np.convolve(delta, np.ones(5) / 5.0, mode="same")
    peak = float(smooth.max()) if smooth.size else 0.0
    rows = [
        y0 + int(index)
        for index, value in enumerate(smooth)
        if value >= peak * 0.62
    ]
    return max(rows) if rows else max(0, eye_top - 36)


def _chin_basin(rgba: np.ndarray) -> tuple[int, int, int, int]:
    """Largest warm connected component below the eye region."""
    alpha = rgba[..., 3]
    x0, y0, x1, y1 = _opaque_bbox(rgba)
    height = y1 - y0
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    warm = (
        (hsv[..., 0] >= 3)
        & (hsv[..., 0] <= 42)
        & (hsv[..., 1] >= 34)
        & (hsv[..., 2] >= 60)
        & (alpha > 16)
    )
    warm[: y0 + int(round(height * 0.53)), :] = False
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        warm.astype(np.uint8),
        connectivity=8,
    )
    choices: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area < max(200, int(width * component_height * 0.12)):
            continue
        choices.append((area, left, top, left + width, top + component_height))
    if choices:
        _, left, top, right, bottom = max(choices)
        return left, top, right, bottom
    return x0, y0 + int(round(height * 0.58)), x1, y1


def analyze_head(
    head: Image.Image,
    *,
    view_name: str | None = None,
    mouth_scale: float = 1.0,
    reanchor_chin: bool = False,
    v4_universal_rig: bool = False,
    v6_master_approved: bool = False,
) -> HeadAnalysis:
    """Infer eyes, level brows, chin basin, mouth yaw and perspective."""
    rgba = np.asarray(head.convert("RGBA"))
    left, right = _pair_eyes(rgba, view_name)
    plate_bottom = _nameplate_bottom(rgba, left, right)
    chin = _chin_basin(rgba)
    chin_left, chin_top, chin_right, chin_bottom = chin
    eye_baseline = (left.center[1] + right.center[1]) / 2.0
    mouth_x = int(round((chin_left + chin_right) / 2.0))
    proposed_y = chin_top + (chin_bottom - chin_top) * 0.30
    mouth_y = int(round(max(eye_baseline + 0.08 * head.height, proposed_y)))
    chin_width = chin_right - chin_left
    is_side = view_name in {"facing_left", "facing_right"}
    if reanchor_chin:
        mouth_y += 36
        turn_shift = max(40, int(round(chin_width * 0.11)))
        if view_name == "facing_left":
            mouth_x -= turn_shift
        elif view_name == "facing_right":
            mouth_x += turn_shift
    mouth_x = int(np.clip(mouth_x, chin_left + 12, chin_right - 12))
    mouth_y = min(mouth_y, chin_bottom - max(8, int(head.height * 0.04)))
    if v4_universal_rig:
        mouth_y += 14
        if view_name == "facing_left":
            mouth_x -= 28
            mouth_y += 16
        elif view_name == "facing_right":
            mouth_x += 28
            mouth_y += 16
    tilt = math.degrees(
        math.atan2(
            right.center[1] - left.center[1],
            right.center[0] - left.center[0],
        )
    )
    tilt = float(np.clip(tilt, -8.0, 8.0))
    perspective = min(left.radius, right.radius) / max(
        left.radius,
        right.radius,
        1,
    )
    if is_side and reanchor_chin:
        perspective *= 0.85
    common_top = min(left.top, right.top)
    # Both brows share exactly one Y coordinate. The lower bound protects the
    # forehead plate, while the upper bound seats the strokes on the rims.
    brow_y = max(plate_bottom + 8, common_top - 7)
    brow_y = min(brow_y, common_top - 2)
    if v6_master_approved:
        brow_y = min(brow_y, common_top - BROW_VERTICAL_CLEARANCE_PX)
    return HeadAnalysis(
        eye_left=left,
        eye_right=right,
        eye_radius=int(round((left.radius + right.radius) / 2.0)),
        nameplate_bottom=plate_bottom,
        mouth_center=(mouth_x, mouth_y),
        mouth_rotation_deg=tilt,
        mouth_target_width=max(
            44,
            int(
                round(
                    chin_width
                    * 0.75
                    * float(np.clip(mouth_scale, 0.25, 1.50))
                    * (0.85 if is_side and reanchor_chin else 1.0)
                )
            ),
        ),
        perspective_scale=float(perspective),
        chin_bbox=chin,
        eyebrow_y=int(brow_y),
    )


def _lid_color(
    head: Image.Image,
    analysis: HeadAnalysis,
    puppet_id: str,
) -> tuple[int, int, int, int]:
    """Chassis-matched shutter metal, never generic dead slate."""
    if "chatgpt" in puppet_id.lower():
        return 234, 217, 184, 255  # warm ivory chassis #ead9b8
    rgba = np.asarray(head.convert("RGBA"))
    samples: list[np.ndarray] = []
    for eye in (analysis.eye_left, analysis.eye_right):
        radius = eye.radius
        x0 = max(0, eye.center[0] - radius)
        x1 = min(head.width, eye.center[0] + radius)
        y0 = max(0, eye.center[1] - radius)
        y1 = max(y0 + 1, eye.center[1] - int(radius * 0.35))
        patch = rgba[y0:y1, x0:x1, :3]
        if patch.size:
            samples.append(patch.reshape(-1, 3))
    if not samples:
        return 76, 48, 30, 255
    color = np.median(np.concatenate(samples, axis=0), axis=0)
    bronze = np.clip(color * 0.68, 28, 174).astype(np.uint8)
    return int(bronze[0]), int(bronze[1]), int(bronze[2]), 255


def _far_side(analysis: HeadAnalysis, view_name: str | None) -> str | None:
    """Receding socket on a three-quarter head: the smaller lens."""
    if view_name not in {"facing_left", "facing_right"}:
        return None
    if analysis.eye_left.radius + 3 < analysis.eye_right.radius:
        return "left"
    if analysis.eye_right.radius + 3 < analysis.eye_left.radius:
        return "right"
    return "left" if view_name == "facing_left" else "right"


def _horizon_tilt(view_name: str | None, side: str, far: str | None) -> float:
    """Tip the far brow onto the receding facial plane."""
    if side != far:
        return 0.0
    if view_name == "facing_left":
        return -7.0
    if view_name == "facing_right":
        return 7.0
    return 0.0


def _v13_lid_adjust(
    puppet_id: str,
    view_name: str | None,
    side: str,
    *,
    half: bool,
) -> tuple[float, float, float]:
    """Golden-seal lid seat: (x, y, size scale) on top of the V11 socket."""
    lowered = puppet_id.lower()
    if "chatgpt" in lowered and view_name == "facing_left":
        if half:
            return -3.0, 0.0, 1.0
        if side == "left":
            return -2.0, 0.0, 1.0
    if "chatgpt" in lowered and view_name == "facing_right" and side == "right":
        return (3.0 if half else 2.0), 0.0, 1.0
    if "claude" in lowered and view_name == "facing_front":
        if not half:
            if side == "left":
                return 1.0, 2.0, 1.02
            return 0.0, 0.0, 1.02
        if side == "left":
            return 2.0, 0.0, 1.01
        return 1.0, -1.0, 1.01
    if (
        "claude" in lowered
        and view_name == "facing_left"
        and side == "left"
        and not half
    ):
        return -2.0, 0.0, 1.0
    if "claude" in lowered and view_name == "facing_right" and side == "right":
        if half:
            return 3.0, 0.0, 1.02
        return 4.0, 0.0, 1.03
    return 0.0, 0.0, 1.0


def _v11_lid_adjust(
    puppet_id: str,
    view_name: str | None,
    side: str,
    *,
    half: bool,
) -> tuple[float, float, float]:
    """Gold-master lid seat: (x, y, size scale) on top of the V10 socket."""
    lowered = puppet_id.lower()
    if "chatgpt" in lowered and view_name == "facing_front":
        if side == "left":
            return -2.0, 0.0, 1.0
        if side == "right":
            return 2.0, 0.0, 1.0
    if "claude" in lowered and view_name == "facing_front":
        if side == "left" and half:
            return 2.0, 0.0, 1.02
        if side == "left":
            return 2.0, 1.0, 1.02
        return 0.0, 0.0, 1.02
    if (
        "claude" in lowered
        and view_name == "facing_left"
        and side == "left"
        and not half
    ):
        return -2.0, 0.0, 1.02
    if "claude" in lowered and view_name == "facing_right" and side == "right":
        if half:
            return 3.0, 0.0, 1.02
        return 4.0, 0.0, 1.03
    return 0.0, 0.0, 1.0


def _v10_lid_shift(
    puppet_id: str,
    view_name: str | None,
    side: str,
    *,
    half: bool,
) -> tuple[float, float]:
    """Sub-pixel socket seat. Claude's front nudge is full-blink only."""
    lowered = puppet_id.lower()
    if "chatgpt" in lowered:
        if view_name == "facing_front" and side == "left":
            return -2.0, 0.0
        if view_name == "facing_front" and side == "right":
            return 2.0, 0.0
        if view_name == "facing_left" and side == "left":
            return -3.0, 0.0
        if view_name == "facing_left" and side == "right":
            return -1.0, 0.0
        if view_name == "facing_right" and side == "right":
            return 3.0, 0.0
    elif "claude" in lowered:
        if view_name == "facing_front" and side == "left" and not half:
            return 1.0, 1.0
        if view_name == "facing_left" and side == "left":
            return -2.0, 0.0
        if view_name == "facing_right" and side == "right":
            return 3.0, 0.0
    return 0.0, 0.0


def _v9_lid_adjust(
    puppet_id: str,
    view_name: str | None,
    side: str,
) -> tuple[float, float, float]:
    """Extra socket shift, then a size scale applied to the finished lid."""
    lowered = puppet_id.lower()
    if "chatgpt" in lowered and view_name == "facing_left" and side == "left":
        return -6.0, 0.0, 1.0
    if "chatgpt" in lowered and view_name == "facing_right" and side == "right":
        return 6.0, 0.0, 1.0
    if "claude" in lowered and view_name == "facing_left" and side == "left":
        return -6.0, 0.0, 1.10
    if "claude" in lowered and view_name == "facing_right" and side == "right":
        return 6.0, 0.0, 1.15
    return 0.0, 0.0, 1.0


def _v8_lid_nudge(
    puppet_id: str,
    view_name: str | None,
    side: str,
) -> tuple[float, float, float]:
    """Screen-space lid seat for the golden master. Half and full share it."""
    lowered = puppet_id.lower()
    if "chatgpt" in lowered and view_name == "facing_left":
        return 0.0, -4.0, 0.0
    if "chatgpt" in lowered and view_name == "facing_right":
        return 0.0, 4.0, 0.0
    if "claude" in lowered and view_name == "facing_left" and side == "left":
        return 0.0, -4.0, 0.0
    if "claude" in lowered and view_name == "facing_right" and side == "right":
        return 2.0, 5.0, 0.0
    return 0.0, 0.0, 0.0


def _v7_lid_nudge(
    puppet_id: str,
    view_name: str | None,
    side: str,
) -> tuple[float, float, float]:
    """Claude's far-right lens: wider lid, seated on the aperture."""
    if (
        "claude" in puppet_id.lower()
        and view_name == "facing_right"
        and side == "right"
    ):
        return 3.0, 4.0, -3.0
    return 0.0, 0.0, 0.0


def _v6_half_nudge(
    puppet_id: str,
    view_name: str | None,
    side: str,
) -> tuple[float, float, float]:
    """Half-blink only: (half-width delta, x shift, y shift) in screen pixels."""
    lowered = puppet_id.lower()
    if "chatgpt" in lowered and view_name == "facing_left" and side == "left":
        return -2.5, -4.0, 0.0
    if "chatgpt" in lowered and view_name == "facing_right" and side == "right":
        return -2.5, 4.0, 0.0
    if "claude" in lowered and view_name == "facing_right" and side == "right":
        return 3.0, 0.0, -5.0
    if "claude" in lowered and view_name == "facing_left" and side == "left":
        return 3.0, 0.0, -5.0
    return 0.0, 0.0, 0.0


def _lid_overlay(
    head: Image.Image,
    analysis: HeadAnalysis,
    fill: tuple[int, int, int, int],
    *,
    half: bool,
    view_name: str | None = None,
    claude_refit: bool = False,
    far_ratio: float = 0.78,
    half_profile: str | None = None,
    puppet_id: str = "",
) -> Image.Image:
    """Build a shutter clipped inside each socket. The far eye is narrower."""
    scale = 4
    overlay = Image.new(
        "RGBA",
        (head.width * scale, head.height * scale),
        (0, 0, 0, 0),
    )
    outline = (18, 21, 28, 255)
    far = _far_side(analysis, view_name)
    eyes = (("left", analysis.eye_left), ("right", analysis.eye_right))
    for side, eye in eyes:
        socket_scale = far_ratio if side == far else 1.0
        layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        x_pos, y_pos = float(eye.center[0]), float(eye.center[1])
        radius = eye.radius
        half_w = radius * socket_scale
        if claude_refit and view_name == "facing_left" and side == "left":
            half_w += 2.0
            x_pos -= 5.0
        elif claude_refit and view_name == "facing_right" and side == "right":
            half_w += 2.0
            x_pos += 5.0
        if half and half_profile in {"v6", "v7", "v8", "v9", "v10", "v11", "v13"}:
            nudge_w, nudge_x, nudge_y = _v6_half_nudge(
                puppet_id,
                view_name,
                side,
            )
            half_w += nudge_w
            x_pos += nudge_x
            y_pos += nudge_y
        if half_profile in {"v7", "v8", "v9", "v10", "v11", "v13"}:
            extra_w, extra_x, extra_y = _v7_lid_nudge(
                puppet_id,
                view_name,
                side,
            )
            half_w += extra_w
            x_pos += extra_x
            y_pos += extra_y
        if half_profile in {"v8", "v9", "v10", "v11", "v13"}:
            extra_w, extra_x, extra_y = _v8_lid_nudge(
                puppet_id,
                view_name,
                side,
            )
            half_w += extra_w
            x_pos += extra_x
            y_pos += extra_y
        if half_profile in {"v9", "v10", "v11", "v13"}:
            nudge_x, nudge_y, size_scale = _v9_lid_adjust(
                puppet_id,
                view_name,
                side,
            )
            x_pos += nudge_x
            y_pos += nudge_y
            half_w *= size_scale
            radius *= size_scale
        if half_profile in {"v10", "v11", "v13"}:
            nudge_x, nudge_y = _v10_lid_shift(
                puppet_id,
                view_name,
                side,
                half=half,
            )
            x_pos += nudge_x
            y_pos += nudge_y
        if half_profile in {"v11", "v13"}:
            nudge_x, nudge_y, size_scale = _v11_lid_adjust(
                puppet_id,
                view_name,
                side,
                half=half,
            )
            x_pos += nudge_x
            y_pos += nudge_y
            half_w *= size_scale
            radius *= size_scale
        if half_profile == "v13":
            nudge_x, nudge_y, size_scale = _v13_lid_adjust(
                puppet_id,
                view_name,
                side,
                half=half,
            )
            x_pos += nudge_x
            y_pos += nudge_y
            half_w *= size_scale
            radius *= size_scale
        box = tuple(
            int(round(value * scale))
            for value in (
                x_pos - half_w,
                y_pos - radius,
                x_pos + half_w,
                y_pos + radius,
            )
        )
        if half:
            eye_mask = Image.new("L", overlay.size, 0)
            ImageDraw.Draw(eye_mask).ellipse(box, fill=255)
            drape_mask = Image.new("L", overlay.size, 0)
            drape_draw = ImageDraw.Draw(drape_mask)
            left_x = (x_pos - half_w) * scale
            right_x = (x_pos + half_w) * scale
            top_y = (y_pos - radius) * scale
            base_y = (y_pos - radius * 0.12) * scale
            curve = []
            for index in range(17):
                progress = index / 16.0
                norm = progress * 2.0 - 1.0
                curve.append(
                    (
                        left_x + (right_x - left_x) * progress,
                        base_y + radius * scale * 0.14 * (1.0 - norm * norm),
                    )
                )
            drape_draw.polygon(
                [(left_x, top_y), (right_x, top_y), *reversed(curve)],
                fill=255,
            )
            alpha = ImageChops.multiply(eye_mask, drape_mask)
            lid_layer = Image.new("RGBA", overlay.size, fill)
            lid_layer.putalpha(alpha)
            layer.alpha_composite(lid_layer)
            draw = ImageDraw.Draw(layer, "RGBA")
            draw.arc(box, 180, 360, fill=outline, width=3 * scale)
            draw.line(curve, fill=outline, width=3 * scale, joint="curve")
        else:
            draw.ellipse(box, fill=fill, outline=outline, width=3 * scale)
            seam_y = int(round(y_pos * scale))
            draw.line(
                (
                    int(round((x_pos - half_w * 0.86) * scale)),
                    seam_y,
                    int(round((x_pos + half_w * 0.86) * scale)),
                    seam_y,
                ),
                fill=outline,
                width=3 * scale,
            )
        socket = Image.new("L", overlay.size, 0)
        ImageDraw.Draw(socket).ellipse(box, fill=255)
        layer.putalpha(ImageChops.multiply(layer.getchannel("A"), socket))
        overlay.alpha_composite(layer)
    return overlay.resize(head.size, Image.Resampling.LANCZOS)


def generate_blink_overlays(
    head: Image.Image,
    analysis: HeadAnalysis,
    puppet_id: str,
    full_destination: Path,
    half_destination: Path,
    *,
    view_name: str | None = None,
    claude_refit: bool = False,
    far_ratio: float = 0.78,
    half_profile: str | None = None,
) -> tuple[Path, Path]:
    """Write matched full and half-closed eyelids for one head view."""
    fill = _lid_color(head, analysis, puppet_id)
    full = _lid_overlay(
        head,
        analysis,
        fill,
        half=False,
        view_name=view_name,
        claude_refit=claude_refit,
        far_ratio=far_ratio,
        half_profile=half_profile,
        puppet_id=puppet_id,
    )
    half = _lid_overlay(
        head,
        analysis,
        fill,
        half=True,
        view_name=view_name,
        claude_refit=claude_refit,
        far_ratio=far_ratio,
        half_profile=half_profile,
        puppet_id=puppet_id,
    )
    full_destination.parent.mkdir(parents=True, exist_ok=True)
    full.save(full_destination, format="PNG", compress_level=1)
    half.save(half_destination, format="PNG", compress_level=1)
    return full_destination, half_destination


def _apply_v5_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """Screen-space brow nudges. Left and right are the viewer's sides."""
    brows["style"] = "ink"
    for side in ("left", "right"):
        brows[side]["scale_x"] = 1.0
        brows[side]["horizon_tilt"] = 0.0
        brows[side]["offset_x"] = 0
        brows[side]["offset_y"] = 0
        brows[side]["width_delta"] = 0
    lowered = puppet_id.lower()
    if "chatgpt" in lowered and view_name == "facing_left":
        brows["left"]["width_delta"] = -4
        brows["left"]["offset_x"] = -4
        brows["left"]["offset_y"] = 6
        brows["left"]["smug_contour"] = "villain"
    elif "chatgpt" in lowered and view_name == "facing_right":
        brows["right"]["width_delta"] = -4
        brows["right"]["offset_x"] = 4
        brows["right"]["offset_y"] = 6
        brows["right"]["smug_contour"] = "villain"
    elif "claude" in lowered and view_name == "facing_left":
        brows["left"]["offset_x"] = -4
        brows["left"]["offset_y"] = 4
        brows["right"]["offset_y"] = -2
    elif "claude" in lowered and view_name == "facing_right":
        brows["right"]["offset_x"] = 4
        brows["right"]["offset_y"] = 4
        brows["left"]["offset_y"] = -2


def _v5_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """Character micro-offsets on top of the approved V4 chin seat."""
    mouth_x, mouth_y = center
    lowered = puppet_id.lower()
    if "chatgpt" in lowered:
        if view_name == "facing_left":
            mouth_x += 4
        elif view_name == "facing_right":
            mouth_x -= 4
    elif "claude" in lowered:
        mouth_y += 3
        if view_name == "facing_left":
            mouth_x -= 7
        elif view_name == "facing_right":
            mouth_x += 7
    return mouth_x, mouth_y


def _apply_v6_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """Straight bars. Deboche is a ±14° V, not a bent brow stroke."""
    _apply_v5_brows(brows, puppet_id, view_name)
    lowered = puppet_id.lower()
    if "chatgpt" in lowered:
        for side in ("left", "right"):
            brows[side].pop("smug_contour", None)
            brows[side]["shape"] = "bevel"
        if view_name == "facing_left":
            brows["left"]["width_scale"] = 0.96
            brows["left"]["offset_x"] = int(brows["left"]["offset_x"]) - 2
            brows["left"]["smug_tilt"] = 14
            brows["right"]["smug_tilt"] = -14
        elif view_name == "facing_right":
            brows["right"]["width_scale"] = 0.96
            brows["right"]["offset_x"] = int(brows["right"]["offset_x"]) + 2
            brows["right"]["smug_tilt"] = -14
            brows["left"]["smug_tilt"] = 14
        else:
            brows["left"]["smug_tilt"] = 14
            brows["right"]["smug_tilt"] = -14
    elif "claude" in lowered:
        for side in ("left", "right"):
            brows[side]["offset_y"] = int(brows[side].get("offset_y") or 0) - 5


def _apply_v7_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """Invert the deboche V so the inner corners point at the nose."""
    _apply_v6_brows(brows, puppet_id, view_name)
    lowered = puppet_id.lower()
    if "chatgpt" in lowered:
        for side in ("left", "right"):
            tilt = brows[side].get("smug_tilt")
            if tilt is not None:
                brows[side]["smug_tilt"] = -int(tilt)
    elif "claude" in lowered:
        for side in ("left", "right"):
            brows[side]["offset_y"] = int(brows[side].get("offset_y") or 0) - 5


def _v7_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """V6 chin seat plus the V7 screen-space corrections."""
    mouth_x, mouth_y = _v6_mouth_center(puppet_id, view_name, center)
    lowered = puppet_id.lower()
    if "chatgpt" in lowered:
        if view_name == "facing_left":
            mouth_x += 5
        elif view_name == "facing_right":
            mouth_x -= 5
    elif "claude" in lowered:
        if view_name == "facing_left":
            mouth_x -= 7
        elif view_name == "facing_right":
            mouth_x += 3
    return mouth_x, mouth_y


def _apply_v8_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """Elegant tapered arches, 3/4 stagger, and the sad/shock poses."""
    _apply_v7_brows(brows, puppet_id, view_name)
    lowered = puppet_id.lower()
    for side in ("left", "right"):
        brows[side]["sad_tilt"] = 14 if side == "left" else -14
        brows[side]["shock_offset_y"] = -12
    if "chatgpt" in lowered:
        for side in ("left", "right"):
            brows[side]["shape"] = "taper"
        if view_name == "facing_left":
            brows["left"]["width_delta"] = float(brows["left"].get("width_delta") or 0) - 6
            brows["left"]["offset_y"] = int(brows["left"].get("offset_y") or 0) + 2
        elif view_name == "facing_right":
            brows["right"]["width_delta"] = float(brows["right"].get("width_delta") or 0) - 6
            brows["right"]["offset_y"] = int(brows["right"].get("offset_y") or 0) + 2
    elif "claude" in lowered:
        for side in ("left", "right"):
            brows[side]["offset_y"] = int(brows[side].get("offset_y") or 0) - 3
        if view_name == "facing_left":
            brows["right"]["offset_y"] = int(brows["right"]["offset_y"]) - 3
        elif view_name == "facing_right":
            brows["left"]["offset_y"] = int(brows["left"]["offset_y"]) - 3


def _v8_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """V7 chin seat plus the V8 screen-space corrections."""
    mouth_x, mouth_y = _v7_mouth_center(puppet_id, view_name, center)
    lowered = puppet_id.lower()
    if "chatgpt" in lowered:
        if view_name == "facing_left":
            mouth_x += 3
        elif view_name == "facing_right":
            mouth_x -= 3
    elif "claude" in lowered:
        if view_name == "facing_left":
            mouth_x -= 5
        elif view_name == "facing_right":
            mouth_x += 2
    return mouth_x, mouth_y


def _apply_v9_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """Straight-blade acting poses plus the 3/4 far-brow foreshorten."""
    _apply_v8_brows(brows, puppet_id, view_name)
    lowered = puppet_id.lower()
    if "chatgpt" in lowered:
        if view_name == "facing_left":
            brows["left"]["width_scale"] = 0.65
            brows["left"]["width_delta"] = 0
            brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) - 7
            brows["left"]["offset_y"] = int(brows["left"].get("offset_y") or 0) + 6
        elif view_name == "facing_right":
            brows["right"]["width_scale"] = 0.65
            brows["right"]["width_delta"] = 0
            brows["right"]["offset_x"] = int(brows["right"].get("offset_x") or 0) + 6
            brows["right"]["offset_y"] = int(brows["right"].get("offset_y") or 0) + 5
    elif "claude" in lowered:
        if view_name == "facing_left":
            brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) - 3
            brows["left"]["offset_y"] = int(brows["left"].get("offset_y") or 0) + 4
        elif view_name == "facing_right":
            brows["right"]["offset_x"] = int(brows["right"].get("offset_x") or 0) + 4
            brows["right"]["offset_y"] = int(brows["right"].get("offset_y") or 0) + 4


def _v9_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """V8 chin seat plus Claude's production drop."""
    mouth_x, mouth_y = _v8_mouth_center(puppet_id, view_name, center)
    if "claude" in puppet_id.lower():
        mouth_y += 3
        if view_name == "facing_left":
            mouth_x -= 6
    return mouth_x, mouth_y


def _scale_brow(node: dict, factor: float) -> None:
    node["width_scale"] = round(float(node.get("width_scale") or 1.0) * factor, 4)


def _apply_v10_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """V9 seats plus the release-candidate width and horizon trims."""
    _apply_v9_brows(brows, puppet_id, view_name)
    if "chatgpt" not in puppet_id.lower():
        return
    if view_name == "facing_left":
        _scale_brow(brows["left"], 1.10)
        brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) - 4
        brows["left"]["offset_y"] = int(brows["left"].get("offset_y") or 0) + 2
        _scale_brow(brows["right"], 0.95)
    elif view_name == "facing_right":
        _scale_brow(brows["left"], 0.90)
        _scale_brow(brows["right"], 1.10)
        brows["right"]["offset_x"] = int(brows["right"].get("offset_x") or 0) + 5
        brows["right"]["offset_y"] = int(brows["right"].get("offset_y") or 0) + 3


def _v10_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """V9 chin seat plus Claude's release lock."""
    mouth_x, mouth_y = _v9_mouth_center(puppet_id, view_name, center)
    if "claude" in puppet_id.lower():
        mouth_y += 2
        if view_name == "facing_left":
            mouth_x -= 4
        elif view_name == "facing_right":
            mouth_x += 2
    return mouth_x, mouth_y


def _steepen_v(angle: float, extra: float = 4.0) -> float:
    """Add degrees in the existing villain-V direction."""
    if angle < 0:
        return angle - extra
    if angle > 0:
        return angle + extra
    return angle


def _apply_v11_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """V10 seats, steeper cara-de-mau, and Claude's heavier blade."""
    _apply_v10_brows(brows, puppet_id, view_name)
    lowered = puppet_id.lower()
    for side, index in (("left", 0), ("right", 1)):
        smug = brows[side].get("smug_tilt")
        if smug is None:
            smug = SCENE_GRAPH_BROW_ANGLES["smug"][index]
        brows[side]["smug_tilt"] = _steepen_v(float(smug))
        angry = brows[side].get("angry_tilt")
        if angry is None:
            angry = SCENE_GRAPH_BROW_ANGLES["angry"][index]
        brows[side]["angry_tilt"] = _steepen_v(float(angry))
    if "chatgpt" in lowered:
        for side in ("left", "right"):
            brows[side]["menace_drop_y"] = 3
        if view_name == "facing_left":
            brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) - 4
            brows["right"]["offset_x"] = int(brows["right"].get("offset_x") or 0) - 2
            brows["left"]["menace_drop_y"] = 5
        elif view_name == "facing_right":
            brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) + 2
    elif "claude" in lowered:
        for side in ("left", "right"):
            brows[side]["blade_stroke_scale"] = 1.12


def _v11_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """V10 chin seat plus Claude's gold-master lock."""
    mouth_x, mouth_y = _v10_mouth_center(puppet_id, view_name, center)
    if "claude" in puppet_id.lower():
        mouth_y += 3
        if view_name == "facing_left":
            mouth_x -= 6
        elif view_name == "facing_right":
            mouth_x += 3
    return mouth_x, mouth_y


def _apply_v12_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """V11 seats plus the definitive lateral shifts and anger drop."""
    _apply_v11_brows(brows, puppet_id, view_name)
    lowered = puppet_id.lower()
    if "chatgpt" in lowered:
        if view_name == "facing_left":
            brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) - 4
            brows["right"]["offset_x"] = int(brows["right"].get("offset_x") or 0) - 3
        elif view_name == "facing_right":
            brows["right"]["offset_x"] = int(brows["right"].get("offset_x") or 0) + 4
            brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) + 3
        for side in ("left", "right"):
            brows[side]["menace_drop_y"] = int(brows[side].get("menace_drop_y") or 0) + 4
        if view_name == "facing_left":
            brows["left"]["menace_drop_y"] = int(brows["left"]["menace_drop_y"]) + 3
        elif view_name == "facing_right":
            brows["right"]["menace_drop_y"] = int(brows["right"]["menace_drop_y"]) + 4
    elif "claude" in lowered:
        for side in ("left", "right"):
            brows[side]["blade_stroke_scale"] = round(
                float(brows[side].get("blade_stroke_scale") or 1.0) * 1.12,
                4,
            )
            brows[side]["menace_drop_y"] = int(brows[side].get("menace_drop_y") or 0) + 3


def _v12_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """V11 chin seat plus Claude's left-view protrusion lock."""
    mouth_x, mouth_y = _v11_mouth_center(puppet_id, view_name, center)
    if "claude" in puppet_id.lower() and view_name == "facing_left":
        mouth_x -= 4
    return mouth_x, mouth_y


def _apply_v13_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """V12 seats plus the golden-seal shifts and the socket-level anger drop."""
    _apply_v12_brows(brows, puppet_id, view_name)
    lowered = puppet_id.lower()
    if "chatgpt" in lowered:
        if view_name == "facing_left":
            brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) - 4
            brows["left"]["offset_y"] = int(brows["left"].get("offset_y") or 0) + 2
            brows["right"]["offset_x"] = int(brows["right"].get("offset_x") or 0) - 3
        elif view_name == "facing_right":
            brows["right"]["offset_x"] = int(brows["right"].get("offset_x") or 0) + 6
            brows["right"]["offset_y"] = int(brows["right"].get("offset_y") or 0) + 2
            brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) + 3
        for side in ("left", "right"):
            brows[side]["menace_drop_y"] = int(brows[side].get("menace_drop_y") or 0) + 6
    elif "claude" in lowered:
        if view_name == "facing_left":
            brows["left"]["offset_x"] = int(brows["left"].get("offset_x") or 0) - 4
        elif view_name == "facing_right":
            brows["right"]["offset_x"] = int(brows["right"].get("offset_x") or 0) + 4


def _v13_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """V12 chin seat plus Claude's golden-seal left lock."""
    mouth_x, mouth_y = _v12_mouth_center(puppet_id, view_name, center)
    if "claude" in puppet_id.lower() and view_name == "facing_left":
        mouth_x -= 5
    return mouth_x, mouth_y


def _apply_v14_brows(brows: dict, puppet_id: str, view_name: str) -> None:
    """V13 seats. Three-quarter smug and angry drop and steepen only."""
    _apply_v13_brows(brows, puppet_id, view_name)
    if view_name not in {"facing_left", "facing_right"}:
        return
    for side in ("left", "right"):
        brows[side]["menace_drop_y"] = int(brows[side].get("menace_drop_y") or 0) + 4
        smug = brows[side].get("smug_tilt")
        if smug is not None:
            brows[side]["smug_tilt"] = _steepen_v(float(smug), 3.0)
        angry = brows[side].get("angry_tilt")
        if angry is not None:
            brows[side]["angry_tilt"] = _steepen_v(float(angry), 3.0)


def _v14_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """V13 chin seat plus Claude's sign-off drop and left lock."""
    mouth_x, mouth_y = _v13_mouth_center(puppet_id, view_name, center)
    if "claude" in puppet_id.lower():
        mouth_y += 3
        if view_name == "facing_left":
            mouth_x -= 3
    return mouth_x, mouth_y


def _v6_mouth_center(
    puppet_id: str,
    view_name: str,
    center: tuple[int, int],
) -> tuple[int, int]:
    """V5 seat plus the chassis jaw protrusion for this view."""
    mouth_x, mouth_y = _v5_mouth_center(puppet_id, view_name, center)
    profile = CHASSIS_JAW_PROTRUSION_PX.get(
        archetype_for(puppet_id),
        CHASSIS_JAW_PROTRUSION_PX["cyber_capsule"],
    )
    dx, dy = profile.get(view_name, (0, 0))
    return mouth_x + dx, mouth_y + dy


def _archive_v4_sprites(puppet_id: str, root: Path) -> None:
    """Mirror the V4 suite into the harness folder. V3 mouths stay in place."""
    dest = V4_MASTER_SHEET.parent / "sprites" / puppet_id
    suite = root / "mouths" / "v4_universal_rig"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = dest / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads((root / "puppet.json").read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = dest / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_blink.png", "eyelid_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)


def _archive_v5_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V5 suite and manifest. V4 files stay in their own folder."""
    dest = V5_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v5_final_calibration"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v5_blink.png", "eyelid_v5_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def _archive_v6_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V6 suite and manifest. V5 files stay in their own folder."""
    dest = V6_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v6_master_approved"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v6_blink.png", "eyelid_v6_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def _archive_v7_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V7 suite and manifest. V6 files stay in their own folder."""
    dest = V7_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v7_pixel_perfect"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v7_blink.png", "eyelid_v7_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def _archive_v8_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V8 suite and manifest. V7 files stay in their own folder."""
    dest = V8_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v8_golden_master"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v8_blink.png", "eyelid_v8_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def _archive_v9_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V9 suite and manifest. V8 files stay in their own folder."""
    dest = V9_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v9_production_final"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v9_blink.png", "eyelid_v9_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def _archive_v14_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V14 suite and manifest. V13 files stay in their own folder."""
    dest = V14_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v14_master_signoff"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v14_blink.png", "eyelid_v14_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def _archive_v13_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V13 suite and manifest. V12 files stay in their own folder."""
    dest = V13_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v13_golden_seal"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v13_blink.png", "eyelid_v13_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def _archive_v12_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V12 suite and manifest. V11 files stay in their own folder."""
    dest = V12_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v12_definitive_release"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v12_blink.png", "eyelid_v12_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def _archive_v11_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V11 suite and manifest. V10 files stay in their own folder."""
    dest = V11_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v11_gold_master"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v11_blink.png", "eyelid_v11_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def _archive_v10_sprites(puppet_id: str, root: Path, manifest_path: Path) -> None:
    """Mirror the V10 suite and manifest. V9 files stay in their own folder."""
    dest = V10_MASTER_SHEET.parent
    sprite_root = dest / "sprites" / puppet_id
    suite = root / "mouths" / "v10_release_candidate"
    for view_name in ("front", "facing_left", "facing_right"):
        source = suite / view_name
        target = sprite_root / "mouths" / view_name
        target.mkdir(parents=True, exist_ok=True)
        for png in source.glob("*.png"):
            shutil.copy2(png, target / png.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view_name in VIEW_ORDER:
        head_parent = (root / manifest["views"][view_name]["head"]).parent
        lid_dest = sprite_root / "views" / view_name
        lid_dest.mkdir(parents=True, exist_ok=True)
        for name in ("eyelid_v10_blink.png", "eyelid_v10_half.png"):
            src = head_parent / name
            if src.is_file():
                shutil.copy2(src, lid_dest / name)
    manifest_dest = dest / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_dest / f"{puppet_id}.json")


def auto_rig_puppet(
    puppet_id: str,
    *,
    mouth_scale: float = 1.0,
    reanchor_chin: bool = False,
    v3_southpark_hybrid: bool = False,
    v4_universal_rig: bool = False,
    v5_final_calibration: bool = False,
    v6_master_approved: bool = False,
    v7_pixel_perfect: bool = False,
    v8_golden_master: bool = False,
    v9_production_final: bool = False,
    v10_release_candidate: bool = False,
    v11_gold_master: bool = False,
    v12_definitive_release: bool = False,
    v13_golden_seal: bool = False,
    v14_master_signoff: bool = False,
) -> Path:
    """Analyze all existing head views and update only facial rig metadata."""
    root = assets_root() / "puppets" / puppet_id
    manifest_path = root / "puppet.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    archetype = archetype_for(puppet_id)
    native_suite = (
        v3_southpark_hybrid
        or v4_universal_rig
        or v5_final_calibration
        or v6_master_approved
        or v7_pixel_perfect
        or v8_golden_master
        or v9_production_final
        or v10_release_candidate
        or v11_gold_master
        or v12_definitive_release
        or v13_golden_seal
        or v14_master_signoff
    )
    if v14_master_signoff:
        suite_dir = root / "mouths" / "v14_master_signoff"
    elif v13_golden_seal:
        suite_dir = root / "mouths" / "v13_golden_seal"
    elif v12_definitive_release:
        suite_dir = root / "mouths" / "v12_definitive_release"
    elif v11_gold_master:
        suite_dir = root / "mouths" / "v11_gold_master"
    elif v10_release_candidate:
        suite_dir = root / "mouths" / "v10_release_candidate"
    elif v9_production_final:
        suite_dir = root / "mouths" / "v9_production_final"
    elif v8_golden_master:
        suite_dir = root / "mouths" / "v8_golden_master"
    elif v7_pixel_perfect:
        suite_dir = root / "mouths" / "v7_pixel_perfect"
    elif v6_master_approved:
        suite_dir = root / "mouths" / "v6_master_approved"
    elif v5_final_calibration:
        suite_dir = root / "mouths" / "v5_final_calibration"
    elif v4_universal_rig:
        suite_dir = root / "mouths" / "v4_universal_rig"
    else:
        suite_dir = None
    if native_suite:
        generate_view_mouth_suites(
            puppet_id,
            archetype=archetype,
            puppet_root=root,
            suite_dir=suite_dir,
        )
    else:
        generate_mouth_suite(puppet_id, archetype=archetype, puppet_root=root)
    art_before = {
        name: (
            view.get("head"),
            view.get("body"),
            view.get("head_xy"),
            view.get("body_xy"),
        )
        for name, view in data["views"].items()
    }
    for view_name in VIEW_ORDER:
        view = data["views"][view_name]
        head_path = root / view["head"]
        with Image.open(head_path) as opened:
            head = opened.convert("RGBA")
        analysis = analyze_head(
            head,
            view_name=view_name,
            mouth_scale=mouth_scale,
            reanchor_chin=reanchor_chin,
            v4_universal_rig=(
                v4_universal_rig
                or v5_final_calibration
                or v6_master_approved
                or v7_pixel_perfect
                or v8_golden_master
                or v9_production_final
                or v10_release_candidate
                or v11_gold_master
                or v12_definitive_release
                or v13_golden_seal
                or v14_master_signoff
            ),
            v6_master_approved=(
                v6_master_approved
                or v7_pixel_perfect
                or v8_golden_master
                or v9_production_final
                or v10_release_candidate
                or v11_gold_master
                or v12_definitive_release
                or v13_golden_seal
                or v14_master_signoff
            ),
        )
        if v14_master_signoff:
            blink_path = head_path.parent / "eyelid_v14_blink.png"
            half_blink_path = head_path.parent / "eyelid_v14_half.png"
        elif v13_golden_seal:
            blink_path = head_path.parent / "eyelid_v13_blink.png"
            half_blink_path = head_path.parent / "eyelid_v13_half.png"
        elif v12_definitive_release:
            blink_path = head_path.parent / "eyelid_v12_blink.png"
            half_blink_path = head_path.parent / "eyelid_v12_half.png"
        elif v11_gold_master:
            blink_path = head_path.parent / "eyelid_v11_blink.png"
            half_blink_path = head_path.parent / "eyelid_v11_half.png"
        elif v10_release_candidate:
            blink_path = head_path.parent / "eyelid_v10_blink.png"
            half_blink_path = head_path.parent / "eyelid_v10_half.png"
        elif v9_production_final:
            blink_path = head_path.parent / "eyelid_v9_blink.png"
            half_blink_path = head_path.parent / "eyelid_v9_half.png"
        elif v8_golden_master:
            blink_path = head_path.parent / "eyelid_v8_blink.png"
            half_blink_path = head_path.parent / "eyelid_v8_half.png"
        elif v7_pixel_perfect:
            blink_path = head_path.parent / "eyelid_v7_blink.png"
            half_blink_path = head_path.parent / "eyelid_v7_half.png"
        elif v6_master_approved:
            blink_path = head_path.parent / "eyelid_v6_blink.png"
            half_blink_path = head_path.parent / "eyelid_v6_half.png"
        elif v5_final_calibration:
            blink_path = head_path.parent / "eyelid_v5_blink.png"
            half_blink_path = head_path.parent / "eyelid_v5_half.png"
        else:
            blink_path = head_path.parent / "eyelid_blink.png"
            half_blink_path = head_path.parent / "eyelid_half.png"
        generate_blink_overlays(
            head,
            analysis,
            puppet_id,
            blink_path,
            half_blink_path,
            view_name=view_name,
            claude_refit=v5_final_calibration and "claude" in puppet_id.lower(),
            far_ratio=(
                FAR_EYE_FORESHORTEN_RATIO
                if (
                    v6_master_approved
                    or v7_pixel_perfect
                    or v8_golden_master
                    or v9_production_final
                    or v10_release_candidate
                    or v11_gold_master
                    or v12_definitive_release
                    or v13_golden_seal
                    or v14_master_signoff
                )
                else 0.78
            ),
            half_profile=(
                "v13"
                if v13_golden_seal or v14_master_signoff
                else "v11"
                if v11_gold_master or v12_definitive_release
                else "v10"
                if v10_release_candidate
                else "v9"
                if v9_production_final
                else "v8"
                if v8_golden_master
                else "v7"
                if v7_pixel_perfect
                else "v6"
                if v6_master_approved
                else None
            ),
        )
        blink_relative = blink_path.relative_to(root).as_posix()
        half_relative = half_blink_path.relative_to(root).as_posix()
        view["facial_rig"] = analysis.scene_graph(
            blink_relative,
            half_relative,
        )
        if native_suite:
            folder = VIEW_FOLDERS[view_name]
            if v14_master_signoff:
                asset_dir = f"mouths/v14_master_signoff/{folder}"
            elif v13_golden_seal:
                asset_dir = f"mouths/v13_golden_seal/{folder}"
            elif v12_definitive_release:
                asset_dir = f"mouths/v12_definitive_release/{folder}"
            elif v11_gold_master:
                asset_dir = f"mouths/v11_gold_master/{folder}"
            elif v10_release_candidate:
                asset_dir = f"mouths/v10_release_candidate/{folder}"
            elif v9_production_final:
                asset_dir = f"mouths/v9_production_final/{folder}"
            elif v8_golden_master:
                asset_dir = f"mouths/v8_golden_master/{folder}"
            elif v7_pixel_perfect:
                asset_dir = f"mouths/v7_pixel_perfect/{folder}"
            elif v6_master_approved:
                asset_dir = f"mouths/v6_master_approved/{folder}"
            elif v5_final_calibration:
                asset_dir = f"mouths/v5_final_calibration/{folder}"
            elif v4_universal_rig:
                asset_dir = f"mouths/v4_universal_rig/{folder}"
            else:
                asset_dir = f"mouths/{folder}"
            view["facial_rig"]["mouth"]["asset_dir"] = asset_dir
            view["facial_rig"]["mouth"]["native_view"] = folder
        if v14_master_signoff:
            mouth_x, mouth_y = _v14_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v14_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
            view["facial_rig"]["expressions"] = {
                "sad_pout": {"mouth": "mouth_sad.png", "brows": "sad"},
                "shock_surprise": {
                    "mouth": "mouth_shock.png",
                    "brows": "shock",
                    "brow_offset_y": -12,
                },
                "angry_rebuttal": {"mouth": "mouth_angry.png", "brows": "angry"},
                "smug_deboche": {"mouth": "mouth_smug.png", "brows": "smug"},
            }
        elif v13_golden_seal:
            mouth_x, mouth_y = _v13_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v13_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
            view["facial_rig"]["expressions"] = {
                "sad_pout": {"mouth": "mouth_sad.png", "brows": "sad"},
                "shock_surprise": {
                    "mouth": "mouth_shock.png",
                    "brows": "shock",
                    "brow_offset_y": -12,
                },
                "angry_rebuttal": {"mouth": "mouth_angry.png", "brows": "angry"},
                "smug_deboche": {"mouth": "mouth_smug.png", "brows": "smug"},
            }
        elif v12_definitive_release:
            mouth_x, mouth_y = _v12_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v12_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
            view["facial_rig"]["expressions"] = {
                "sad_pout": {"mouth": "mouth_sad.png", "brows": "sad"},
                "shock_surprise": {
                    "mouth": "mouth_shock.png",
                    "brows": "shock",
                    "brow_offset_y": -12,
                },
                "angry_rebuttal": {"mouth": "mouth_angry.png", "brows": "angry"},
                "smug_deboche": {"mouth": "mouth_smug.png", "brows": "smug"},
            }
        elif v11_gold_master:
            mouth_x, mouth_y = _v11_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v11_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
            view["facial_rig"]["expressions"] = {
                "sad_pout": {"mouth": "mouth_sad.png", "brows": "sad"},
                "shock_surprise": {
                    "mouth": "mouth_shock.png",
                    "brows": "shock",
                    "brow_offset_y": -12,
                },
                "angry_rebuttal": {"mouth": "mouth_angry.png", "brows": "angry"},
                "smug_deboche": {"mouth": "mouth_smug.png", "brows": "smug"},
            }
        elif v10_release_candidate:
            mouth_x, mouth_y = _v10_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v10_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
            view["facial_rig"]["expressions"] = {
                "sad_pout": {"mouth": "mouth_sad.png", "brows": "sad"},
                "shock_surprise": {
                    "mouth": "mouth_shock.png",
                    "brows": "shock",
                    "brow_offset_y": -12,
                },
                "angry_rebuttal": {"mouth": "mouth_angry.png", "brows": "angry"},
                "smug_deboche": {"mouth": "mouth_smug.png", "brows": "smug"},
            }
        elif v9_production_final:
            mouth_x, mouth_y = _v9_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v9_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
            view["facial_rig"]["expressions"] = {
                "sad_pout": {"mouth": "mouth_sad.png", "brows": "sad"},
                "shock_surprise": {
                    "mouth": "mouth_shock.png",
                    "brows": "shock",
                    "brow_offset_y": -12,
                },
                "angry_rebuttal": {"mouth": "mouth_angry.png", "brows": "angry"},
                "smug_deboche": {"mouth": "mouth_smug.png", "brows": "smug"},
            }
        elif v8_golden_master:
            mouth_x, mouth_y = _v8_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v8_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
            view["facial_rig"]["expressions"] = {
                "sad_pout": {"mouth": "mouth_sad.png", "brows": "sad"},
                "shock_surprise": {
                    "mouth": "mouth_shock.png",
                    "brows": "shock",
                    "brow_offset_y": -12,
                },
            }
        elif v7_pixel_perfect:
            mouth_x, mouth_y = _v7_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v7_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
        elif v6_master_approved:
            mouth_x, mouth_y = _v6_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v6_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
        elif v5_final_calibration:
            mouth_x, mouth_y = _v5_mouth_center(
                puppet_id,
                view_name,
                tuple(view["facial_rig"]["mouth"]["center"]),
            )
            view["facial_rig"]["mouth"]["center"] = [mouth_x, mouth_y]
            _apply_v5_brows(view["facial_rig"]["eyebrows"], puppet_id, view_name)
        elif v4_universal_rig:
            far = _far_side(analysis, view_name)
            brows = view["facial_rig"]["eyebrows"]
            brows["style"] = (
                "ink" if archetype == "cyber_capsule" else "brass"
            )
            for side in ("left", "right"):
                brows[side]["scale_x"] = 0.80 if side == far else 1.0
                brows[side]["horizon_tilt"] = _horizon_tilt(
                    view_name,
                    side,
                    far,
                )
        view["mouth_archetype"] = archetype
        view["landmark_analysis"] = {
            **asdict(analysis),
            "eye_left": asdict(analysis.eye_left),
            "eye_right": asdict(analysis.eye_right),
        }
        # Compatibility coordinates for the existing renderer and pipeline.
        mouth_center = view["facial_rig"]["mouth"]["center"]
        view["mouth"] = {
            "x": int(mouth_center[0]),
            "y": int(mouth_center[1]),
            "rot_deg": round(analysis.mouth_rotation_deg, 3),
            "scale": round(analysis.perspective_scale, 4),
        }
        left_brow = view["facial_rig"]["eyebrows"]["left"]
        right_brow = view["facial_rig"]["eyebrows"]["right"]
        view["eyebrow_left"] = {
            "x": int(left_brow["center"][0] + int(left_brow.get("offset_x") or 0)),
            "y": int(left_brow["center"][1] + int(left_brow.get("offset_y") or 0)),
            "rot_deg": 0.0,
        }
        view["eyebrow_right"] = {
            "x": int(right_brow["center"][0] + int(right_brow.get("offset_x") or 0)),
            "y": int(right_brow["center"][1] + int(right_brow.get("offset_y") or 0)),
            "rot_deg": 0.0,
        }
        view["eye_lid_left"] = {
            "x": analysis.eye_left.center[0],
            "y": analysis.eye_left.center[1],
        }
        view["eye_lid_right"] = {
            "x": analysis.eye_right.center[0],
            "y": analysis.eye_right.center[1],
        }
        view["nameplate_bottom"] = analysis.nameplate_bottom
    art_after = {
        name: (
            view.get("head"),
            view.get("body"),
            view.get("head_xy"),
            view.get("body_xy"),
        )
        for name, view in data["views"].items()
    }
    if art_after != art_before:
        raise RuntimeError(f"refusing to rewrite approved view art in {manifest_path}")
    data["facial_scene_graph"] = {
        "coordinate_space": "head_local",
        "root": "head",
        "hierarchy": {
            "head": ["eyes_lids", "eyebrows", "mouth"],
            "eyes_lids": [],
            "eyebrows": [],
            "mouth": [],
        },
        "brow_angles": {
            name: list(angles)
            for name, angles in SCENE_GRAPH_BROW_ANGLES.items()
        },
        "mouth_archetype": archetype,
        "expressions": (
            {
                "sad_pout": {"mouth": "mouth_sad.png", "brows": "sad"},
                "shock_surprise": {
                    "mouth": "mouth_shock.png",
                    "brows": "shock",
                    "brow_offset_y": -12,
                },
                "angry_rebuttal": {"mouth": "mouth_angry.png", "brows": "angry"},
                "smug_deboche": {"mouth": "mouth_smug.png", "brows": "smug"},
            }
            if (
                v14_master_signoff
                or v13_golden_seal
                or v12_definitive_release
                or v11_gold_master
                or v10_release_candidate
                or v9_production_final
            )
            else {
                "sad_pout": {"mouth": "mouth_sad.png", "brows": "sad"},
                "shock_surprise": {
                    "mouth": "mouth_shock.png",
                    "brows": "shock",
                    "brow_offset_y": -12,
                },
            }
            if v8_golden_master
            else {}
        ),
        "version": (
            "v14_master_signoff"
            if v14_master_signoff
            else "v13_golden_seal"
            if v13_golden_seal
            else "v12_definitive_release"
            if v12_definitive_release
            else "v11_gold_master"
            if v11_gold_master
            else "v10_release_candidate"
            if v10_release_candidate
            else "v9_production_final"
            if v9_production_final
            else "v8_golden_master"
            if v8_golden_master
            else "v7_pixel_perfect"
            if v7_pixel_perfect
            else "v6_master_approved"
            if v6_master_approved
            else "v5_final_calibration"
            if v5_final_calibration
            else "v4_universal_rig"
            if v4_universal_rig
            else "v3_southpark_hybrid"
            if v3_southpark_hybrid
            else "v2_master"
        ),
    }
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if v14_master_signoff:
        _archive_v14_sprites(puppet_id, root, manifest_path)
    elif v13_golden_seal:
        _archive_v13_sprites(puppet_id, root, manifest_path)
    elif v12_definitive_release:
        _archive_v12_sprites(puppet_id, root, manifest_path)
    elif v11_gold_master:
        _archive_v11_sprites(puppet_id, root, manifest_path)
    elif v10_release_candidate:
        _archive_v10_sprites(puppet_id, root, manifest_path)
    elif v9_production_final:
        _archive_v9_sprites(puppet_id, root, manifest_path)
    elif v8_golden_master:
        _archive_v8_sprites(puppet_id, root, manifest_path)
    elif v7_pixel_perfect:
        _archive_v7_sprites(puppet_id, root, manifest_path)
    elif v6_master_approved:
        _archive_v6_sprites(puppet_id, root, manifest_path)
    elif v5_final_calibration:
        _archive_v5_sprites(puppet_id, root, manifest_path)
    elif v4_universal_rig:
        _archive_v4_sprites(puppet_id, root)
    print(f"auto-rigged: {manifest_path}")
    return manifest_path


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _inspection_portrait(
    puppet_id: str,
    view_name: str,
    state: str,
    *,
    v3_southpark_hybrid: bool = False,
    v4_universal_rig: bool = False,
    v5_final_calibration: bool = False,
    v6_master_approved: bool = False,
    v7_pixel_perfect: bool = False,
    v8_golden_master: bool = False,
    v9_production_final: bool = False,
    v10_release_candidate: bool = False,
    v11_gold_master: bool = False,
    v12_definitive_release: bool = False,
    v13_golden_seal: bool = False,
    v14_master_signoff: bool = False,
) -> Image.Image:
    root = assets_root() / "puppets" / puppet_id
    manifest = json.loads((root / "puppet.json").read_text(encoding="utf-8"))
    view = manifest["views"][view_name]
    with Image.open(root / view["head"]) as opened:
        head = opened.convert("RGBA")
    rig = view["facial_rig"]
    native = (
        v3_southpark_hybrid
        or v4_universal_rig
        or v5_final_calibration
        or v6_master_approved
        or v7_pixel_perfect
        or v8_golden_master
        or v9_production_final
        or v10_release_candidate
        or v11_gold_master
        or v12_definitive_release
        or v13_golden_seal
        or v14_master_signoff
    )
    native_dir = root / rig["mouth"]["asset_dir"] if native else None
    if state.startswith("viseme_"):
        code = state.removeprefix("viseme_")
        mouth_path = (
            native_dir / f"mouth_{code}.png"
            if native_dir is not None
            else root / "mouths" / "visemes" / f"mouth_{code}.png"
        )
        expression = "neutral"
        blink = None
    elif state == "speech":
        mouth_path = (
            native_dir / "mouth_D.png"
            if native_dir is not None
            else root / "mouths" / "visemes" / "mouth_D.png"
        )
        expression = "neutral"
        blink = None
    elif state == "smug":
        mouth_path = (
            native_dir / "mouth_smug.png"
            if native_dir is not None
            else root / "mouths" / "expressions" / "mouth_smug.png"
        )
        expression = "smug"
        blink = None
    elif state == "sad":
        mouth_path = (
            native_dir / "mouth_sad.png"
            if native_dir is not None
            else root / "mouths" / "expressions" / "mouth_sad.png"
        )
        expression = "sad"
        blink = None
    elif state == "shock":
        mouth_path = (
            native_dir / "mouth_shock.png"
            if native_dir is not None
            else root / "mouths" / "expressions" / "mouth_shock.png"
        )
        expression = "shock"
        blink = None
    elif state == "angry":
        mouth_path = (
            native_dir / "mouth_angry.png"
            if native_dir is not None
            else root / "mouths" / "expressions" / "mouth_angry.png"
        )
        expression = "angry"
        blink = None
    else:
        mouth_path = (
            native_dir / "mouth_X.png"
            if native_dir is not None
            else root / "mouths" / "visemes" / "mouth_X.png"
        )
        expression = "neutral"
        lid_key = (
            "half_asset"
            if state == "half_blink"
            else "full_asset"
            if state == "blink"
            else None
        )
        blink = (
            Image.open(root / rig["eyes_lids"][lid_key]).convert("RGBA")
            if lid_key is not None
            else None
        )
    with Image.open(mouth_path) as opened:
        mouth = opened.convert("RGBA")
    bezel = ARCHETYPES[view["mouth_archetype"]].bezel
    try:
        return compose_scene_graph_head(
            head,
            rig,
            mouth,
            bezel=bezel,
            expression=expression,
            blink_overlay=blink,
        )
    finally:
        if blink is not None:
            blink.close()


def _fit_portrait(
    portrait: Image.Image,
    budget_w: int,
    budget_h: int,
) -> Image.Image:
    scale = min(budget_w / portrait.width, budget_h / portrait.height)
    return portrait.resize(
        (
            max(1, int(round(portrait.width * scale))),
            max(1, int(round(portrait.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )


def export_master_sheet(
    puppet_ids: tuple[str, ...],
    destination: Path = MASTER_SHEET,
    *,
    v3_southpark_hybrid: bool = False,
    v4_universal_rig: bool = False,
    v5_final_calibration: bool = False,
    v6_master_approved: bool = False,
    v7_pixel_perfect: bool = False,
    v8_golden_master: bool = False,
    v9_production_final: bool = False,
    v10_release_candidate: bool = False,
    v11_gold_master: bool = False,
    v12_definitive_release: bool = False,
    v13_golden_seal: bool = False,
    v14_master_signoff: bool = False,
) -> Path:
    """Write both character sections as three-view acting matrices."""
    if (
        v14_master_signoff
        or v13_golden_seal
        or v12_definitive_release
        or v11_gold_master
        or v10_release_candidate
        or v9_production_final
    ):
        states = (
            ("neutral", "NEUTRAL REST · ARCHED BROWS"),
            ("speech", "OPEN SPEECH · VISEME D"),
            ("smug", "SMUG DEBOCHE · STRAIGHT BLADE"),
            ("angry", "ANGRY REBUTTAL · STRAIGHT BLADE"),
            ("sad", "SAD MELANCHOLY · ARCHED INVERTED V"),
            ("shock", "SHOCK PERPLEXED · HIGH ARCH"),
            ("half_blink", "HALF-BLINK · SOCKET LOCK"),
            ("blink", "FULL BLINK"),
        )
        cell_h = 640
    elif v8_golden_master:
        states = (
            ("neutral", "NEUTRAL REST · ELEGANT BROWS"),
            ("speech", "OPEN SPEECH · VISEME D"),
            ("smug", "SMUG DEBOCHE · COCKED VILLAIN V"),
            ("sad", "SAD MELANCHOLY · INVERTED V"),
            ("shock", "SHOCK PERPLEXED · HIGH ARCHED BROWS"),
            ("half_blink", "HALF-BLINK · CORRECTED COVERAGE"),
            ("blink", "FULL BLINK"),
        )
        cell_h = 700
    elif v7_pixel_perfect:
        states = (
            ("neutral", "NEUTRAL REST + CALIBRATED LEVELED BROWS"),
            ("speech", "OPEN SPEECH · VISEME D"),
            ("smug", "SMUG DEBOCHE · TRUE ANGRY V"),
            ("phonetics", "KEY PHONETICS · B  C  E  F"),
            ("half_blink", "HALF-BLINK · FAR-LENS COVERAGE"),
            ("blink", "FULL BLINK"),
        )
        cell_h = 700
    elif v6_master_approved:
        states = (
            ("neutral", "NEUTRAL REST + CALIBRATED LEVELED BROWS"),
            ("speech", "OPEN SPEECH · VISEME D"),
            ("smug", "SMUG DEBOCHE · ROTATED V"),
            ("phonetics", "KEY PHONETICS · B  C  E  F"),
            ("half_blink", "HALF-BLINK · FAR-LENS COVERAGE"),
            ("blink", "FULL BLINK"),
        )
        cell_h = 700
    elif v5_final_calibration:
        states = (
            ("neutral", "NEUTRAL REST + CALIBRATED LEVELED BROWS"),
            ("speech", "OPEN SPEECH · VISEME D"),
            ("smug", "SMUG DEBOCHE · TRIANGULAR SMIRK + ARCHED BROW"),
            ("phonetics", "KEY PHONETICS · B  C  E  F"),
            ("half_blink", "HALF-BLINK · FAR-LENS COVERAGE"),
            ("blink", "FULL BLINK"),
        )
        cell_h = 700
    elif v4_universal_rig:
        states = (
            ("neutral", "NEUTRAL REST + LEVELED HIGH-CONTRAST BROWS"),
            ("speech", "OPEN SPEECH · VISEME D"),
            ("smug", "SMUG DEBOCHE · TRIANGULAR V-SMIRK"),
            ("phonetics", "KEY PHONETICS · B  C  E  F"),
            ("half_blink", "HALF-BLINK · CLIPPED"),
            ("blink", "FULL BLINK"),
        )
        cell_h = 700
    else:
        states = (
            ("neutral", "NEUTRAL REST + LEVEL BROWS"),
            ("speech", "OPEN SPEECH · VISEME D"),
            ("smug", "SMUG DEBOCHE"),
            ("angry", "ANGRY REBUTTAL"),
            ("half_blink", "HALF-BLINK · SKEPTICAL"),
            ("blink", "FULL BLINK · SHUTTERS"),
        )
        cell_h = 610
    cell_w = 540
    section_header = 74
    view_header = 52
    section_h = section_header + view_header + cell_h * len(states)
    sheet = Image.new(
        "RGB",
        (cell_w * len(VIEW_ORDER), section_h * len(puppet_ids)),
        (10, 11, 14),
    )
    title_font = _load_font(32, bold=True)
    header_font = _load_font(24, bold=True)
    label_font = _load_font(20, bold=True)
    for section, puppet_id in enumerate(puppet_ids):
        archetype = archetype_for(puppet_id)
        section_y = section * section_h
        pen = ImageDraw.Draw(sheet)
        if v14_master_signoff:
            version = "  ·  V14 MASTER SIGN-OFF"
        elif v13_golden_seal:
            version = "  ·  V13 GOLDEN SEAL"
        elif v12_definitive_release:
            version = "  ·  V12 DEFINITIVE RELEASE"
        elif v11_gold_master:
            version = "  ·  V11 GOLD MASTER"
        elif v10_release_candidate:
            version = "  ·  V10 RELEASE CANDIDATE"
        elif v9_production_final:
            version = "  ·  V9 PRODUCTION FINAL"
        elif v8_golden_master:
            version = "  ·  V8 GOLDEN MASTER"
        elif v7_pixel_perfect:
            version = "  ·  V7 PIXEL PERFECT"
        elif v6_master_approved:
            version = "  ·  V6 MASTER APPROVED"
        elif v5_final_calibration:
            version = "  ·  V5 FINAL CALIBRATION"
        elif v4_universal_rig:
            version = "  ·  V4 UNIVERSAL RIG"
        elif v3_southpark_hybrid:
            version = "  ·  V3 SOUTH PARK MECHA HYBRID"
        else:
            version = ""
        title = f"{puppet_id.upper()}  ·  {archetype.upper()}{version}"
        pen.rectangle(
            (0, section_y, sheet.width, section_y + section_header),
            fill=(24, 19, 16),
        )
        pen.text(
            (24, section_y + 18),
            title,
            fill=(244, 214, 150),
            font=title_font,
        )
        for column, view_name in enumerate(VIEW_ORDER):
            x0 = column * cell_w
            pen.rectangle(
                (
                    x0,
                    section_y + section_header,
                    x0 + cell_w,
                    section_y + section_header + view_header,
                ),
                fill=(31, 27, 23),
            )
            pen.text(
                (x0 + 16, section_y + section_header + 12),
                VIEW_LABELS[view_name],
                fill=(230, 202, 151),
                font=header_font,
            )
            for row, (state, label) in enumerate(states):
                y0 = (
                    section_y
                    + section_header
                    + view_header
                    + row * cell_h
                )
                cell = Image.new("RGB", (cell_w, cell_h), (16, 17, 20))
                cell_pen = ImageDraw.Draw(cell)
                cell_pen.text(
                    (14, 12),
                    f"{row + 1}. {label}",
                    fill=(231, 223, 205),
                    font=label_font,
                )
                if state == "phonetics":
                    codes = ("B", "C", "E", "F")
                    quad_w = (cell_w - 12) // 2
                    quad_h = (cell_h - 46) // 2
                    for index, code in enumerate(codes):
                        portrait = _inspection_portrait(
                            puppet_id,
                            view_name,
                            f"viseme_{code}",
                            v3_southpark_hybrid=v3_southpark_hybrid,
                            v4_universal_rig=v4_universal_rig,
                            v5_final_calibration=v5_final_calibration,
                            v6_master_approved=v6_master_approved,
                            v7_pixel_perfect=v7_pixel_perfect,
                            v8_golden_master=v8_golden_master,
                            v9_production_final=v9_production_final,
                            v10_release_candidate=v10_release_candidate,
                            v11_gold_master=v11_gold_master,
                            v12_definitive_release=v12_definitive_release,
                            v13_golden_seal=v13_golden_seal,
                            v14_master_signoff=v14_master_signoff,
                        )
                        fitted = _fit_portrait(
                            portrait,
                            quad_w - 8,
                            quad_h - 24,
                        )
                        qx = 6 + (index % 2) * quad_w
                        qy = 42 + (index // 2) * quad_h
                        cell_pen.text(
                            (qx + 4, qy),
                            f"[{code}]",
                            fill=(231, 223, 205),
                            font=label_font,
                        )
                        cell.paste(
                            fitted.convert("RGB"),
                            (
                                qx + (quad_w - fitted.width) // 2,
                                qy + 22,
                            ),
                            fitted.split()[-1],
                        )
                else:
                    portrait = _inspection_portrait(
                        puppet_id,
                        view_name,
                        state,
                        v3_southpark_hybrid=v3_southpark_hybrid,
                        v4_universal_rig=v4_universal_rig,
                        v5_final_calibration=v5_final_calibration,
                        v6_master_approved=v6_master_approved,
                        v7_pixel_perfect=v7_pixel_perfect,
                        v8_golden_master=v8_golden_master,
                        v9_production_final=v9_production_final,
                        v10_release_candidate=v10_release_candidate,
                        v11_gold_master=v11_gold_master,
                        v12_definitive_release=v12_definitive_release,
                        v13_golden_seal=v13_golden_seal,
                        v14_master_signoff=v14_master_signoff,
                    )
                    fitted = _fit_portrait(
                        portrait,
                        cell_w - 30,
                        cell_h - 54,
                    )
                    cell.paste(
                        fitted.convert("RGB"),
                        (
                            (cell_w - fitted.width) // 2,
                            48 + (cell_h - 48 - fitted.height) // 2,
                        ),
                        fitted.split()[-1],
                    )
                sheet.paste(cell, (x0, y0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", compress_level=1)
    print(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-rig isolated puppet heads and export a master sheet."
    )
    parser.add_argument("--auto-rig", action="store_true")
    parser.add_argument("--puppets", required=True)
    parser.add_argument("--inspect-master-sheet", action="store_true")
    parser.add_argument("--v3-southpark-hybrid", action="store_true")
    parser.add_argument("--inspect-sheet-v3", action="store_true")
    parser.add_argument("--v4-universal-rig", action="store_true")
    parser.add_argument("--inspect-sheet-v4", action="store_true")
    parser.add_argument("--v5-final-calibration", action="store_true")
    parser.add_argument("--inspect-sheet-v5", action="store_true")
    parser.add_argument("--v6-master-approved", action="store_true")
    parser.add_argument("--inspect-sheet-v6", action="store_true")
    parser.add_argument("--render-contraplano-v6", action="store_true")
    parser.add_argument("--v7-pixel-perfect", action="store_true")
    parser.add_argument("--inspect-sheet-v7", action="store_true")
    parser.add_argument("--render-contraplano-v7", action="store_true")
    parser.add_argument("--v8-golden-master", action="store_true")
    parser.add_argument("--inspect-sheet-v8", action="store_true")
    parser.add_argument("--render-contraplano-v8", action="store_true")
    parser.add_argument("--v9-production-final", action="store_true")
    parser.add_argument("--inspect-sheet-v9", action="store_true")
    parser.add_argument("--render-contraplano-v9", action="store_true")
    parser.add_argument("--v10-release-candidate", action="store_true")
    parser.add_argument("--inspect-sheet-v10", action="store_true")
    parser.add_argument("--render-contraplano-v10", action="store_true")
    parser.add_argument("--v11-gold-master", action="store_true")
    parser.add_argument("--inspect-sheet-v11", action="store_true")
    parser.add_argument("--render-contraplano-v11", action="store_true")
    parser.add_argument("--v12-definitive-release", action="store_true")
    parser.add_argument("--inspect-sheet-v12", action="store_true")
    parser.add_argument("--render-contraplano-v12", action="store_true")
    parser.add_argument("--v13-golden-seal", action="store_true")
    parser.add_argument("--inspect-sheet-v13", action="store_true")
    parser.add_argument("--render-contraplano-v13", action="store_true")
    parser.add_argument("--v14-master-signoff", action="store_true")
    parser.add_argument("--inspect-sheet-v14", action="store_true")
    parser.add_argument("--render-contraplano-v14", action="store_true")
    parser.add_argument(
        "--scale-mouths",
        type=float,
        default=1.0,
        help="Multiplier applied to detected chin-relative mouth width.",
    )
    parser.add_argument(
        "--reanchor-chin",
        action="store_true",
        help="Move mouths down and toward the three-quarter chin turn.",
    )
    parser.add_argument("--output", type=Path, default=MASTER_SHEET)
    args = parser.parse_args(argv)
    puppet_ids = tuple(
        token.strip()
        for token in args.puppets.split(",")
        if token.strip()
    )
    if not args.auto_rig and not (
        args.inspect_master_sheet
        or args.inspect_sheet_v3
        or args.inspect_sheet_v4
        or args.inspect_sheet_v5
        or args.inspect_sheet_v6
        or args.render_contraplano_v6
        or args.inspect_sheet_v7
        or args.render_contraplano_v7
        or args.inspect_sheet_v8
        or args.render_contraplano_v8
        or args.inspect_sheet_v9
        or args.render_contraplano_v9
        or args.inspect_sheet_v10
        or args.render_contraplano_v10
        or args.inspect_sheet_v11
        or args.render_contraplano_v11
        or args.inspect_sheet_v12
        or args.render_contraplano_v12
        or args.inspect_sheet_v13
        or args.render_contraplano_v13
        or args.inspect_sheet_v14
        or args.render_contraplano_v14
    ):
        parser.error(
            "pass --auto-rig, --inspect-master-sheet, "
            "--inspect-sheet-v3, --inspect-sheet-v4, --inspect-sheet-v5, "
            "--inspect-sheet-v6, --inspect-sheet-v7, --inspect-sheet-v8, "
            "--inspect-sheet-v9, --inspect-sheet-v10, --inspect-sheet-v11, "
            "--inspect-sheet-v12, --inspect-sheet-v13, --inspect-sheet-v14, "
            "--render-contraplano-v6, --render-contraplano-v7, "
            "--render-contraplano-v8, --render-contraplano-v9, "
            "--render-contraplano-v10, --render-contraplano-v11, "
            "--render-contraplano-v12, --render-contraplano-v13, "
            "or --render-contraplano-v14"
        )
    calibrated = (
        args.v3_southpark_hybrid
        or args.v4_universal_rig
        or args.v5_final_calibration
        or args.v6_master_approved
        or args.v7_pixel_perfect
        or args.v8_golden_master
        or args.v9_production_final
        or args.v10_release_candidate
        or args.v11_gold_master
        or args.v12_definitive_release
        or args.v13_golden_seal
        or args.v14_master_signoff
    )
    v3_scale = (
        0.75
        if calibrated and args.scale_mouths == 1.0
        else args.scale_mouths
    )
    if args.auto_rig:
        for puppet_id in puppet_ids:
            auto_rig_puppet(
                puppet_id,
                mouth_scale=v3_scale,
                reanchor_chin=args.reanchor_chin or calibrated,
                v3_southpark_hybrid=args.v3_southpark_hybrid,
                v4_universal_rig=args.v4_universal_rig,
                v5_final_calibration=args.v5_final_calibration,
                v6_master_approved=args.v6_master_approved,
                v7_pixel_perfect=args.v7_pixel_perfect,
                v8_golden_master=args.v8_golden_master,
                v9_production_final=args.v9_production_final,
                v10_release_candidate=args.v10_release_candidate,
                v11_gold_master=args.v11_gold_master,
                v12_definitive_release=args.v12_definitive_release,
                v13_golden_seal=args.v13_golden_seal,
                v14_master_signoff=args.v14_master_signoff,
            )
    if args.inspect_master_sheet:
        export_master_sheet(puppet_ids, args.output)
    if args.inspect_sheet_v3:
        export_master_sheet(
            puppet_ids,
            V3_MASTER_SHEET,
            v3_southpark_hybrid=True,
        )
    if args.inspect_sheet_v4:
        export_master_sheet(
            puppet_ids,
            V4_MASTER_SHEET,
            v4_universal_rig=True,
        )
    if args.inspect_sheet_v5:
        export_master_sheet(
            puppet_ids,
            V5_MASTER_SHEET,
            v5_final_calibration=True,
        )
    if args.inspect_sheet_v6:
        export_master_sheet(
            puppet_ids,
            V6_MASTER_SHEET,
            v6_master_approved=True,
        )
    if args.render_contraplano_v6:
        from ..pipeline import render_v6_contraplano

        render_v6_contraplano(V6_VIDEO)
    if args.inspect_sheet_v7:
        export_master_sheet(
            puppet_ids,
            V7_MASTER_SHEET,
            v7_pixel_perfect=True,
        )
    if args.render_contraplano_v7:
        from ..pipeline import render_v7_contraplano

        render_v7_contraplano(V7_VIDEO)
    if args.inspect_sheet_v8:
        export_master_sheet(
            puppet_ids,
            V8_MASTER_SHEET,
            v8_golden_master=True,
        )
    if args.render_contraplano_v8:
        from ..pipeline import render_v8_contraplano

        render_v8_contraplano(V8_VIDEO)
    if args.inspect_sheet_v9:
        export_master_sheet(
            puppet_ids,
            V9_MASTER_SHEET,
            v9_production_final=True,
        )
    if args.render_contraplano_v9:
        from ..pipeline import render_v9_contraplano

        render_v9_contraplano(V9_VIDEO)
    if args.inspect_sheet_v10:
        export_master_sheet(
            puppet_ids,
            V10_MASTER_SHEET,
            v10_release_candidate=True,
        )
    if args.render_contraplano_v10:
        from ..pipeline import render_v10_contraplano

        render_v10_contraplano(V10_VIDEO)
    if args.inspect_sheet_v11:
        export_master_sheet(
            puppet_ids,
            V11_MASTER_SHEET,
            v11_gold_master=True,
        )
    if args.render_contraplano_v11:
        from ..pipeline import render_v11_contraplano

        render_v11_contraplano(V11_VIDEO)
    if args.inspect_sheet_v12:
        export_master_sheet(
            puppet_ids,
            V12_MASTER_SHEET,
            v12_definitive_release=True,
        )
    if args.render_contraplano_v12:
        from ..pipeline import render_v12_contraplano

        render_v12_contraplano(V12_VIDEO)
    if args.inspect_sheet_v13:
        export_master_sheet(
            puppet_ids,
            V13_MASTER_SHEET,
            v13_golden_seal=True,
        )
    if args.render_contraplano_v13:
        from ..pipeline import render_v13_contraplano

        render_v13_contraplano(V13_VIDEO)
    if args.inspect_sheet_v14:
        export_master_sheet(
            puppet_ids,
            V14_MASTER_SHEET,
            v14_master_signoff=True,
        )
    if args.render_contraplano_v14:
        from ..pipeline import render_v14_contraplano

        render_v14_contraplano(V14_VIDEO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
