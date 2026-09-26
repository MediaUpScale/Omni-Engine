"""Emotional acting poses, kept separate from the phonetic Rhubarb mouths.

Phonetic lip-sync stays in ``visemes.py``. This module draws the acting
mouths from that same Llama patch, plus the eyebrow sprites and the pose
matrix that pairs them.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.pipeline_paths import assets_root, outputs_root

from .viseme_inspector import _approved_head, _chin_anchor, _chin_plate
from .visemes import (
    MOUTH_DOCK_CHIN_RATIO,
    PALETTES,
    draw_expression_mouth,
    draw_eyebrow,
)

EXPRESSIONS: dict[str, dict] = {
    "smug_deboche": {
        "mouth": "mouths/expressions/mouth_smug.png",
        "eyebrow_left": {"offset_y": -16, "rot_deg": 12},
        "eyebrow_right": {"offset_y": 8, "rot_deg": -8},
    },
    "angry_rebuttal": {
        "mouth": "mouths/expressions/mouth_angry.png",
        "eyebrow_left": {"offset_y": 10, "rot_deg": -15},
        "eyebrow_right": {"offset_y": 10, "rot_deg": 15},
    },
    "shock_surprise": {
        "mouth": "mouths/expressions/mouth_shock.png",
        "eyebrow_left": {"offset_y": -22, "rot_deg": 5},
        "eyebrow_right": {"offset_y": -22, "rot_deg": -5},
    },
    "neutral_speaking": {
        "mode": "rhubarb_visemes",
        "eyebrow_left": {"offset_y": 0, "rot_deg": 0},
        "eyebrow_right": {"offset_y": 0, "rot_deg": 0},
    },
}

# The inspection column is a closed smile. Speech still uses Rhubarb.
_SHEET = (
    ("smug_deboche", "SMUG DEBOCHE", "mouth_smug.png"),
    ("angry_rebuttal", "ANGRY REBUTTAL", "mouth_angry.png"),
    ("shock_surprise", "SHOCK SURPRISE", "mouth_shock.png"),
    ("neutral_rest", "NEUTRAL REST", "mouth_neutral.png"),
)
CELL_W = 640
CELL_H = 860
LABEL_H = 78


def _eyebrow_for_side(bezel: tuple[int, int, int, int], side: str) -> Image.Image:
    """One resting brow. Pose rotation is applied later, not baked in."""
    brow = draw_eyebrow(bezel)
    if side == "left":
        return brow.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return brow


def generate_character_acting(character_id: str) -> Path:
    """Write eyebrows, expression mouths, and merge the pose matrix."""
    palette = PALETTES[character_id]
    root = assets_root() / "puppets" / character_id
    brows = root / "eyebrows"
    mouths = root / "mouths" / "expressions"
    brows.mkdir(parents=True, exist_ok=True)
    mouths.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        _eyebrow_for_side(palette.bezel, side).save(brows / f"eyebrow_{side}.png", format="PNG", compress_level=1)
    for name in ("smug", "angry", "shock", "sad", "triumph", "neutral"):
        draw_expression_mouth(name, palette.bezel).save(mouths / f"mouth_{name}.png", format="PNG", compress_level=1)
    _merge_expressions(root / "puppet.json")
    print(f"acting: {root}")
    return root


def _merge_expressions(manifest_path: Path) -> None:
    """Add the emotion matrix without moving any docked view."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    views_before = json.dumps(data.get("views"), sort_keys=True)
    data["expressions"] = EXPRESSIONS
    if json.dumps(data.get("views"), sort_keys=True) != views_before:
        raise RuntimeError(f"refusing to rewrite views in {manifest_path}")
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Optic:
    """One circular eye socket, in head-image pixels."""

    center: tuple[int, int]
    radius: float

    @property
    def top(self) -> int:
        return int(round(self.center[1] - self.radius))


@dataclass(frozen=True, slots=True)
class FaceLayout:
    """Sockets, the forehead nameplate's lower edge, and the chin seat."""

    left: Optic
    right: Optic
    nameplate_bottom: int
    chin_anchor: tuple[int, int]
    chin_width: int


def face_layout(head: Image.Image) -> FaceLayout:
    """Locate the two lenses and the nameplate edge that brows must stay under."""
    rgba = np.asarray(head.convert("RGBA"))
    height, width = rgba.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY), (0, 0), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=70,
        param1=80,
        param2=28,
        minRadius=28,
        maxRadius=80,
    )
    glow = _glow_optics(rgba)
    candidates: list[tuple[float, float, float]] = []
    if circles is not None:
        for x_pos, y_pos, radius in circles[0]:
            if 0.32 * height < y_pos < 0.62 * height and 0.08 * width < x_pos < 0.94 * width:
                candidates.append((float(x_pos), float(y_pos), float(radius)))
    lenses = [circle for circle in candidates if _bright_lens(rgba, circle)]
    pair = glow or _closest_pair(lenses if len(lenses) >= 2 else candidates)
    if pair is None:
        pair = (
            (width * 0.34, height * 0.46, height * 0.07),
            (width * 0.66, height * 0.46, height * 0.07),
        )
    ordered = sorted(pair, key=lambda optic: optic[0])
    left = Optic((int(round(ordered[0][0])), int(round(ordered[0][1]))), ordered[0][2])
    right = Optic((int(round(ordered[1][0])), int(round(ordered[1][1]))), ordered[1][2])
    plate = _chin_plate(head)
    return FaceLayout(
        left=left,
        right=right,
        nameplate_bottom=_nameplate_bottom(rgba, left, right),
        chin_anchor=_chin_anchor(head),
        chin_width=plate[2] - plate[0],
    )


def _glow_optics(rgba: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Two lens cores from the cyan or amber glow, ignoring grilles and dials."""
    rgb = rgba[..., :3]
    height, width = rgb.shape[:2]
    glow = (
        (rgb[..., 1] > 180) & (rgb[..., 2] > 160) & (rgb[..., 0] < 220)
    ) | (
        (rgb[..., 0] > 200) & (rgb[..., 1] > 150) & (rgb[..., 2] < 150)
    )
    y0, y1 = int(height * 0.34), int(height * 0.60)
    mask = glow.copy()
    mask[:y0, :] = False
    mask[y1:, :] = False
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    found: list[tuple[int, float, float, float]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        blob_w = int(stats[index, cv2.CC_STAT_WIDTH])
        if area < 120 or not 12 <= blob_w <= 48:
            continue
        x_pos, y_pos = float(centroids[index][0]), float(centroids[index][1])
        if not (0.08 * width < x_pos < 0.94 * width):
            continue
        found.append((area, x_pos, y_pos, float(blob_w)))
    found.sort(reverse=True)
    chosen: list[tuple[int, float, float, float]] = []
    for blob in found:
        if any(abs(blob[1] - other[1]) < 70 for other in chosen):
            continue
        if chosen and abs(blob[2] - chosen[0][2]) > 48:
            continue
        chosen.append(blob)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return None
    optics = tuple(
        (x_pos, y_pos, max(50.0, blob_w * 2.3))
        for _area, x_pos, y_pos, blob_w in sorted(chosen, key=lambda item: item[1])
    )
    return optics  # type: ignore[return-value]


def _bright_lens(rgba: np.ndarray, circle: tuple[float, float, float]) -> bool:
    """True when the circle's core is a glowing lens, not a bronze grille."""
    height, width = rgba.shape[:2]
    x_pos, y_pos, radius = circle
    reach = max(4, int(round(radius * 0.35)))
    x0 = max(0, int(x_pos) - reach)
    x1 = min(width, int(x_pos) + reach)
    y0 = max(0, int(y_pos) - reach)
    y1 = min(height, int(y_pos) + reach)
    core = rgba[y0:y1, x0:x1, :3]
    if core.size == 0:
        return False
    mean = core.mean(axis=(0, 1))
    return float(mean.max()) > 200 and float(mean[1]) > 140


def _closest_pair(
    candidates: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    best: tuple[float, tuple[float, float, float], tuple[float, float, float]] | None = None
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if abs(left[1] - right[1]) > 40 or abs(left[0] - right[0]) < 80:
                continue
            score = abs(left[2] - right[2]) + abs(left[1] - right[1])
            if best is None or score < best[0]:
                best = (score, left, right)
    if best is None:
        return None
    return best[1], best[2]


def _nameplate_bottom(rgba: np.ndarray, left: Optic, right: Optic) -> int:
    """Lowest strong horizontal edge above the sockets. That edge is the plate."""
    height, width = rgba.shape[:2]
    eye_top = min(left.top, right.top)
    x0, x1 = int(width * 0.28), int(width * 0.72)
    y0 = int(height * 0.12)
    y1 = max(y0 + 8, eye_top - 12)
    band = rgba[y0:y1, x0:x1, :3].astype(np.int16)
    if band.shape[0] < 3:
        return max(0, eye_top - 36)
    diff = np.abs(np.diff(band, axis=0)).mean(axis=(1, 2))
    smooth = np.convolve(diff, np.ones(5) / 5.0, mode="same")
    peak = float(smooth.max()) if smooth.size else 0.0
    rows = [int(index) + y0 for index, value in enumerate(smooth) if value >= peak * 0.62]
    if not rows:
        return max(0, eye_top - 36)
    return max(rows)


def _pose_for(name: str) -> dict:
    if name == "neutral_rest":
        return {
            "eyebrow_left": {"offset_y": 0, "rot_deg": 0},
            "eyebrow_right": {"offset_y": 0, "rot_deg": 0},
        }
    return EXPRESSIONS[name]


def _opaque_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = np.asarray(image)[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    if xs.size == 0:
        return 0, 0, image.width, image.height
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _place_brow(
    plate: Image.Image,
    brow: Image.Image,
    optic: Optic,
    pose: dict,
    nameplate_bottom: int,
) -> None:
    """Sit the brow on the socket rim and keep it under the forehead plate."""
    target_w = max(24, int(round(optic.radius * 2.35)))
    scale = target_w / brow.width
    resized = brow.resize(
        (target_w, max(1, int(round(brow.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    rotated = resized.rotate(float(pose["rot_deg"]), resample=Image.Resampling.BICUBIC, expand=True)
    left, top, right, bottom = _opaque_bbox(rotated)
    ink_h = max(1, bottom - top)
    gap = max(8, optic.top - nameplate_bottom - 4)
    if ink_h > gap:
        shrink = gap / ink_h
        rotated = rotated.resize(
            (max(1, int(round(rotated.width * shrink))), max(1, int(round(rotated.height * shrink)))),
            Image.Resampling.LANCZOS,
        )
        left, top, right, bottom = _opaque_bbox(rotated)
        ink_h = max(1, bottom - top)
    # Bottom of the ink rests 2px above the socket. Pose offset can lift it,
    # but the ink top is hard-clamped to the nameplate's lower edge.
    paste_y = optic.top - 2 - bottom + int(pose["offset_y"])
    if paste_y + top < nameplate_bottom + 2:
        paste_y = nameplate_bottom + 2 - top
    if paste_y + bottom > optic.top - 1:
        paste_y = optic.top - 1 - bottom
    plate.alpha_composite(rotated, (optic.center[0] - rotated.width // 2, paste_y))


def _place_mouth(plate: Image.Image, mouth: Image.Image, anchor: tuple[int, int], chin_width: int) -> None:
    target_w = max(32, int(round(chin_width * MOUTH_DOCK_CHIN_RATIO)))
    scale = target_w / mouth.width
    resized = mouth.resize(
        (target_w, max(1, int(round(mouth.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    plate.alpha_composite(resized, (anchor[0] - resized.width // 2, anchor[1] - resized.height // 2))


def compose_acting_head(character_id: str, expression: str, mouth_name: str) -> Image.Image:
    """Return a preview head. The approved head file is only read."""
    head = _approved_head(character_id)
    plate = head.copy()
    layout = face_layout(head)
    root = assets_root() / "puppets" / character_id
    with Image.open(root / "mouths" / "expressions" / mouth_name) as opened:
        mouth = opened.convert("RGBA")
    _place_mouth(plate, mouth, layout.chin_anchor, layout.chin_width)
    pose = _pose_for(expression)
    with Image.open(root / "eyebrows" / "eyebrow_right.png") as opened:
        character_right_brow = opened.convert("RGBA")
    with Image.open(root / "eyebrows" / "eyebrow_left.png") as opened:
        character_left_brow = opened.convert("RGBA")
    # A front-facing head shows the character's right eye on the image left.
    _place_brow(plate, character_right_brow, layout.left, pose["eyebrow_right"], layout.nameplate_bottom)
    _place_brow(plate, character_left_brow, layout.right, pose["eyebrow_left"], layout.nameplate_bottom)
    return plate


def export_emotion_sheet(destination: Path | None = None) -> Path:
    """ChatGPT on the top row, Claude on the bottom, four attitudes across."""
    sheet = Image.new("RGB", (CELL_W * 4, CELL_H * 2), (14, 12, 10))
    try:
        font = ImageFont.truetype("arialbd.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    for row, character_id in enumerate(PALETTES):
        title = "CHATGPT" if character_id.startswith("chatgpt") else "CLAUDE"
        for column, (expression, label, mouth_name) in enumerate(_SHEET):
            portrait = compose_acting_head(character_id, expression, mouth_name)
            budget_w = CELL_W - 28
            budget_h = CELL_H - LABEL_H - 24
            scale = min(budget_w / portrait.width, budget_h / portrait.height)
            fitted = portrait.resize(
                (max(1, int(portrait.width * scale)), max(1, int(portrait.height * scale))),
                Image.Resampling.LANCZOS,
            )
            cell = Image.new("RGB", (CELL_W, CELL_H), (22, 18, 14))
            banner = ImageDraw.Draw(cell)
            banner.rectangle((0, 0, CELL_W, LABEL_H), fill=(28, 22, 16))
            banner.text((16, 12), title, fill=(244, 214, 150), font=font)
            banner.text((16, 44), label, fill=(210, 186, 140), font=font)
            cell.paste(
                fitted.convert("RGB"),
                ((CELL_W - fitted.width) // 2, LABEL_H + (CELL_H - LABEL_H - fitted.height) // 2),
                fitted.split()[-1],
            )
            sheet.paste(cell, (column * CELL_W, row * CELL_H))
    destination = destination or (outputs_root() / "aiwake" / "_test_harness" / "facial_expressions_preview.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", compress_level=1)
    return destination


# Per-view mouth yaw and resting brow tilt. Coordinates themselves are measured
# on that view's head so a three-quarter chin is not given the front anchor.
_VIEW_MOUTH = {
    "facing_front": {"rot_deg": 0.0, "scale": 1.0, "brow_rot": (0.0, 0.0)},
    "facing_right": {"rot_deg": 5.5, "scale": 0.98, "brow_rot": (4.0, 6.0)},
    "facing_left": {"rot_deg": -5.5, "scale": 0.98, "brow_rot": (-6.0, -4.0)},
}


def _brow_y(eye_top: int, nameplate_bottom: int) -> int:
    """Sit the brow on the bezel and keep it under the forehead plate."""
    y_pos = eye_top - 7
    floor = nameplate_bottom + 8
    if y_pos < floor:
        y_pos = floor
    if y_pos > eye_top - 3:
        y_pos = eye_top - 3
    return int(y_pos)


def install_view_anchors(character_id: str) -> Path:
    """Write mouth, brow, and lid points into each existing view.

    Head and body paths stay untouched. ``facing_front`` is the front camera;
    it is not copied to a second view key that would lack art.
    """
    root = assets_root() / "puppets" / character_id
    manifest_path = root / "puppet.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    art_before = {
        name: (view.get("head"), view.get("body"), view.get("head_xy"), view.get("body_xy"))
        for name, view in data["views"].items()
    }
    for name, pose in _VIEW_MOUTH.items():
        view = data["views"][name]
        with Image.open(root / view["head"]) as opened:
            head = opened.convert("RGBA")
        layout = face_layout(head)
        left_rot, right_rot = pose["brow_rot"]
        view["mouth"] = {
            "x": int(layout.chin_anchor[0]),
            "y": int(layout.chin_anchor[1]),
            "rot_deg": pose["rot_deg"],
            "scale": pose["scale"],
        }
        view["eyebrow_left"] = {
            "x": int(layout.left.center[0]),
            "y": _brow_y(layout.left.top, layout.nameplate_bottom),
            "rot_deg": left_rot,
        }
        view["eyebrow_right"] = {
            "x": int(layout.right.center[0]),
            "y": _brow_y(layout.right.top, layout.nameplate_bottom),
            "rot_deg": right_rot,
        }
        view["eye_lid_left"] = {"x": int(layout.left.center[0]), "y": int(layout.left.center[1])}
        view["eye_lid_right"] = {"x": int(layout.right.center[0]), "y": int(layout.right.center[1])}
        view["nameplate_bottom"] = int(layout.nameplate_bottom)
    art_after = {
        name: (view.get("head"), view.get("body"), view.get("head_xy"), view.get("body_xy"))
        for name, view in data["views"].items()
    }
    if art_after != art_before:
        raise RuntimeError(f"refusing to rewrite view art in {manifest_path}")
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"anchors: {manifest_path}")
    return manifest_path


def _stamp(
    plate: Image.Image,
    sprite: Image.Image,
    x_pos: int,
    y_pos: int,
    rot_deg: float,
    *,
    ink_center: bool = False,
) -> None:
    rotated = sprite.rotate(-rot_deg, resample=Image.Resampling.BICUBIC, expand=True)
    if ink_center:
        alpha = np.asarray(rotated)[..., 3]
        ys, xs = np.nonzero(alpha > 16)
        if xs.size:
            center_x = float(xs.min() + xs.max()) / 2.0
            center_y = float(ys.min() + ys.max()) / 2.0
        else:
            center_x = rotated.width / 2.0
            center_y = rotated.height / 2.0
    else:
        center_x = rotated.width / 2.0
        center_y = rotated.height / 2.0
    plate.alpha_composite(rotated, (int(round(x_pos - center_x)), int(round(y_pos - center_y))))


def _pilot_cell(
    head: Image.Image,
    layout: FaceLayout,
    mouth: Image.Image,
    mouth_spec: dict,
    brow_left: Image.Image,
    brow_right: Image.Image,
    pose: tuple[tuple[int, float], tuple[int, float]],
) -> Image.Image:
    """One head with a docked mouth and brows clamped under the nameplate."""
    plate = head.copy()
    chin_width = layout.chin_width
    target_w = max(32, int(round(chin_width * MOUTH_DOCK_CHIN_RATIO * float(mouth_spec["scale"]))))
    scale = target_w / mouth.width
    resized = mouth.resize(
        (max(1, int(round(mouth.width * scale))), max(1, int(round(mouth.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    _stamp(plate, resized, int(mouth_spec["x"]), int(mouth_spec["y"]), float(mouth_spec["rot_deg"]))
    for optic, brow, (delta_y, delta_rot) in (
        (layout.left, brow_left, pose[0]),
        (layout.right, brow_right, pose[1]),
    ):
        target = max(24, int(round(optic.radius * 2.35)))
        brow_scale = target / brow.width
        fitted = brow.resize(
            (target, max(1, int(round(brow.height * brow_scale)))),
            Image.Resampling.LANCZOS,
        )
        y_pos = _brow_y(optic.top, layout.nameplate_bottom) + int(delta_y)
        floor = layout.nameplate_bottom + 8
        if y_pos < floor:
            y_pos = floor
        if y_pos > optic.top - 3:
            y_pos = optic.top - 3
        _stamp(plate, fitted, optic.center[0], y_pos, float(delta_rot), ink_center=True)
    return plate


def export_pilot_facial_sheet(character_id: str = "chatgpt_cyborg_v1", destination: Path | None = None) -> Path:
    """Front and right three-quarter: visemes on row 1, acting faces on row 2."""
    from .visemes import viseme_file

    root = assets_root() / "puppets" / character_id
    manifest = json.loads((root / "puppet.json").read_text(encoding="utf-8"))
    palette = PALETTES[character_id]
    brow_left = _eyebrow_for_side(palette.bezel, "left")
    brow_right = _eyebrow_for_side(palette.bezel, "right")
    columns = ("X", "D", "C", "E")
    emotions = (
        ("SMUG", "mouth_smug.png", ((-14, 14.0), (8, -8.0))),
        ("ANGRY", "mouth_angry.png", ((10, -16.0), (10, 16.0))),
        ("SHOCK", "mouth_shock.png", ((-18, 5.0), (-18, -5.0))),
        ("NEUTRAL", "mouth_neutral.png", ((0, 0.0), (0, 0.0))),
    )
    cell_w, cell_h, label_h = 340, 420, 36
    views = ("facing_front", "facing_right")
    sheet = Image.new("RGB", (cell_w * 8, label_h + cell_h * 2), (14, 12, 10))
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for group, view_name in enumerate(views):
        view = manifest["views"][view_name]
        with Image.open(root / view["head"]) as opened:
            head = opened.convert("RGBA")
        layout = face_layout(head)
        title = "FRONT" if view_name == "facing_front" else "FACING RIGHT"
        draw.text((group * cell_w * 4 + 12, 6), title, fill=(244, 214, 150), font=font)
        rest_rot = (
            float(view["eyebrow_left"]["rot_deg"]),
            float(view["eyebrow_right"]["rot_deg"]),
        )
        rest_pose = ((0, rest_rot[0]), (0, rest_rot[1]))
        for column, code in enumerate(columns):
            with Image.open(viseme_file(root, code)) as opened:
                mouth = opened.convert("RGBA")
            portrait = _pilot_cell(
                head, layout, mouth, view["mouth"], brow_left, brow_right, rest_pose
            )
            _paste_pilot_cell(sheet, portrait, group * 4 + column, 0, cell_w, cell_h, label_h, code)
        for column, (label, filename, pose) in enumerate(emotions):
            acting = (
                (pose[0][0], pose[0][1] + rest_rot[0]),
                (pose[1][0], pose[1][1] + rest_rot[1]),
            )
            with Image.open(root / "mouths" / "expressions" / filename) as opened:
                mouth = opened.convert("RGBA")
            portrait = _pilot_cell(head, layout, mouth, view["mouth"], brow_left, brow_right, acting)
            _paste_pilot_cell(sheet, portrait, group * 4 + column, 1, cell_w, cell_h, label_h, label)
    destination = destination or (
        outputs_root() / "aiwake" / "_test_harness" / "chatgpt_pilot_facial_inspection.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", compress_level=1)
    print(destination)
    return destination


def _paste_pilot_cell(
    sheet: Image.Image,
    portrait: Image.Image,
    column: int,
    row: int,
    cell_w: int,
    cell_h: int,
    label_h: int,
    label: str,
) -> None:
    budget_w = cell_w - 16
    budget_h = cell_h - 28
    scale = min(budget_w / portrait.width, budget_h / portrait.height)
    fitted = portrait.resize(
        (max(1, int(portrait.width * scale)), max(1, int(portrait.height * scale))),
        Image.Resampling.LANCZOS,
    )
    origin_x = column * cell_w
    origin_y = label_h + row * cell_h
    cell = Image.new("RGB", (cell_w, cell_h), (22, 18, 14))
    pen = ImageDraw.Draw(cell)
    pen.text((8, 4), label, fill=(244, 214, 150))
    cell.paste(
        fitted.convert("RGB"),
        ((cell_w - fitted.width) // 2, 24 + (cell_h - 24 - fitted.height) // 2),
        fitted.split()[-1],
    )
    sheet.paste(cell, (origin_x, origin_y))


INK_BROW = (16, 18, 22, 255)       # #101216
BRASS_BROW = (245, 222, 152, 255)  # #F5DE98


def _contrast_brow(style: str, width: int) -> Image.Image:
    """High-contrast arch drawn at the placed head width.

    ChatGPT is solid ink. Claude is warm brass with a 2px ink edge.
    """
    height = max(18, int(round(width * 0.18)))
    scale = 4
    canvas = Image.new(
        "RGBA",
        (width * scale, (height + 10) * scale),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas, "RGBA")
    bounds = (
        2 * scale,
        1 * scale,
        (width - 2) * scale,
        (height + 12) * scale,
    )
    if style == "ink":
        stroke = max(5, int(round(width * 0.08)))
        draw.arc(bounds, 200, 340, fill=INK_BROW, width=stroke * scale)
    else:
        body = max(5, int(round(width * 0.06)))
        draw.arc(bounds, 200, 340, fill=INK_BROW, width=(body + 4) * scale)
        draw.arc(bounds, 200, 340, fill=BRASS_BROW, width=body * scale)
    return canvas.resize((width, height + 10), Image.Resampling.LANCZOS)


def _straight_blade_brow(width: int, *, stroke_scale: float = 1.0) -> Image.Image:
    """Linear wedge: thick inner root, razor outer tip, no arch."""
    height = max(18, int(round(width * 0.16)))
    scale = 4
    canvas = Image.new(
        "RGBA",
        (width * scale, (height + 10) * scale),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas, "RGBA")
    inner = max(4.0, width * 0.075 * 0.75) * scale * 1.18 * stroke_scale
    outer = max(1.0, scale * 0.55) * 1.18 * stroke_scale
    y_pos = height * 0.55 * scale
    x0 = 2 * scale
    x1 = (width - 2) * scale
    draw.polygon(
        [
            (x0, y_pos - inner / 2),
            (x1, y_pos - outer / 2),
            (x1, y_pos + outer / 2),
            (x0, y_pos + inner / 2),
        ],
        fill=INK_BROW,
    )
    return canvas.resize((width, height + 10), Image.Resampling.LANCZOS)


def _tapered_arch_brow(width: int, *, bow: float = 0.34) -> Image.Image:
    """Slight anime arch. Outer tip tapers; stroke is 25% under the bevel bar."""
    height = max(20, int(round(width * 0.22)))
    scale = 4
    canvas = Image.new(
        "RGBA",
        (width * scale, (height + 14) * scale),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas, "RGBA")
    base = max(4.0, width * 0.075 * 0.75) * scale
    samples = 32
    centerline: list[tuple[float, float]] = []
    for index in range(samples + 1):
        progress = index / samples
        x_pos = (3 + (width - 6) * progress) * scale
        rise = 4 * progress * (1.0 - progress)
        y_pos = (height * 0.78 - height * bow * rise) * scale
        centerline.append((x_pos, y_pos))
    upper: list[tuple[float, float]] = []
    lower: list[tuple[float, float]] = []
    for index, (x_pos, y_pos) in enumerate(centerline):
        progress = index / samples
        thick = base * (1.0 - 0.78 * (progress ** 1.15))
        if index == 0:
            dx, dy = centerline[1][0] - x_pos, centerline[1][1] - y_pos
        elif index == samples:
            dx = x_pos - centerline[index - 1][0]
            dy = y_pos - centerline[index - 1][1]
        else:
            dx = centerline[index + 1][0] - centerline[index - 1][0]
            dy = centerline[index + 1][1] - centerline[index - 1][1]
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        nx, ny = -dy / length, dx / length
        upper.append((x_pos + nx * thick / 2, y_pos + ny * thick / 2))
        lower.append((x_pos - nx * thick / 2, y_pos - ny * thick / 2))
    draw.polygon([*upper, *reversed(lower)], fill=INK_BROW)
    return canvas.resize((width, height + 14), Image.Resampling.LANCZOS)


def _bevel_brow(width: int) -> Image.Image:
    """Clean straight ink bar. Expression comes from rotation, not a bent path."""
    height = max(16, int(round(width * 0.14)))
    scale = 4
    canvas = Image.new(
        "RGBA",
        (width * scale, (height + 8) * scale),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas, "RGBA")
    stroke = max(5, int(round(width * 0.075)))
    y_pos = int((height * 0.55) * scale)
    draw.line(
        (4 * scale, y_pos, (width - 4) * scale, y_pos),
        fill=INK_BROW,
        width=stroke * scale,
    )
    cap = stroke * scale // 2
    for x_pos in (4 * scale, (width - 4) * scale):
        draw.ellipse(
            (x_pos - cap, y_pos - cap, x_pos + cap, y_pos + cap),
            fill=INK_BROW,
        )
    return canvas.resize((width, height + 8), Image.Resampling.LANCZOS)


def _villain_brow(width: int) -> Image.Image:
    """Sharp triangular smirk brow. The peak sits toward the inner corner."""
    height = max(22, int(round(width * 0.28)))
    scale = 4
    canvas = Image.new(
        "RGBA",
        (width * scale, (height + 6) * scale),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas, "RGBA")
    stroke = max(5, int(round(width * 0.09)))
    points = (
        (3 * scale, int(height * 0.78) * scale),
        (int(width * 0.62) * scale, 2 * scale),
        ((width - 3) * scale, int(height * 0.46) * scale),
    )
    draw.line(points, fill=INK_BROW, width=stroke * scale, joint="miter")
    return canvas.resize((width, height + 6), Image.Resampling.LANCZOS)


SCENE_GRAPH_BROW_ANGLES: dict[str, tuple[float, float]] = {
    "neutral": (0.0, 0.0),
    "smug": (-8.0, 4.0),
    "angry": (-12.0, 12.0),
    "sad": (8.0, -8.0),
    "shock": (0.0, 0.0),
}


def stamp_scene_brows(
    plate: Image.Image,
    brow_nodes: dict,
    expression: str,
    bezel: tuple[int, int, int, int],
) -> None:
    """Paste the rig brows at their stored width, stroke, and anchor."""
    style = str(brow_nodes.get("style") or "")
    legacy_brow = None if style in {"ink", "brass"} else draw_eyebrow(bezel)
    left_angle, right_angle = SCENE_GRAPH_BROW_ANGLES.get(
        expression,
        SCENE_GRAPH_BROW_ANGLES["neutral"],
    )
    for side, angle in (("left", left_angle), ("right", right_angle)):
        node = brow_nodes[side]
        target = max(24, int(round(float(node["width"]))))
        scale_x = float(node.get("scale_x") or 1.0)
        horizon = float(node.get("horizon_tilt") or 0.0)
        width_delta = float(node.get("width_delta") or 0.0)
        width_scale = float(node.get("width_scale") or 1.0)
        placed_w = max(
            8,
            int(round(target * scale_x * width_scale + width_delta)),
        )
        pose = float(angle)
        if expression == "smug" and node.get("smug_tilt") is not None:
            pose = float(node["smug_tilt"])
        elif expression == "angry" and node.get("angry_tilt") is not None:
            pose = float(node["angry_tilt"])
        elif expression == "sad" and node.get("sad_tilt") is not None:
            pose = float(node["sad_tilt"])
        if expression in {"smug", "angry"}:
            fitted = _straight_blade_brow(
                placed_w,
                stroke_scale=float(node.get("blade_stroke_scale") or 1.0),
            )
        elif node.get("shape") == "taper":
            bow = 0.62 if expression == "shock" else 0.34
            fitted = _tapered_arch_brow(placed_w, bow=bow)
        elif node.get("shape") == "bevel":
            fitted = _bevel_brow(placed_w)
        elif expression == "smug" and node.get("smug_contour") == "villain":
            fitted = _villain_brow(placed_w)
        elif style in {"ink", "brass"}:
            fitted = _contrast_brow(style, placed_w)
        else:
            assert legacy_brow is not None
            uniform = target / legacy_brow.width
            fitted = legacy_brow.resize(
                (
                    placed_w,
                    max(1, int(round(legacy_brow.height * uniform))),
                ),
                Image.Resampling.LANCZOS,
            )
        if side == "left":
            fitted = fitted.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        rotated = fitted.rotate(
            pose + horizon,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
        alpha = np.asarray(rotated)[..., 3]
        ys, xs = np.nonzero(alpha > 16)
        center_x = (
            float(xs.min() + xs.max()) / 2.0
            if xs.size
            else rotated.width / 2.0
        )
        center_y = (
            float(ys.min() + ys.max()) / 2.0
            if ys.size
            else rotated.height / 2.0
        )
        anchor_x = float(node["center"][0]) + float(node.get("offset_x") or 0.0)
        anchor_y = float(node["center"][1]) + float(node.get("offset_y") or 0.0)
        if expression == "shock":
            lift = node.get("shock_offset_y")
            anchor_y += -12.0 if lift is None else float(lift)
        elif expression in {"smug", "angry"}:
            anchor_y += float(node.get("menace_drop_y") or 0.0)
        plate.alpha_composite(
            rotated,
            (
                int(round(anchor_x - center_x)),
                int(round(anchor_y - center_y)),
            ),
        )


def compose_scene_graph_head(
    head: Image.Image,
    rig: dict,
    mouth: Image.Image,
    *,
    bezel: tuple[int, int, int, int],
    expression: str = "neutral",
    blink_overlay: Image.Image | None = None,
) -> Image.Image:
    """Composite children in head-local coordinates.

    The input ``rig`` is the per-view scene graph written by the vision
    analyzer.  No child owns a stage/global coordinate: lids, brows and mouth
    all remain attached to the supplied head sprite.
    """
    plate = head.convert("RGBA").copy()
    mouth_node = rig["mouth"]
    target_w = max(32, int(round(float(mouth_node["target_width"]))))
    mouth_scale = target_w / mouth.width
    fitted_mouth = mouth.resize(
        (
            target_w,
            max(1, int(round(mouth.height * mouth_scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    fitted_mouth = fitted_mouth.rotate(
        -float(mouth_node.get("rot_deg") or 0.0),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    mouth_x, mouth_y = mouth_node["center"]
    plate.alpha_composite(
        fitted_mouth,
        (
            int(round(mouth_x - fitted_mouth.width / 2)),
            int(round(mouth_y - fitted_mouth.height / 2)),
        ),
    )

    if blink_overlay is not None:
        plate.alpha_composite(blink_overlay.convert("RGBA"), (0, 0))

    stamp_scene_brows(plate, rig["eyebrows"], expression, bezel)
    return plate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate mecha expressions, eyebrows, and the emotion sheet.")
    parser.add_argument("--generate-expressions-and-eyebrows", action="store_true")
    parser.add_argument("--inspect-emotions", action="store_true")
    args = parser.parse_args(argv)
    if not args.generate_expressions_and_eyebrows and not args.inspect_emotions:
        parser.error("pass --generate-expressions-and-eyebrows and/or --inspect-emotions")
    if args.generate_expressions_and_eyebrows:
        for character_id in PALETTES:
            generate_character_acting(character_id)
    if args.inspect_emotions:
        print(export_emotion_sheet())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
