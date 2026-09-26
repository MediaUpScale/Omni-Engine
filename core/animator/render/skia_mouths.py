"""Deterministic Skia mouth library for the two cyborg lineages.

Every sprite is authored on a 512x512 transparent canvas at 3x resolution,
then downsampled once.  The public generator writes the complete Rhubarb and
acting suites; callers never need to mix geometry from different characters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import skia
from PIL import Image

from utils.pipeline_paths import assets_root

from ..types import VISEMES

SPRITE_SIZE = 512
DRAW_SCALE = 3
EMOTIONS = ("smug", "angry", "shock", "sad", "triumph")

INK = (18, 21, 28, 255)           # #12151c
CAVITY = (14, 16, 20, 255)        # #0e1014
IVORY = (244, 238, 218, 255)
CARAMEL = (205, 151, 88, 255)
CYBER_LIP = (30, 22, 18, 255)      # espresso #1e1612
ANTIQUE_BRASS = (200, 147, 62, 255)  # #c8933e
GOLD_HIGHLIGHT = (212, 160, 84, 255)  # #d4a054
CLAUDE_LIP = (74, 50, 30, 255)     # oil-rubbed bronze #4a321e
COPPER = (190, 111, 46, 255)

VIEW_FOLDERS = {
    "facing_front": "front",
    "front": "front",
    "facing_left": "facing_left",
    "facing_right": "facing_right",
}


@dataclass(frozen=True, slots=True)
class MouthArchetype:
    name: str
    bezel: tuple[int, int, int, int]


ARCHETYPES: dict[str, MouthArchetype] = {
    "cyber_capsule": MouthArchetype("cyber_capsule", CARAMEL),
    "organic_brass_cutout": MouthArchetype(
        "organic_brass_cutout",
        CLAUDE_LIP,
    ),
    # Read compatibility for manifests produced before the V3 redesign.
    "industrial_louver": MouthArchetype("industrial_louver", CLAUDE_LIP),
}

PUPPET_ARCHETYPE = {
    "chatgpt_cyborg_v1": "cyber_capsule",
    "gemini_cyborg_v2": "cyber_capsule",
    "claude_cyborg_v1": "organic_brass_cutout",
    "llama_cyborg_v2": "organic_brass_cutout",
}


def archetype_for(puppet_id: str) -> str:
    """Resolve a puppet id without silently sharing an unrelated design."""
    if puppet_id in PUPPET_ARCHETYPE:
        return PUPPET_ARCHETYPE[puppet_id]
    lowered = puppet_id.lower()
    if "chatgpt" in lowered or "gemini" in lowered:
        return "cyber_capsule"
    if "claude" in lowered or "llama" in lowered:
        return "organic_brass_cutout"
    raise KeyError(f"no Skia mouth archetype reserved for {puppet_id}")


def _color(rgba: tuple[int, int, int, int]) -> int:
    red, green, blue, alpha = rgba
    return skia.ColorSetARGB(alpha, red, green, blue)


def _paint(
    rgba: tuple[int, int, int, int],
    *,
    stroke: bool = False,
    width: float = 1.0,
) -> skia.Paint:
    paint = skia.Paint(AntiAlias=True, Color=_color(rgba))
    paint.setStyle(skia.Paint.kStroke_Style if stroke else skia.Paint.kFill_Style)
    if stroke:
        paint.setStrokeWidth(width * DRAW_SCALE)
        paint.setStrokeCap(skia.Paint.kRound_Cap)
        paint.setStrokeJoin(skia.Paint.kRound_Join)
    return paint


def _surface() -> tuple[skia.Surface, skia.Canvas]:
    surface = skia.Surface(SPRITE_SIZE * DRAW_SCALE, SPRITE_SIZE * DRAW_SCALE)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color4f(0, 0, 0, 0))
    return surface, canvas


def _finish(surface: skia.Surface) -> Image.Image:
    bgra = surface.makeImageSnapshot().toarray()
    rgba = bgra[..., [2, 1, 0, 3]]
    return Image.fromarray(rgba).resize(
        (SPRITE_SIZE, SPRITE_SIZE),
        Image.Resampling.LANCZOS,
    )


def _rect(box: tuple[float, float, float, float]) -> skia.Rect:
    left, top, right, bottom = box
    return skia.Rect.MakeLTRB(
        left * DRAW_SCALE,
        top * DRAW_SCALE,
        right * DRAW_SCALE,
        bottom * DRAW_SCALE,
    )


def _round_rect(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    radius: float,
    fill: tuple[int, int, int, int] | None,
    *,
    stroke: tuple[int, int, int, int] | None = None,
    stroke_width: float = 0.0,
) -> None:
    rect = _rect(box)
    scaled_radius = radius * DRAW_SCALE
    if fill is not None:
        canvas.drawRoundRect(rect, scaled_radius, scaled_radius, _paint(fill))
    if stroke is not None and stroke_width > 0:
        canvas.drawRoundRect(
            rect,
            scaled_radius,
            scaled_radius,
            _paint(stroke, stroke=True, width=stroke_width),
        )


def _oval(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int, int] | None,
    *,
    stroke: tuple[int, int, int, int] | None = None,
    stroke_width: float = 0.0,
) -> None:
    rect = _rect(box)
    if fill is not None:
        canvas.drawOval(rect, _paint(fill))
    if stroke is not None and stroke_width > 0:
        canvas.drawOval(rect, _paint(stroke, stroke=True, width=stroke_width))


def _curve(
    canvas: skia.Canvas,
    points: tuple[tuple[float, float], ...],
    color: tuple[int, int, int, int],
    width: float,
) -> None:
    path = skia.Path()
    path.moveTo(points[0][0] * DRAW_SCALE, points[0][1] * DRAW_SCALE)
    if len(points) == 3:
        path.quadTo(
            points[1][0] * DRAW_SCALE,
            points[1][1] * DRAW_SCALE,
            points[2][0] * DRAW_SCALE,
            points[2][1] * DRAW_SCALE,
        )
    else:
        path.cubicTo(
            points[1][0] * DRAW_SCALE,
            points[1][1] * DRAW_SCALE,
            points[2][0] * DRAW_SCALE,
            points[2][1] * DRAW_SCALE,
            points[3][0] * DRAW_SCALE,
            points[3][1] * DRAW_SCALE,
        )
    canvas.drawPath(path, _paint(color, stroke=True, width=width))


def _curved_teeth(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    *,
    drop: float = 0.0,
) -> None:
    """One continuous ivory ribbon attached to the arched upper lip."""
    left, top, right, bottom = box
    top += drop
    bottom += drop
    path = skia.Path()
    path.moveTo(left * DRAW_SCALE, (top + 7) * DRAW_SCALE)
    path.cubicTo(
        (left + (right - left) * 0.28) * DRAW_SCALE,
        (top - 5) * DRAW_SCALE,
        (left + (right - left) * 0.72) * DRAW_SCALE,
        (top - 4) * DRAW_SCALE,
        right * DRAW_SCALE,
        (top + 7) * DRAW_SCALE,
    )
    path.lineTo(right * DRAW_SCALE, (bottom - 3) * DRAW_SCALE)
    path.cubicTo(
        (left + (right - left) * 0.70) * DRAW_SCALE,
        bottom * DRAW_SCALE,
        (left + (right - left) * 0.30) * DRAW_SCALE,
        bottom * DRAW_SCALE,
        left * DRAW_SCALE,
        (bottom - 3) * DRAW_SCALE,
    )
    path.close()
    canvas.drawPath(path, _paint(IVORY))
    canvas.drawPath(path, _paint(INK, stroke=True, width=1.6))


def _closed_slit(
    canvas: skia.Canvas,
    lip: tuple[int, int, int, int],
    *,
    tight: bool,
) -> None:
    """Confident rest slit, or the closed P/B/M line."""
    if tight:
        points = ((124, 258), (256, 254), (388, 258))
        ink_width, lip_width = 9.0, 5.0
    else:
        points = ((102, 270), (256, 238), (410, 268))
        ink_width, lip_width = 11.0, 6.5
    _curve(canvas, points, INK, ink_width)
    _curve(canvas, points, lip, lip_width)


def _organic_cutout(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    *,
    lip: tuple[int, int, int, int],
    teeth: bool,
    tongue: bool = False,
    teeth_on_lower: bool = False,
) -> None:
    """Imperfect cut-out aperture; deliberately not a CAD stadium."""
    left, top, right, bottom = box
    mid = (top + bottom) * 0.5
    path = skia.Path()
    path.moveTo((left + 3) * DRAW_SCALE, (mid + 2) * DRAW_SCALE)
    path.cubicTo(
        (left + 18) * DRAW_SCALE,
        (top + 5) * DRAW_SCALE,
        (right - 38) * DRAW_SCALE,
        (top - 4) * DRAW_SCALE,
        (right - 3) * DRAW_SCALE,
        (mid - 4) * DRAW_SCALE,
    )
    path.cubicTo(
        (right - 18) * DRAW_SCALE,
        (bottom - 1) * DRAW_SCALE,
        (left + 30) * DRAW_SCALE,
        (bottom + 5) * DRAW_SCALE,
        (left + 3) * DRAW_SCALE,
        (mid + 2) * DRAW_SCALE,
    )
    path.close()
    canvas.drawPath(path, _paint(CAVITY))
    canvas.drawPath(path, _paint(INK, stroke=True, width=12.0))
    canvas.drawPath(path, _paint(lip, stroke=True, width=7.0))
    if teeth:
        inset = (right - left) * 0.10
        if teeth_on_lower:
            band = max(18.0, (bottom - top) * 0.26)
            teeth_box = (
                left + inset,
                bottom - band,
                right - inset,
                bottom - 3,
            )
        else:
            teeth_box = (
                left + inset,
                top + 5,
                right - inset,
                top + max(18.0, (bottom - top) * 0.31),
            )
        canvas.save()
        canvas.clipPath(path, True)
        _curved_teeth(canvas, teeth_box)
        canvas.restore()
    if tongue:
        _oval(canvas, (226, bottom - 27, 290, bottom - 7), COPPER)


def _top_ribbon(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
) -> None:
    """A single cream ribbon hugging the top of a cyber capsule."""
    left, top, right, bottom = box
    height = bottom - top
    _round_rect(
        canvas,
        (left, top, right, bottom),
        height * 0.5,
        IVORY,
        stroke=INK,
        stroke_width=1.6,
    )


def _cyber_capsule(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    *,
    teeth: bool,
    tongue: bool = False,
    teeth_on_lower: bool = False,
) -> None:
    _organic_cutout(
        canvas,
        box,
        lip=CYBER_LIP,
        teeth=teeth,
        tongue=tongue,
        teeth_on_lower=teeth_on_lower,
    )


_CYBER_BOXES = {
    "A": (108, 232, 404, 282),
    "B": (88, 224, 424, 290),
    "C": (64, 210, 448, 302),
    "D": (72, 192, 440, 322),
    "E": (170, 202, 342, 310),
    "F": (206, 220, 306, 292),
    "G": (96, 218, 416, 294),
    "H": (84, 204, 428, 312),
}


def _draw_cyber_viseme(code: str) -> Image.Image:
    surface, canvas = _surface()
    if code == "X":
        _closed_slit(canvas, CYBER_LIP, tight=False)
    elif code == "A":
        _closed_slit(canvas, CYBER_LIP, tight=True)
    elif code == "D":
        _joy_smile(canvas, bezel=CYBER_LIP, box=_CYBER_BOXES["D"])
    else:
        _cyber_capsule(
            canvas,
            _CYBER_BOXES[code],
            teeth=code in {"B", "C", "G", "H"},
            tongue=code == "H",
            teeth_on_lower=code == "G",
        )
    return _finish(surface)


def _louver(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    *,
    teeth: bool,
    tongue: bool = False,
    teeth_on_lower: bool = False,
) -> None:
    """Organic Llama-lineage cutout with oil-rubbed bronze trim.

    The old rectilinear radio box is deliberately gone: both outer and inner
    contours are true stadiums with continuous cream teeth.
    """
    _organic_cutout(
        canvas,
        box,
        lip=CLAUDE_LIP,
        teeth=teeth,
        tongue=tongue,
        teeth_on_lower=teeth_on_lower,
    )


_LOUVER_BOXES = {
    "A": (122, 234, 390, 278),
    "B": (98, 206, 414, 306),
    "C": (94, 180, 418, 332),
    "D": (112, 150, 400, 362),
    "E": (184, 162, 328, 354),
    "F": (210, 188, 302, 326),
    "G": (104, 194, 408, 318),
    "H": (112, 168, 400, 346),
}


def _draw_louver_viseme(code: str) -> Image.Image:
    surface, canvas = _surface()
    if code == "X":
        _closed_slit(canvas, CLAUDE_LIP, tight=False)
    elif code == "A":
        _closed_slit(canvas, CLAUDE_LIP, tight=True)
    elif code == "D":
        _joy_smile(
            canvas,
            bezel=CLAUDE_LIP,
            box=_LOUVER_BOXES["D"],
            teeth_drop=6.0,
        )
    else:
        _louver(
            canvas,
            _LOUVER_BOXES[code],
            teeth=code in {"B", "C", "G", "H"},
            tongue=code == "H",
            teeth_on_lower=code == "G",
        )
    return _finish(surface)


def _joy_smile(
    canvas: skia.Canvas,
    *,
    bezel: tuple[int, int, int, int],
    box: tuple[float, float, float, float],
    smirk_px: float = 0.0,
    teeth_drop: float = 0.0,
    sharp_apex: bool = False,
) -> None:
    """Signature arched smile with an inverted rounded-triangle cavity."""
    left, top, right, bottom = box
    path = skia.Path()
    tip_x = (left + right) * 0.5 + smirk_px * 0.8
    if sharp_apex:
        peak_x = left + (right - left) * 0.62
        path.moveTo(left * DRAW_SCALE, (top + smirk_px) * DRAW_SCALE)
        path.lineTo(peak_x * DRAW_SCALE, (top - 18) * DRAW_SCALE)
        path.lineTo(right * DRAW_SCALE, (top - smirk_px) * DRAW_SCALE)
        path.lineTo(tip_x * DRAW_SCALE, bottom * DRAW_SCALE)
    else:
        path.moveTo(left * DRAW_SCALE, (top + smirk_px) * DRAW_SCALE)
        path.cubicTo(
            (left + (right - left) * 0.30) * DRAW_SCALE,
            (top - 7) * DRAW_SCALE,
            (left + (right - left) * 0.70) * DRAW_SCALE,
            (top - 5) * DRAW_SCALE,
            right * DRAW_SCALE,
            (top - smirk_px) * DRAW_SCALE,
        )
        path.quadTo(
            (right - (right - left) * 0.20) * DRAW_SCALE,
            (top + (bottom - top) * 0.55) * DRAW_SCALE,
            tip_x * DRAW_SCALE,
            bottom * DRAW_SCALE,
        )
        path.quadTo(
            (left + (right - left) * 0.20) * DRAW_SCALE,
            (top + (bottom - top) * 0.55) * DRAW_SCALE,
            left * DRAW_SCALE,
            (top + smirk_px) * DRAW_SCALE,
        )
    path.close()
    canvas.drawPath(path, _paint(CAVITY))
    canvas.drawPath(path, _paint(INK, stroke=True, width=12.0))
    canvas.drawPath(path, _paint(bezel, stroke=True, width=7.0))
    inset = (right - left) * 0.10
    canvas.save()
    canvas.clipPath(path, True)
    _curved_teeth(
        canvas,
        (
            left + inset,
            top + 3,
            right - inset,
            top + 28,
        ),
        drop=teeth_drop,
    )
    canvas.restore()


def _draw_cyber_emotion(name: str) -> Image.Image:
    surface, canvas = _surface()
    if name == "smug":
        _joy_smile(
            canvas,
            bezel=CYBER_LIP,
            box=(108, 230, 404, 310),
            smirk_px=12,
            sharp_apex=True,
        )
    elif name == "angry":
        _cyber_capsule(canvas, (100, 224, 412, 290), teeth=True)
        for x_pos in (180, 256, 332):
            _round_rect(canvas, (x_pos - 4, 240, x_pos + 4, 276), 2, CAVITY)
    elif name == "shock":
        _oval(canvas, (194, 144, 318, 370), CAVITY, stroke=INK, stroke_width=12)
        _oval(canvas, (194, 144, 318, 370), None, stroke=CYBER_LIP, stroke_width=7)
    elif name == "sad":
        points = ((108, 246), (256, 296), (404, 246))
        _curve(canvas, points, INK, 11)
        _curve(canvas, points, CYBER_LIP, 6.5)
    elif name == "triumph":
        _joy_smile(
            canvas,
            bezel=CYBER_LIP,
            box=(48, 208, 464, 322),
        )
    else:
        raise ValueError(name)
    return _finish(surface)


def _draw_louver_emotion(name: str) -> Image.Image:
    surface, canvas = _surface()
    if name == "smug":
        _joy_smile(
            canvas,
            bezel=CLAUDE_LIP,
            box=(104, 228, 408, 310),
            smirk_px=11,
        )
    elif name == "angry":
        _louver(canvas, (90, 216, 422, 300), teeth=True)
    elif name == "shock":
        _oval(canvas, (176, 132, 336, 380), CAVITY, stroke=INK, stroke_width=12)
        _oval(
            canvas,
            (176, 132, 336, 380),
            None,
            stroke=ANTIQUE_BRASS,
            stroke_width=7,
        )
    elif name == "sad":
        points = ((112, 242), (256, 304), (400, 242))
        _curve(canvas, points, INK, 11)
        _curve(canvas, points, CLAUDE_LIP, 6.5)
    elif name == "triumph":
        _joy_smile(
            canvas,
            bezel=CLAUDE_LIP,
            box=(72, 202, 440, 326),
        )
    else:
        raise ValueError(name)
    return _finish(surface)


def _view_geometry(image: Image.Image, view_name: str) -> Image.Image:
    """Bake asymmetric foreshortening into a native per-view sprite."""
    folder = VIEW_FOLDERS.get(view_name, view_name)
    if folder == "front":
        return image
    side = float(SPRITE_SIZE - 1)
    source = np.float32(
        [[0, 0], [side, 0], [side, side], [0, side]]
    )
    if folder == "facing_left":
        target = np.float32(
            [[54, 30], [side - 5, 0], [side, side], [78, side - 33]]
        )
    elif folder == "facing_right":
        target = np.float32(
            [[5, 0], [side - 54, 30], [side - 78, side - 33], [0, side]]
        )
    else:
        raise KeyError(view_name)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(
        np.asarray(image),
        matrix,
        (SPRITE_SIZE, SPRITE_SIZE),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped)


def draw_viseme(
    archetype: str,
    code: str,
    *,
    view_name: str = "front",
) -> Image.Image:
    code = code.upper()[:1]
    if code not in VISEMES:
        raise ValueError(f"unknown Rhubarb viseme {code}")
    if archetype == "cyber_capsule":
        image = _draw_cyber_viseme(code)
    elif archetype in {"industrial_louver", "organic_brass_cutout"}:
        image = _draw_louver_viseme(code)
    else:
        raise KeyError(archetype)
    return _view_geometry(image, view_name)


def draw_emotion(
    archetype: str,
    name: str,
    *,
    view_name: str = "front",
) -> Image.Image:
    if name not in EMOTIONS:
        raise ValueError(name)
    if archetype == "cyber_capsule":
        image = _draw_cyber_emotion(name)
    elif archetype in {"industrial_louver", "organic_brass_cutout"}:
        image = _draw_louver_emotion(name)
    else:
        raise KeyError(archetype)
    return _view_geometry(image, view_name)


def _save(image: Image.Image, path: Path) -> None:
    alpha = np.asarray(image)[..., 3]
    if int(alpha.max()) == 0 or int((alpha == 0).sum()) == 0:
        raise RuntimeError(f"{path.name} does not have a clean alpha ground")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=1)


def generate_mouth_suite(
    puppet_id: str,
    *,
    archetype: str | None = None,
    puppet_root: Path | None = None,
) -> Path:
    """Generate 9 visemes and 5 acting mouths for one puppet."""
    resolved = archetype or archetype_for(puppet_id)
    if resolved not in ARCHETYPES:
        raise KeyError(resolved)
    root = puppet_root or (assets_root() / "puppets" / puppet_id)
    viseme_dir = root / "mouths" / "visemes"
    emotion_dir = root / "mouths" / "expressions"
    for code in VISEMES:
        _save(draw_viseme(resolved, code), viseme_dir / f"mouth_{code}.png")
    for name in EMOTIONS:
        _save(draw_emotion(resolved, name), emotion_dir / f"mouth_{name}.png")
    # Runtime-neutral expression remains a named sprite, but is generated
    # from the same archetype's canonical rest shape.
    _save(draw_viseme(resolved, "X"), emotion_dir / "mouth_neutral.png")
    return viseme_dir


def generate_view_mouth_suites(
    puppet_id: str,
    *,
    archetype: str | None = None,
    puppet_root: Path | None = None,
    suite_dir: Path | None = None,
) -> dict[str, Path]:
    """Write independent front/left/right suites without touching legacy sets."""
    resolved = archetype or archetype_for(puppet_id)
    if resolved not in ARCHETYPES:
        raise KeyError(resolved)
    root = puppet_root or (assets_root() / "puppets" / puppet_id)
    base = suite_dir or (root / "mouths")
    written: dict[str, Path] = {}
    for view_name in ("front", "facing_left", "facing_right"):
        destination = base / view_name
        for code in VISEMES:
            _save(
                draw_viseme(resolved, code, view_name=view_name),
                destination / f"mouth_{code}.png",
            )
        for name in EMOTIONS:
            _save(
                draw_emotion(resolved, name, view_name=view_name),
                destination / f"mouth_{name}.png",
            )
        _save(
            draw_viseme(resolved, "X", view_name=view_name),
            destination / "mouth_neutral.png",
        )
        written[view_name] = destination
    return written


__all__ = [
    "ARCHETYPES",
    "EMOTIONS",
    "SPRITE_SIZE",
    "archetype_for",
    "draw_emotion",
    "draw_viseme",
    "generate_mouth_suite",
    "generate_view_mouth_suites",
]
