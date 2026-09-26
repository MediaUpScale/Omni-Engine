"""Character-specific Rhubarb mouths for the cyborg puppets.

ChatGPT is a sleek aerodynamic capsule: a thin caramel stroke and one ivory
ribbon. Claude is an antique microphone aperture: a chunky terracotta bezel
and a taller rounded-rectangle slot. Strokes are anti-aliased Skia paths.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import skia
from PIL import Image, ImageDraw

from utils.pipeline_paths import assets_root

from ..types import VISEMES
from .skia_mouths import (
    archetype_for as skia_archetype_for,
    draw_emotion as draw_skia_emotion,
    draw_viseme as draw_skia_viseme,
    generate_mouth_suite,
)

SPRITE_SIZE = 512

# Measured on llama_cyborg_v2: the orange chin plate is 476px wide and the
# wide Eh mouth (viseme C) is 230px. The same patch is centered in this
# 512px sprite, so docking the sprite at ``chin_width * 512/476`` puts the
# mouth on the chin at Llama's ratio.
LLAMA_CHIN_WIDTH = 476
LLAMA_CHIN_HEIGHT = 362
LLAMA_WIDE_MOUTH = 230
MOUTH_DOCK_CHIN_RATIO = SPRITE_SIZE / LLAMA_CHIN_WIDTH

CAVITY = (42, 20, 20, 255)  # llama cavity #2A1414
DEEP_CAVITY = (13, 14, 18, 255)  # cybernetic capsule #0d0e12
TEETH = (245, 240, 235, 255)  # llama ivory #F5F0EB
TONGUE = (184, 115, 51, 255)  # llama copper tongue #B87333
INK = (43, 26, 21, 255)  # llama ink #2B1A15


@dataclass(frozen=True, slots=True)
class MechaPalette:
    """Outer lip metal for one puppet."""

    character_id: str
    bezel: tuple[int, int, int, int]


PALETTES: dict[str, MechaPalette] = {
    "chatgpt_cyborg_v1": MechaPalette("chatgpt_cyborg_v1", (212, 164, 102, 255)),
    "claude_cyborg_v1": MechaPalette("claude_cyborg_v1", (176, 84, 52, 255)),
}


def metal_stroke(bezel: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Character metal, pulled dark enough to read on a matching chin plate."""
    red, green, blue, _alpha = bezel
    return (
        int(round(red * 0.62 + 18 * 0.38)),
        int(round(green * 0.50 + 14 * 0.50)),
        int(round(blue * 0.42 + 12 * 0.58)),
        255,
    )


_DRAW_SCALE = 3
# ChatGPT's lip stays thin next to Claude's machined frame, with an ink
# edge so the caramel still reads on a matching bronze chin.
_GPT_STROKE = 7.0
_CLAUDE_FRAME = 22.0


def _skia_color(color: tuple[int, int, int, int]) -> int:
    red, green, blue, alpha = color
    return skia.ColorSetARGB(alpha, red, green, blue)


def _fill_paint(color: tuple[int, int, int, int]) -> skia.Paint:
    paint = skia.Paint(AntiAlias=True, Color=_skia_color(color))
    paint.setStyle(skia.Paint.kFill_Style)
    return paint


def _stroke_paint(color: tuple[int, int, int, int], width: float) -> skia.Paint:
    paint = skia.Paint(AntiAlias=True, Color=_skia_color(color))
    paint.setStyle(skia.Paint.kStroke_Style)
    paint.setStrokeWidth(width * _DRAW_SCALE)
    paint.setStrokeCap(skia.Paint.kRound_Cap)
    paint.setStrokeJoin(skia.Paint.kRound_Join)
    return paint


def _canvas() -> tuple[skia.Surface, skia.Canvas]:
    side = SPRITE_SIZE * _DRAW_SCALE
    surface = skia.Surface(side, side)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color4f(0, 0, 0, 0))
    return surface, canvas


def _finish(surface: skia.Surface) -> Image.Image:
    # Skia's snapshot array is BGRA. Pillow reads the buffer as RGBA.
    bgra = surface.makeImageSnapshot().toarray()
    rgba = bgra[..., [2, 1, 0, 3]]
    image = Image.fromarray(rgba)
    return image.resize((SPRITE_SIZE, SPRITE_SIZE), Image.Resampling.LANCZOS)


def _rect(box: tuple[float, float, float, float]) -> skia.Rect:
    left, top, right, bottom = box
    return skia.Rect.MakeLTRB(
        left * _DRAW_SCALE,
        top * _DRAW_SCALE,
        right * _DRAW_SCALE,
        bottom * _DRAW_SCALE,
    )


def _round_rect(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    radius: float,
    fill: tuple[int, int, int, int] | None,
    stroke: tuple[int, int, int, int] | None = None,
    stroke_width: float = 0.0,
) -> None:
    rect = _rect(box)
    corner = radius * _DRAW_SCALE
    if fill is not None:
        canvas.drawRoundRect(rect, corner, corner, _fill_paint(fill))
    if stroke is not None and stroke_width > 0:
        canvas.drawRoundRect(rect, corner, corner, _stroke_paint(stroke, stroke_width))


def _ellipse(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int, int] | None,
    stroke: tuple[int, int, int, int] | None = None,
    stroke_width: float = 0.0,
) -> None:
    rect = _rect(box)
    if fill is not None:
        canvas.drawOval(rect, _fill_paint(fill))
    if stroke is not None and stroke_width > 0:
        canvas.drawOval(rect, _stroke_paint(stroke, stroke_width))


def _quad(
    canvas: skia.Canvas,
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int, int],
    width: float,
) -> None:
    path = skia.Path()
    path.moveTo(start[0] * _DRAW_SCALE, start[1] * _DRAW_SCALE)
    path.quadTo(control[0] * _DRAW_SCALE, control[1] * _DRAW_SCALE, end[0] * _DRAW_SCALE, end[1] * _DRAW_SCALE)
    canvas.drawPath(path, _stroke_paint(color, width))


def _ribbon(canvas: skia.Canvas, box: tuple[float, float, float, float]) -> None:
    """One continuous ivory bar. No tooth grid."""
    height = box[3] - box[1]
    _round_rect(canvas, box, radius=height * 0.5, fill=TEETH, stroke=INK, stroke_width=1.4)


def _slats(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    cavity: tuple[int, int, int, int],
    *,
    count: int = 3,
) -> None:
    """Ivory ribbon broken into a few mechanical slats. Not a tooth grid."""
    _ribbon(canvas, box)
    span = box[2] - box[0]
    gap = max(3.5, span * 0.04)
    for index in range(1, count):
        x_pos = box[0] + span * index / count
        _round_rect(
            canvas,
            (x_pos - gap * 0.5, box[1] + 1.5, x_pos + gap * 0.5, box[3] - 1.5),
            1.2,
            cavity,
        )


def _gpt_smile(canvas: skia.Canvas, *, width: float, lift: float, frown: bool = False) -> None:
    """Wide, shallow caramel smile. The resting shape is the widest."""
    stroke = (212, 164, 102, 255)
    half = width / 2
    bow = -lift if frown else lift
    start = (256 - half, 268)
    end = (256 + half, 264)
    control = (256, 256 - bow)
    _quad(canvas, start, control, end, INK, _GPT_STROKE + 3.0)
    _quad(canvas, start, control, end, stroke, _GPT_STROKE)


def _gpt_capsule(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    *,
    ribbon: bool,
    tongue: bool = False,
    cavity: tuple[int, int, int, int] = DEEP_CAVITY,
    slats: int = 3,
) -> None:
    stroke = (212, 164, 102, 255)
    _ellipse(canvas, box, cavity, INK, _GPT_STROKE + 3.5)
    _ellipse(canvas, box, None, stroke, _GPT_STROKE)
    if ribbon:
        inset_x = (box[2] - box[0]) * 0.08
        top = box[1] + (box[3] - box[1]) * 0.16
        _slats(
            canvas,
            (box[0] + inset_x, top, box[2] - inset_x, top + max(8.0, (box[3] - box[1]) * 0.28)),
            cavity,
            count=slats,
        )
    if tongue:
        _ellipse(
            canvas,
            (256 - 28, box[3] - 22, 256 + 28, box[3] - 4),
            TONGUE,
        )


def _draw_chatgpt(code: str) -> Image.Image:
    surface, canvas = _canvas()
    code = code.upper()[:1]
    if code == "X":
        _gpt_smile(canvas, width=320, lift=28)
    elif code == "A":
        _gpt_smile(canvas, width=286, lift=16)
    elif code == "B":
        _gpt_capsule(canvas, (86, 228, 426, 284), ribbon=True)
    elif code == "C":
        _gpt_capsule(canvas, (64, 214, 448, 298), ribbon=True)
    elif code == "D":
        _gpt_capsule(canvas, (78, 198, 434, 314), ribbon=True, tongue=True)
    elif code == "E":
        _gpt_capsule(canvas, (168, 206, 344, 306), ribbon=False)
    elif code == "F":
        _gpt_capsule(canvas, (206, 226, 306, 286), ribbon=False)
    elif code == "G":
        _gpt_capsule(canvas, (96, 222, 416, 290), ribbon=True)
    elif code == "H":
        _gpt_capsule(canvas, (84, 208, 428, 308), ribbon=True, tongue=True)
    else:
        raise ValueError(code)
    return _finish(surface)


def _claude_aperture(
    canvas: skia.Canvas,
    box: tuple[float, float, float, float],
    *,
    ribbon: bool,
    tongue: bool = False,
) -> None:
    """Chunky terracotta frame around a taller rounded slot."""
    bezel = (176, 84, 52, 255)
    frame = _CLAUDE_FRAME
    radius = min(28.0, (box[3] - box[1]) * 0.28)
    _round_rect(canvas, box, radius, bezel, INK, 3.2)
    inner = (box[0] + frame, box[1] + frame * 0.72, box[2] - frame, box[3] - frame * 0.72)
    if inner[2] - inner[0] < 16 or inner[3] - inner[1] < 10:
        return
    inner_radius = max(6.0, radius - frame * 0.35)
    _round_rect(canvas, inner, inner_radius, CAVITY)
    if ribbon:
        bar_h = max(9.0, (inner[3] - inner[1]) * 0.22)
        _ribbon(
            canvas,
            (
                inner[0] + 8,
                inner[1] + 5,
                inner[2] - 8,
                inner[1] + 5 + bar_h,
            ),
        )
    if tongue:
        _round_rect(
            canvas,
            (256 - 36, inner[3] - 28, 256 + 36, inner[3] - 6),
            10,
            TONGUE,
        )


def _draw_claude(code: str) -> Image.Image:
    surface, canvas = _canvas()
    code = code.upper()[:1]
    if code == "X":
        _claude_aperture(canvas, (108, 232, 404, 280), ribbon=False)
    elif code == "A":
        _claude_aperture(canvas, (124, 238, 388, 274), ribbon=False)
    elif code == "B":
        _claude_aperture(canvas, (96, 200, 416, 312), ribbon=True)
    elif code == "C":
        _claude_aperture(canvas, (100, 176, 412, 336), ribbon=True)
    elif code == "D":
        _claude_aperture(canvas, (118, 148, 394, 364), ribbon=True, tongue=True)
    elif code == "E":
        _claude_aperture(canvas, (186, 156, 326, 356), ribbon=False)
    elif code == "F":
        _claude_aperture(canvas, (210, 184, 302, 328), ribbon=False)
    elif code == "G":
        _claude_aperture(canvas, (102, 190, 410, 322), ribbon=True)
    elif code == "H":
        _claude_aperture(canvas, (112, 164, 400, 348), ribbon=True, tongue=True)
    else:
        raise ValueError(code)
    return _finish(surface)


def _archetype_id(bezel: tuple[int, int, int, int]) -> str:
    for palette in PALETTES.values():
        if palette.bezel[:3] == tuple(bezel[:3]):
            return palette.character_id
    return "chatgpt_cyborg_v1"


def draw_viseme(code: str, palette: MechaPalette) -> Image.Image:
    """Compatibility facade over the unified two-archetype Skia engine."""
    code = code.upper()[:1]
    if code not in VISEMES:
        raise ValueError(f"unknown viseme {code}")
    return draw_skia_viseme(
        skia_archetype_for(palette.character_id),
        code,
    )


def _chatgpt_expression(name: str) -> Image.Image:
    """Acting mouths for the cybernetic capsule. Each one is its own shape."""
    surface, canvas = _canvas()
    stroke = (212, 164, 102, 255)
    if name == "smug":
        # Subtle curved triangle. The peak is shallow, not a sharp wedge.
        path = skia.Path()
        path.moveTo(140 * _DRAW_SCALE, 274 * _DRAW_SCALE)
        path.quadTo(196 * _DRAW_SCALE, 228 * _DRAW_SCALE, 256 * _DRAW_SCALE, 210 * _DRAW_SCALE)
        path.quadTo(316 * _DRAW_SCALE, 228 * _DRAW_SCALE, 372 * _DRAW_SCALE, 272 * _DRAW_SCALE)
        canvas.drawPath(path, _stroke_paint(INK, _GPT_STROKE + 5.0))
        canvas.drawPath(path, _stroke_paint(stroke, _GPT_STROKE + 1.5))
    elif name == "angry":
        box = (108, 228, 404, 286)
        _ellipse(canvas, box, DEEP_CAVITY, INK, _GPT_STROKE + 3.5)
        _ellipse(canvas, box, None, stroke, _GPT_STROKE)
        # Clenched tooth-lock: slats fill the slot and nearly meet.
        _slats(canvas, (132, 242, 380, 272), DEEP_CAVITY, count=4)
    elif name == "shock":
        _ellipse(canvas, (196, 148, 316, 368), DEEP_CAVITY, INK, _GPT_STROKE + 3.5)
        _ellipse(canvas, (196, 148, 316, 368), None, stroke, _GPT_STROKE)
    elif name == "sad":
        _gpt_smile(canvas, width=360, lift=30, frown=True)
    elif name == "triumph":
        _gpt_capsule(canvas, (48, 214, 464, 300), ribbon=True, slats=4)
    elif name == "neutral":
        _gpt_smile(canvas, width=320, lift=28)
    else:
        raise ValueError(name)
    return _finish(surface)


def draw_expression_mouth(name: str, bezel: tuple[int, int, int, int]) -> Image.Image:
    """Compatibility facade over the unified acting-mouth generator."""
    palette = PALETTES[_archetype_id(bezel)]
    archetype = skia_archetype_for(palette.character_id)
    if name == "neutral":
        return draw_skia_viseme(archetype, "X")
    return draw_skia_emotion(archetype, name)


def draw_eyebrow(bezel: tuple[int, int, int, int]) -> Image.Image:
    """Flat arch. A short gap under the nameplate still leaves a readable stroke."""
    scale = 4
    width, height = 360, 42
    canvas = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    stroke = metal_stroke(bezel)
    bounds = (10 * scale, 4 * scale, (width - 10) * scale, (height + 18) * scale)
    draw.arc(bounds, 206, 334, fill=INK, width=16 * scale)
    inset = (14 * scale, 8 * scale, (width - 14) * scale, (height + 12) * scale)
    draw.arc(inset, 210, 330, fill=stroke, width=7 * scale)
    return canvas.resize((width, height), Image.Resampling.LANCZOS)


def _save_sprite(sprite: Image.Image, path: Path) -> None:
    alpha = np.asarray(sprite)[..., 3]
    if int(alpha.max()) == 0 or int((alpha == 0).sum()) == 0:
        raise RuntimeError(f"{path.name} is missing a transparent ground")
    path.parent.mkdir(parents=True, exist_ok=True)
    sprite.save(path, format="PNG", compress_level=1)


def viseme_file(puppet_root: Path, code: str) -> Path:
    """Prefer the viseme suite, then the legacy mouths folder."""
    preferred = puppet_root / "mouths" / "visemes" / f"mouth_{code}.png"
    if preferred.is_file():
        return preferred
    return puppet_root / "mouths" / f"mouth_{code}.png"


def draw_mouths(puppet_id: str, archetype_config: dict | None = None) -> Path:
    """Write the full suite through :mod:`render.skia_mouths`."""
    config = dict(archetype_config or {})
    root = Path(config["destination"]) if config.get("destination") else (
        assets_root() / "puppets" / puppet_id
    )
    requested = str(config.get("style") or "")
    style_aliases = {
        "cybernetic_capsule": "cyber_capsule",
        "aperture": "industrial_louver",
    }
    archetype = style_aliases.get(requested, requested) or skia_archetype_for(
        puppet_id
    )
    visemes = generate_mouth_suite(
        puppet_id,
        archetype=archetype,
        puppet_root=root,
    )
    print(f"mouths: {visemes}")
    return visemes


def generate_viseme_set(character_id: str, destination: Path | None = None) -> Path:
    """Write mouth_A.png through mouth_X.png for one character."""
    config = {"destination": destination} if destination else None
    return draw_mouths(character_id, config)


def generate_all_visemes(character_ids: tuple[str, ...] = tuple(PALETTES)) -> list[Path]:
    return [generate_viseme_set(character_id) for character_id in character_ids]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Llama-blueprint Rhubarb visemes and an inspection sheet.")
    parser.add_argument("--generate-all-visemes", action="store_true")
    parser.add_argument("--inspect-sheet", action="store_true")
    args = parser.parse_args(argv)
    if not args.generate_all_visemes and not args.inspect_sheet:
        parser.error("pass --generate-all-visemes and/or --inspect-sheet")
    if args.generate_all_visemes:
        generate_all_visemes()
    if args.inspect_sheet:
        from .viseme_inspector import export_inspection_sheet

        print(export_inspection_sheet())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
