# -*- coding: utf-8 -*-
"""Procedural fallback generator for puppet skins.

If a character's PNG layers do not exist yet under ``{ASSETS_PATH}/puppets/``,
this module stylizes high-tech cybernetic avatar sprites with
``PIL.ImageDraw`` (gradient shading, drop shadows, specular highlights) so
the pipeline runs out of the box with zero external art dependencies. Ships
two default skins:

* ``gemini_robot`` — sleek angular android: dark metallic casing, glowing
  cyan scan-line visor, audio-reactive hex chest core, articulated speech
  aperture.
* ``llama_robot`` — industrial armored mech: riveted amber/copper plating,
  warm optical lenses with a glint, articulated speech aperture.

Any other ``character_id`` gets a generic robot palette derived from its
name (deterministic per id), so a brand-new skin directory still renders
something reasonable before a real artist swaps in hand-made PNGs.

Generated skins are written under the factory's ``ASSETS_PATH`` (Google
Drive), never into the git repo — see :data:`DEFAULT_PUPPETS_DIR`.
"""
from __future__ import annotations

import json
import hashlib
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .puppet import ALL_LAYER_KEYS, LAYER_KEYS, PuppetSkin, viseme_layer_key
from .types import VISEMES

_LOG = logging.getLogger("animator.asset_generator")

MODULE_ROOT = Path(__file__).resolve().parent


def _default_puppets_dir() -> Path:
    """``{ASSETS_PATH}/puppets`` — the factory (Google Drive) asset root.

    Falls back to a module-local ``assets/puppets`` directory only if
    ``utils.pipeline_paths`` cannot be imported at all (fully standalone
    extraction of this package with no parent factory on the path) — never
    silently back to a repo-tracked location when the factory *is* present.
    """
    try:
        from utils.pipeline_paths import assets_root  # noqa: PLC0415

        return assets_root() / "puppets"
    except Exception as exc:  # noqa: BLE001 — standalone extraction, no parent factory
        _LOG.warning(
            "utils.pipeline_paths unavailable (%s); falling back to a local "
            "assets/puppets dir. Set ASSETS_PATH so generated skins land on "
            "the Drive, not in the repo.",
            exc,
        )
        return MODULE_ROOT / "assets" / "puppets"


DEFAULT_PUPPETS_DIR = _default_puppets_dir()

CANVAS_SIZE = (480, 760)
V2_CANVAS_SIZE = (720, 1080)
SHARED_BACKGROUND_DIRNAME = "shared_backgrounds"
SHARED_PANORAMA_FILENAME = "aiwake_arena_panorama_v2.png"
ARTIST_ASSET_REVISION = 5
_ANCHORS = {
    "head_pivot": [240, 250],
    "neck_pivot": [240, 370],
    "eyes": [240, 222],
    "mouth": [240, 300],
}

_DEFAULT_MANIFESTS: dict[str, dict] = {
    "gemini_robot": {
        "character_id": "gemini_robot",
        "anchors": _ANCHORS,
        "theme": {"glow_color": "#00F0FF", "glow_radius": 46},
        "palette": {
            "style": "sleek",
            "shell": (20, 24, 34),
            "shell_light": (46, 56, 76),
            "shell_dark": (10, 12, 18),
            "accent": (0, 224, 255),
            "accent2": (150, 245, 255),
            "visor": (4, 12, 20),
        },
    },
    "llama_robot": {
        "character_id": "llama_robot",
        "anchors": _ANCHORS,
        "theme": {"glow_color": "#FF8A00", "glow_radius": 46},
        "palette": {
            "style": "industrial",
            "shell": (40, 28, 20),
            "shell_light": (94, 62, 34),
            "shell_dark": (18, 12, 8),
            "accent": (255, 140, 0),
            "accent2": (255, 200, 120),
            "visor": (26, 14, 6),
        },
    },
    "gemini_cyborg_v2": {
        "character_id": "gemini_cyborg_v2",
        "skin_version": "v2",
        "canvas_size": V2_CANVAS_SIZE,
        "anchors": _ANCHORS,
        "theme": {"glow_color": "#00F0FF", "glow_radius": 48},
        "palette": {
            "style": "humanoid_cyborg",
            "brand": "GEMINI",
            "shell": (23, 31, 43),
            "shell_light": (84, 105, 126),
            "shell_dark": (7, 11, 18),
            "accent": (0, 240, 255),
            "accent2": (175, 250, 255),
            "visor": (2, 16, 24),
        },
    },
    "llama_cyborg_v2": {
        "character_id": "llama_cyborg_v2",
        "skin_version": "v2",
        "canvas_size": V2_CANVAS_SIZE,
        "anchors": _ANCHORS,
        "theme": {"glow_color": "#FFB300", "glow_radius": 48},
        "palette": {
            "style": "titan_cyborg",
            "brand": "LLAMA",
            "shell": (50, 34, 24),
            "shell_light": (122, 83, 48),
            "shell_dark": (17, 10, 7),
            "accent": (255, 179, 0),
            "accent2": (255, 220, 145),
            "visor": (25, 12, 4),
        },
    },
}


def _hash_palette(character_id: str) -> dict:
    """Deterministic accent colour for an unknown ``character_id``."""
    digest = sum(character_id.encode("utf-8"))
    hue = (digest * 37) % 360
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 1.0)
    accent = (int(r * 255), int(g * 255), int(b * 255))
    r2, g2, b2 = colorsys.hsv_to_rgb(hue / 360.0, 0.35, 1.0)
    return {
        "style": "sleek",
        "shell": (26, 26, 32),
        "shell_light": (54, 54, 64),
        "shell_dark": (12, 12, 16),
        "accent": accent,
        "accent2": (int(r2 * 255), int(g2 * 255), int(b2 * 255)),
        "visor": (8, 8, 12),
    }


def _manifest_for(character_id: str) -> dict:
    if character_id in _DEFAULT_MANIFESTS:
        return _DEFAULT_MANIFESTS[character_id]
    accent = _hash_palette(character_id)
    glow_hex = "#%02X%02X%02X" % accent["accent"]
    return {
        "character_id": character_id,
        "anchors": _ANCHORS,
        "theme": {"glow_color": glow_hex, "glow_radius": 40},
        "palette": accent,
    }


def ensure_puppet_manifest(puppet_dir: Path) -> Path:
    """Write a default ``puppet.json`` for ``puppet_dir`` if one is missing."""
    puppet_dir = Path(puppet_dir)
    puppet_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = puppet_dir / "puppet.json"
    if manifest_path.is_file():
        return manifest_path
    character_id = puppet_dir.name
    manifest = _manifest_for(character_id)
    canvas_size = tuple(manifest.get("canvas_size", CANVAS_SIZE))
    body_path = puppet_dir / "body.png"
    if body_path.is_file():
        with Image.open(body_path) as body:
            canvas_size = body.size
    scale_x = canvas_size[0] / float(CANVAS_SIZE[0])
    scale_y = canvas_size[1] / float(CANVAS_SIZE[1])
    anchors = {
        name: [int(round(point[0] * scale_x)), int(round(point[1] * scale_y))]
        for name, point in manifest["anchors"].items()
    }
    layers = {key: f"{key}.png" for key in ALL_LAYER_KEYS}
    payload = {
        "character_id": manifest["character_id"],
        "skin_version": manifest.get("skin_version", "unversioned"),
        "canvas_size": list(canvas_size),
        "anchors": anchors,
        "theme": manifest["theme"],
        "layers": layers,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info("wrote default puppet manifest -> %s", manifest_path)
    return manifest_path


def archive_v1_retro_skins(*, puppets_dir: Path | None = None) -> dict[str, Path]:
    """Copy the current unversioned retro skins into immutable V1 IDs.

    Existing archives are never overwritten. This makes the operation safe
    to run from setup, tests, and production without mutating an established
    milestone.
    """
    root = Path(puppets_dir) if puppets_dir else DEFAULT_PUPPETS_DIR
    archived: dict[str, Path] = {}
    for source_id, archive_id in (
        ("gemini_robot", "gemini_robot_v1"),
        ("llama_robot", "llama_robot_v1"),
    ):
        source = root / source_id
        destination = root / archive_id
        if not source.is_dir():
            raise FileNotFoundError(f"cannot archive missing V1 skin: {source}")
        if not destination.exists():
            shutil.copytree(source, destination)
            manifest_path = destination / "puppet.json"
            if manifest_path.is_file():
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload["character_id"] = archive_id
                payload["skin_version"] = "v1"
                payload["archived_from"] = source_id
                manifest_path.write_text(
                    json.dumps(payload, indent=2) + "\n",
                    encoding="utf-8",
                )
            _LOG.info("archived V1 skin without altering source: %s -> %s", source, destination)
        archived[archive_id] = destination
    return archived


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _blank() -> Image.Image:
    return Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))


# --------------------------------------------------------------------------- #
# Gradient / depth helpers — shared shading toolkit for every drawn layer.
# --------------------------------------------------------------------------- #
def _linear_gradient_rgb(size: tuple[int, int], top_rgb: tuple[int, int, int], bottom_rgb: tuple[int, int, int]) -> Image.Image:
    w, h = size
    top = np.array(top_rgb, dtype=np.float32)
    bottom = np.array(bottom_rgb, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
    grad = top.reshape(1, 1, 3) * (1 - ramp) + bottom.reshape(1, 1, 3) * ramp
    arr = np.repeat(grad, w, axis=1).astype(np.uint8)
    return Image.fromarray(arr)


def _shape_mask(size: tuple[int, int], draw_fn) -> Image.Image:
    """``draw_fn(ImageDraw.ImageDraw) -> None`` painted onto a white-on-black mask."""
    mask = Image.new("L", size, 0)
    draw_fn(ImageDraw.Draw(mask))
    return mask


def _gradient_fill(mask: Image.Image, top_rgb: tuple[int, int, int], bottom_rgb: tuple[int, int, int]) -> Image.Image:
    """A vertical-gradient RGBA cutout shaped by ``mask`` (its own alpha)."""
    grad = _linear_gradient_rgb(mask.size, top_rgb, bottom_rgb).convert("RGBA")
    grad.putalpha(mask)
    return grad


def _drop_shadow(size: tuple[int, int], *, cx: int, cy: int, rx: int, ry: int, blur: int = 22, alpha: int = 110) -> Image.Image:
    """Soft blurred ellipse — grounds a character against the arena floor."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(0, 0, 0, alpha))
    return img.filter(ImageFilter.GaussianBlur(blur))


def _brand_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_cyborg_body(palette: dict) -> Image.Image:
    """Broad humanoid shoulders, armored chest, and neck hydraulics."""
    img = _blank()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = CANVAS_SIZE
    cx = w // 2
    titan = palette.get("style") == "titan_cyborg"
    shell = palette["shell"]
    shell_light = palette["shell_light"]
    shell_dark = palette["shell_dark"]
    accent = palette["accent"]

    img.alpha_composite(_drop_shadow(CANVAS_SIZE, cx=cx, cy=h - 25, rx=205, ry=34, blur=20, alpha=140))
    torso_top = 355
    torso = [
        (38 if titan else 58, h - 35),
        (48 if titan else 68, torso_top + 58),
        (132, torso_top - 8),
        (cx - 48, torso_top - 28),
        (cx + 48, torso_top - 28),
        (w - 132, torso_top - 8),
        (w - (48 if titan else 68), torso_top + 58),
        (w - (38 if titan else 58), h - 35),
    ]
    mask = _shape_mask(CANVAS_SIZE, lambda d: d.polygon(torso, fill=255))
    img.alpha_composite(_gradient_fill(mask, shell_light, shell_dark))

    # Carbon-fiber neck core and paired hydraulic pistons.
    draw.rounded_rectangle([cx - 58, 286, cx + 58, 390], radius=18, fill=(*shell_dark, 255))
    for offset in (-34, -17, 17, 34):
        draw.line([(cx + offset, 300), (cx + offset * 1.25, 377)], fill=(*shell_light, 220), width=7)
        draw.line([(cx + offset + 2, 301), (cx + offset * 1.25 + 2, 376)], fill=(*accent, 95), width=2)

    # Sculpted pectoral armor plates with a central sternum channel.
    left_plate = [(75, 420), (cx - 18, 390), (cx - 20, 555), (88, 535)]
    right_plate = [(w - 75, 420), (cx + 18, 390), (cx + 20, 555), (w - 88, 535)]
    for plate in (left_plate, right_plate):
        draw.polygon(plate, fill=(*shell, 245), outline=(*shell_light, 190))
    draw.polygon(
        [(cx - 18, 390), (cx + 18, 390), (cx + 25, h - 70), (cx - 25, h - 70)],
        fill=(*shell_dark, 245),
    )
    draw.line([(cx, 408), (cx, h - 85)], fill=(*accent, 95), width=3)

    # Layered shoulder caps establish a humanoid silhouette.
    for side in (-1, 1):
        sx = cx + side * (158 if titan else 150)
        box = [sx - 72, 356, sx + 72, 485]
        shoulder = _shape_mask(CANVAS_SIZE, lambda d, b=box: d.ellipse(b, fill=255))
        img.alpha_composite(_gradient_fill(shoulder, shell_light, shell))
        draw.arc(box, 192, 348, fill=(*accent, 170), width=4)
        if titan:
            for bolt_x in (sx - 32, sx, sx + 32):
                draw.ellipse([bolt_x - 4, 375, bolt_x + 4, 383], fill=(*palette["accent2"], 220))

    # Reactor is restrained and anatomical—set into the sternum.
    core_y = 568
    draw.ellipse([cx - 44, core_y - 44, cx + 44, core_y + 44], fill=(*accent, 38))
    draw.regular_polygon((cx, core_y, 31), n_sides=6, fill=(*shell_dark, 255), outline=(*accent, 235))
    draw.regular_polygon((cx, core_y, 18), n_sides=6, fill=(*accent, 190))
    return img.filter(ImageFilter.SMOOTH_MORE)


def _draw_cyborg_head(palette: dict) -> Image.Image:
    """Human-contoured android face with branded forehead plate."""
    img = _blank()
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = _ANCHORS["head_pivot"]
    titan = palette.get("style") == "titan_cyborg"
    shell = palette["shell"]
    shell_light = palette["shell_light"]
    shell_dark = palette["shell_dark"]
    accent = palette["accent"]
    accent2 = palette["accent2"]
    brand = str(palette.get("brand") or "CYBORG")

    # Crown -> temples -> cheekbones -> jaw -> chin: recognizably humanoid.
    face = [
        (cx - 62, cy - 158),
        (cx - 88, cy - 120),
        (cx - 98, cy - 48),
        (cx - 84, cy + 38),
        (cx - 62, cy + 88),
        (cx - 34, cy + 119),
        (cx, cy + 132),
        (cx + 34, cy + 119),
        (cx + 62, cy + 88),
        (cx + 84, cy + 38),
        (cx + 98, cy - 48),
        (cx + 88, cy - 120),
        (cx + 62, cy - 158),
    ]
    face_mask = _shape_mask(CANVAS_SIZE, lambda d: d.polygon(face, fill=255))
    img.alpha_composite(_gradient_fill(face_mask, shell_light, shell_dark))

    # Cheek and cyber-jaw planes.
    draw.polygon(
        [(cx - 88, cy + 12), (cx - 42, cy + 30), (cx - 32, cy + 105), (cx - 62, cy + 88)],
        fill=(*shell, 240),
        outline=(*accent, 75),
    )
    draw.polygon(
        [(cx + 88, cy + 12), (cx + 42, cy + 30), (cx + 32, cy + 105), (cx + 62, cy + 88)],
        fill=(*shell, 240),
        outline=(*accent, 75),
    )
    draw.line([(cx - 34, cy + 119), (cx, cy + 132), (cx + 34, cy + 119)], fill=(*accent2, 95), width=3)

    # Branded forehead badge, embossed and laser-lit.
    badge = [cx - 67, cy - 137, cx + 67, cy - 77]
    draw.rounded_rectangle(
        badge,
        radius=10 if not titan else 5,
        fill=(*(shell_dark if not titan else (42, 25, 13)), 255),
        outline=(*accent, 230),
        width=3,
    )
    draw.line([(badge[0] + 10, badge[1] + 9), (badge[2] - 10, badge[1] + 9)], fill=(*shell_light, 150), width=2)
    font = _brand_font(22 if brand == "GEMINI" else 24)
    draw.text((cx + 1, cy - 105 + 2), brand, font=font, fill=(*shell_dark, 220), anchor="mm")
    draw.text((cx, cy - 105), brand, font=font, fill=(*accent, 255), anchor="mm")

    # Optic recess and anatomical nose bridge.
    visor_box = [cx - 77, cy - 63, cx + 77, cy - 8]
    draw.rounded_rectangle(
        visor_box,
        radius=22 if not titan else 12,
        fill=(*palette["visor"], 255),
        outline=(*accent, 125),
        width=3,
    )
    draw.polygon(
        [(cx - 13, cy - 10), (cx + 13, cy - 10), (cx + 19, cy + 43), (cx, cy + 56), (cx - 19, cy + 43)],
        fill=(*shell, 245),
        outline=(*shell_light, 130),
    )
    draw.line([(cx, cy - 4), (cx, cy + 44)], fill=(*accent2, 75), width=2)

    if titan:
        # Rugged temple fasteners and reinforced jaw hinges.
        for side in (-1, 1):
            sx = cx + side * 88
            draw.ellipse([sx - 9, cy - 2, sx + 9, cy + 16], fill=(*accent2, 220), outline=(*shell_dark, 255))
            draw.rounded_rectangle([sx - 12, cy + 43, sx + 12, cy + 85], radius=6, fill=(*shell_dark, 245))
    else:
        # Gemini's clean cranial seams and cyan temple light rails.
        draw.arc([cx - 88, cy - 145, cx + 88, cy + 70], 198, 342, fill=(*accent, 105), width=3)

    return img.filter(ImageFilter.SMOOTH_MORE)


# --------------------------------------------------------------------------- #
# Body
# --------------------------------------------------------------------------- #
def _draw_body(palette: dict) -> Image.Image:
    """Shoulders/torso shell — the static base layer every state paints over."""
    if palette.get("style") in {"humanoid_cyborg", "titan_cyborg"}:
        return _draw_cyborg_body(palette)
    img = _blank()
    w, h = CANVAS_SIZE
    cx = w // 2
    shell = palette["shell"]
    shell_light = palette["shell_light"]
    shell_dark = palette.get("shell_dark", (10, 10, 14))
    accent = palette["accent"]
    industrial = palette.get("style") == "industrial"

    # Ground shadow first so the chassis prints over its own footing.
    img.alpha_composite(_drop_shadow(CANVAS_SIZE, cx=cx, cy=h - 30, rx=170, ry=34, blur=18, alpha=130))

    torso_top = 360
    torso_poly = [
        (cx - 150, h - 40),
        (cx - 170, torso_top + 60),
        (cx - 110, torso_top),
        (cx + 110, torso_top),
        (cx + 170, torso_top + 60),
        (cx + 150, h - 40),
    ]
    torso_mask = _shape_mask(CANVAS_SIZE, lambda d: d.polygon(torso_poly, fill=255))
    # Gradient shaded chassis: lighter at the top (key light), darker toward
    # the base — reads as a lit 3D shell instead of a flat silhouette.
    img.alpha_composite(_gradient_fill(torso_mask, shell_light, shell_dark))

    draw = ImageDraw.Draw(img, "RGBA")
    # Specular highlight streak down the sternum.
    draw.line([(cx, torso_top + 10), (cx, h - 60)], fill=(*shell_light, 120), width=6)
    draw.line([(cx - 4, torso_top + 10), (cx - 4, h - 60)], fill=(255, 255, 255, 40), width=2)

    # Shoulder plates — gradient-shaded ellipses with a rim highlight.
    for side in (-1, 1):
        sx0 = cx + side * 205 if side < 0 else cx + 95
        sx1 = cx + side * 95 if side < 0 else cx + 205
        box = [min(sx0, sx1), torso_top - 10, max(sx0, sx1), torso_top + 90]
        shoulder_mask = _shape_mask(CANVAS_SIZE, lambda d, b=box: d.ellipse(b, fill=255))
        img.alpha_composite(_gradient_fill(shoulder_mask, shell_light, shell))
        draw.arc(box, 200, 340, fill=(*accent, 140), width=3)
        if industrial:
            # Rivets along the shoulder plate rim — heavy-armor detailing.
            for t_frac in (0.2, 0.5, 0.8):
                rx = box[0] + (box[2] - box[0]) * t_frac
                ry = box[1] + 14
                draw.ellipse([rx - 4, ry - 4, rx + 4, ry + 4], fill=(*palette.get("accent2", accent), 200))
                draw.ellipse([rx - 2, ry - 2, rx + 2, ry + 2], fill=(255, 245, 230, 160))

    # Chest core light — audio-reactive hex, with a soft under-glow disc.
    core_y = torso_top + 140
    draw.ellipse([cx - 46, core_y - 40, cx + 46, core_y + 40], fill=(*accent, 55))
    draw.regular_polygon((cx, core_y, 34), n_sides=6, fill=(*accent, 235), outline=(*palette.get("accent2", (255, 255, 255)), 255))
    draw.regular_polygon((cx, core_y, 20), n_sides=6, fill=(255, 255, 255, 190))
    draw.regular_polygon((cx, core_y, 34), n_sides=6, outline=(255, 255, 255, 90), width=1)

    # Neck collar joining to the head.
    collar_mask = _shape_mask(
        CANVAS_SIZE, lambda d: d.rectangle([cx - 46, torso_top - 20, cx + 46, torso_top + 20], fill=255)
    )
    img.alpha_composite(_gradient_fill(collar_mask, shell_light, shell))

    if industrial:
        # Armor seams: horizontal panel lines across the torso.
        for seam_y in (torso_top + 60, torso_top + 100, h - 90):
            draw.line([(cx - 130, seam_y), (cx + 130, seam_y)], fill=(*shell_dark, 200), width=3)
            draw.line([(cx - 130, seam_y - 1), (cx + 130, seam_y - 1)], fill=(*shell_light, 60), width=1)

    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img


# --------------------------------------------------------------------------- #
# Head
# --------------------------------------------------------------------------- #
def _draw_head(palette: dict) -> Image.Image:
    """Angular head shell (no eyes/mouth — those are separate layers)."""
    if palette.get("style") in {"humanoid_cyborg", "titan_cyborg"}:
        return _draw_cyborg_head(palette)
    img = _blank()
    cx, cy = _ANCHORS["head_pivot"]
    shell = palette["shell"]
    shell_light = palette["shell_light"]
    shell_dark = palette.get("shell_dark", (10, 10, 14))
    accent = palette["accent"]
    industrial = palette.get("style") == "industrial"

    helmet_poly = [
        (cx - 90, cy + 110),
        (cx - 100, cy + 10),
        (cx - 70, cy - 90),
        (cx - 20, cy - 130),
        (cx + 20, cy - 130),
        (cx + 70, cy - 90),
        (cx + 100, cy + 10),
        (cx + 90, cy + 110),
    ]
    helmet_mask = _shape_mask(CANVAS_SIZE, lambda d: d.polygon(helmet_poly, fill=255))
    img.alpha_composite(_gradient_fill(helmet_mask, shell_light, shell_dark))

    draw = ImageDraw.Draw(img, "RGBA")
    # Rim-light along the crown so the helmet reads as curved, not flat.
    draw.line([(cx - 70, cy - 88), (cx - 20, cy - 128)], fill=(*shell_light, 160), width=4)
    draw.line([(cx + 20, cy - 128), (cx + 70, cy - 88)], fill=(255, 255, 255, 30), width=2)

    # Visor recess (dark glass band where eyes will sit) with an inner
    # gradient so it reads as a lit lens, not a flat cutout.
    visor_box = [cx - 78, cy - 30, cx + 78, cy + 34]
    visor_mask = _shape_mask(CANVAS_SIZE, lambda d: d.rounded_rectangle(visor_box, radius=22, fill=255))
    img.alpha_composite(_gradient_fill(visor_mask, palette["visor"], shell_dark))
    draw.rounded_rectangle(visor_box, radius=22, outline=(*accent, 90), width=2)

    if not industrial:
        # Sleek variant: thin cyan scan-lines across the visor glass.
        for i, frac in enumerate((0.3, 0.5, 0.7)):
            y = visor_box[1] + (visor_box[3] - visor_box[1]) * frac
            alpha = 70 if i != 1 else 110
            draw.line([(visor_box[0] + 10, y), (visor_box[2] - 10, y)], fill=(*accent, alpha), width=1)
    else:
        # Industrial variant: a perforated sensor grille behind the optics,
        # so the visor recess reads as machined hardware rather than a void.
        for row in range(3):
            gy = visor_box[1] + 14 + row * 18
            for col in range(11):
                gx = visor_box[0] + 14 + col * 13
                draw.ellipse([gx - 2, gy - 2, gx + 2, gy + 2], fill=(*shell_dark, 210))
                draw.ellipse([gx - 2, gy - 2, gx + 1, gy + 1], fill=(*accent, 55))
        draw.line(
            [(cx, visor_box[1] + 6), (cx, visor_box[3] - 6)],
            fill=(*palette.get("accent2", accent), 90),
            width=2,
        )

    # Antenna / accent trim.
    draw.line([(cx, cy - 130), (cx, cy - 165)], fill=(*accent, 255), width=6)
    draw.ellipse([cx - 8, cy - 178, cx + 8, cy - 162], fill=(*accent, 255))
    draw.ellipse([cx - 4, cy - 174, cx + 2, cy - 168], fill=(255, 255, 255, 180))

    # Side accent studs / lens mounts.
    for side in (-1, 1):
        sx = cx + side * 92
        draw.ellipse([sx - 8, cy + 30, sx + 8, cy + 46], fill=(*palette.get("accent2", accent), 220))
        if industrial:
            draw.ellipse([sx - 12, cy + 26, sx + 12, cy + 50], outline=(*shell_dark, 220), width=2)

    if industrial:
        # Heavy plate seams + rivet rows across the brow, matching the
        # "armored mechanical chassis" brief.
        draw.line([(cx - 70, cy - 40), (cx + 70, cy - 40)], fill=(*shell_dark, 200), width=3)
        for dx in (-55, -25, 25, 55):
            draw.ellipse([cx + dx - 4, cy - 55, cx + dx + 4, cy - 47], fill=(*palette.get("accent2", accent), 200))

    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img


# --------------------------------------------------------------------------- #
# Eyes
# --------------------------------------------------------------------------- #
def _draw_eyes(state: str, palette: dict) -> Image.Image:
    img = _blank()
    draw = ImageDraw.Draw(img, "RGBA")
    ex, ey = _ANCHORS["eyes"]
    accent = palette["accent"]
    industrial = palette.get("style") == "industrial"
    cyborg = palette.get("style") == "humanoid_cyborg"
    titan = palette.get("style") == "titan_cyborg"
    gap = 42

    if cyborg and state == "eyes_open":
        # Gemini V2: one luminous optic visor with a focused central iris.
        draw.rounded_rectangle(
            [ex - 70, ey - 13, ex + 70, ey + 13],
            radius=12,
            fill=(*accent, 120),
            outline=(*palette.get("accent2", accent), 245),
            width=3,
        )
        draw.line([(ex - 56, ey), (ex + 56, ey)], fill=(230, 255, 255, 235), width=4)
        draw.ellipse([ex - 10, ey - 10, ex + 10, ey + 10], fill=(*accent, 255))
        draw.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(255, 255, 255, 255))
    elif cyborg:
        draw.line([(ex - 67, ey), (ex + 67, ey)], fill=(*accent, 210), width=4)
    elif state == "eyes_open":
        for dx in (-gap, gap):
            box = [ex + dx - 20, ey - 12, ex + dx + 20, ey + 12]
            draw.ellipse(box, fill=(*accent, 255))
            if industrial or titan:
                # Warm optical lens with a bright glint highlight.
                draw.ellipse(box, outline=(*palette.get("shell_dark", (0, 0, 0)), 200), width=2)
                draw.ellipse([box[0] + 6, box[1] + 2, box[0] + 14, box[1] + 8], fill=(255, 255, 255, 210))
            else:
                draw.ellipse([box[0] + 4, box[1] + 2, box[2] - 12, box[3] - 6], fill=(*palette.get("accent2", accent), 160))
    elif state == "eyes_half":
        for dx in (-gap, gap):
            draw.rounded_rectangle(
                [ex + dx - 20, ey - 5, ex + dx + 20, ey + 5], radius=4, fill=(*accent, 230)
            )
    else:  # eyes_closed
        for dx in (-gap, gap):
            draw.line([(ex + dx - 18, ey), (ex + dx + 18, ey)], fill=(*accent, 200), width=3)

    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img


# --------------------------------------------------------------------------- #
# Mouth — nine anatomical cybernetic speech apertures. These use continuous
# lip rims, cavities, teeth, and tongue silhouettes; no equalizer bars.
# --------------------------------------------------------------------------- #
_VISEME_SHAPE: dict[str, tuple[float, float]] = {
    "A": (0.62, 0.06),  # closed — p/b/m
    "B": (0.66, 0.20),  # slightly open, teeth together
    "C": (0.78, 0.55),  # open — "eh"
    "D": (0.94, 1.00),  # wide open — "ah"
    "E": (0.60, 0.62),  # rounded — "oh"
    "F": (0.38, 0.44),  # puckered — "oo"/"w"
    "G": (0.70, 0.28),  # bite — f/v, upper teeth on lower lip
    "H": (0.76, 0.70),  # tongue — "l"
    "X": (0.56, 0.03),  # rest / silence
}

_MOUTH_MAX_HALF_W = 48
_MOUTH_MAX_HALF_H = 30

def _draw_viseme(viseme: str, palette: dict) -> Image.Image:
    """Draw one articulated cybernetic mouth overlay."""
    img = _blank()
    draw = ImageDraw.Draw(img, "RGBA")
    mx, my = _ANCHORS["mouth"]
    accent = palette["accent"]
    accent2 = palette.get("accent2", (255, 255, 255))
    visor = palette["visor"]
    shell_dark = palette.get("shell_dark", (0, 0, 0))
    industrial = palette.get("style") == "industrial"

    width_frac, open_frac = _VISEME_SHAPE[viseme]
    half_w = int(_MOUTH_MAX_HALF_W * width_frac)
    half_h = max(4, int(_MOUTH_MAX_HALF_H * open_frac))
    housing_w = 60
    housing_h = 42
    draw.rounded_rectangle(
        [mx - housing_w, my - housing_h, mx + housing_w, my + housing_h],
        radius=14 if not industrial else 8,
        fill=(*visor, 245),
        outline=(*shell_dark, 255),
        width=3,
    )
    # Mounting bolts make the overlay read as a speech module.
    for bx in (mx - housing_w + 9, mx + housing_w - 9):
        draw.ellipse([bx - 3, my - 3, bx + 3, my + 3], fill=(*accent2, 180))

    cavity = (mx - half_w, my - half_h, mx + half_w, my + half_h)
    lip_width = 4 if industrial else 3
    cavity_fill = (2, 3, 6, 255)
    teeth = (225, 238, 240, 245)
    tongue = (150, 55, 70, 235) if industrial else (70, 150, 170, 235)

    if viseme in {"A", "X"}:
        # Closed horizontal seam; X is dimmer rest, A is a firm bilabial.
        alpha = 150 if viseme == "X" else 245
        draw.line([(mx - half_w, my), (mx + half_w, my)], fill=(*accent, alpha), width=lip_width)
        draw.line([(mx - half_w + 5, my + 3), (mx + half_w - 5, my + 3)], fill=(*shell_dark, 220), width=2)
    elif viseme == "B":
        # Narrow consonant aperture with a continuous teeth line.
        draw.rounded_rectangle(cavity, radius=half_h, fill=cavity_fill, outline=(*accent, 240), width=lip_width)
        draw.rounded_rectangle(
            [mx - half_w + 5, my - 3, mx + half_w - 5, my + 2],
            radius=2,
            fill=teeth,
        )
    elif viseme == "C":
        # Wide horizontal vowel aperture (E/I).
        draw.ellipse(cavity, fill=cavity_fill, outline=(*accent, 250), width=lip_width)
        draw.pieslice(
            [mx - half_w + 5, my - half_h + 4, mx + half_w - 5, my + half_h],
            180,
            360,
            fill=teeth,
        )
    elif viseme == "D":
        # Tall, wide A-vowel cavity with upper teeth and lower tongue.
        draw.ellipse(cavity, fill=cavity_fill, outline=(*accent, 255), width=lip_width)
        draw.pieslice(
            [mx - half_w + 7, my - half_h + 5, mx + half_w - 7, my + 8],
            180,
            360,
            fill=teeth,
        )
        draw.pieslice(
            [mx - half_w + 9, my + 3, mx + half_w - 9, my + half_h + 8],
            180,
            360,
            fill=tongue,
        )
    elif viseme in {"E", "F"}:
        # Rounded O and small puckered U/W apertures.
        radius_x = half_w if viseme == "E" else max(9, half_w - 4)
        radius_y = half_h if viseme == "E" else max(10, half_h)
        rounded = (mx - radius_x, my - radius_y, mx + radius_x, my + radius_y)
        draw.ellipse(rounded, fill=cavity_fill, outline=(*accent, 255), width=5)
        draw.ellipse(
            [mx - radius_x + 6, my - radius_y + 6, mx + radius_x - 6, my + radius_y - 6],
            outline=(*accent2, 100),
            width=2,
        )
    elif viseme == "G":
        # Dental bite: bright upper teeth overlap a tucked lower lip.
        draw.rounded_rectangle(cavity, radius=half_h, fill=cavity_fill, outline=(*accent, 240), width=lip_width)
        draw.rectangle([mx - half_w + 5, my - half_h + 4, mx + half_w - 5, my + 2], fill=teeth)
        draw.arc(
            [mx - half_w + 7, my - 2, mx + half_w - 7, my + half_h + 8],
            0,
            180,
            fill=(*accent2, 255),
            width=5,
        )
    else:  # H — open mouth with a raised tongue silhouette for L.
        draw.ellipse(cavity, fill=cavity_fill, outline=(*accent, 250), width=lip_width)
        draw.rounded_rectangle(
            [mx - half_w + 10, my + 2, mx + half_w - 10, my + half_h + 4],
            radius=max(5, half_h // 2),
            fill=tongue,
        )
        draw.line([(mx - 12, my + 3), (mx + 12, my + 3)], fill=(*accent2, 190), width=2)
    return img


# --------------------------------------------------------------------------- #
# Glow
# --------------------------------------------------------------------------- #
def _draw_glow(palette: dict, radius: int) -> Image.Image:
    """Radial aura behind the head/chest, pulsed via alpha at composite time."""
    img = _blank()
    cx, cy = _ANCHORS["head_pivot"]
    accent = palette["accent"]

    glow = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    max_r = radius * 4
    steps = 24
    for i in range(steps, 0, -1):
        frac = i / steps
        r = int(max_r * frac)
        alpha = int(70 * (1 - frac) ** 2)
        draw.ellipse([cx - r, cy - r + 40, cx + r, cy + r + 40], fill=(*accent, alpha))
    img.alpha_composite(glow)
    img = img.filter(ImageFilter.GaussianBlur(radius / 3))
    return img


def _draw_background(palette: dict) -> Image.Image:
    """Deep sci-fi studio plate used when no external ``bg.png`` exists."""
    w, h = CANVAS_SIZE
    shell_dark = palette.get("shell_dark", (6, 8, 14))
    accent = palette["accent"]
    img = _linear_gradient_rgb((w, h), (10, 15, 27), shell_dark).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    horizon = int(h * 0.60)
    draw.line([(0, horizon), (w, horizon)], fill=(*accent, 36), width=2)
    for y in range(horizon + 28, h, 42):
        draw.line([(0, y), (w, y)], fill=(*accent, 16), width=1)
    vanish_x = w // 2
    for x in range(-w, w * 2, 70):
        draw.line([(vanish_x, horizon), (x, h)], fill=(*accent, 13), width=1)
    floor_glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(floor_glow).ellipse(
        [w * 0.10, h * 0.63, w * 0.90, h * 0.91],
        fill=(*accent, 34),
    )
    img.alpha_composite(floor_glow.filter(ImageFilter.GaussianBlur(35)))
    return img


def ensure_shared_panorama(*, puppets_dir: Path | None = None) -> Path:
    """Create the shared two-camera V2 arena once and return its path."""
    root = Path(puppets_dir) if puppets_dir else DEFAULT_PUPPETS_DIR
    destination = root / SHARED_BACKGROUND_DIRNAME / SHARED_PANORAMA_FILENAME
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = 2160, 1920
    top = np.array((11, 16, 29), dtype=np.float32)
    bottom = np.array((2, 4, 9), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1, 1)
    gradient = top.reshape(1, 1, 3) * (1 - ramp) + bottom.reshape(1, 1, 3) * ramp
    panorama = Image.fromarray(
        np.repeat(gradient, width, axis=1).astype(np.uint8),
    ).convert("RGBA")
    draw = ImageDraw.Draw(panorama, "RGBA")

    horizon = 1020
    centre = width // 2
    # Rear-wall architecture is continuous across both camera crops.
    for x in range(0, width + 1, 180):
        draw.line([(x, 120), (x, horizon)], fill=(75, 91, 122, 24), width=2)
    for y in range(180, horizon, 150):
        draw.line([(0, y), (width, y)], fill=(75, 91, 122, 20), width=2)
    draw.rounded_rectangle(
        [centre - 330, 230, centre + 330, 840],
        radius=80,
        fill=(7, 13, 23, 180),
        outline=(102, 124, 160, 45),
        width=5,
    )
    draw.ellipse(
        [centre - 220, 320, centre + 220, 760],
        outline=(125, 145, 185, 34),
        width=5,
    )

    # One floor and one vanishing point shared by both reverse angles.
    draw.line([(0, horizon), (width, horizon)], fill=(112, 132, 168, 55), width=3)
    for y in range(horizon + 55, height, 85):
        draw.line([(0, y), (width, y)], fill=(92, 110, 146, 30), width=2)
    for floor_x in range(-width, width * 2, 150):
        draw.line([(centre, horizon), (floor_x, height)], fill=(92, 110, 146, 28), width=2)

    # Atmospheric pools live at opposite sides but share room geometry.
    atmosphere = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    adraw = ImageDraw.Draw(atmosphere, "RGBA")
    adraw.ellipse([-400, 220, 1250, 1720], fill=(0, 240, 255, 34))
    adraw.ellipse([910, 220, 2560, 1720], fill=(255, 179, 0, 32))
    panorama.alpha_composite(atmosphere.filter(ImageFilter.GaussianBlur(180)))
    panorama.convert("RGB").save(destination, quality=95)
    _LOG.info("generated shared V2 panoramic arena -> %s", destination)
    return destination


_GHIBLI_OUTLINE = (43, 26, 21, 255)   # #2B1A15
_GHIBLI_CAVITY = (42, 20, 20, 255)    # #2A1414
_GHIBLI_TEETH = (245, 240, 235, 255)  # #F5F0EB
_GHIBLI_TONGUE = (184, 115, 51, 255)  # #B87333 copper bevel/lip


def _detect_ghibli_chin_plate(head: Image.Image) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Detect the frontal orange chin plate and its articulation center."""
    import cv2  # noqa: PLC0415

    rgba = np.asarray(head.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    alpha_bbox = head.convert("RGBA").getchannel("A").getbbox()
    if alpha_bbox is None:
        raise ValueError("artist head has no opaque pixels")
    x0, y0, x1, y1 = alpha_bbox
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)

    # The plate occupies the lower quarter of the head. Restricting the
    # orange segmentation to that band separates its frontal panel from the
    # bronze side cheek and upper facial shell.
    head_h = y1 - y0
    lower_cut = y0 + int(round(head_h * 0.77))
    warm = (
        (hsv[..., 0] >= 5)
        & (hsv[..., 0] <= 35)
        & (hsv[..., 1] >= 60)
        & (hsv[..., 2] >= 80)
        & (alpha > 0)
    )
    warm[:lower_cut, :] = False
    count, _, stats, centroids = cv2.connectedComponentsWithStats(warm.astype(np.uint8))
    candidates: list[tuple[int, int]] = []
    alpha_cx = (x0 + x1) / 2.0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        component_cx = float(centroids[index][0])
        # Prefer the broad frontal plate near/left of the three-quarter
        # portrait's alpha center, not the narrow right cheek panel.
        if area >= 500 and component_cx <= alpha_cx + (x1 - x0) * 0.08:
            candidates.append((area, index))
    if not candidates:
        raise ValueError("could not isolate warm orange chin plate in artist head")
    _, selected = max(candidates)
    px = int(stats[selected, cv2.CC_STAT_LEFT])
    py = int(stats[selected, cv2.CC_STAT_TOP])
    pw = int(stats[selected, cv2.CC_STAT_WIDTH])
    ph = int(stats[selected, cv2.CC_STAT_HEIGHT])

    # Segmentation starts inside the plate by design; restore its upper
    # blank area while retaining the detected left/right/bottom boundaries.
    plate_top = max(y0, py - int(round(head_h * 0.09)))
    bbox = (px, plate_top, px + pw, py + ph)
    anchor = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return bbox, anchor


def _draw_ghibli_mouth_patch(
    viseme: str,
    *,
    plate_width: int,
    plate_height: int,
    outline: tuple[int, int, int, int] = _GHIBLI_OUTLINE,
    cavity: tuple[int, int, int, int] = _GHIBLI_CAVITY,
    teeth: tuple[int, int, int, int] = _GHIBLI_TEETH,
    tongue: tuple[int, int, int, int] = _GHIBLI_TONGUE,
) -> Image.Image:
    """Supersampled hand-drawn anime mouth patch for one Rhubarb cue."""
    scale = 4
    patch_w = max(240, int(round(plate_width * 0.66)))
    patch_h = max(150, int(round(plate_height * 0.48)))
    canvas = Image.new("RGBA", (patch_w * scale, patch_h * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    cx, cy = patch_w * scale // 2, patch_h * scale // 2
    outline_w = 4 * scale

    def box(width: float, height: float, y_shift: float = 0.0) -> tuple[int, int, int, int]:
        half_w = int(round(width * scale / 2))
        half_h = int(round(height * scale / 2))
        y = cy + int(round(y_shift * scale))
        return cx - half_w, y - half_h, cx + half_w, y + half_h

    def ellipse_outline(bounds, *, fill, width=outline_w) -> None:
        draw.ellipse(bounds, fill=fill, outline=outline, width=width)

    max_w = patch_w * 0.73
    if viseme in {"A", "X"}:
        seam_w = max_w * (0.72 if viseme == "A" else 0.58)
        y = cy + (2 * scale if viseme == "A" else 0)
        draw.arc(
            box(seam_w, 22),
            8,
            172,
            fill=outline,
            width=outline_w,
        )
        if viseme == "A":
            draw.line(
                [(cx - int(seam_w * scale * 0.35), y + 3 * scale),
                 (cx + int(seam_w * scale * 0.35), y + 3 * scale)],
                fill=cavity,
                width=scale,
            )
    elif viseme == "B":
        bounds = box(max_w * 0.82, patch_h * 0.23)
        draw.rounded_rectangle(
            bounds,
            radius=12 * scale,
            fill=cavity,
            outline=outline,
            width=outline_w,
        )
        x0, y0, x1, y1 = bounds
        mid = (y0 + y1) // 2
        draw.rounded_rectangle(
            [x0 + 7 * scale, y0 + 5 * scale, x1 - 7 * scale, mid],
            radius=3 * scale,
            fill=teeth,
        )
        draw.rounded_rectangle(
            [x0 + 10 * scale, mid + scale, x1 - 10 * scale, y1 - 5 * scale],
            radius=3 * scale,
            fill=teeth,
        )
        draw.line([(x0 + 8 * scale, mid), (x1 - 8 * scale, mid)], fill=outline, width=scale)
    elif viseme == "C":
        bounds = box(max_w, patch_h * 0.50)
        ellipse_outline(bounds, fill=cavity)
        x0, y0, x1, y1 = bounds
        draw.pieslice(
            [x0 + 7 * scale, y0 + 5 * scale, x1 - 7 * scale, y0 + 35 * scale],
            180,
            360,
            fill=teeth,
        )
        draw.arc(
            [x0 + 14 * scale, y1 - 30 * scale, x1 - 14 * scale, y1 + 4 * scale],
            180,
            360,
            fill=tongue,
            width=8 * scale,
        )
    elif viseme == "D":
        bounds = box(max_w * 0.78, patch_h * 0.86)
        x0, y0, x1, y1 = bounds
        inset = int(round((x1 - x0) * 0.10))
        aperture = [
            (x0 + inset, y0),
            (x1 - inset, y0),
            (x1, y1 - 8 * scale),
            (x1 - 8 * scale, y1),
            (x0 + 8 * scale, y1),
            (x0, y1 - 8 * scale),
        ]
        draw.polygon(aperture, fill=cavity)
        draw.line(
            aperture + [aperture[0]],
            fill=outline,
            width=outline_w,
            joint="curve",
        )
        draw.pieslice(
            [x0 + inset + 5 * scale, y0 + 5 * scale, x1 - inset - 5 * scale, y0 + 39 * scale],
            180,
            360,
            fill=teeth,
        )
        draw.pieslice(
            [x0 + 12 * scale, y1 - 45 * scale, x1 - 12 * scale, y1 + 8 * scale],
            180,
            360,
            fill=tongue,
        )
    elif viseme == "E":
        ellipse_outline(box(patch_h * 0.57, patch_h * 0.68), fill=cavity)
    elif viseme == "F":
        outer = box(patch_h * 0.34, patch_h * 0.39)
        ellipse_outline(outer, fill=tongue, width=4 * scale)
        draw.ellipse(box(patch_h * 0.16, patch_h * 0.19), fill=cavity)
    elif viseme == "G":
        bounds = box(max_w * 0.82, patch_h * 0.38)
        draw.rounded_rectangle(
            bounds,
            radius=18 * scale,
            fill=cavity,
            outline=outline,
            width=outline_w,
        )
        x0, y0, x1, y1 = bounds
        draw.rectangle(
            [x0 + 8 * scale, y0 + 5 * scale, x1 - 8 * scale, cy + 2 * scale],
            fill=teeth,
        )
        draw.arc(
            [x0 + 12 * scale, cy - 2 * scale, x1 - 12 * scale, y1 + 7 * scale],
            0,
            180,
            fill=tongue,
            width=7 * scale,
        )
    else:  # H: open L posture with raised tongue tip.
        bounds = box(max_w * 0.88, patch_h * 0.65)
        ellipse_outline(bounds, fill=cavity)
        x0, _, x1, y1 = bounds
        draw.rounded_rectangle(
            [x0 + 15 * scale, cy + 2 * scale, x1 - 15 * scale, y1 + 3 * scale],
            radius=18 * scale,
            fill=tongue,
            outline=outline,
            width=2 * scale,
        )
        # Raised tongue tip in the center is the key L-viseme silhouette.
        draw.ellipse(
            [cx - 19 * scale, cy - 10 * scale, cx + 19 * scale, cy + 25 * scale],
            fill=tongue,
            outline=outline,
            width=2 * scale,
        )

    return canvas.resize((patch_w, patch_h), Image.Resampling.LANCZOS)


def _draw_artist_eyelids(
    canvas_size: tuple[int, int],
    eye_bboxes: tuple[tuple[int, int, int, int], ...],
    *,
    casing: tuple[int, int, int, int],
    closed: bool,
    ink: tuple[int, int, int, int] = _GHIBLI_OUTLINE,
) -> Image.Image:
    """Supersampled metal eyelid overlays aligned to finished artist optics."""
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    scale = 4
    for x0, y0, x1, y1 in eye_bboxes:
        width, height = x1 - x0, y1 - y0
        patch = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch, "RGBA")
        bounds = (2 * scale, 2 * scale, (width - 2) * scale, (height - 2) * scale)
        if closed:
            draw.ellipse(bounds, fill=casing, outline=ink, width=4 * scale)
            cy = height * scale // 2
            draw.line(
                [(10 * scale, cy), ((width - 10) * scale, cy)],
                fill=ink,
                width=4 * scale,
            )
        else:
            # Upper lid covers half the lens; its curved lower edge follows
            # the circular housing rather than reading as a rectangular mask.
            draw.pieslice(bounds, 180, 360, fill=casing, outline=ink, width=4 * scale)
            cy = height * scale // 2
            draw.arc(
                (8 * scale, cy - 12 * scale, (width - 8) * scale, cy + 18 * scale),
                180,
                360,
                fill=ink,
                width=4 * scale,
            )
        patch = patch.resize((width, height), Image.Resampling.LANCZOS)
        layer.alpha_composite(patch, (x0, y0))
    return layer


def generate_ghibli_llama_assets(puppet_dir: Path) -> dict[str, object]:
    """Ingest official Llama cels and generate calibrated anime visemes.

    Artist sources are never modified. The descriptively named torso source
    is copied to the canonical ``body.png`` path when needed; all generated
    mouth overlays are written under ``mouths/`` on the same full canvas.
    """
    puppet_dir = Path(puppet_dir)
    head_path = puppet_dir / "head.png"
    body_path = puppet_dir / "body.png"
    torso_source = puppet_dir / "Vintage_robot_bust_portrait_2K_20260921234728.png"
    if not head_path.is_file():
        raise FileNotFoundError(f"official Ghibli head is missing: {head_path}")
    if not body_path.is_file():
        if not torso_source.is_file():
            raise FileNotFoundError(
                f"official Ghibli body is missing: expected {body_path} or {torso_source}"
            )
        shutil.copy2(torso_source, body_path)
        _LOG.info("ingested official Ghibli torso non-destructively -> %s", body_path)

    with Image.open(head_path) as opened:
        head = opened.convert("RGBA")
    with Image.open(body_path) as opened:
        body = opened.convert("RGBA")
    if head.size != body.size:
        raise ValueError(f"Ghibli head/body canvases must match: head={head.size}, body={body.size}")

    plate_bbox, detected_mouth_anchor = _detect_ghibli_chin_plate(head)
    mouth_anchor = (detected_mouth_anchor[0] - 18, detected_mouth_anchor[1])
    head_digest = hashlib.sha256(head_path.read_bytes()).hexdigest()
    mouths_dir = puppet_dir / "mouths"
    mouths_dir.mkdir(parents=True, exist_ok=True)
    plate_width = plate_bbox[2] - plate_bbox[0]
    plate_height = plate_bbox[3] - plate_bbox[1]

    for viseme in VISEMES:
        patch = _draw_ghibli_mouth_patch(
            viseme,
            plate_width=plate_width,
            plate_height=plate_height,
        )
        layer = Image.new("RGBA", head.size, (0, 0, 0, 0))
        layer.alpha_composite(
            patch,
            (
                mouth_anchor[0] - patch.width // 2,
                mouth_anchor[1] - patch.height // 2,
            ),
        )
        layer.save(mouths_dir / f"mouth_{viseme}.png")

    eye_bboxes = ((330, 935, 540, 1175), (680, 905, 895, 1150))
    # The artist head already contains its optical sensors. Open is
    # transparent; half/closed overlays are matching dark-bronze eyelids.
    transparent = Image.new("RGBA", head.size, (0, 0, 0, 0))
    transparent.save(puppet_dir / "eyes_open.png")
    _draw_artist_eyelids(
        head.size,
        eye_bboxes,
        casing=(76, 46, 36, 255),
        closed=False,
    ).save(puppet_dir / "eyes_half.png")
    _draw_artist_eyelids(
        head.size,
        eye_bboxes,
        casing=(76, 46, 36, 255),
        closed=True,
    ).save(puppet_dir / "eyes_blink.png")

    manifest_path = puppet_dir / "puppet.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    payload.update(
        {
            "character_id": "llama_cyborg_v2",
            "skin_version": "v2",
            "asset_profile": "ghibli_cel_v2",
            "asset_revision": ARTIST_ASSET_REVISION,
            "canvas_size": list(head.size),
            "anchors": {
                "head_pivot": [head.size[0] // 2, int(round(head.size[1] * 0.40))],
                "neck_pivot": [detected_mouth_anchor[0] + 45, plate_bbox[3] - 63],
                "eyes": [
                    detected_mouth_anchor[0],
                    int(round(plate_bbox[1] - plate_height * 0.58)),
                ],
                "mouth": list(mouth_anchor),
            },
            "calibration": {
                "chin_plate_bbox": list(plate_bbox),
                "eye_bboxes": [list(box) for box in eye_bboxes],
                "layer_offsets": {"head": [0, 0], "body": [0, 0]},
                "source_head_sha256": head_digest,
            },
        }
    )
    layers = dict(payload.get("layers") or {})
    layers.update(
        {
            "body": "body.png",
            "head": "head.png",
            "eyes_open": "eyes_open.png",
            "eyes_half": "eyes_half.png",
            "eyes_blink": "eyes_blink.png",
        }
    )
    layers.update({f"mouth_{viseme}": f"mouths/mouth_{viseme}.png" for viseme in VISEMES})
    payload["layers"] = layers
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _LOG.info(
        "calibrated Ghibli Llama: canvas=%s chin=%s mouth_anchor=%s",
        head.size,
        plate_bbox,
        mouth_anchor,
    )
    return {
        "canvas_size": head.size,
        "chin_plate_bbox": plate_bbox,
        "mouth_anchor": mouth_anchor,
        "mouths_dir": mouths_dir,
    }


def generate_gemini_anime_assets(puppet_dir: Path) -> dict[str, object]:
    """Calibrate the high-resolution Gemini cel and replace vector mouth UI."""
    puppet_dir = Path(puppet_dir)
    head_path = puppet_dir / "head.png"
    body_path = puppet_dir / "body.png"
    if not head_path.is_file() or not body_path.is_file():
        raise FileNotFoundError(f"Gemini artist head/body are incomplete under {puppet_dir}")
    with Image.open(head_path) as opened:
        head = opened.convert("RGBA")
    with Image.open(body_path) as opened:
        body = opened.convert("RGBA")
    if head.size != body.size:
        raise ValueError(f"Gemini head/body canvases must match: head={head.size}, body={body.size}")

    alpha_bbox = head.getchannel("A").getbbox()
    if alpha_bbox is None:
        raise ValueError("Gemini artist head has no opaque pixels")
    x0, y0, x1, y1 = alpha_bbox
    face_w, face_h = x1 - x0, y1 - y0
    # Lower facial plate beneath the twin optics. Proportional calibration
    # remains stable if the artist re-exports this cel at another resolution.
    plate_bbox = (
        x0 + int(round(face_w * 0.30)),
        y0 + int(round(face_h * 0.60)),
        x0 + int(round(face_w * 0.86)),
        y0 + int(round(face_h * 0.92)),
    )
    prior_mouth_anchor = (
        (plate_bbox[0] + plate_bbox[2]) // 2 + 75,
        (plate_bbox[1] + plate_bbox[3]) // 2 + 55,
    )
    mouth_anchor = (prior_mouth_anchor[0] + 18, prior_mouth_anchor[1] - 20)
    neck_pivot = (
        (plate_bbox[0] + plate_bbox[2]) // 2 - 25,
        y0 + int(round(face_h * 0.91)),
    )
    mouths_dir = puppet_dir / "mouths"
    mouths_dir.mkdir(parents=True, exist_ok=True)
    anime_palette = {
        "outline": (21, 32, 38, 255),     # #152026 charcoal
        "cavity": (14, 23, 28, 255),      # #0E171C blue-black
        "teeth": (238, 240, 232, 255),   # soft off-white
        "tongue": (90, 133, 141, 255),    # #5A858D cyan metal bevel
    }
    for viseme in VISEMES:
        patch = _draw_ghibli_mouth_patch(
            viseme,
            plate_width=plate_bbox[2] - plate_bbox[0],
            plate_height=plate_bbox[3] - plate_bbox[1],
            **anime_palette,
        )
        patch = patch.rotate(-4.0, resample=Image.Resampling.BICUBIC, expand=False)
        patch_pixels = np.asarray(patch, dtype=np.uint8).copy()
        patch_pixels[patch_pixels[..., 3] < 8, :3] = 0
        patch = Image.fromarray(patch_pixels)
        layer = Image.new("RGBA", head.size, (0, 0, 0, 0))
        layer.alpha_composite(
            patch,
            (
                mouth_anchor[0] - patch.width // 2,
                mouth_anchor[1] - patch.height // 2,
            ),
        )
        layer.save(mouths_dir / f"mouth_{viseme}.png")

    eye_bboxes = ((650, 1005, 810, 1180), (1020, 1040, 1135, 1190))
    # Optics are already finished in the artist head. Open is transparent;
    # the other states add graphite upper lids or closed seams.
    transparent = Image.new("RGBA", head.size, (0, 0, 0, 0))
    transparent.save(puppet_dir / "eyes_open.png")
    _draw_artist_eyelids(
        head.size,
        eye_bboxes,
        casing=(51, 65, 75, 255),
        closed=False,
        ink=(21, 32, 38, 255),
    ).save(puppet_dir / "eyes_half.png")
    _draw_artist_eyelids(
        head.size,
        eye_bboxes,
        casing=(51, 65, 75, 255),
        closed=True,
        ink=(21, 32, 38, 255),
    ).save(puppet_dir / "eyes_blink.png")

    manifest_path = puppet_dir / "puppet.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    payload.update(
        {
            "character_id": "gemini_cyborg_v2",
            "skin_version": "v2",
            "asset_profile": "gemini_anime_cel_v2",
            "asset_revision": ARTIST_ASSET_REVISION,
            "canvas_size": list(head.size),
            "anchors": {
                "head_pivot": [head.size[0] // 2, int(round(head.size[1] * 0.40))],
                "neck_pivot": list(neck_pivot),
                "eyes": [prior_mouth_anchor[0], y0 + int(round(face_h * 0.48))],
                "mouth": list(mouth_anchor),
            },
            "calibration": {
                "facial_plate_bbox": list(plate_bbox),
                "eye_bboxes": [list(box) for box in eye_bboxes],
                "mouth_rotation_deg": -4.0,
                "layer_offsets": {"head": [0, 0], "body": [0, 0]},
                "source_head_sha256": hashlib.sha256(head_path.read_bytes()).hexdigest(),
            },
        }
    )
    layers = dict(payload.get("layers") or {})
    layers.update(
        {
            "body": "body.png",
            "head": "head.png",
            "eyes_open": "eyes_open.png",
            "eyes_half": "eyes_half.png",
            "eyes_blink": "eyes_blink.png",
        }
    )
    layers.update({f"mouth_{viseme}": f"mouths/mouth_{viseme}.png" for viseme in VISEMES})
    payload["layers"] = layers
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _LOG.info(
        "calibrated Gemini anime face: canvas=%s plate=%s mouth_anchor=%s neck_pivot=%s",
        head.size,
        plate_bbox,
        mouth_anchor,
        neck_pivot,
    )
    return {
        "canvas_size": head.size,
        "facial_plate_bbox": plate_bbox,
        "mouth_anchor": mouth_anchor,
        "neck_pivot": neck_pivot,
        "mouths_dir": mouths_dir,
    }


def generate_puppet_assets(skin: PuppetSkin, *, missing: list[str] | None = None) -> None:
    """Write every missing PNG layer for ``skin`` into its own directory."""
    skin.root.mkdir(parents=True, exist_ok=True)
    palette = _manifest_for(skin.character_id).get("palette") or _hash_palette(skin.character_id)
    wanted = set(missing) if missing is not None else set(ALL_LAYER_KEYS)

    generators = {
        "body": lambda: _draw_body(palette),
        "head": lambda: _draw_head(palette),
        "eyes_open": lambda: _draw_eyes("eyes_open", palette),
        "eyes_half": lambda: _draw_eyes("eyes_half", palette),
        "eyes_blink": lambda: _draw_eyes("eyes_closed", palette),
        "glow": lambda: _draw_glow(palette, skin.theme.glow_radius or 40),
        "bg": lambda: _draw_background(palette),
    }
    for viseme in VISEMES:
        generators[viseme_layer_key(viseme)] = lambda v=viseme: _draw_viseme(v, palette)

    # Match an existing external art canvas. Procedural functions draw on
    # the canonical 480x760 canvas, then scale only generated overlays to
    # the artist's body/head resolution; existing high-res files are never
    # overwritten or downscaled.
    target_size = skin.reference_canvas_size
    for anchor_key in ("body", "head", "eyes_open"):
        anchor_path = skin.layer_path(anchor_key)
        if anchor_path.is_file():
            with Image.open(anchor_path) as anchor_image:
                target_size = anchor_image.size
            break

    for key in ALL_LAYER_KEYS:
        if key not in wanted:
            continue
        image = generators[key]()
        if key != "bg" and image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        dest = skin.layer_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest)
        _LOG.debug("generated %s -> %s", key, dest)


def generate_default_puppet(
    character_id: str,
    *,
    puppets_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Convenience: manifest + every layer (base + nine visemes) for one of
    the built-in skins (or any arbitrary id, which gets a deterministic
    hashed palette)."""
    root = Path(puppets_dir) if puppets_dir else DEFAULT_PUPPETS_DIR
    puppet_dir = root / character_id
    ensure_puppet_manifest(puppet_dir)
    artist_profiles = {
        "gemini_cyborg_v2": ("gemini_anime_cel_v2", generate_gemini_anime_assets),
        "llama_cyborg_v2": ("ghibli_cel_v2", generate_ghibli_llama_assets),
    }
    artist_profile = artist_profiles.get(character_id)
    if artist_profile is not None and (puppet_dir / "head.png").is_file():
        with Image.open(puppet_dir / "head.png") as candidate_head:
            high_resolution_artist_head = candidate_head.width >= 1000
        if high_resolution_artist_head:
            expected_profile, generator = artist_profile
            manifest_path = puppet_dir / "puppet.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            current_digest = hashlib.sha256((puppet_dir / "head.png").read_bytes()).hexdigest()
            configured_digest = str(
                (payload.get("calibration") or {}).get("source_head_sha256") or ""
            )
            mouths_ready = all(
                (puppet_dir / "mouths" / f"mouth_{viseme}.png").is_file()
                for viseme in VISEMES
            )
            if (
                payload.get("asset_profile") != expected_profile
                or payload.get("asset_revision") != ARTIST_ASSET_REVISION
                or configured_digest != current_digest
                or not (puppet_dir / "body.png").is_file()
                or not mouths_ready
                or not (puppet_dir / "eyes_half.png").is_file()
                or not (payload.get("anchors") or {}).get("neck_pivot")
            ):
                generator(puppet_dir)
    skin = PuppetSkin.load(puppet_dir)
    missing = list(ALL_LAYER_KEYS) if force else skin.missing_layers()
    if missing:
        generate_puppet_assets(skin, missing=missing)
    return puppet_dir


def generate_all_defaults(*, puppets_dir: Path | None = None) -> list[Path]:
    return [generate_default_puppet(cid, puppets_dir=puppets_dir) for cid in _DEFAULT_MANIFESTS]


__all__ = [
    "CANVAS_SIZE",
    "DEFAULT_PUPPETS_DIR",
    "SHARED_BACKGROUND_DIRNAME",
    "SHARED_PANORAMA_FILENAME",
    "V2_CANVAS_SIZE",
    "archive_v1_retro_skins",
    "ensure_puppet_manifest",
    "ensure_shared_panorama",
    "generate_all_defaults",
    "generate_default_puppet",
    "generate_gemini_anime_assets",
    "generate_ghibli_llama_assets",
    "generate_puppet_assets",
]


if __name__ == "__main__":  # pragma: no cover — manual asset regen
    logging.basicConfig(level=logging.INFO)
    for path in generate_all_defaults():
        print(f"generated {path}")
