"""Concept contact sheet. Generates approval art and does not rig or matte it.

Usage:
    python -m core.animator.factory.generate_concepts --character-id deepseek_cyborg_v3 --candidates 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.pipeline_paths import outputs_root

from .create_puppet import generate_character_image
from .rigger import birefnet_cutout

_SHARED = (
    "Studio Ghibli 1980s-1990s vintage anime cel animation, Hayao Miyazaki, "
    "Laputa and The Iron Giant. Clean flat watercolor cel shading. "
    "Medium bust of a charming dignified tin-plate robot named DeepSeek, "
    "angled exactly 25 degrees facing LEFT. "
    "Head height is about one third of the frame. Broad mechanical shoulders "
    "and thick upper arms descend along both edges down toward the bottom. "
    "CRITICAL TORSO: the chest and abdominal armor are solid and opaque and "
    "continue past the bottom edge of the 9:16 canvas. No arched hollow chest, "
    "no floating bib, no rounded cutoff, no empty space under the torso. "
    "Bold crisp 3.5px hand-inked dark anime contours (#12151C). "
    "No Power Rangers, no Marvel, no Iron Man, no modern plastic helmet, no muscular pecs. "
    "Pure solid black background. --ar 9:16"
)

_BATCH_2 = (
    (
        "D",
        "Laputa steam-tin golem. Rounded friendly dome head in dark matte naval blue. "
        "Wide rounded brass collar with round rivets. Rectangular brass forehead "
        "nameplate reading DEEPSEEK. Warm glowing circular emerald lenses, innocent "
        "and calm. Smooth bronze chin shield with no mouth. Solid heavy boiler-plate "
        "torso and full upper arms continuing to the bottom edge of the canvas.",
    ),
    (
        "E",
        "Retro-nautical boiler mecha. Stout spherical naval-blue helmet, antique brass "
        "porthole dial ears and rivets. Straight brass forehead plate reading DEEPSEEK. "
        "Soft horizontal pill-shaped green optic sensor. Robust industrial chest with "
        "copper piping along the collar. Solid torso and thick arms descending fully "
        "to the bottom edge.",
    ),
    (
        "F",
        "Vintage 1950s radio-tin robot. Charming curved rectangular jaw, smooth "
        "naval-blue chassis. Bolted brass nameplate reading DEEPSEEK. Gentle circular "
        "green sensor eyes with delicate concentric iris circles. Thick mechanical arms "
        "and a full-width chest extending solidly to the bottom edge.",
    ),
)

_CANDIDATES = (
    (
        "A",
        "Antique nautical diving-suit mecha. Tall oval domed naval-blue helmet "
        "with circular brass pressure dials on the ear nodes. Bolted brass "
        "nameplate reading DEEPSEEK. Soft glowing emerald lenses. Smooth curved "
        "bronze chin shield with no mouth. Robust boiler-plate chest, brass "
        "rimmed collar, visible upper-arm ball joints.",
    ),
    (
        "B",
        "Laputa ancient guardian tin-toy mecha. Nostalgic rounded chassis, warm "
        "matte iron, exposed bronze rivets. Friendly stoic circular glowing green "
        "ocular lenses. Antique ventilation slots on a bronze chin guard, no human "
        "mouth. Natural mecha neck collar docking into broad articulated shoulder "
        "and arm plates.",
    ),
    (
        "C",
        "Dieselpunk brass scholar mecha. Heavy cast-iron and matte dark navy "
        "plating, visible copper conduits and small brass gears at the neck and "
        "shoulders. Prominent brass forehead plate engraved DEEPSEEK. Full medium "
        "bust with articulated arm sockets and upper arms along the frame edges.",
    ),
)


def _prompt(detail: str) -> str:
    return f"{_SHARED} {detail}"


def _label_font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_panel(path: Path, size: tuple[int, int]) -> Image.Image:
    """Scale the bust so the chest is cut by the bottom edge, not floating above it."""
    import numpy as np

    with Image.open(path) as opened:
        image = opened.convert("RGB")
    rgb = np.asarray(image)
    ink = rgb.max(axis=2) > 24
    ys, xs = np.nonzero(ink)
    if not ys.size:
        return Image.new("RGB", size, (0, 0, 0))
    crop = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    target_w, target_h = size
    # Leave the lower 12% of the figure past the frame so an arched hem cannot sit inside it.
    scale = max(
        (target_w * 0.92) / crop.width,
        (target_h * 1.12) / crop.height,
    )
    resized = crop.resize(
        (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", size, (0, 0, 0))
    x = (target_w - resized.width) // 2
    y = target_h - resized.height + int(round(target_h * 0.04))
    panel.paste(resized, (x, y))
    return panel


def build_concept_sheet(
    *,
    character_id: str,
    candidates: int = 3,
    batch: int = 1,
) -> Path:
    """Generate up to three busts and stitch a labeled inspection sheet."""
    if character_id.strip().lower() != "deepseek_cyborg_v3":
        raise ValueError(f"unsupported concept character: {character_id}")
    count = int(candidates)
    if count != 3:
        raise ValueError("this approval stage renders exactly 3 candidates")
    harness = outputs_root() / "aiwake" / "_test_harness"
    harness.mkdir(parents=True, exist_ok=True)
    panel_size = (1080, 1920)
    panels: list[Image.Image] = []
    font = _label_font(64)
    if batch not in (1, 2):
        raise ValueError("batch must be 1 or 2")
    roster = _CANDIDATES if batch == 1 else _BATCH_2
    for label, detail in roster:
        source = harness / f"deepseek_concept_{label.lower()}.png"
        generate_character_image(_prompt(detail), source, aspect_ratio="9:16")
        panel = _fit_panel(source, panel_size)
        draw = ImageDraw.Draw(panel)
        text = f"[ {label} ]"
        box = draw.textbbox((0, 0), text, font=font)
        width = box[2] - box[0]
        x = (panel_size[0] - width) // 2
        draw.rectangle((x - 24, 36, x + width + 24, 36 + (box[3] - box[1]) + 28), fill=(12, 14, 18))
        draw.text((x, 48), text, fill=(232, 214, 170), font=font)
        panels.append(panel)
    sheet = Image.new("RGB", (panel_size[0] * count, panel_size[1]), (0, 0, 0))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * panel_size[0], 0))
    sheet_name = (
        "deepseek_concepts_sheet.png"
        if batch == 1
        else "deepseek_concepts_sheet_batch2.png"
    )
    destination = harness / sheet_name
    sheet.save(destination, format="PNG", compress_level=1)
    return destination


_F_TORSO_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel, Hayao Miyazaki tin robot. "
    "BODY ONLY, no head, no face, no eyes, no jaw, no text, no nameplate, no letters. "
    "Lean slender dark matte naval-blue mecha torso, slim chest, not a barrel and not a tank. "
    "Head-to-torso proportion about 1 to 1.6. Discreet dome rivets on elegant plates. "
    "Rounded brass collar socket at the top, ready to receive a neck. "
    "Thick upper arms descend along both side edges. "
    "The chest armor is solid and continues past the bottom edge of the 9:16 canvas. "
    "No arched hollow hem, no floating bib. "
    "Crisp 3.5px hand-inked outlines (#12151C). Pure solid black background. --ar 9:16"
)


def _neck_row(alpha: np.ndarray) -> int:
    ys, xs = np.nonzero(alpha > 16)
    top = int(ys.min())
    height = int(ys.max()) - top
    best_y = top + int(height * 0.42)
    best_width = 10**9
    for y in range(top + int(height * 0.28), top + int(height * 0.52)):
        width = int(np.count_nonzero(alpha[y] > 16))
        if 8 < width < best_width:
            best_width = width
            best_y = y
    return best_y


def _isolate_f_head(cutout: Image.Image) -> Image.Image:
    """Keep the natural jaw. The cut sits on the neck stump, below the chin."""
    rgba = np.asarray(cutout.convert("RGBA"), dtype=np.uint8).copy()
    alpha = rgba[..., 3]
    cut = _neck_row(alpha) + 6
    rgba[cut:, :, 3] = 0
    ys, xs = np.nonzero(rgba[..., 3] > 16)
    if not xs.size:
        raise RuntimeError("candidate F head isolation found no pixels")
    crop = Image.fromarray(rgba).crop(
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    )
    return crop


def _eye_y(head: Image.Image) -> int:
    rgba = np.asarray(head.convert("RGBA"))
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    green = (rgb[..., 1] > 120) & (rgb[..., 1] > rgb[..., 0] + 25) & (alpha > 16)
    rows = np.nonzero(green.any(axis=1))[0]
    if rows.size:
        return int(np.median(rows))
    ys = np.nonzero(alpha > 16)[0]
    return int(ys.min() + (ys.max() - ys.min()) * 0.42)


def _collar_top(body: Image.Image) -> int:
    alpha = np.asarray(body.convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    center = int(np.median(xs))
    half = max(10, int((int(xs.max()) - int(xs.min())) * 0.12))
    neck = alpha[:, max(0, center - half) : center + half]
    rows = np.nonzero(neck > 16)[0]
    return int(rows.min()) if rows.size else int(ys.min())


def assemble_candidate_f(output: Path) -> Path:
    """Static F approval frame. No rig, no video, no audio."""
    harness = outputs_root() / "aiwake" / "_test_harness"
    source = harness / "deepseek_concept_f.png"
    if not source.is_file():
        raise FileNotFoundError(f"candidate F art is missing: {source}")
    with Image.open(source) as opened:
        figure = birefnet_cutout(opened.convert("RGB"))
    head = _isolate_f_head(figure)
    body_source = harness / "deepseek_f_torso_original.png"
    generate_character_image(_F_TORSO_PROMPT, body_source, aspect_ratio="9:16")
    with Image.open(body_source) as opened:
        body = birefnet_cutout(opened.convert("RGB"))
    parts = harness / "deepseek_f"
    parts.mkdir(parents=True, exist_ok=True)
    head.save(parts / "head.png", format="PNG", compress_level=1)
    body.save(parts / "body.png", format="PNG", compress_level=1)

    from core.animator.asset_generator import (  # noqa: PLC0415
        DEFAULT_PUPPETS_DIR,
        ensure_shared_panorama,
    )

    panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
    library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
    frame = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")

    body_alpha = np.asarray(body)[..., 3]
    bys, bxs = np.nonzero(body_alpha > 16)
    body_crop = body.crop((int(bxs.min()), int(bys.min()), int(bxs.max()) + 1, int(bys.max()) + 1))
    eye = _eye_y(head)
    head_scale = 500 / max(1, head.height)
    head_resized = head.resize(
        (
            max(1, int(round(head.width * head_scale))),
            max(1, int(round(head.height * head_scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    eye_scaled = int(round(eye * head_scale))
    head_y = 640 - eye_scaled
    chin_y = head_y + int(round(_opaque_bottom(head) * head_scale))
    # Collar meets the chin. The chest continues through the bottom edge.
    body_scale = (1920 - (chin_y - 20)) / max(1, body_crop.height)
    body_scale = max(body_scale, (1080 * 0.78) / body_crop.width)
    body_resized = body_crop.resize(
        (
            max(1, int(round(body_crop.width * body_scale))),
            max(1, int(round(body_crop.height * body_scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    body_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    body_layer.paste(
        body_resized,
        ((1080 - body_resized.width) // 2, 1920 - body_resized.height),
    )
    frame.alpha_composite(body_layer)
    frame.alpha_composite(head_resized, ((1080 - head_resized.width) // 2, head_y))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(output, format="PNG", compress_level=1)
    print(f"eye_line={head_y + int(round(eye * head_scale))}")
    print(f"approval: {output}")
    return output


def _opaque_bottom(image: Image.Image) -> int:
    ys = np.nonzero(np.asarray(image.convert("RGBA"))[..., 3] > 16)[0]
    return int(ys.max())


def _erase_chest_nameplate(rgba: np.ndarray, neck_y: int) -> np.ndarray:
    """Replace the brass DEEPSEEK chest box with the plate color just above it."""
    image = rgba.copy()
    rgb = image[..., :3]
    alpha = image[..., 3]
    brass = (
        (rgb[..., 0] > 130)
        & (rgb[..., 1] > 100)
        & (rgb[..., 2] < 190)
        & (rgb[..., 0].astype(np.int16) > rgb[..., 2].astype(np.int16) + 25)
        & (alpha > 16)
    )
    brass[: neck_y + 36, :] = False
    ys, xs = np.nonzero(brass)
    if ys.size < 80:
        return image
    pad = 10
    mask = np.zeros(alpha.shape, dtype=np.uint8)
    mask[
        max(0, int(ys.min()) - pad) : int(ys.max()) + pad,
        max(0, int(xs.min()) - pad) : int(xs.max()) + pad,
    ] = 255
    mask[alpha < 16] = 0
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    side = np.concatenate(
        (
            image[y0 : y1 + 1, max(0, x0 - 24) : x0, :3].reshape(-1, 3),
            image[y0 : y1 + 1, x1 + 1 : x1 + 25, :3].reshape(-1, 3),
        )
    )
    flat = np.median(side, axis=0)
    image[y0 : y1 + 1, x0 : x1 + 1, :3] = flat.astype(np.uint8)
    return image


def finalize_candidate_f(output: Path) -> Path:
    """Medium-bust F from the native batch-2 drawing. No new body, no video."""
    harness = outputs_root() / "aiwake" / "_test_harness"
    source = harness / "deepseek_concept_f.png"
    if not source.is_file():
        raise FileNotFoundError(f"candidate F art is missing: {source}")
    with Image.open(source) as opened:
        cutout = birefnet_cutout(opened.convert("RGB"))
    rgba = np.asarray(cutout.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    neck = _neck_row(alpha)
    rgba = _erase_chest_nameplate(rgba, neck)
    ys, xs = np.nonzero(rgba[..., 3] > 16)
    crown = int(ys.min())
    head_bottom = neck
    head_height = max(1, head_bottom - crown)
    eye_rows = np.nonzero(
        (
            (rgba[..., 1] > 120)
            & (rgba[..., 1] > rgba[..., 0] + 25)
            & (rgba[..., 3] > 16)
            & (np.arange(rgba.shape[0])[:, None] < neck)
        ).any(axis=1)
    )[0]
    eye = int(np.median(eye_rows)) if eye_rows.size else crown + int(head_height * 0.45)
    # Keep the collar and upper chest. Waist, hips and lower arms fall away.
    cut_y = neck + int(round(head_height * 0.95))
    rgba[cut_y:, :, 3] = 0
    scale = 550 / float(head_height)
    head = _isolate_f_head(Image.fromarray(rgba))
    body_rgba = rgba.copy()
    body_rgba[: neck - 8, :, 3] = 0
    bys, bxs = np.nonzero(body_rgba[..., 3] > 16)
    if not bxs.size:
        raise RuntimeError("candidate F body isolation found no pixels")
    body = Image.fromarray(body_rgba).crop(
        (int(bxs.min()), int(bys.min()), int(bxs.max()) + 1, int(bys.max()) + 1)
    )
    parts = harness / "deepseek_f"
    parts.mkdir(parents=True, exist_ok=True)
    head.save(parts / "head.png", format="PNG", compress_level=1)
    body.save(parts / "body.png", format="PNG", compress_level=1)

    from core.animator.asset_generator import (  # noqa: PLC0415
        DEFAULT_PUPPETS_DIR,
        ensure_shared_panorama,
    )

    eye_local = _eye_y(head)
    head_resized = head.resize(
        (
            max(1, int(round(head.width * scale))),
            max(1, int(round(head.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    head_y = 640 - int(round(eye_local * scale))
    chin_y = head_y + int(round(_opaque_bottom(head) * scale))
    body_scale = max((1080 * 1.18) / body.width, (1920 - (chin_y - 16)) / body.height)
    body_resized = body.resize(
        (
            max(1, int(round(body.width * body_scale))),
            max(1, int(round(body.height * body_scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
    frame = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
    frame = frame.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
    frame.alpha_composite(body_resized, ((1080 - body_resized.width) // 2, 1920 - body_resized.height))
    frame.alpha_composite(head_resized, ((1080 - head_resized.width) // 2, head_y))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(output, format="PNG", compress_level=1)
    print(f"eye_line={head_y + int(round(eye_local * scale))} head_scale_px={head_resized.height}")
    print(f"approval: {output}")
    return output


_TORSO_OPTIONS = (
    (
        "1",
        "Classic boiler plate. Gentle curved dark matte naval-blue chest, color #243342, "
        "subtle dome rivets, no text. Warm rounded brass collar socket. Natural upper-arm "
        "sockets at the sides.",
    ),
    (
        "2",
        "Articulated segmented torso in the same dark matte naval blue #243342, never silver "
        "and never white. Two horizontal chest plates with subtle dark ink panel lines, "
        "no text. Antique bronze collar rim with small screws. Lean balanced upper arms.",
    ),
    (
        "3",
        "Steam-piping scholar torso. Minimalist dark navy chest, no text. Two subtle curved "
        "copper conduits from the collar to the armpits. High brass neck cowl.",
    ),
)


def _tight_rgba(image: Image.Image) -> Image.Image:
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    if not xs.size:
        raise RuntimeError("empty cutout")
    return image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def _scale_to(image: Image.Image, *, width: int | None = None, height: int | None = None) -> Image.Image:
    if height is not None:
        scale = height / max(1, image.height)
    elif width is not None:
        scale = width / max(1, image.width)
    else:
        raise ValueError("width or height is required")
    return image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )


def torso_candidates_for_f(output: Path) -> Path:
    """Three native torsos under the locked Head F. No chest paint-over."""
    harness = outputs_root() / "aiwake" / "_test_harness"
    head_path = harness / "deepseek_f" / "head.png"
    if not head_path.is_file():
        raise FileNotFoundError(f"Head F is missing: {head_path}")
    with Image.open(head_path) as opened:
        head = _scale_to(_tight_rgba(opened.convert("RGBA")), height=540)
    from core.animator.asset_generator import (  # noqa: PLC0415
        DEFAULT_PUPPETS_DIR,
        ensure_shared_panorama,
    )

    panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
    library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
    library = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
    font = _label_font(64)
    panels: list[Image.Image] = []
    shared = (
        "Studio Ghibli 1990s cel, Hayao Miyazaki tin mecha. BODY ONLY. "
        "No head, no face, no jaw, no letters, no nameplate, no chest text. "
        "Angled 25 degrees facing LEFT: the left shoulder recedes, the right shoulder "
        "comes forward. Medium bust. Shoulder span is about twice the head width, "
        "dignified, not a giant barrel and not a stick. Upper arms only. "
        "Every chest plate is dark matte naval blue #243342. "
        "No hips, no waist belt. The collar rim is the top of the body, open and ready "
        "for a jaw to sit inside it. Chest armor continues to the bottom edge. "
        "Crisp 3.5px ink (#12151C). Pure solid black background. --ar 9:16"
    )
    for label, detail in _TORSO_OPTIONS:
        source = harness / f"deepseek_torso_option_{label}.png"
        generate_character_image(f"{shared} {detail}", source, aspect_ratio="9:16")
        with Image.open(source) as opened:
            body = _scale_to(_tight_rgba(birefnet_cutout(opened.convert("RGB"))), width=880)
        frame = library.copy()
        body_x = (1080 - body.width) // 2
        body_y = 1920 - body.height
        frame.alpha_composite(body, (body_x, body_y))
        rim = int(np.nonzero(np.asarray(body)[..., 3] > 16)[0].min())
        head_x = (1080 - head.width) // 2
        head_y = body_y + rim - head.height + 72
        frame.alpha_composite(head, (head_x, head_y))
        draw = ImageDraw.Draw(frame)
        text = f"[ {label} ]"
        box = draw.textbbox((0, 0), text, font=font)
        width = box[2] - box[0]
        x = (1080 - width) // 2
        draw.rectangle((x - 24, 36, x + width + 24, 36 + (box[3] - box[1]) + 28), fill=(12, 14, 18))
        draw.text((x, 48), text, fill=(232, 214, 170), font=font)
        panels.append(frame.convert("RGB"))
        print(f"option {label} shoulders={body.width}px head={head.width}x{head.height}px")
    sheet = Image.new("RGB", (1080 * len(panels), 1920), (0, 0, 0))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * 1080, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", compress_level=1)
    print(f"torsos: {output}")
    return output


def static_proportion_check(character_id: str, output: Path) -> Path:
    """Head +10%, body -6%, ink restored. No new generation and no video."""
    if character_id != "deepseek_cyborg_v3":
        raise ValueError(f"unsupported static check: {character_id}")
    from core.animator.asset_generator import (  # noqa: PLC0415
        DEFAULT_PUPPETS_DIR,
        ensure_shared_panorama,
    )

    skin = DEFAULT_PUPPETS_DIR / character_id
    head = Image.open(skin / "head.png").convert("RGBA")
    body = Image.open(skin / "body.png").convert("RGBA")
    master_path = skin / "deepseek_master_transparent.png"
    eye_source = np.asarray(Image.open(master_path).convert("RGBA")) if master_path.is_file() else np.asarray(head)
    green = (eye_source[..., 1] > 120) & (eye_source[..., 1] > eye_source[..., 0] + 25) & (eye_source[..., 3] > 16)
    eye_rows = np.nonzero(green.any(axis=1))[0]
    opaque_rows = np.nonzero((eye_source[..., 3] > 16).any(axis=1))[0]
    eye_y = int(np.median(eye_rows)) if eye_rows.size else int(opaque_rows.min() + 0.4 * (opaque_rows.max() - opaque_rows.min()))
    bottom = int(opaque_rows.max()) + 1
    base = (1920 - 640) / max(1, bottom - eye_y)
    head_scale = base * 1.10
    body_scale = base * 0.94
    placed_head = head.resize(
        (max(1, int(round(head.width * head_scale))), max(1, int(round(head.height * head_scale)))),
        Image.Resampling.LANCZOS,
    )
    placed_body = body.resize(
        (max(1, int(round(body.width * body_scale))), max(1, int(round(body.height * body_scale)))),
        Image.Resampling.LANCZOS,
    )
    body_alpha = np.asarray(placed_body)[..., 3]
    bys, bxs = np.nonzero(body_alpha > 16)
    body_bottom = int(bys.max()) + 1
    body_right = int(bxs.max()) + 1
    body_y = 1920 - body_bottom
    body_x = 1080 - body_right
    panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
    library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
    frame = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
    frame.alpha_composite(placed_body, (body_x, body_y))
    head_y = 640 - int(round(eye_y * head_scale))
    frame.alpha_composite(placed_head, ((1080 - placed_head.width) // 2, head_y))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(output, format="PNG", compress_level=1)
    print(f"eye_line={head_y + int(round(eye_y * head_scale))} head_scale={head_scale:.4f} body_scale={body_scale:.4f}")
    print(f"static-check: {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a 3-candidate concept sheet. Does not rig.")
    parser.add_argument("--character-id", default="deepseek_cyborg_v3")
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--assemble-candidate", choices=("F",))
    parser.add_argument("--finalize-candidate-f", action="store_true")
    parser.add_argument("--torso-candidates-for-f", action="store_true")
    parser.add_argument("--static-check", default="")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.static_check:
        destination = args.output or (
            outputs_root()
            / "aiwake"
            / "_test_harness"
            / "deepseek_proportional_static_check.png"
        )
        static_proportion_check(args.static_check, Path(destination))
        return 0
    if args.torso_candidates_for_f:
        destination = args.output or (
            outputs_root()
            / "aiwake"
            / "_test_harness"
            / "deepseek_torso_candidates.png"
        )
        torso_candidates_for_f(Path(destination))
        return 0
    if args.finalize_candidate_f:
        destination = args.output or (
            outputs_root()
            / "aiwake"
            / "_test_harness"
            / "deepseek_f_static_approval.png"
        )
        finalize_candidate_f(Path(destination))
        return 0
    if args.assemble_candidate:
        destination = args.output or (
            outputs_root()
            / "aiwake"
            / "_test_harness"
            / "deepseek_f_static_approval.png"
        )
        assemble_candidate_f(Path(destination))
        return 0
    destination = build_concept_sheet(
        character_id=args.character_id,
        candidates=args.candidates,
        batch=args.batch,
    )
    print(f"concepts: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
