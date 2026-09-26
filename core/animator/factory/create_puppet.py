"""End-to-end prompt/image-to-puppet command-line utility.

Usage:
    python -m core.animator.factory.create_puppet \
        --character-id example --prompt "robot bust" --facing left
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from utils.pipeline_paths import assets_root

from .layer_slicer import slice_jaw_contour
from .rigger import (
    auto_rig_character,
    birefnet_cutout,
    chroma_key_cutout,
    solidify_character_alpha,
)

_LOG = logging.getLogger("animator.factory.create_puppet")
# Budget tier. Imagen 3 generate-002/001 404 on this key, and preview/pro
# image models are the slow expensive path. Flash image is the fast endpoint.
_FAST_IMAGE_MODEL = "models/gemini-2.5-flash-image"

GHIBLI_MECHA_PROMPT_TEMPLATE = (
    "Studio Ghibli character design, vintage 90s anime style, 2d cel animation, clean flat shading, "
    "Hayao Miyazaki retro-mecha aesthetic. Medium close-up bust shot of a humanoid mecha robot named '{name}', "
    "angled 25 degrees facing toward the {facing}. Broad mechanical shoulders extending past the frame. "
    "The opaque armored torso continues beyond the bottom edge of the 9:16 canvas; no visible lower-body cutoff. "
    "Matte metallic plating with visible hand-drawn dark ink outlines and subtle watercolor shading. "
    "On its forehead, a clean metal plate prominently displaying the text '{name}'. "
    "Expressive glowing retro optic lenses. The lower half of the face is a smooth, blank metallic shield plate "
    "with absolutely no mouth drawn. Pristine condition, warm nostalgic ambient lighting, no heavy shadows. "
    "Rendered on a pure solid black background. --ar 9:16"
)

DEEPSEEK_MASTER_PROMPT = (
    "Studio Ghibli character design, vintage 90s anime style, 2d cel animation, flat shading, muted pastel colors. "
    "Medium bust shot of a charming, dignified vintage mechanical robot, facing 25 degrees to the left. "
    "Smooth metallic chassis in muted naval blue, slate grey, and antique bronze trim with retro steampunk rivets. "
    "On its forehead, a sturdy metal plate with the text 'DEEPSEEK'. "
    "Soft glowing emerald green eyes with subtle concentric circles. Closed mouth on a smooth curved bronze chin plate. "
    "Broad mechanical shoulders with attached upper arms descending naturally along the borders. "
    "Nostalgic warm anime lighting, solid pure black background. --ar 9:16"
)

DEEPSEEK_STAGE1_PROMPT = (
    "Studio Ghibli vintage 1990s anime cel animation, Hayao Miyazaki mecha character design, clean flat cel shading. "
    "Medium bust shot of an advanced naval cybernetic humanoid robot named 'DeepSeek', "
    "STRICTLY angled 25 degrees facing toward the LEFT (looking across the frame at an interlocutor). "
    "Polished pristine matte navy-blue armor plating with vintage brushed bronze trim, exposed round rivets, zero rust, zero scratches. "
    "On its forehead, a curved brass plate neatly engraved with the word 'DEEPSEEK'. "
    "Glowing pill-shaped retro neon-green optic lenses. "
    "LOWER FACE: A clean, smooth, wide metallic bronze chin guard shield plate with NO human mouth and NO protruding beak/exhaust, "
    "providing a clean flat surface for mechanical visemes. "
    "Broad mechanical shoulders extending fully across the lower frame with a sturdy cylindrical neck column. "
    "Crisp hand-drawn dark ink outlines (#151820). Warm nostalgic library ambient light. "
    "Solid pure black background. --ar 9:16"
)

PROMPT_DEEPSEEK_GHIBLI = (
    "Vintage 1990s anime cel animation, Studio Ghibli mecha character design, Hayao Miyazaki retro-futurism. "
    "Pulled-back waist-up portrait of a vintage naval cybernetic robot named 'DeepSeek'. "
    "STRICT POSE LOCK: exactly 25 degrees facing left in three-quarter anime perspective, never dead-front. "
    "STRICT PROPORTION BLUEPRINT: crown-to-chin head height is exactly 27 percent of the 9:16 portrait height; "
    "the armored torso fills the lower 73 percent. Keep ample headroom. The shoulder span is at least 1.6 times "
    "the helmet width and exits both horizontal edges. LAYOUT GUIDE: helmet crown at y=12%, optical sensors at "
    "y=25%, chin at y=39%, collar at y=43%; helmet width no more than 42% of the canvas. This is a pulled-back "
    "medium-bust broadcast frame, never a close-up or macro portrait. "
    "Round curved iron chassis, vintage dark oceanic blue and matte brass hardware, exposed rivets. "
    "Soft watercolor flat shading with thick hand-inked anime contours (#1a1a24). "
    "A rectangular brass nameplate bolted to its curved forehead clearly embossed with 'DEEPSEEK'. "
    "Soft, warm round/pill-shaped glowing green optical sensor lenses. "
    "Lower face is a smooth curved vintage ventilation grill plate or blank iron chin guard, absolutely no human mouth drawn. "
    "The head docks into a short, sturdy industrial collar socket; no long accordion tube or giraffe neck. "
    "Broad mechanical iron shoulders extending off the side edges. The solid opaque chest armor continues naturally "
    "past the bottom edge of the 9:16 canvas, with no visible torso hem, rounded bust cutoff, fade, fog, or empty space below it. "
    "MANDATORY COMPOSITION: the camera frame physically cuts through the robot's mid-torso armor at the bottom image edge. "
    "Never show the lower silhouette or bottom boundary of the bust; chest plating must visibly exit the canvas on the bottom. "
    "Warm antique-library atmospheric lighting, zero 3D glossy render. "
    "Pristine factory condition, polished smooth matte navy-blue automotive finish, zero rust, zero chipped paint, "
    "zero scratches, zero battle damage, perfectly maintained museum-grade vintage mecha. "
    "Solid pure black background. --ar 9:16"
)

DEEPSEEK_VINTAGE_GHIBLI_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha character design, clean flat watercolor cel shading. "
    "Medium bust shot of a charming, dignified retro-futuristic tin-plate humanoid robot named 'DeepSeek', "
    "angled 25 degrees facing toward the LEFT (reverse-shot dialogue perspective). "
    "HEAD: Smooth, gently rounded dome helmet in dark matte naval blue. "
    "On its forehead, a clean rectangular brass plate neatly bolted with the text 'DEEPSEEK'. "
    "Large, expressive, gently rounded pill-shaped glowing green optic lenses (warm, intelligent, calm expression, NO angry squint). "
    "Ears are circular antique brass dials with small mechanical rivets. "
    "LOWER FACE: A smooth, continuous, curved bronze chin guard plate with a clean flat surface, absolutely NO mouth drawn and NO notches. "
    "BODY: Smooth, rounded barrel/boiler-plate naval blue chest with a wide curved brass collar rim, rounded dome rivets, "
    "and broad rounded mechanical shoulders extending off-screen. NO muscular pecs, NO six-pack armor plates. "
    "The head rests snugly inside the warm brass collar socket with a short sturdy neck. "
    "Thick, hand-inked dark anime contour outlines (#151820). Pure solid black background. --ar 9:16"
)

PROMPT_DEEPSEEK_HEAD_ONLY = (
    DEEPSEEK_VINTAGE_GHIBLI_PROMPT
    + " ISOLATED COMPONENT: draw ONLY the head. Smooth rounded dome, rectangular DEEPSEEK plate, "
    "large warm calm green pill lenses, circular brass ear dials, continuous curved bronze chin "
    "with a closed rounded bottom contour. No neck, no torso, no shoulders, no machinery, no mouth."
)

PROMPT_DEEPSEEK_BODY_ONLY = (
    DEEPSEEK_VINTAGE_GHIBLI_PROMPT
    + " ISOLATED COMPONENT: draw ONLY the body. Rounded barrel boiler-plate chest, wide curved "
    "brass collar rim, short sturdy neck inside that rim, broad rounded shoulders past both edges, "
    "chest exiting the bottom. No head, no face, no jaw, no piston pole, no pecs, no stray machines."
)


def _generation_prompt(
    character_id: str,
    facing: str,
    custom_prompt: str,
) -> str:
    if character_id.strip().lower() == "deepseek_cyborg_v3":
        return DEEPSEEK_VINTAGE_GHIBLI_PROMPT
    detail = custom_prompt.strip()
    if detail:
        return detail
    name = character_id.split("_cyborg", 1)[0].replace("_", " ").strip().upper()
    name = name or character_id.replace("_", " ").strip().upper()
    return GHIBLI_MECHA_PROMPT_TEMPLATE.format(name=name, facing=facing)


def _safe_character_id(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if not normalized or any(
        not (character.isalnum() or character == "_")
        for character in normalized
    ):
        raise argparse.ArgumentTypeError(
            "character id must contain only letters, numbers, underscores, or hyphens"
        )
    return normalized


def _save_google_image(image: Any, destination: Path) -> bool:
    if image is None:
        return False
    save = getattr(image, "save", None)
    if callable(save):
        try:
            save(str(destination))
            if destination.is_file() and destination.stat().st_size:
                return True
        except (OSError, TypeError, ValueError):
            pass
    image_bytes = getattr(image, "image_bytes", None)
    if image_bytes:
        destination.write_bytes(bytes(image_bytes))
        return True
    pil_image = getattr(image, "_pil_image", None)
    if isinstance(pil_image, Image.Image):
        pil_image.save(destination, format="PNG")
        return True
    return False


def _save_generated_response(response: Any, destination: Path) -> bool:
    for generated in getattr(response, "generated_images", None) or ():
        if _save_google_image(getattr(generated, "image", generated), destination):
            return True
    for candidate in getattr(response, "candidates", None) or ():
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or ():
            as_image = getattr(part, "as_image", None)
            if callable(as_image) and _save_google_image(as_image(), destination):
                return True
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None)
            if data:
                destination.write_bytes(bytes(data))
                return True
    return False


def generate_imagen3_image(
    prompt: str,
    destination: Path,
    *,
    aspect_ratio: str = "9:16",
    model: str = "imagen-3.0-generate-002",
) -> str:
    """Generate one image on Imagen 3. Does not fall back to Flash."""
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required when --image-path is omitted")
    from core.google_guardrail import (  # noqa: PLC0415
        estimate_call_cost_usd,
        get_google_cost_tracker,
        make_guarded_gemini_client,
    )
    from google.genai import types  # noqa: PLC0415

    client = make_guarded_gemini_client(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"active image model: {model}")
    tracker = get_google_cost_tracker()
    estimate = estimate_call_cost_usd(model, images=1, kind="image")
    tracker.preflight(estimate, model=model, source="factory.create_puppet", kind="image")
    try:
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                output_mime_type="image/png",
            ),
        )
    except Exception as exc:
        if "404" not in str(exc):
            raise
        raise RuntimeError(
            "imagen-3.0-generate-002 is unavailable; gemini-3-pro-image is prohibited"
        ) from exc
    if not _save_generated_response(response, destination):
        raise RuntimeError(f"{model} response contained no image payload")
    tracker.record(
        model=model,
        kind="image",
        images=1,
        source="factory.create_puppet",
        status="ok",
    )
    return model


def generate_character_image(
    prompt: str,
    destination: Path,
    *,
    aspect_ratio: str = "9:16",
    reference_path: Path | None = None,
) -> str:
    """Generate one image on the fast Gemini Flash image tier."""
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required when --image-path is omitted")
    from core.google_guardrail import (  # noqa: PLC0415
        guarded_generate_content,
        make_guarded_gemini_client,
    )
    from google.genai import types  # noqa: PLC0415

    client = make_guarded_gemini_client(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = _FAST_IMAGE_MODEL
    if "gemini-3-pro-image" in model:
        raise RuntimeError("gemini-3-pro-image is prohibited")
    print(f"active image model: {model}")
    image_config = types.ImageConfig(aspect_ratio=aspect_ratio)
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=image_config,
    )
    contents: Any = prompt
    if reference_path is not None:
        contents = [
            types.Part.from_bytes(data=reference_path.read_bytes(), mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ]
        print(f"reference image: {reference_path}")
    response = guarded_generate_content(
        client,
        model=model,
        contents=contents,
        config=config,
        kind="image",
        source="factory.create_puppet",
    )
    if not _save_generated_response(response, destination):
        raise RuntimeError(f"{model} response contained no image payload")
    return model


def _fallback_remove_background(image: Image.Image) -> Image.Image:
    """Remove a flat stage with contiguous Magic-Wand tolerance 22."""
    return chroma_key_cutout(image, tolerance=22.0)


def remove_background(source: Path, destination: Path) -> str:
    """Write continuous-alpha BiRefNet art, with an offline chroma fallback."""
    with Image.open(source) as opened:
        image = opened.convert("RGBA")
    try:
        cleaned = birefnet_cutout(image)
        backend = "rembg_birefnet-general"
    except Exception as exc:  # noqa: BLE001 - missing model/runtime uses local fallback
        _LOG.warning(
            "BiRefNet unavailable; using continuous-alpha OpenCV cutout: %s",
            exc,
        )
        cleaned = _fallback_remove_background(image)
        backend = "opencv_continuous_chroma"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(destination, format="PNG", compress_level=1)
    return backend


def _torso_reaches_canvas_bottom(path: Path) -> bool:
    with Image.open(path) as image:
        alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3]
    required = max(8, int(round(alpha.shape[1] * 0.20)))
    return int(np.count_nonzero(alpha[-2:] > 240)) >= required * 2


def _crop_bust_through_canvas_bottom(path: Path) -> None:
    """Uniformly enlarge isolated art so real chest pixels exit the canvas."""
    with Image.open(path) as opened:
        art = opened.convert("RGBA")
    alpha = np.asarray(art, dtype=np.uint8)[..., 3]
    binary = np.where(alpha > 200, 255, 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        raise RuntimeError("cannot frame an empty character alpha mask")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    art_array = np.asarray(art, dtype=np.uint8).copy()
    art_array[..., 3][labels != largest] = 0
    art = Image.fromarray(art_array)
    component_y = int(stats[largest, cv2.CC_STAT_TOP])
    component_height = int(stats[largest, cv2.CC_STAT_HEIGHT])
    width, height = art.size
    scale = max(1.0, (height * 1.04) / max(1, component_height))
    resized = art.resize(
        (int(round(width * scale)), int(round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", art.size, (0, 0, 0, 0))
    x = (width - resized.width) // 2
    y = int(round(height * 0.04 - component_y * scale))
    canvas.alpha_composite(resized, (x, y))
    canvas.save(path, format="PNG", compress_level=1)


def _format_isolated_component(
    isolated_path: Path,
    destination: Path,
    *,
    canvas_size: tuple[int, int] = (768, 1344),
    target_height: int | None = None,
    target_width: int | None = None,
    top: int | None = None,
    bottom_anchor: bool = False,
) -> None:
    """Uniformly place one clean component on the shared puppet canvas."""
    with Image.open(isolated_path) as opened:
        component = opened.convert("RGBA")
    alpha = np.asarray(component, dtype=np.uint8)[..., 3]
    ys, xs = np.nonzero(alpha > 8)
    if not xs.size:
        raise RuntimeError(f"component has no visible pixels: {isolated_path}")
    crop = component.crop(
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    )
    if target_height is not None:
        scale = target_height / float(crop.height)
    elif target_width is not None:
        scale = target_width / float(crop.width)
    else:
        raise ValueError("target_height or target_width is required")
    resized = crop.resize(
        (
            max(1, int(round(crop.width * scale))),
            max(1, int(round(crop.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    x = (canvas_size[0] - resized.width) // 2
    y = (
        canvas_size[1] - resized.height
        if bottom_anchor
        else int(top or 0)
    )
    canvas.alpha_composite(resized, (x, y))
    cleaned, _filled = solidify_character_alpha(canvas)
    cleaned.save(destination, format="PNG", compress_level=1)


def _collar_top(alpha: np.ndarray) -> int:
    ys, xs = np.nonzero(alpha > 16)
    center = int(np.median(xs))
    half = max(12, int((int(xs.max()) - int(xs.min())) * 0.12))
    neck = alpha[:, max(0, center - half) : center + half]
    neck_ys = np.nonzero(neck > 16)[0]
    return int(neck_ys.min()) if neck_ys.size else int(ys.min())


def _dock_head_into_collar(head_path: Path, body_path: Path) -> None:
    """Put the eyes on the 770 line and grow the barrel until the chin sits in the rim."""
    with Image.open(head_path) as opened:
        head = opened.convert("RGBA")
    with Image.open(body_path) as opened:
        body = opened.convert("RGBA")
    head_alpha = np.asarray(head, dtype=np.uint8)[..., 3]
    rgb = np.asarray(head, dtype=np.uint8)[..., :3]
    head_ys, head_xs = np.nonzero(head_alpha > 16)
    if not head_xs.size:
        raise RuntimeError("cannot dock an empty head")
    green = (rgb[..., 1] > 140) & (rgb[..., 1] > rgb[..., 0] + 30) & (head_alpha > 16)
    eye_rows = np.nonzero(green.any(axis=1))[0]
    eye_y = int(np.median(eye_rows)) if eye_rows.size else int(np.median(head_ys))
    shift = 770 - eye_y
    seated = Image.new("RGBA", head.size, (0, 0, 0, 0))
    seated.alpha_composite(head, (0, shift))
    seated.save(head_path, format="PNG", compress_level=1)
    chin_y = int(head_ys.max()) + shift
    body_alpha = np.asarray(body, dtype=np.uint8)[..., 3]
    body_ys, body_xs = np.nonzero(body_alpha > 16)
    if not body_xs.size:
        raise RuntimeError("cannot dock an empty body")
    collar_y = _collar_top(body_alpha)
    body_bottom = int(body_ys.max()) + 1
    span = max(1, body_bottom - collar_y)
    target_collar = chin_y - 24
    scale = max(1.0, (body.size[1] - target_collar) / float(span))
    enlarged = body.resize(
        (
            max(1, int(round(body.width * scale))),
            max(1, int(round(body.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", body.size, (0, 0, 0, 0))
    canvas.alpha_composite(
        enlarged,
        ((body.width - enlarged.width) // 2, body.height - enlarged.height),
    )
    canvas.save(body_path, format="PNG", compress_level=1)


def _prepare_deepseek_components(
    skin_dir: Path,
    *,
    regenerate: bool,
) -> tuple[str, str]:
    """Generate and format independent head-only and body-only source art."""
    head_source = skin_dir / "source_head_original.png"
    body_source = skin_dir / "source_body_original.png"
    models: list[str] = []
    if regenerate or not head_source.is_file():
        models.append(
            generate_character_image(
                PROMPT_DEEPSEEK_HEAD_ONLY,
                head_source,
                aspect_ratio="1:1",
            )
        )
    if regenerate or not body_source.is_file():
        models.append(
            generate_character_image(
                PROMPT_DEEPSEEK_BODY_ONLY,
                body_source,
                aspect_ratio="9:16",
            )
        )

    head_isolated = skin_dir / "_head_isolated.png"
    body_isolated = skin_dir / "_body_isolated.png"
    head_backend = remove_background(head_source, head_isolated)
    body_backend = remove_background(body_source, body_isolated)
    _format_isolated_component(
        head_isolated,
        skin_dir / "head.png",
        canvas_size=(1080, 1920),
        target_height=550,
        top=220,
    )
    _format_isolated_component(
        body_isolated,
        skin_dir / "body.png",
        canvas_size=(1080, 1920),
        target_width=1040,
        bottom_anchor=True,
    )
    _dock_head_into_collar(skin_dir / "head.png", skin_dir / "body.png")

    with Image.open(skin_dir / "body.png") as opened:
        assembled = opened.convert("RGBA")
    with Image.open(skin_dir / "head.png") as opened:
        assembled.alpha_composite(opened.convert("RGBA"))
    assembled.save(skin_dir / "character.png", format="PNG", compress_level=1)
    Image.new("RGBA", assembled.size, (0, 0, 0, 0)).save(
        skin_dir / "collar.png",
        format="PNG",
        compress_level=1,
    )
    acquisition = "+".join(dict.fromkeys(models)) if models else "component-cache"
    return acquisition, f"head:{head_backend};body:{body_backend}"


def _jaw_is_closed_contour(head_path: Path) -> bool:
    """A guillotine crop ends on one straight row; a drawn jaw curves down."""
    with Image.open(head_path) as opened:
        alpha = np.asarray(opened.convert("RGBA"), dtype=np.uint8)[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    if xs.size < 32:
        return False
    left, right = int(xs.min()), int(xs.max())
    span = max(1, right - left)
    band_left = left + int(span * 0.12)
    band_right = right - int(span * 0.12)
    bottoms: list[int] = []
    for x in range(band_left, band_right + 1):
        column = np.nonzero(alpha[:, x] > 16)[0]
        if column.size:
            bottoms.append(int(column.max()))
    if len(bottoms) < 16:
        return False
    lowest = max(bottoms)
    on_cut = sum(1 for y in bottoms if y >= lowest - 1)
    return (on_cut / len(bottoms)) < 0.72 and (lowest - min(bottoms)) >= 6


def build_deepseek_stage1(*, regenerate: bool = True) -> dict[str, Any]:
    """Generate isolated head and body. Never slice the jaw with a flat cut."""
    skin_dir = assets_root() / "puppets" / "deepseek_cyborg_v3"
    skin_dir.mkdir(parents=True, exist_ok=True)
    manifest = skin_dir / "puppet.json"
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("skin_version") != "v3-auto-rig":
            raise RuntimeError(f"refusing to replace artist manifest: {manifest}")
        manifest.unlink()
    acquisition, backend = _prepare_deepseek_components(
        skin_dir,
        regenerate=regenerate,
    )
    if not _jaw_is_closed_contour(skin_dir / "head.png"):
        raise RuntimeError(
            "DeepSeek head.png still has a flat guillotine jaw; refusing to rig"
        )
    auto_rig_character(
        skin_dir,
        "deepseek_cyborg_v3",
        facing="left",
        component_mode=True,
    )
    return {
        "source_head": str(skin_dir / "source_head_original.png"),
        "source_body": str(skin_dir / "source_body_original.png"),
        "character": str(skin_dir / "character.png"),
        "head": str(skin_dir / "head.png"),
        "body": str(skin_dir / "body.png"),
        "acquisition": acquisition,
        "background_removal": backend,
        "component_mode": True,
    }


def create_puppet(
    *,
    character_id: str,
    prompt: str,
    facing: str,
    image_path: Path | None = None,
    puppets_dir: Path | None = None,
    rerig: bool = False,
    regenerate: bool = False,
) -> dict[str, Any]:
    """Acquire, isolate, and rig a character, returning run metadata."""
    root = Path(puppets_dir) if puppets_dir else assets_root() / "puppets"
    skin_dir = root / character_id
    manifest_path = skin_dir / "puppet.json"
    prior_factory: dict[str, Any] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not rerig and not regenerate:
            return {
                "character_id": character_id,
                "skin_dir": str(skin_dir),
                "manifest_path": str(manifest_path),
                "mode": "artist_override",
                "rigging_seconds": 0.0,
            }
        if existing.get("skin_version") != "v3-auto-rig":
            raise RuntimeError(
                f"refusing to re-rig artist-owned manifest: {manifest_path}"
            )
        prior_factory = dict(existing.get("factory") or {})
        if not prompt.strip():
            prompt = str(
                prior_factory.get("custom_prompt")
                or prior_factory.get("prompt")
                or ""
            )
        manifest_path.unlink()

    started = time.perf_counter()
    standardized_prompt = _generation_prompt(character_id, facing, prompt)
    skin_dir.mkdir(parents=True, exist_ok=True)
    original_path = skin_dir / "source_original.png"
    acquisition_model = str(prior_factory.get("acquisition") or "local")
    is_deepseek = character_id.strip().lower() == "deepseek_cyborg_v3"
    component_sources_exist = all(
        (skin_dir / filename).is_file()
        for filename in (
            "source_head_original.png",
            "source_body_original.png",
        )
    )
    component_mode = bool(
        is_deepseek
        and image_path is None
        and (regenerate or component_sources_exist)
    )
    if component_mode:
        acquisition_model, background_backend = _prepare_deepseek_components(
            skin_dir,
            regenerate=regenerate,
        )
    else:
        if rerig and not regenerate and image_path is None and original_path.is_file():
            image_path = original_path
        if image_path is not None:
            local_source = Path(image_path).expanduser().resolve()
            if not local_source.is_file():
                raise FileNotFoundError(f"source image does not exist: {local_source}")
            if local_source != original_path.resolve():
                with Image.open(local_source) as source:
                    source.convert("RGBA").save(
                        original_path,
                        format="PNG",
                        compress_level=1,
                    )
        else:
            if not prompt.strip():
                raise ValueError("--prompt is required when --image-path is omitted")
            acquisition_model = generate_character_image(
                standardized_prompt,
                original_path,
            )
        background_backend = remove_background(
            original_path,
            skin_dir / "character.png",
        )
        if is_deepseek:
            character_path = skin_dir / "character.png"
            if not _torso_reaches_canvas_bottom(character_path):
                _crop_bust_through_canvas_bottom(character_path)
            if not _torso_reaches_canvas_bottom(character_path):
                raise RuntimeError(
                    "generated DeepSeek torso does not exit the bottom canvas edge"
                )
    rig_started = time.perf_counter()
    auto_rig_character(
        skin_dir,
        character_id,
        facing=facing,
        component_mode=component_mode,
    )
    rigging_seconds = time.perf_counter() - rig_started

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["facing"] = facing
    manifest["factory"] = {
        "acquisition": acquisition_model,
        "background_removal": background_backend,
        "prompt": standardized_prompt,
        "custom_prompt": prompt,
        "component_mode": component_mode,
        "rigging_seconds": round(rigging_seconds, 4),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "character_id": character_id,
        "skin_dir": str(skin_dir),
        "manifest_path": str(manifest_path),
        "source_path": str(
            skin_dir / "source_head_original.png"
            if component_mode
            else original_path
        ),
        "character_path": str(skin_dir / "character.png"),
        "acquisition": acquisition_model,
        "background_removal": background_backend,
        "rigging_seconds": rigging_seconds,
        "total_seconds": time.perf_counter() - started,
        "mode": "auto_rig",
    }


def build_deepseek_master() -> dict[str, Any]:
    """One 9:16 bust, matted, then split on the jaw curve. No separate torso prompt."""
    skin_dir = assets_root() / "puppets" / "deepseek_cyborg_v3"
    skin_dir.mkdir(parents=True, exist_ok=True)
    source = skin_dir / "source_master.png"
    model = generate_character_image(DEEPSEEK_MASTER_PROMPT, source, aspect_ratio="9:16")
    transparent = skin_dir / "deepseek_master_transparent.png"
    backend = remove_background(source, transparent)
    with Image.open(transparent) as opened:
        master = opened.convert("RGBA")
    head, body = slice_jaw_contour(master)
    head.save(skin_dir / "head.png", format="PNG", compress_level=1)
    body.save(skin_dir / "body.png", format="PNG", compress_level=1)

    from core.animator.asset_generator import (  # noqa: PLC0415
        DEFAULT_PUPPETS_DIR,
        ensure_shared_panorama,
    )
    from utils.pipeline_paths import outputs_root  # noqa: PLC0415

    rgba = np.asarray(master)
    green = (rgba[..., 1] > 120) & (rgba[..., 1] > rgba[..., 0] + 25) & (rgba[..., 3] > 16)
    eye_rows = np.nonzero(green.any(axis=1))[0]
    opaque_rows = np.nonzero((rgba[..., 3] > 16).any(axis=1))[0]
    eye_y = int(np.median(eye_rows)) if eye_rows.size else int(np.median(opaque_rows))
    bottom = int(opaque_rows.max()) + 1
    scale = (1920 - 640) / max(1, bottom - eye_y)
    placed_body = body.resize(
        (max(1, int(round(body.width * scale))), max(1, int(round(body.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    placed_head = head.resize(
        (max(1, int(round(head.width * scale))), max(1, int(round(head.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
    library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
    frame = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
    paste_y = 1920 - placed_body.height
    paste_x = (1080 - placed_body.width) // 2
    frame.alpha_composite(placed_body, (paste_x, paste_y))
    frame.alpha_composite(placed_head, (paste_x, paste_y))
    destination = outputs_root() / "aiwake" / "_test_harness" / "deepseek_single_9x16_approved.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(destination, format="PNG", compress_level=1)
    placed_eye = paste_y + int(round(eye_y * scale))
    print(f"eye_line={placed_eye} scale={scale:.4f}")
    print(f"approved: {destination}")
    return {
        "acquisition": model,
        "background_removal": backend,
        "transparent": str(transparent),
        "head": str(skin_dir / "head.png"),
        "body": str(skin_dir / "body.png"),
        "approved": str(destination),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a production-ready V3 puppet from a prompt or image."
    )
    parser.add_argument("--character-id", required=True, type=_safe_character_id)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--facing", choices=("left", "right"))
    parser.add_argument("--master-blueprint", action="store_true")
    parser.add_argument("--image-path", type=Path)
    parser.add_argument(
        "--rerig",
        action="store_true",
        help="Rebuild an existing v3-auto-rig skin; artist manifests remain protected.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Generate fresh source art before rebuilding a v3-auto-rig skin.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.master_blueprint:
        print(json.dumps(build_deepseek_master(), indent=2))
        return 0
    if not args.facing:
        build_parser().error("--facing is required unless --master-blueprint is set")
    result = create_puppet(
        character_id=args.character_id,
        prompt=args.prompt,
        facing=args.facing,
        image_path=args.image_path,
        rerig=args.rerig,
        regenerate=args.regenerate,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GHIBLI_MECHA_PROMPT_TEMPLATE",
    "PROMPT_DEEPSEEK_GHIBLI",
    "DEEPSEEK_VINTAGE_GHIBLI_PROMPT",
    "DEEPSEEK_MASTER_PROMPT",
    "create_puppet",
    "generate_character_image",
    "remove_background",
]
