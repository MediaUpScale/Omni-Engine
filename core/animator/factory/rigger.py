"""Autonomous conversion of raw character art into a V2-compatible puppet."""
from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..types import VISEMES
from .color_extractor import PuppetPalette, extract_palette
from .landmark_detector import FacialLandmarks, detect_landmarks
from .layer_slicer import slice_character_layers

_SOURCE_NAMES = (
    "character.png",
    "portrait.png",
    "bust.png",
    "base.png",
    "head.png",
    "body.png",
)
_NON_CHARACTER_NAMES = {
    "bg.png",
    "glow.png",
    "eyes_open.png",
    "eyes_half.png",
    "eyes_blink.png",
    "eyes_closed.png",
    "collar.png",
}


def _source_image(skin_dir: Path) -> Path:
    for name in _SOURCE_NAMES:
        candidate = skin_dir / name
        if candidate.is_file():
            return candidate
    candidates = sorted(
        path
        for path in skin_dir.glob("*.png")
        if path.name.lower() not in _NON_CHARACTER_NAMES
        and not path.name.lower().startswith("mouth_")
    )
    if not candidates:
        raise FileNotFoundError(
            f"no character PNG found in manifestless skin directory: {skin_dir}"
        )
    return candidates[0]


def has_character_imagery(skin_dir: Path) -> bool:
    try:
        _source_image(Path(skin_dir))
    except FileNotFoundError:
        return False
    return True


def _rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        alpha,
    )


def _mouth_layer(
    canvas_size: tuple[int, int],
    center: tuple[int, int],
    mouth_width: int,
    viseme: str,
    palette: PuppetPalette,
) -> Image.Image:
    scale = 4
    width = max(20, mouth_width)
    ratios = {
        "A": (0.82, 0.10),
        "B": (0.86, 0.16),
        "C": (0.76, 0.28),
        "D": (1.00, 0.26),
        "E": (0.62, 0.22),
        "F": (0.54, 0.20),
        "O": (0.48, 0.30),
        "G": (0.88, 0.20),
        "H": (0.72, 0.26),
        "X": (0.72, 0.08),
    }
    width_ratio, height_ratio = ratios[viseme]
    patch_w = max(12, int(round(width * width_ratio)))
    patch_h = max(5, int(round(width * height_ratio)))
    padding = max(5, int(round(width * 0.08)))
    patch = Image.new(
        "RGBA",
        ((patch_w + padding * 2) * scale, (patch_h + padding * 2) * scale),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(patch, "RGBA")
    box = (
        padding * scale,
        padding * scale,
        (padding + patch_w) * scale,
        (padding + patch_h) * scale,
    )
    outline = _rgba(palette["ink_outline"])
    cavity = _rgba(palette["cavity_interior"])
    teeth = (238, 240, 232, 255)
    stroke = max(scale * 2, int(round(width * 0.045 * scale)))

    if viseme in {"A", "X"}:
        y = (padding + patch_h / 2.0) * scale
        line_stroke = 4 * scale
        start_x = padding * scale
        end_x = (padding + patch_w) * scale
        draw.line(
            [(start_x, y), (end_x, y)],
            fill=outline,
            width=line_stroke,
        )
        cap = line_stroke // 2
        for x in (start_x, end_x):
            draw.ellipse(
                (x - cap, y - cap, x + cap, y + cap),
                fill=outline,
            )
    else:
        if viseme == "D":
            skew = max(scale * 2, int(patch_w * scale * 0.10))
            points = [
                (box[0] + skew, box[1]),
                (box[2] - skew // 2, box[1]),
                (box[2], box[3]),
                (box[0], box[3]),
            ]
            draw.polygon(points, fill=cavity)
            draw.line(points + [points[0]], fill=outline, width=stroke, joint="curve")
        else:
            radius = (
                min(patch_w, patch_h) * scale // 2
                if viseme in {"F", "O"}
                else max(scale * 2, min(patch_w, patch_h) * scale // 3)
            )
            draw.rounded_rectangle(
                box,
                radius=radius,
                fill=cavity,
                outline=outline,
                width=stroke,
            )
        if viseme in {"B", "C", "D", "E", "G"}:
            tooth_height = max(scale * 2, int(patch_h * scale * 0.25))
            inset = max(stroke, int(patch_w * scale * 0.10))
            draw.rounded_rectangle(
                (
                    box[0] + inset,
                    box[1] + stroke // 2,
                    box[2] - inset,
                    min(box[3] - stroke, box[1] + tooth_height),
                ),
                radius=scale,
                fill=teeth,
            )
            if viseme in {"B", "C"}:
                draw.rounded_rectangle(
                    (
                        box[0] + inset,
                        max(box[1] + stroke, box[3] - tooth_height),
                        box[2] - inset,
                        box[3] - stroke // 2,
                    ),
                    radius=scale,
                    fill=teeth,
                )
        if viseme == "G":
            bite_x = int((padding + patch_w * 0.55) * scale)
            draw.line(
                [(bite_x, box[1]), (bite_x, box[3])],
                fill=teeth,
                width=max(scale, stroke // 2),
            )
        if viseme == "H":
            tongue_y = int((padding + patch_h * 0.68) * scale)
            draw.arc(
                (box[0] + stroke, tongue_y - stroke, box[2] - stroke, box[3]),
                180,
                350,
                fill=teeth,
                width=max(scale, stroke // 2),
            )

    patch = patch.resize(
        (patch.width // scale, patch.height // scale),
        Image.Resampling.LANCZOS,
    )
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    layer.alpha_composite(
        patch,
        (center[0] - patch.width // 2, center[1] - patch.height // 2),
    )
    return layer


def _save_layer(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=False, compress_level=1)


@lru_cache(maxsize=1)
def _birefnet_session():
    """Load the studio V3 matte once; model initialization is expensive."""
    from rembg import new_session  # type: ignore[import-not-found]

    return new_session("birefnet-general")


def birefnet_cutout(image: Image.Image) -> Image.Image:
    """Return BiRefNet RGBA art with its continuous 8-bit alpha untouched."""
    from rembg import remove  # type: ignore[import-not-found]

    result = remove(
        image.convert("RGB"),
        session=_birefnet_session(),
        alpha_matting=False,
    )
    if isinstance(result, bytes):
        from io import BytesIO

        result = Image.open(BytesIO(result))
    if not isinstance(result, Image.Image):
        raise TypeError(f"unexpected BiRefNet result: {type(result)!r}")
    # Native BiRefNet alpha. No threshold, dilation, or recolor.
    return result.convert("RGBA")


def chroma_key_cutout(
    image: Image.Image,
    *,
    tolerance: float = 22.0,
) -> Image.Image:
    """Photoshop-style corner chroma key with a one-pixel clean feather."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    corner_alpha = np.asarray(
        (
            rgba[0, 0, 3],
            rgba[0, -1, 3],
            rgba[-1, 0, 3],
            rgba[-1, -1, 3],
        )
    )
    # Already-isolated art must retain its authored alpha.
    if int(np.count_nonzero(corner_alpha < 16)) >= 3:
        return Image.fromarray(rgba)

    # Generated full-bleed torsos may legitimately cover both lower corners.
    # Derive the key from corner neighbourhoods plus the top stage strip so a
    # shoulder colour can never be mistaken for the backdrop.
    sample_size = max(2, min(rgba.shape[:2]) // 64)
    samples = np.concatenate(
        (
            rgba[:sample_size, :sample_size, :3].reshape(-1, 3),
            rgba[:sample_size, -sample_size:, :3].reshape(-1, 3),
            rgba[-sample_size:, :sample_size, :3].reshape(-1, 3),
            rgba[-sample_size:, -sample_size:, :3].reshape(-1, 3),
            rgba[:sample_size, :, :3].reshape(-1, 3),
        )
    )
    quantized = (samples // 8).astype(np.int32)
    packed = (
        quantized[:, 0] * 1024
        + quantized[:, 1] * 32
        + quantized[:, 2]
    )
    dominant = int(np.bincount(packed).argmax())
    background_rgb = np.median(
        samples[packed == dominant],
        axis=0,
    ).astype(np.float32)
    delta = rgba[..., :3].astype(np.float32) - background_rgb
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    candidate = (distance <= float(tolerance) * 2.55).astype(np.uint8)

    # Magic Wand contiguous mode: only matching regions reached from a corner
    # are background. This preserves equally dark ink cavities inside the art.
    count, labels = cv2.connectedComponents(candidate, connectivity=8)
    edge_labels = (
        set(labels[0])
        | set(labels[-1])
        | set(labels[:, 0])
        | set(labels[:, -1])
    )
    background = np.isin(labels, tuple(edge_labels - {0}))
    if count <= 1:
        background = candidate.astype(bool)
    foreground = ~background

    # Restore dark ink pixels touching a confirmed non-key surface. This is
    # critical for #10141C contours on a black generation stage.
    dark_ink = rgba[..., :3].max(axis=2) <= 45
    non_key_neighbor = cv2.dilate(
        (~candidate.astype(bool)).astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1,
    ).astype(bool)
    foreground |= dark_ink & non_key_neighbor

    # Continuous 8-bit fallback matte. BiRefNet is the production path, but
    # even the offline chroma fallback must never quantize edges to 0/192/255.
    alpha = cv2.GaussianBlur(
        foreground.astype(np.float32) * 255.0,
        (0, 0),
        sigmaX=0.65,
        sigmaY=0.65,
        borderType=cv2.BORDER_REPLICATE,
    )
    alpha = np.clip(np.rint(alpha), 0, 255).astype(np.uint8)

    # Remove key-colour contamination according to each boundary pixel's
    # actual coverage rather than assuming one fixed feather opacity.
    fraction = alpha.astype(np.float32) / 255.0
    boundary = (alpha > 0) & (alpha < 255) & ~dark_ink
    safe_fraction = np.maximum(fraction, 1.0 / 255.0)
    corrected = (
        rgba[..., :3].astype(np.float32)
        - background_rgb.reshape(1, 1, 3) * (1.0 - fraction[..., None])
    ) / safe_fraction[..., None]
    rgba[..., :3][boundary] = np.clip(
        corrected[boundary],
        0,
        255,
    ).astype(np.uint8)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba)


def solidify_character_alpha(image: Image.Image) -> tuple[Image.Image, int]:
    """Fill internal silhouette holes without expanding the authored contour."""
    keyed = chroma_key_cutout(image)
    rgba = np.asarray(keyed.convert("RGBA"), dtype=np.uint8).copy()
    original_alpha = rgba[..., 3].copy()
    binary = np.where(original_alpha > 8, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    if not np.any(binary):
        return Image.fromarray(rgba), 0
    # Flood only the transparent area connected to the canvas exterior.
    # A filled external contour incorrectly seals legitimate concavities
    # between jaw, neck and shoulders into large rectangular matte artifacts.
    flood = np.pad(binary, 1, mode="constant", constant_values=0)
    cv2.floodFill(flood, None, (0, 0), 128)
    enclosed_holes = flood[1:-1, 1:-1] == 0
    silhouette = binary.copy()
    silhouette[enclosed_holes] = 255
    holes = enclosed_holes & (original_alpha < 250)
    filled_pixels = int(np.count_nonzero(holes))
    # Preserve every BiRefNet/chroma sub-pixel value. Only genuinely enclosed
    # transparent holes are repaired; the authored outer matte is immutable.
    repaired_alpha = original_alpha.copy()
    repaired_alpha[holes] = 255
    rgba[..., 3] = repaired_alpha
    return Image.fromarray(rgba), filled_pixels


def _ensure_support_layers(
    skin_dir: Path,
    source_path: Path,
    character: Image.Image,
    canvas_size: tuple[int, int],
    landmarks: FacialLandmarks,
    palette: PuppetPalette,
    facing: str,
    *,
    component_mode: bool = False,
) -> tuple[dict[str, str], dict[str, int | str]]:
    blank = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    slicing: dict[str, int | str] = {}
    if component_mode:
        body_name = "body.png"
        head_name = "head.png"
        _save_layer(blank, skin_dir / "collar.png")
        slicing = {
            "component_mode": "isolated_head_body",
            "neck_overlap_px": 0,
            "facing": facing,
        }
    elif source_path.name == "character.png":
        facial_plate = None
        if landmarks.facial_plate_bbox is not None:
            x0, y0, x1, y1 = landmarks.facial_plate_bbox
            head_x, head_y, head_right, head_bottom = landmarks.head_bbox
            head_width = head_right - head_x
            head_height = head_bottom - head_y
            facial_plate = (
                int(round(head_x + x0 * head_width)),
                int(round(head_y + y0 * head_height)),
                int(round(head_x + x1 * head_width)),
                int(round(head_y + y1 * head_height)),
            )
        sliced = slice_character_layers(
            character,
            neck_pivot=landmarks.pixel_point(landmarks.neck_pivot),
            chin_boundary=landmarks.pixel_point(landmarks.chin_base)[1],
            facing=facing,
            facial_plate_bbox=facial_plate,
            overlap_px=45,
        )
        _save_layer(sliced.body, skin_dir / "body.png")
        _save_layer(sliced.head, skin_dir / "head.png")
        _save_layer(sliced.collar, skin_dir / "collar.png")
        body_name = "body.png"
        head_name = "head.png"
        slicing = {
            "collar_y": sliced.collar_y,
            "neck_overlap_px": sliced.overlap_px,
            "facing": sliced.facing,
        }
    elif source_path.name == "body.png":
        body_name = "body.png"
        if not (skin_dir / "head.png").is_file():
            _save_layer(blank, skin_dir / "head.png")
        head_name = "head.png"
    elif source_path.name == "head.png":
        head_name = "head.png"
        if not (skin_dir / "body.png").is_file():
            _save_layer(blank, skin_dir / "body.png")
        body_name = "body.png"
    else:
        body_name = source_path.name
        head_name = "head.png"
        if not (skin_dir / head_name).is_file():
            _save_layer(blank, skin_dir / head_name)

    overwrite_generated = source_path.name == "character.png"
    if overwrite_generated or not (skin_dir / "eyes_open.png").is_file():
        _save_layer(blank, skin_dir / "eyes_open.png")
    shutter_color = _rgba(palette["cavity_interior"])
    ink_color = _rgba(palette["ink_outline"])
    for name, closed_amount in (("eyes_half.png", 0.55), ("eyes_blink.png", 1.0)):
        path = skin_dir / name
        if path.is_file() and not overwrite_generated:
            continue
        eyelids = blank.copy()
        draw = ImageDraw.Draw(eyelids, "RGBA")
        for eye in (landmarks.left_eye, landmarks.right_eye):
            x, y = landmarks.pixel_point(eye.center)
            radius = max(3, int(round(eye.iris_radius * (landmarks.head_bbox[2] - landmarks.head_bbox[0]))))
            shutter_bottom = int(round(y - radius + (radius * 2 * closed_amount)))
            draw.rectangle(
                (x - radius, y - radius, x + radius, shutter_bottom),
                fill=shutter_color,
            )
            draw.line(
                [(x - radius, shutter_bottom), (x + radius, shutter_bottom)],
                fill=ink_color,
                width=5,
            )
        _save_layer(eyelids, path)
    for name in ("glow.png", "bg.png"):
        path = skin_dir / name
        if not path.is_file():
            _save_layer(blank, path)
    layers = {
        "body": body_name,
        "head": head_name,
        "eyes_open": "eyes_open.png",
        "eyes_half": "eyes_half.png",
        "eyes_blink": "eyes_blink.png",
        "glow": "glow.png",
        "bg": "bg.png",
    }
    if (skin_dir / "collar.png").is_file():
        layers["collar"] = "collar.png"
    return layers, slicing


def _generate_mouths(
    skin_dir: Path,
    canvas_size: tuple[int, int],
    landmarks: FacialLandmarks,
    palette: PuppetPalette,
) -> dict[str, str]:
    mouths_dir = skin_dir / "mouths"
    mouths_dir.mkdir(parents=True, exist_ok=True)
    center = landmarks.pixel_point(landmarks.mouth_center)
    head_width = landmarks.head_bbox[2] - landmarks.head_bbox[0]
    mouth_width = max(
        20,
        min(
            int(canvas_size[0] * 0.20),
            int(head_width * 0.16),
            int(round(landmarks.mouth_width * head_width)),
        ),
    )
    layers: dict[str, str] = {}
    for viseme in (*VISEMES, "O"):
        image = _mouth_layer(canvas_size, center, mouth_width, viseme, palette)
        destination = mouths_dir / f"mouth_{viseme}.png"
        _save_layer(image, destination)
        # Legacy manifestless callers historically looked in the skin root.
        # Byte-copy aliases preserve that API without changing V3 layer paths.
        shutil.copyfile(destination, skin_dir / f"mouth_{viseme}.png")
        layers[f"mouth_{viseme}"] = f"mouths/mouth_{viseme}.png"

    rest_files = {
        "mouth_X_neutral": ("mouth_X_neutral.png", "X"),
        "mouth_smug_smile": ("mouth_smug_smile.png", "A"),
        "mouth_stressed_grimace": ("mouth_stressed_grimace.png", "B"),
    }
    for key, (filename, source_viseme) in rest_files.items():
        source = mouths_dir / f"mouth_{source_viseme}.png"
        shutil.copyfile(source, mouths_dir / filename)
        shutil.copyfile(source, skin_dir / filename)
        layers[key] = f"mouths/{filename}"
    return layers


def _calibration(landmarks: FacialLandmarks) -> dict:
    head_width = max(1, landmarks.head_bbox[2] - landmarks.head_bbox[0])
    head_height = max(1, landmarks.head_bbox[3] - landmarks.head_bbox[1])
    head_x, head_y = landmarks.head_bbox[:2]
    eyes = (landmarks.left_eye, landmarks.right_eye)
    eye_bboxes = []
    lens_circles = []
    for eye in eyes:
        x, y = landmarks.pixel_point(eye.center)
        radius = max(1, int(round(eye.iris_radius * head_width)))
        eye_bboxes.append([x - radius, y - radius, x + radius, y + radius])
        lens_circles.append([x, y, radius])
    calibration = {
        "detector": landmarks.backend,
        "confidence": round(landmarks.confidence, 4),
        "head_bbox": list(landmarks.head_bbox),
        "eye_bboxes": eye_bboxes,
        "lens_circles": lens_circles,
        "landmarks_uv": {
            "left_eye": list(landmarks.left_eye.center.__dict__.values())
            if hasattr(landmarks.left_eye.center, "__dict__")
            else [landmarks.left_eye.center.x, landmarks.left_eye.center.y, landmarks.left_eye.center.z],
            "right_eye": [landmarks.right_eye.center.x, landmarks.right_eye.center.y, landmarks.right_eye.center.z],
            "mouth_center": [landmarks.mouth_center.x, landmarks.mouth_center.y, landmarks.mouth_center.z],
            "mouth_width": landmarks.mouth_width,
            "jawline": [[point.x, point.y, point.z] for point in landmarks.jawline],
            "chin_base": [landmarks.chin_base.x, landmarks.chin_base.y, landmarks.chin_base.z],
            "neck_pivot": [landmarks.neck_pivot.x, landmarks.neck_pivot.y, landmarks.neck_pivot.z],
        },
    }
    for key, box in (
        ("facial_plate_bbox", landmarks.facial_plate_bbox),
        ("forehead_plate_bbox", landmarks.forehead_plate_bbox),
    ):
        if box is not None:
            calibration[key] = [
                int(round(head_x + box[0] * head_width)),
                int(round(head_y + box[1] * head_height)),
                int(round(head_x + box[2] * head_width)),
                int(round(head_y + box[3] * head_height)),
            ]
    return calibration


def _detect_optic_bboxes(
    image: Image.Image,
    calibration: dict,
) -> list[list[int]]:
    """Tighten robotic eye boxes around luminous green pill sensors."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    green = (
        (hsv[..., 0] >= 35)
        & (hsv[..., 0] <= 95)
        & (hsv[..., 1] >= 55)
        & (hsv[..., 2] >= 90)
        & (rgba[..., 3] > 8)
    )
    height, width = green.shape
    boxes: list[list[int]] = []
    for cx, cy, radius in calibration.get("lens_circles") or ():
        search = max(24, int(round(float(radius) * 2.5)))
        x0, y0 = max(0, cx - search), max(0, cy - search)
        x1, y1 = min(width, cx + search + 1), min(height, cy + search + 1)
        local = np.where(green[y0:y1, x0:x1], 255, 0).astype(np.uint8)
        contours = cv2.findContours(
            local,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        candidates = [
            contour for contour in contours if cv2.contourArea(contour) >= 20.0
        ]
        if not candidates:
            boxes.append([cx - radius, cy - radius, cx + radius, cy + radius])
            continue
        local_cx, local_cy = cx - x0, cy - y0

        def optic_score(contour: np.ndarray) -> tuple[int, float, float]:
            bx, by, bw, bh = cv2.boundingRect(contour)
            contains = int(
                bx <= local_cx <= bx + bw
                and by <= local_cy <= by + bh
            )
            distance = (
                (bx + bw * 0.5 - local_cx) ** 2
                + (by + bh * 0.5 - local_cy) ** 2
            )
            return contains, float(cv2.contourArea(contour)), -distance

        bx, by, bw, bh = cv2.boundingRect(max(candidates, key=optic_score))
        pad = 2
        boxes.append(
            [
                max(0, x0 + bx - pad),
                max(0, y0 + by - pad),
                min(width, x0 + bx + bw + pad),
                min(height, y0 + by + bh + pad),
            ]
        )
    return boxes


def auto_rig_character(
    skin_dir: Path,
    character_id: str,
    *,
    facing: str = "right",
    component_mode: bool = False,
) -> Path:
    """Create a V3 rig unless an artist-owned manifest already exists."""
    skin_dir = Path(skin_dir)
    manifest_path = skin_dir / "puppet.json"
    if manifest_path.is_file():
        return manifest_path
    skin_dir.mkdir(parents=True, exist_ok=True)
    source_path = _source_image(skin_dir)
    with Image.open(source_path) as source:
        character, filled_alpha_pixels = solidify_character_alpha(source)
        canvas_size = character.size
        if source_path.name == "character.png":
            _save_layer(character, source_path)
        landmarks = detect_landmarks(character)
        palette = extract_palette(character)
    if character_id.strip().lower() == "deepseek_cyborg_v3":
        palette = {
            **palette,
            "cavity_interior": "#151820",
        }

    base_layers, slicing = _ensure_support_layers(
        skin_dir,
        source_path,
        character,
        canvas_size,
        landmarks,
        palette,
        facing,
        component_mode=component_mode,
    )
    mouth_layers = _generate_mouths(
        skin_dir,
        canvas_size,
        landmarks,
        palette,
    )
    left_eye = landmarks.pixel_point(landmarks.left_eye.center)
    right_eye = landmarks.pixel_point(landmarks.right_eye.center)
    mouth = landmarks.pixel_point(landmarks.mouth_center)
    chin = landmarks.pixel_point(landmarks.chin_base)
    neck = landmarks.pixel_point(landmarks.neck_pivot)
    head_width = max(1, landmarks.head_bbox[2] - landmarks.head_bbox[0])
    eye_radius = max(
        1,
        int(
            round(
                (landmarks.left_eye.iris_radius + landmarks.right_eye.iris_radius)
                * 0.5
                * head_width
            )
        ),
    )
    calibration = _calibration(landmarks)
    optic_bboxes = _detect_optic_bboxes(character, calibration)
    if len(optic_bboxes) == 2:
        calibration["eye_bboxes"] = optic_bboxes
    manifest = {
        "character_id": character_id,
        "skin_version": "v3-auto-rig",
        "schema_version": 3,
        "canvas_size": list(canvas_size),
        "anchors": {
            "mouth": list(mouth),
            "left_eye": list(left_eye),
            "right_eye": list(right_eye),
            "eye_radius": eye_radius,
            "head_pivot": [
                (left_eye[0] + right_eye[0]) // 2,
                (left_eye[1] + mouth[1]) // 2,
            ],
            "neck_pivot": list(neck),
            "chin_base": list(chin),
        },
        # Keep the renderer's established style contract; V3 describes the
        # ingestion path, not a new runtime mouth renderer.
        "mouth_style": "ghibli_mecha",
        "eye_style": "circular_slit",
        "brows": {
            "style": "acute_mecha",
            "ink_color": palette["ink_outline"],
            "stroke_width_px": 5.5,
            "width_px": 48,
            "gap_px": 4,
            "angles": {
                "neutral": 0.0,
                "inquisitor": -5.5,
                "skeptical": 12.0,
                "conceded": 12.0,
                "defeated": 12.0,
                "troubled": 12.0,
            },
        },
        "framing": {
            "full_bleed_torso": source_path.name == "character.png",
            "bottom_anchor": source_path.name == "character.png",
            "normalize_head_height": source_path.name == "character.png",
            "puppet_matrix": source_path.name == "character.png",
            "target_eye_y": 770,
            "target_head_height": 550,
            "target_head_width": 480,
            "neck_pivot_y": 820,
            "shoulder_min_width": 850,
            "body_anchor_y": 1920,
            "harmonic_head_body_scale": True,
        },
        "palette": {
            **palette,
            "accent_color": palette["teeth_color"],
        },
        "theme": {
            "glow_color": palette["teeth_color"],
            "glow_radius": max(12, eye_radius),
        },
        "layers": {**base_layers, **mouth_layers},
        "calibration": {
            **calibration,
            "source_image": source_path.name,
            "alpha_holes_filled_px": filled_alpha_pixels,
            **slicing,
        },
    }

    # Exclusive creation is the final artist-override guard. If another
    # process supplied a manifest while detection ran, it wins unchanged.
    try:
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    except FileExistsError:
        pass
    return manifest_path


__all__ = [
    "auto_rig_character",
    "birefnet_cutout",
    "chroma_key_cutout",
    "has_character_imagery",
    "solidify_character_alpha",
]
