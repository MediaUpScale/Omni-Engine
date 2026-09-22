# -*- coding: utf-8 -*-
"""Generic high-resolution sprite-matrix loader.

External art lives under ``{ASSETS_PATH}/puppets/<character_id>/`` and may
provide body/head/eye layers, nine ``mouth_<viseme>.png`` overlays, and a
studio ``bg.png``. Missing files are filled by the procedural generator.
Nothing here knows about any specific channel, robot, or debate.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .types import REST_VISEME, VISEMES, PuppetAnchors, PuppetTheme

_LOG = logging.getLogger("animator.puppet")

# Canonical external sprite-matrix schema.
LAYER_KEYS: tuple[str, ...] = (
    "body",
    "head",
    "eyes_open",
    "eyes_half",
    "eyes_blink",
    "glow",
    "bg",
)


def viseme_layer_key(viseme: str) -> str:
    """Layer key for one Rhubarb mouth shape (``"D"`` -> ``"mouth_D"``)."""
    return f"mouth_{viseme.upper()[:1]}"


#: The phonetic mouth set: one sprite per canonical Rhubarb shape.
VISEME_LAYER_KEYS: tuple[str, ...] = tuple(viseme_layer_key(v) for v in VISEMES)

#: Everything a fully-featured skin ships: base layers + the viseme set.
ALL_LAYER_KEYS: tuple[str, ...] = LAYER_KEYS + VISEME_LAYER_KEYS

_EYE_LAYER_BY_STATE = {0: "eyes_open", 1: "eyes_half", 2: "eyes_blink"}

#: Coarse RMS state -> viseme, for skins/callers still driving the legacy
#: 3-level mouth instead of a phonetic track.
_STATE_VISEME = {0: "A", 1: "C", 2: "D"}
_EMOTION_BROW_ANGLES = {
    "skeptical": (0.0, 0.0),
    "neutral": (1.0, -1.0),
    "inquisitor": (6.0, -6.0),
    "resolute": (3.5, -3.5),
    "conceded": (0.0, 0.0),
}
BROW_STATES: tuple[str, ...] = (
    "neutral",
    "skeptical",
    "inquisitor",
    "resolute",
    "conceded",
)


def emotion_brow_angles(emotion: str, pulse_deg: float = 0.0) -> tuple[float, float]:
    """Legacy numeric telemetry for callers migrating to brow sprites."""
    baseline = _EMOTION_BROW_ANGLES.get((emotion or "neutral").lower(), (0.0, 0.0))
    pulse = float(np.clip(pulse_deg, 0.0, 1.5))
    return tuple(
        angle + (pulse if angle >= 0.0 else -pulse)
        for angle in baseline
    )


def emotion_brow_state(emotion: str) -> str:
    state = (emotion or "neutral").strip().lower()
    return state if state in BROW_STATES else "neutral"

@dataclass(frozen=True, slots=True)
class PuppetSkin:
    """Validated view of one ``puppet.json`` plus the directory it lives in."""

    character_id: str
    anchors: PuppetAnchors
    theme: PuppetTheme
    root: Path
    layer_files: dict[str, str]
    reference_canvas_size: tuple[int, int] = (480, 760)
    eye_bboxes: tuple[tuple[int, int, int, int], ...] = ()

    @classmethod
    def load(cls, puppet_dir: Path) -> "PuppetSkin":
        """Load ``puppet.json`` from ``puppet_dir``. Raises if it is missing."""
        manifest_path = Path(puppet_dir) / "puppet.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls._from_manifest(data, Path(puppet_dir))

    @classmethod
    def load_or_create(cls, puppet_dir: Path) -> "PuppetSkin":
        """Load an existing skin, or synthesize a default manifest + assets.

        This is the entry point most callers want: it guarantees a fully
        usable, on-disk puppet directory (manifest *and* PNG layers) no
        matter what was there before.
        """
        from .asset_generator import ensure_puppet_manifest  # noqa: PLC0415

        puppet_dir = Path(puppet_dir)
        manifest_path = puppet_dir / "puppet.json"
        if not manifest_path.is_file():
            _LOG.info("no puppet.json at %s — generating a default skin", puppet_dir)
            ensure_puppet_manifest(puppet_dir)
        skin = cls.load(puppet_dir)
        skin.ensure_assets()
        return skin

    @classmethod
    def _from_manifest(cls, data: dict, root: Path) -> "PuppetSkin":
        anchors_raw = data.get("anchors", {})
        default_w = _DEFAULT_CANVAS_SIZE[0]
        anchors = PuppetAnchors(
            mouth=_as_vec2(anchors_raw.get("mouth", [default_w // 2, 320])),
            eyes=_as_vec2(anchors_raw.get("eyes", [default_w // 2, 230])),
            head_pivot=_as_vec2(anchors_raw.get("head_pivot", [default_w // 2, 260])),
            neck_pivot=_as_vec2(
                anchors_raw.get(
                    "neck_pivot",
                    anchors_raw.get("head_pivot", [default_w // 2, 360]),
                )
            ),
        )
        theme_raw = data.get("theme", {})
        theme = PuppetTheme(
            glow_color=str(theme_raw.get("glow_color", "#00F0FF")),
            glow_radius=int(theme_raw.get("glow_radius", 25)),
        )
        raw_layers = data.get("layers", {})

        def resolve_file(key: str, *legacy_keys: str) -> str:
            configured = raw_layers.get(key)
            if configured:
                return str(configured)
            for legacy in legacy_keys:
                configured = raw_layers.get(legacy)
                if configured:
                    return str(configured)
                legacy_path = root / f"{legacy}.png"
                if legacy_path.is_file():
                    return legacy_path.name
            return f"{key}.png"

        layer_files = {
            "body": resolve_file("body"),
            "head": resolve_file("head"),
            "eyes_open": resolve_file("eyes_open"),
            "eyes_half": resolve_file("eyes_half", "eyes_blink", "eyes_closed"),
            "eyes_blink": resolve_file("eyes_blink", "eyes_closed", "eyes_half"),
            "glow": resolve_file("glow"),
            "bg": resolve_file("bg"),
        }
        for viseme in VISEMES:
            key = viseme_layer_key(viseme)
            layer_files[key] = resolve_file(key)

        canvas_raw = data.get("canvas_size", _DEFAULT_CANVAS_SIZE)
        reference_canvas_size = _as_vec2(canvas_raw)
        eye_bboxes = tuple(
            tuple(int(value) for value in box)
            for box in data.get("calibration", {}).get("eye_bboxes", ())
            if len(box) == 4
        )
        character_id = str(data.get("character_id") or root.name)
        return cls(
            character_id=character_id,
            anchors=anchors,
            theme=theme,
            root=root,
            layer_files=layer_files,
            reference_canvas_size=reference_canvas_size,
            eye_bboxes=eye_bboxes,
        )

    def layer_path(self, key: str) -> Path:
        return self.root / self.layer_files[key]

    def missing_layers(self) -> list[str]:
        return [key for key in ALL_LAYER_KEYS if not self.layer_path(key).is_file()]

    def available_visemes(self) -> list[str]:
        """Viseme shapes this skin actually ships a sprite for."""
        return [v for v in VISEMES if self.layer_path(viseme_layer_key(v)).is_file()]

    def ensure_assets(self) -> None:
        """Procedurally generate any PNG layer that is not already on disk."""
        missing = self.missing_layers()
        if not missing:
            return
        from .asset_generator import generate_puppet_assets  # noqa: PLC0415

        _LOG.info("generating %d missing layer(s) for %r: %s", len(missing), self.character_id, missing)
        generate_puppet_assets(self, missing=missing)


def _as_vec2(value) -> tuple[int, int]:
    x, y = value
    return int(x), int(y)


# Fallback canvas-size guess used only when a hand-authored ``puppet.json``
# omits an anchor entirely. Real generation always uses ``asset_generator``'s
# canonical canvas size, which matches this default.
_DEFAULT_CANVAS_SIZE: tuple[int, int] = (480, 760)


class PuppetRig:
    """Loads a :class:`PuppetSkin`'s layers and composites animated frames.

    All layers share one canvas size. The body remains grounded while the
    head, eyes, and mouth are grouped into a separate articulated plane that
    can rotate subtly around the manifest's neck pivot.
    """

    def __init__(self, skin: PuppetSkin) -> None:
        skin.ensure_assets()
        self.skin = skin
        layers: dict[str, Image.Image] = {
            key: Image.open(skin.layer_path(key)).convert("RGBA") for key in LAYER_KEYS
        }
        for viseme in VISEMES:
            key = viseme_layer_key(viseme)
            layers[key] = Image.open(skin.layer_path(key)).convert("RGBA")

        # The body defines the aligned sprite canvas. External high-res
        # overlays are accepted as-is when aligned, or normalized to that
        # canvas when an artist exported one layer at a different size.
        self.canvas_size = layers["body"].size
        if layers["head"].size != self.canvas_size:
            raise ValueError(
                f"puppet {skin.character_id!r} head/body canvases differ: "
                f"head={layers['head'].size}, body={self.canvas_size}"
            )
        for key, image in tuple(layers.items()):
            if key == "bg" or image.size == self.canvas_size:
                continue
            _LOG.warning(
                "resizing puppet %r layer %s from %s to body canvas %s",
                skin.character_id,
                key,
                image.size,
                self.canvas_size,
            )
            layers[key] = _contain_rgba(image, self.canvas_size)
        self.background = layers["bg"]
        self._body = np.asarray(layers["body"], dtype=np.uint8).copy()
        self._body_rgba = self._body
        static_stack = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        static_stack.alpha_composite(layers["body"])
        static_stack.alpha_composite(layers["head"])
        static_stack.alpha_composite(layers["eyes_open"])
        self._static_rgba = np.asarray(static_stack, dtype=np.uint8).copy()

        # The glow layer's own alpha channel, as a float array, so per-frame
        # intensity pulsing is one numpy multiply instead of a PIL re-encode.
        # Cropped to its own tight bounding box up front — most of the
        # canvas is fully transparent glow, and blending the whole 480x760
        # frame for a small aura every frame is pure wasted float math.
        glow_rgb_full, glow_alpha_full = self._split_rgba_np(layers["glow"])
        self._glow_bbox = _alpha_bbox(glow_alpha_full)
        x0, y0, x1, y1 = self._glow_bbox
        self._glow_rgb_u8 = glow_rgb_full[y0:y1, x0:x1].astype(np.uint8)
        self._glow_alpha_u8 = glow_alpha_full[y0:y1, x0:x1].astype(np.uint8)

        # Cache nine head+mouth cels, then apply one of the tiny eyelid
        # overlays at runtime. Prebuilding all 9x3 full-canvas combinations
        # would exceed a gigabyte for two 1536x2752 artist rigs.
        self._head_stack: dict[str, np.ndarray] = {}
        self._head_bbox: dict[str, tuple[int, int, int, int]] = {}
        self._mouth_overlay: dict[str, np.ndarray] = {}
        self._mouth_bbox: dict[str, tuple[int, int, int, int]] = {}
        for viseme in VISEMES:
            mouth_layer = layers[viseme_layer_key(viseme)]
            mouth_arr = np.asarray(mouth_layer, dtype=np.uint8).copy()
            self._mouth_overlay[viseme] = mouth_arr
            self._mouth_bbox[viseme] = _alpha_bbox(
                mouth_arr[..., 3].astype(np.float32),
                pad=3,
            )
            head_stack = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
            head_stack.alpha_composite(layers["head"])
            head_stack.alpha_composite(layers["eyes_open"])
            head_stack.alpha_composite(mouth_layer)
            head_arr = np.asarray(head_stack, dtype=np.uint8).copy()
            self._head_stack[viseme] = head_arr
            self._head_bbox[viseme] = _alpha_bbox(
                head_arr[..., 3].astype(np.float32),
                pad=80,
            )

        self._eye_overlay: dict[int, np.ndarray] = {}
        self._eye_bbox: dict[int, tuple[int, int, int, int]] = {}
        for eye_state, eye_key in _EYE_LAYER_BY_STATE.items():
            overlay = np.asarray(layers[eye_key], dtype=np.uint8).copy()
            self._eye_overlay[eye_state] = overlay
            self._eye_bbox[eye_state] = _alpha_bbox(
                overlay[..., 3].astype(np.float32),
                pad=8,
            )
        base_stack = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        base_stack.alpha_composite(layers["body"])
        base_stack.alpha_composite(layers["head"])
        self._content_bbox = _alpha_bbox(
            np.asarray(base_stack, dtype=np.uint8)[..., 3].astype(np.float32),
            pad=80,
        )

        ref_w, ref_h = self.skin.reference_canvas_size
        scale_x = self.canvas_size[0] / float(max(1, ref_w))
        scale_y = self.canvas_size[1] / float(max(1, ref_h))
        self._pivot = (
            int(round(self.skin.anchors.head_pivot[0] * scale_x)),
            int(round(self.skin.anchors.head_pivot[1] * scale_y)),
        )
        self._neck_pivot = (
            int(round(self.skin.anchors.neck_pivot[0] * scale_x)),
            int(round(self.skin.anchors.neck_pivot[1] * scale_y)),
        )
        self._eye_bboxes = tuple(
            (
                int(round(x0 * scale_x)),
                int(round(y0 * scale_y)),
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
            )
            for x0, y0, x1, y1 in self.skin.eye_bboxes
        )
        self._brow_cache: dict[
            tuple[str, bool],
            tuple[np.ndarray, tuple[int, int, int, int]] | None,
        ] = {}
        self._articulated_head_cache: OrderedDict[
            tuple[str, int, str, bool, float],
            tuple[np.ndarray, tuple[int, int, int, int]],
        ] = OrderedDict()

    # -- Camera geometry ------------------------------------------------- #
    def closeup_box(self) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` medium close-up crop: chest through head.

        Derived from the skin's own anchors rather than hardcoded pixels,
        so a re-skin with a differently-proportioned character still frames
        correctly. Horizontally it is the full shoulder span; vertically it
        runs from above the crown down to mid-torso, which is exactly the
        "chest to head" framing the shot-reverse-shot director asks for.
        """
        w, h = self.canvas_size
        _, pivot_y = self._pivot
        half_w = int(w * 0.46)
        cx = w // 2
        x0 = max(0, cx - half_w)
        x1 = min(w, cx + half_w)
        y0 = max(0, pivot_y - int(h * 0.25))
        y1 = min(h, pivot_y + int(h * 0.41))
        return x0, y0, x1, y1

    @staticmethod
    def _split_rgba_np(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(img, dtype=np.float32)
        return arr[..., :3], arr[..., 3]

    def compose(
        self,
        *,
        viseme: str | None = None,
        mouth_state: int = 0,
        eye_state: int = 0,
        y_offset: float = 0.0,
        scale: float = 1.0,
        brightness: float = 1.0,
        glow_intensity: float = 0.0,
        head_angle: float = 0.0,
        emotion: str = "neutral",
        previous_emotion: str = "neutral",
        emotion_mix: float = 1.0,
        brow_pulse_deg: float = 0.0,
    ) -> np.ndarray:
        """Composite one frame of this puppet at the given animation state.

        ``viseme`` (one of :data:`~core.animator.types.VISEMES`) selects the
        phonetic mouth sprite. When omitted, the legacy coarse
        ``mouth_state`` (0/1/2) is mapped onto the equivalent shape.

        Hot path: called once per output video frame. Everything here is
        numpy/OpenCV (translation/resize rather than PIL transforms) — roughly a
        30-40x speedup over the equivalent PIL calls, which is what keeps a
        45s clip renderable in well under 15s wall time.
        """
        shape = (viseme or _STATE_VISEME.get(mouth_state, REST_VISEME)).upper()[:1]
        if shape not in self._head_stack:
            shape = REST_VISEME
        head = self._head_stack[shape]
        if eye_state in (1, 2):
            head = _alpha_composite_rgba(
                head,
                self._eye_overlay[eye_state],
                self._eye_bbox[eye_state],
            )
        mix = float(np.clip(emotion_mix, 0.0, 1.0))
        brow_state = emotion_brow_state(
            emotion if mix >= 0.5 else previous_emotion
        )
        brow = self._brow_overlay(brow_state, brow_pulse_deg > 0.01)
        if brow is not None:
            head = _alpha_composite_crop(head, brow[0], brow[1])
        if abs(head_angle) > 0.001:
            head = self._rotate_head(head, head_angle)
        frame = _alpha_composite_rgba(
            self._body,
            head,
            self._head_bbox[shape],
        )

        if abs(brightness - 1.0) > 0.001:
            frame = frame.copy()
            # Restricted to this state's own (padded) content bbox — most of
            # the 480x760 canvas is fully transparent background, and
            # scaling every one of those pixels every frame was the single
            # largest cost in this hot path.
            bx0, by0, bx1, by1 = self._content_bbox
            region = frame[by0:by1, bx0:bx1, :3]
            frame[by0:by1, bx0:bx1, :3] = cv2.convertScaleAbs(region, alpha=max(0.0, brightness), beta=0)

        if abs(scale - 1.0) > 0.001:
            frame = self._apply_scale(frame, scale)

        canvas = self._composite_glow(frame, glow_intensity)

        if abs(y_offset) > 0.01:
            canvas = self._apply_y_offset(canvas, y_offset)

        return canvas

    @property
    def static_rgba(self) -> np.ndarray:
        """Immutable body + neutral head used by the dirty-rectangle fast path."""
        return self._static_rgba

    @property
    def body_rgba(self) -> np.ndarray:
        """Immutable body plane used beneath the articulated head crop."""
        return self._body_rgba

    def mouth_overlay(
        self,
        viseme: str,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        shape = (viseme or REST_VISEME).upper()[:1]
        if shape not in self._mouth_overlay:
            shape = REST_VISEME
        bbox = self._mouth_bbox[shape]
        x0, y0, x1, y1 = bbox
        return self._mouth_overlay[shape][y0:y1, x0:x1], bbox

    def eye_overlay(
        self,
        eye_state: int,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        if eye_state not in (1, 2):
            return None
        bbox = self._eye_bbox[eye_state]
        x0, y0, x1, y1 = bbox
        return self._eye_overlay[eye_state][y0:y1, x0:x1], bbox

    def brow_overlay(
        self,
        state: str,
        emphasized: bool = False,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        return self._brow_overlay(state, emphasized)

    def _brow_overlay(
        self,
        state: str,
        emphasized: bool = False,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        """Build one of five hand-inked Ghibli mecha-brow sprite states.

        Each state has its own curved polygon silhouette. No sprite is rotated,
        so the result reads as expressive face art rather than clock hands.
        """
        brow_state = emotion_brow_state(state)
        key = (brow_state, bool(emphasized))
        if key in self._brow_cache:
            return self._brow_cache[key]
        if len(self._eye_bboxes) < 2:
            self._brow_cache[key] = None
            return None

        gemini = "gemini" in self.skin.character_id.lower()
        outline = (21, 32, 38, 255) if gemini else (43, 26, 21, 255)
        metal = (83, 105, 112, 245) if gemini else (157, 96, 54, 245)
        highlight = (151, 174, 180, 210) if gemini else (226, 154, 93, 210)
        layer = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        for index, (x0, y0, x1, y1) in enumerate(self._eye_bboxes[:2]):
            eye_w = x1 - x0
            plate_w = max(58, int(round(eye_w * 0.80)))
            plate_h = 18
            scale = 4
            pad = 24
            patch = Image.new(
                "RGBA",
                ((plate_w + pad * 2) * scale, (plate_h + pad * 2) * scale),
                (0, 0, 0, 0),
            )
            draw = ImageDraw.Draw(patch, "RGBA")
            left = float(pad * scale)
            right = float((pad + plate_w) * scale)
            middle = (left + right) * 0.5
            center = float((pad + plate_h // 2) * scale)
            inner_is_right = index == 0

            outer_y = center
            mid_y = center - 2 * scale
            inner_y = center
            thickness = 7 * scale
            vertical_shift = 0
            if brow_state == "skeptical":
                if index == 0:
                    outer_y -= 10 * scale
                    mid_y -= 7 * scale
                    inner_y -= 2 * scale
                else:
                    mid_y -= 1 * scale
            elif brow_state == "inquisitor":
                inner_y += (15 + (3 if emphasized else 0)) * scale
                mid_y += 7 * scale
                thickness += 2 * scale
            elif brow_state == "resolute":
                inner_y += (8 + (2 if emphasized else 0)) * scale
                mid_y += 5 * scale
                vertical_shift = 7 * scale
                thickness += 3 * scale
            elif brow_state == "conceded":
                outer_y += 3 * scale
                mid_y -= 7 * scale
                inner_y += 2 * scale
                thickness -= 1 * scale
            elif emphasized:
                thickness += 1 * scale

            if not inner_is_right:
                outer_y, inner_y = inner_y, outer_y
            top = [
                (left, outer_y + vertical_shift),
                (middle, mid_y + vertical_shift),
                (right, inner_y + vertical_shift),
            ]
            bottom = [
                (right, inner_y + vertical_shift + thickness),
                (middle, mid_y + vertical_shift + thickness * 0.72),
                (left, outer_y + vertical_shift + thickness * 0.55),
            ]
            points = top + bottom
            draw.polygon(points, fill=metal)
            draw.line(
                points + [points[0]],
                fill=outline,
                width=3 * scale,
                joint="curve",
            )
            draw.line(
                top,
                fill=highlight,
                width=max(2, scale),
                joint="curve",
            )
            patch = patch.resize(
                (plate_w + pad * 2, plate_h + pad * 2),
                Image.Resampling.LANCZOS,
            )
            center_x = (x0 + x1) // 2
            center_y = y0 - max(5, plate_h // 3) + 18
            if (self.skin.character_id.lower().find("gemini") >= 0 and index == 1):
                center_y += 4
            layer.alpha_composite(
                patch,
                (center_x - patch.width // 2, center_y - patch.height // 2),
            )

        rgba = np.asarray(layer, dtype=np.uint8)
        bbox = _alpha_bbox(rgba[..., 3].astype(np.float32), pad=4)
        x0, y0, x1, y1 = bbox
        result = (rgba[y0:y1, x0:x1].copy(), bbox)
        self._brow_cache[key] = result
        return result

    def articulated_head_overlay(
        self,
        *,
        viseme: str,
        eye_state: int,
        brow_state: str,
        brow_emphasized: bool,
        angle_deg: float,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Return a cached, neck-pivoted head crop with all facial sprites attached."""
        shape = (viseme or REST_VISEME).upper()[:1]
        if shape not in self._head_stack:
            shape = REST_VISEME
        state = emotion_brow_state(brow_state)
        angle = float(np.clip(round(float(angle_deg) / 0.3) * 0.3, -1.2, 1.2))
        cache_key = (shape, int(eye_state), state, bool(brow_emphasized), angle)
        cached = self._articulated_head_cache.get(cache_key)
        if cached is not None:
            self._articulated_head_cache.move_to_end(cache_key)
            return cached

        head = self._head_stack[shape]
        if eye_state in (1, 2):
            head = _alpha_composite_rgba(
                head,
                self._eye_overlay[eye_state],
                self._eye_bbox[eye_state],
            )
        brow = self._brow_overlay(state, brow_emphasized)
        if brow is not None:
            head = _alpha_composite_crop(head, brow[0], brow[1])

        hx0, hy0, hx1, hy1 = self._head_bbox[shape]
        pivot_x, pivot_y = self._neck_pivot
        pad = 100
        x0 = max(0, min(hx0, pivot_x - pad))
        y0 = max(0, min(hy0, pivot_y - pad))
        x1 = min(self.canvas_size[0], max(hx1, pivot_x + pad))
        y1 = min(self.canvas_size[1], max(hy1, pivot_y + pad))
        crop = head[y0:y1, x0:x1].copy()
        if abs(angle) > 0.001:
            matrix = cv2.getRotationMatrix2D(
                (pivot_x - x0, pivot_y - y0),
                angle,
                1.0,
            )
            crop = cv2.warpAffine(
                crop,
                matrix,
                (crop.shape[1], crop.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
        result = (crop, (x0, y0, x1, y1))
        self._articulated_head_cache[cache_key] = result
        while len(self._articulated_head_cache) > 96:
            self._articulated_head_cache.popitem(last=False)
        return result

    def _rotate_head(self, head: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate the articulated head plane around the collar joint."""
        h, w = head.shape[:2]
        matrix = cv2.getRotationMatrix2D(self._neck_pivot, float(angle_deg), 1.0)
        return cv2.warpAffine(
            head,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    def compose_image(self, **kwargs) -> Image.Image:
        """PIL-Image-returning variant of :meth:`compose`, for callers (or
        tests) that want a savable/inspectable frame instead of the raw
        numpy array the compositor's hot path consumes directly."""
        return Image.fromarray(self.compose(**kwargs), mode="RGBA")

    def _apply_scale(self, arr: np.ndarray, factor: float) -> np.ndarray:
        w, h = self.canvas_size
        new_w, new_h = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((h, w, 4), dtype=np.uint8)
        # Keep the head-pivot anchor visually stationary while the puppet
        # scales up/down, so a 1.03x "lean forward" reads as a lean, not a
        # canvas-corner zoom.
        px, py = self._pivot
        offset_x = int(round(px - px * factor))
        offset_y = int(round(py - py * factor))
        _paste_np(out, resized, offset_x, offset_y)
        return out

    def _composite_glow(self, body: np.ndarray, glow_intensity: float) -> np.ndarray:
        """Blend the aura *behind* the body, restricted to its own bounding
        box and driven entirely by ``cv2``'s SIMD uint8 ops (no numpy
        float32 temporaries) — the naive full-canvas float version was the
        single largest cost in the whole compose() hot path.

        The glow only shows through where the body is not already opaque
        there (``inv_body_a``); it never darkens or discolors an opaque
        pixel, matching a true "body over glow" composite for every pixel
        that is not sitting inside a 1-2px antialiased body edge — an
        imperceptible approximation in exchange for a large speedup.
        """
        if glow_intensity <= 0.001:
            return body
        x0, y0, x1, y1 = self._glow_bbox
        out = body.copy()
        region = out[y0:y1, x0:x1]
        intensity = min(1.0, max(0.0, glow_intensity))

        glow_a = cv2.convertScaleAbs(self._glow_alpha_u8, alpha=intensity)
        body_a = region[..., 3]
        inv_body_a = cv2.bitwise_not(body_a)
        add_a = cv2.multiply(glow_a, inv_body_a, scale=1.0 / 255.0)

        add_a_3ch = cv2.merge([add_a, add_a, add_a])
        glow_contrib = cv2.multiply(self._glow_rgb_u8, add_a_3ch, scale=1.0 / 255.0)

        region[..., :3] = cv2.add(region[..., :3], glow_contrib)
        region[..., 3] = cv2.add(body_a, add_a)
        return out

    @staticmethod
    def _apply_y_offset(arr: np.ndarray, y_offset: float) -> np.ndarray:
        h, w = arr.shape[:2]
        out = np.zeros_like(arr)
        _paste_np(out, arr, 0, int(round(y_offset)))
        return out


def _contain_rgba(image: Image.Image, canvas_size: tuple[int, int]) -> Image.Image:
    """Uniformly contain an auxiliary layer without changing pixel aspect."""
    target_w, target_h = canvas_size
    scale = min(target_w / float(image.width), target_h / float(image.height))
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas.alpha_composite(
        resized,
        ((target_w - width) // 2, (target_h - height) // 2),
    )
    return canvas


def _alpha_bbox(alpha: np.ndarray, *, pad: int = 2) -> tuple[int, int, int, int]:
    """Tight ``(x0, y0, x1, y1)`` bounding box of an alpha channel's non-zero
    region, padded slightly and clamped to the array's own bounds."""
    h, w = alpha.shape[:2]
    ys, xs = np.nonzero(alpha > 0.5)
    if ys.size == 0:
        return 0, 0, w, h
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def _alpha_composite_rgba(
    background: np.ndarray,
    foreground: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Fast RGBA-over-RGBA blend limited to the articulated head region."""
    out = background.copy()
    x0, y0, x1, y1 = bbox
    src = foreground[y0:y1, x0:x1]
    dst = out[y0:y1, x0:x1]
    alpha = src[..., 3]
    inv_alpha = cv2.bitwise_not(alpha)
    alpha_3 = cv2.merge([alpha, alpha, alpha])
    inv_alpha_3 = cv2.merge([inv_alpha, inv_alpha, inv_alpha])
    dst[..., :3] = cv2.add(
        cv2.multiply(src[..., :3], alpha_3, scale=1.0 / 255.0),
        cv2.multiply(dst[..., :3], inv_alpha_3, scale=1.0 / 255.0),
    )
    dst[..., 3] = cv2.add(
        alpha,
        cv2.multiply(dst[..., 3], inv_alpha, scale=1.0 / 255.0),
    )
    return out


def _alpha_composite_crop(
    background: np.ndarray,
    foreground_crop: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """RGBA-over-RGBA blend for an already cropped overlay."""
    out = background.copy()
    x0, y0, x1, y1 = bbox
    dst = out[y0:y1, x0:x1]
    alpha = foreground_crop[..., 3]
    inv_alpha = cv2.bitwise_not(alpha)
    alpha_3 = cv2.merge([alpha, alpha, alpha])
    inv_alpha_3 = cv2.merge([inv_alpha, inv_alpha, inv_alpha])
    dst[..., :3] = cv2.add(
        cv2.multiply(foreground_crop[..., :3], alpha_3, scale=1.0 / 255.0),
        cv2.multiply(dst[..., :3], inv_alpha_3, scale=1.0 / 255.0),
    )
    dst[..., 3] = cv2.add(
        alpha,
        cv2.multiply(dst[..., 3], inv_alpha, scale=1.0 / 255.0),
    )
    return out


def _paste_np(dest: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Alpha-free numpy paste (like ``Image.paste`` with no mask): copies
    the overlapping region of ``src`` into ``dest`` at offset ``(x, y)``,
    clipped to ``dest``'s bounds. Used for pure translation/placement where
    no blending against existing pixels is needed (dest starts transparent)."""
    dh, dw = dest.shape[:2]
    sh, sw = src.shape[:2]
    dst_x0, dst_y0 = max(0, x), max(0, y)
    dst_x1, dst_y1 = min(dw, x + sw), min(dh, y + sh)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x0, src_y0 = dst_x0 - x, dst_y0 - y
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
    dest[dst_y0:dst_y1, dst_x0:dst_x1] = src[src_y0:src_y1, src_x0:src_x1]


__all__ = [
    "ALL_LAYER_KEYS",
    "BROW_STATES",
    "LAYER_KEYS",
    "PuppetRig",
    "PuppetSkin",
    "VISEME_LAYER_KEYS",
    "emotion_brow_angles",
    "emotion_brow_state",
    "viseme_layer_key",
]
