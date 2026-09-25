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


REST_MOUTH_STATES: tuple[str, ...] = (
    "neutral",
    "smug_smile",
    "stressed_grimace",
)


def rest_mouth_layer_key(state: str) -> str:
    normalized = (state or "neutral").strip().lower()
    normalized = {
        "smug": "smug_smile",
        "defeated": "stressed_grimace",
        "stressed": "stressed_grimace",
    }.get(normalized, normalized)
    if normalized not in REST_MOUTH_STATES:
        normalized = "neutral"
    if normalized == "neutral":
        return "mouth_X_neutral"
    return f"mouth_{normalized}"


#: The phonetic mouth set: one sprite per canonical Rhubarb shape.
VISEME_LAYER_KEYS: tuple[str, ...] = tuple(viseme_layer_key(v) for v in VISEMES)
REST_MOUTH_LAYER_KEYS: tuple[str, ...] = tuple(
    rest_mouth_layer_key(state) for state in REST_MOUTH_STATES
)

#: Everything a fully-featured skin ships: base layers + the viseme set.
ALL_LAYER_KEYS: tuple[str, ...] = (
    LAYER_KEYS + VISEME_LAYER_KEYS + REST_MOUTH_LAYER_KEYS
)

_EYE_LAYER_BY_STATE = {0: "eyes_open", 1: "eyes_half", 2: "eyes_blink"}

#: Coarse RMS state -> viseme, for skins/callers still driving the legacy
#: 3-level mouth instead of a phonetic track.
_STATE_VISEME = {0: "A", 1: "C", 2: "D"}
_EMOTION_BROW_ANGLES = {
    "skeptical": (12.0, -12.0),
    "neutral": (0.0, 0.0),
    "inquisitor": (-5.5, 5.5),
    "resolute": (0.0, 0.0),
    "conceded": (12.0, -12.0),
    "defeated": (12.0, -12.0),
    "disbelief": (12.0, -12.0),
    "troubled": (12.0, -12.0),
    "concerned": (12.0, -12.0),
}
BROW_STATES: tuple[str, ...] = (
    "neutral",
    "skeptical",
    "inquisitor",
    "resolute",
    "conceded",
    "troubled",
)


def emotion_brow_angles(emotion: str, pulse_deg: float = 0.0) -> tuple[float, float]:
    """Legacy numeric telemetry for callers migrating to brow sprites."""
    baseline = _EMOTION_BROW_ANGLES.get((emotion or "neutral").lower(), (0.0, 0.0))
    if baseline == (0.0, 0.0):
        return baseline
    pulse = float(np.clip(pulse_deg, 0.0, 1.5))
    return tuple(
        angle + (pulse if angle >= 0.0 else -pulse)
        for angle in baseline
    )


def emotion_brow_state(emotion: str) -> str:
    state = (emotion or "neutral").strip().lower()
    if state in {"defeated", "stressed"}:
        return "conceded"
    if state in {"disbelief", "troubled", "concerned"}:
        return "troubled"
    return state if state in BROW_STATES else "neutral"


def emotion_rest_mouth_state(emotion: str) -> str:
    state = (emotion or "neutral").strip().lower()
    if state in {"inquisitor", "confident"}:
        return "smug_smile"
    if state in {"conceded", "defeated", "stressed"}:
        return "stressed_grimace"
    return "neutral"

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
    lens_circles: tuple[tuple[float, float, float], ...] = ()
    mouth_style: str = "ghibli_mecha"
    palette: dict[str, str] | None = None
    skin_version: str = "unversioned"
    eye_style: str = "circular"
    brow_config: dict | None = None
    framing: dict | None = None

    @classmethod
    def load(cls, puppet_dir: Path) -> "PuppetSkin":
        """Load ``puppet.json`` from ``puppet_dir``. Raises if it is missing."""
        manifest_path = Path(puppet_dir) / "puppet.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls._from_manifest(data, Path(puppet_dir))

    @classmethod
    def load_or_create(cls, puppet_dir: Path) -> "PuppetSkin":
        """Load an artist skin, auto-rig raw art, or synthesize a fallback.

        This is the entry point most callers want: it guarantees a fully
        usable, on-disk puppet directory (manifest *and* PNG layers) no
        matter what was there before.
        """
        puppet_dir = Path(puppet_dir)
        manifest_path = puppet_dir / "puppet.json"
        if manifest_path.is_file():
            # Artist mode has absolute priority. In particular, do not call
            # the legacy manifest upgrader, which rewrites calibrated JSON.
            skin = cls.load(puppet_dir)
            skin.ensure_assets()
            return skin

        from .factory.rigger import (  # noqa: PLC0415
            auto_rig_character,
            has_character_imagery,
        )

        if has_character_imagery(puppet_dir):
            _LOG.info("no puppet.json at %s — invoking V3 auto-rigger", puppet_dir)
            auto_rig_character(puppet_dir, puppet_dir.name)
        else:
            from .asset_generator import ensure_puppet_manifest  # noqa: PLC0415

            _LOG.info("no character art at %s — generating a default skin", puppet_dir)
            ensure_puppet_manifest(puppet_dir)
        skin = cls.load(puppet_dir)
        skin.ensure_assets()
        return skin

    @classmethod
    def _from_manifest(cls, data: dict, root: Path) -> "PuppetSkin":
        anchors_raw = data.get("anchors", {})
        canvas_raw = data.get("canvas_size", _DEFAULT_CANVAS_SIZE)
        reference_canvas_size = _as_vec2(canvas_raw)
        default_w = reference_canvas_size[0]
        calibration = data.get("calibration", {})
        eye_bboxes = tuple(
            tuple(int(value) for value in box)
            for box in calibration.get("eye_bboxes", ())
            if len(box) == 4
        )
        legacy_eyes = _as_vec2(
            anchors_raw.get("eyes", [default_w // 2, 230])
        )
        if len(eye_bboxes) >= 2:
            inferred_left_eye = (
                (eye_bboxes[0][0] + eye_bboxes[0][2]) // 2,
                (eye_bboxes[0][1] + eye_bboxes[0][3]) // 2,
            )
            inferred_right_eye = (
                (eye_bboxes[1][0] + eye_bboxes[1][2]) // 2,
                (eye_bboxes[1][1] + eye_bboxes[1][3]) // 2,
            )
            inferred_radius = int(
                round(
                    sum(
                        min(box[2] - box[0], box[3] - box[1]) / 2.0
                        for box in eye_bboxes[:2]
                    )
                    / 2.0
                )
            )
        else:
            inferred_radius = int(anchors_raw.get("eye_radius", 55))
            inferred_left_eye = (
                legacy_eyes[0] - inferred_radius,
                legacy_eyes[1],
            )
            inferred_right_eye = (
                legacy_eyes[0] + inferred_radius,
                legacy_eyes[1],
            )
        anchors = PuppetAnchors(
            mouth=_as_vec2(anchors_raw.get("mouth", [default_w // 2, 320])),
            left_eye=_as_vec2(
                anchors_raw.get("left_eye", inferred_left_eye)
            ),
            right_eye=_as_vec2(
                anchors_raw.get("right_eye", inferred_right_eye)
            ),
            eye_radius=max(
                1,
                int(anchors_raw.get("eye_radius", inferred_radius)),
            ),
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
        if raw_layers.get("collar"):
            layer_files["collar"] = str(raw_layers["collar"])
        for viseme in VISEMES:
            key = viseme_layer_key(viseme)
            layer_files[key] = resolve_file(key)
        for state in REST_MOUTH_STATES:
            key = rest_mouth_layer_key(state)
            layer_files[key] = resolve_file(key)

        lens_circles = tuple(
            (float(circle[0]), float(circle[1]), float(circle[2]))
            for circle in calibration.get("lens_circles", ())
            if len(circle) == 3
        )
        character_id = str(data.get("character_id") or root.name)
        palette = {
            str(key): str(value)
            for key, value in (data.get("palette") or {}).items()
            if isinstance(value, str)
        }
        return cls(
            character_id=character_id,
            anchors=anchors,
            theme=theme,
            root=root,
            layer_files=layer_files,
            reference_canvas_size=reference_canvas_size,
            eye_bboxes=eye_bboxes,
            lens_circles=lens_circles,
            mouth_style=str(data.get("mouth_style") or "ghibli_mecha"),
            palette=palette,
            skin_version=str(data.get("skin_version") or "unversioned"),
            eye_style=str(data.get("eye_style") or "circular"),
            brow_config=dict(data.get("brows") or {}),
            framing=dict(data.get("framing") or {}),
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
        if "collar" in skin.layer_files and skin.layer_path("collar").is_file():
            layers["collar"] = Image.open(skin.layer_path("collar")).convert("RGBA")
        for viseme in VISEMES:
            key = viseme_layer_key(viseme)
            layers[key] = Image.open(skin.layer_path(key)).convert("RGBA")
        for state in REST_MOUTH_STATES:
            key = rest_mouth_layer_key(state)
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
        collar_layer = layers.get(
            "collar",
            Image.new("RGBA", self.canvas_size, (0, 0, 0, 0)),
        )
        self._collar_rgba = np.asarray(collar_layer, dtype=np.uint8).copy()
        self._collar_bbox = _alpha_bbox(
            self._collar_rgba[..., 3].astype(np.float32),
            pad=3,
        )
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
        self._rest_head_cache: dict[str, np.ndarray] = {}
        self._rest_mouth_crop: dict[
            str,
            tuple[np.ndarray, tuple[int, int, int, int]],
        ] = {}
        self._head_bbox: dict[str, tuple[int, int, int, int]] = {}
        self._mouth_overlay: dict[str, np.ndarray] = {}
        self._mouth_bbox: dict[str, tuple[int, int, int, int]] = {}
        for viseme in VISEMES:
            mouth_layer = layers[viseme_layer_key(viseme)]
            mouth_arr = np.asarray(mouth_layer, dtype=np.uint8).copy()
            mouth_bbox = _alpha_bbox(
                mouth_arr[..., 3].astype(np.float32),
                pad=3,
            )
            mx0, my0, mx1, my1 = mouth_bbox
            self._mouth_overlay[viseme] = mouth_arr[my0:my1, mx0:mx1].copy()
            self._mouth_bbox[viseme] = mouth_bbox
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
        bare_head = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        bare_head.alpha_composite(layers["head"])
        bare_head.alpha_composite(layers["eyes_open"])
        self._bare_head = np.asarray(bare_head, dtype=np.uint8).copy()
        self._head_content_bbox = _alpha_bbox(
            self._bare_head[..., 3].astype(np.float32),
        )
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            np.where(self._bare_head[..., 3] > 8, 255, 0).astype(np.uint8),
            connectivity=8,
        )
        if component_count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            self._head_crown_y = int(stats[largest, cv2.CC_STAT_TOP])
        else:
            self._head_crown_y = self._head_content_bbox[1]
        for state in REST_MOUTH_STATES:
            mouth = np.asarray(
                layers[rest_mouth_layer_key(state)],
                dtype=np.uint8,
            )
            bbox = _alpha_bbox(mouth[..., 3].astype(np.float32), pad=3)
            x0, y0, x1, y1 = bbox
            self._rest_mouth_crop[state] = (mouth[y0:y1, x0:x1].copy(), bbox)

        ref_w, ref_h = self.skin.reference_canvas_size
        reference_scale = min(
            self.canvas_size[0] / float(max(1, ref_w)),
            self.canvas_size[1] / float(max(1, ref_h)),
        )
        reference_offset_x = (
            self.canvas_size[0] - ref_w * reference_scale
        ) * 0.5
        reference_offset_y = (
            self.canvas_size[1] - ref_h * reference_scale
        ) * 0.5

        def reference_x(value: float) -> float:
            return float(value) * reference_scale + reference_offset_x

        def reference_y(value: float) -> float:
            return float(value) * reference_scale + reference_offset_y

        self._raw_lens_circles = tuple(
            (
                reference_x(cx),
                reference_y(cy),
                float(radius) * reference_scale,
            )
            for cx, cy, radius in self.skin.lens_circles
        )
        self._lens_circles = tuple(
            (
                cx + self._eye_stencil_nudge(index)[0],
                cy + self._eye_stencil_nudge(index)[1],
                radius,
            )
            for index, (cx, cy, radius) in enumerate(self._raw_lens_circles)
        )
        self._eye_bboxes = tuple(
            (
                int(round(reference_x(x0))),
                int(round(reference_y(y0))),
                int(round(reference_x(x1))),
                int(round(reference_y(y1))),
            )
            for x0, y0, x1, y1 in self.skin.eye_bboxes
        )
        self._eye_overlay: dict[int, np.ndarray] = {}
        self._eye_bbox: dict[int, tuple[int, int, int, int]] = {}
        for eye_state, eye_key in _EYE_LAYER_BY_STATE.items():
            overlay = self._stencil_to_lenses(
                self._translate_eye_overlay(
                    np.asarray(layers[eye_key], dtype=np.uint8).copy()
                )
            )
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

        self._pivot = (
            int(round(reference_x(self.skin.anchors.head_pivot[0]))),
            int(round(reference_y(self.skin.anchors.head_pivot[1]))),
        )
        self._neck_pivot = (
            int(round(reference_x(self.skin.anchors.neck_pivot[0]))),
            int(round(reference_y(self.skin.anchors.neck_pivot[1]))),
        )
        self._eye_centers = (
            (
                int(round(reference_x(self.skin.anchors.left_eye[0]))),
                int(round(reference_y(self.skin.anchors.left_eye[1]))),
            ),
            (
                int(round(reference_x(self.skin.anchors.right_eye[0]))),
                int(round(reference_y(self.skin.anchors.right_eye[1]))),
            ),
        )
        self._eye_radius = max(
            1,
            int(
                round(
                    self.skin.anchors.eye_radius
                    * reference_scale
                )
            ),
        )
        self._brow_cache: dict[
            tuple[str, float],
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
        mix = float(np.clip(emotion_mix, 0.0, 1.0))
        brow_state = emotion_brow_state(
            emotion if mix >= 0.5 else previous_emotion
        )
        rest_state = emotion_rest_mouth_state(
            emotion if mix >= 0.5 else previous_emotion
        )
        head = (
            self._rest_head(rest_state)
            if shape == REST_VISEME
            else self._head_stack[shape]
        )
        if eye_state in (1, 2):
            head = _alpha_composite_rgba(
                head,
                self._eye_overlay[eye_state],
                self._eye_bbox[eye_state],
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
        frame = _alpha_composite_rgba(
            frame,
            self._collar_rgba,
            self._collar_bbox,
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

    @property
    def head_content_bbox(self) -> tuple[int, int, int, int]:
        """Tight native-pixel bounds used for consistent camera framing."""
        return self._head_content_bbox

    @property
    def framing_head_height(self) -> int:
        """Visible crown-to-chin height, excluding the hidden neck socket."""
        mouth_y = float(self.skin.anchors.mouth[1])
        neck_y = float(self.skin.anchors.neck_pivot[1])
        inferred_chin_y = mouth_y + max(0.0, neck_y - mouth_y) * 0.35
        return max(1, int(round(inferred_chin_y - self._head_crown_y)))

    def collar_overlay(
        self,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        """Optional collar armor composited after the articulated head."""
        if "collar" not in self.skin.layer_files:
            return None
        x0, y0, x1, y1 = self._collar_bbox
        return self._collar_rgba[y0:y1, x0:x1], self._collar_bbox

    def mouth_overlay(
        self,
        viseme: str,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        shape = (viseme or REST_VISEME).upper()[:1]
        if shape not in self._mouth_overlay:
            shape = REST_VISEME
        bbox = self._mouth_bbox[shape]
        return self._mouth_overlay[shape], bbox

    def _rest_head(self, state: str) -> np.ndarray:
        if state == "neutral":
            return self._head_stack[REST_VISEME]
        cached = self._rest_head_cache.get(state)
        if cached is not None:
            return cached
        crop, bbox = self._rest_mouth_crop[state]
        head = _alpha_composite_crop(self._bare_head, crop, bbox)
        self._rest_head_cache[state] = head
        return head

    def _stencil_to_lenses(self, overlay: np.ndarray) -> np.ndarray:
        """Clip eyelid metal to calibrated circular or pill-shaped optics."""
        if overlay.shape[2] < 4 or not self._lens_circles:
            return overlay
        height, width = overlay.shape[:2]
        keep_u8 = np.zeros((height, width), dtype=np.uint8)
        use_pills = (
            self.skin.eye_style == "circular_slit"
            and len(self._eye_bboxes) >= len(self._lens_circles)
        )
        if use_pills:
            for x0, y0, x1, y1 in self._eye_bboxes[: len(self._lens_circles)]:
                box_w = max(1, x1 - x0)
                box_h = max(1, y1 - y0)
                radius = max(1, min(box_w, box_h) // 2)
                cy = (y0 + y1) // 2
                if box_w > box_h:
                    cv2.rectangle(
                        keep_u8,
                        (x0 + radius, y0),
                        (x1 - radius, y1),
                        255,
                        thickness=cv2.FILLED,
                    )
                    cv2.circle(keep_u8, (x0 + radius, cy), radius, 255, -1)
                    cv2.circle(keep_u8, (x1 - radius, cy), radius, 255, -1)
                else:
                    cv2.ellipse(
                        keep_u8,
                        ((x0 + x1) // 2, cy),
                        (box_w // 2, box_h // 2),
                        0,
                        0,
                        360,
                        255,
                        thickness=cv2.FILLED,
                    )
            keep = keep_u8 > 0
        else:
            grid_y, grid_x = np.ogrid[:height, :width]
            keep = np.zeros((height, width), dtype=bool)
            for cx, cy, radius in self._lens_circles:
                keep |= (
                    (grid_x + 0.5 - cx) ** 2
                    + (grid_y + 0.5 - cy) ** 2
                    <= radius * radius
                )
        if bool(keep.all()):
            return overlay
        clipped = overlay.copy()
        clipped[~keep] = 0
        return clipped

    def _eye_stencil_nudge(self, index: int) -> tuple[int, int]:
        """Micron calibration in artist-canvas pixels for one optic."""
        character = self.skin.character_id.lower()
        if "gemini" in character and index == 1:
            return (2, 0)  # viewer's-right / far optic
        if "llama" in character and index == 0:
            return (-2, -2)  # viewer's-left / near optic
        return (0, 0)

    def _translate_eye_overlay(self, overlay: np.ndarray) -> np.ndarray:
        """Move each shutter with its calibrated stencil before clipping."""
        if not self._raw_lens_circles:
            return overlay
        shifted = np.zeros_like(overlay)
        height, width = overlay.shape[:2]
        for index, (cx, cy, radius) in enumerate(self._raw_lens_circles):
            dx, dy = self._eye_stencil_nudge(index)
            pad = int(np.ceil(radius)) + 4
            x0 = max(0, int(np.floor(cx)) - pad)
            y0 = max(0, int(np.floor(cy)) - pad)
            x1 = min(width, int(np.ceil(cx)) + pad + 1)
            y1 = min(height, int(np.ceil(cy)) + pad + 1)
            tx0, ty0, tx1, ty1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
            sx0 = x0 + max(0, -tx0)
            sy0 = y0 + max(0, -ty0)
            sx1 = x1 - max(0, tx1 - width)
            sy1 = y1 - max(0, ty1 - height)
            tx0, ty0 = max(0, tx0), max(0, ty0)
            tx1, ty1 = min(width, tx1), min(height, ty1)
            if sx1 > sx0 and sy1 > sy0:
                shifted[ty0:ty1, tx0:tx1] = overlay[sy0:sy1, sx0:sx1]
        return shifted

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
        angle_override: float | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        return self._brow_overlay(state, emphasized, angle_override)

    def brow_angle(self, state: str) -> float:
        """Return the manifest-convention scalar brow angle for one state."""
        brow_state = emotion_brow_state(state)
        configured = (self.skin.brow_config or {}).get("angles") or {}
        if brow_state in configured:
            return float(configured[brow_state])
        if brow_state == "inquisitor":
            return -5.5
        if brow_state in {"conceded", "skeptical", "troubled"}:
            return 12.0
        return 0.0

    def _brow_overlay(
        self,
        state: str,
        emphasized: bool = False,
        angle_override: float | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        """Build anti-aliased brow bars locked to the optical-lens rims."""
        brow_state = emotion_brow_state(state)
        brow_angle = (
            self.brow_angle(brow_state)
            if angle_override is None
            else float(angle_override)
        )
        key = (brow_state, round(brow_angle, 2))
        if key in self._brow_cache:
            return self._brow_cache[key]
        if len(self._eye_centers) < 2:
            self._brow_cache[key] = None
            return None

        brow_config = self.skin.brow_config or {}
        ink_hex = str(
            brow_config.get("ink_color")
            or (self.skin.palette or {}).get("ink_outline")
            or "#152026"
        ).lstrip("#")
        try:
            outline = (
                int(ink_hex[0:2], 16),
                int(ink_hex[2:4], 16),
                int(ink_hex[4:6], 16),
                255,
            )
        except (TypeError, ValueError):
            outline = (21, 32, 38, 255)
        layer = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        for index, (eye_x, eye_y) in enumerate(self._eye_centers[:2]):
            configured_width = brow_config.get("width_px")
            bar_w = (
                max(24, int(round(float(configured_width))))
                if configured_width is not None
                else max(48, int(round(self._eye_radius * 1.45)))
            )
            scale = 4
            pad = 36
            patch_h = max(52, int(round(self._eye_radius * 0.70)))
            patch = Image.new(
                "RGBA",
                ((bar_w + pad * 2) * scale, (patch_h + pad * 2) * scale),
                (0, 0, 0, 0),
            )
            draw = ImageDraw.Draw(patch, "RGBA")
            left = float(pad * scale)
            right = float((pad + bar_w) * scale)
            center = float((pad + patch_h // 2) * scale)
            inner_is_right = index == 0

            # Manifest convention: negative = acute attack (\ /),
            # positive = troubled/defeated (/ \).
            inner_delta = (
                -np.tan(np.deg2rad(brow_angle))
                * bar_w
                * scale
            )
            inner_y = center + inner_delta
            outer_point = (left, center)
            inner_point = (right, inner_y)
            if not inner_is_right:
                outer_point = (right, center)
                inner_point = (left, inner_y)
            character = self.skin.character_id.lower()
            is_gemini = "gemini" in character
            is_llama = "llama" in character
            # Near brow (viewer's left) is closer to camera, so it is wider.
            # Far brow (viewer's right) stays at the 7.5px optic ink.
            if is_gemini and index == 0:
                stroke_px = 9.5
            elif is_gemini:
                stroke_px = 7.5
            elif brow_config.get("style") == "acute_mecha":
                stroke_px = float(brow_config.get("stroke_width_px") or 5.5)
            else:
                stroke_px = 5.5
            ink_width = max(1, int(round(stroke_px * scale)))
            draw.line(
                [outer_point, inner_point],
                fill=outline,
                width=ink_width,
            )
            radius = ink_width // 2
            for px, py in (outer_point, inner_point):
                draw.ellipse(
                    [px - radius, py - radius, px + radius, py + radius],
                    fill=outline,
                )
            patch = patch.resize(
                (bar_w + pad * 2, patch_h + pad * 2),
                Image.Resampling.LANCZOS,
            )
            if index < len(self._eye_bboxes):
                brow_y = self._eye_bboxes[index][1] - int(
                    round(float(brow_config.get("gap_px") or 4))
                )
            else:
                brow_y = eye_y - self._eye_radius - 4
            x_nudge = 0
            y_nudge = 0
            if is_gemini and index == 1:
                x_nudge = 4  # far optic, viewer's right
            elif is_gemini and index == 0:
                y_nudge = -2  # near optic sits higher
            elif is_llama and index == 0:
                x_nudge = -3  # viewer's left
            layer.alpha_composite(
                patch,
                (
                    eye_x - patch.width // 2 + x_nudge,
                    brow_y - (pad + patch_h // 2) + y_nudge,
                ),
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
        brow_angle_deg: float | None = None,
        rest_mouth_state: str | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Return a cached, neck-pivoted head crop with all facial sprites attached."""
        shape = (viseme or REST_VISEME).upper()[:1]
        if shape not in self._head_stack:
            shape = REST_VISEME
        state = emotion_brow_state(brow_state)
        angle = float(np.clip(round(float(angle_deg) / 0.3) * 0.3, -1.5, 1.5))
        rest_state = rest_mouth_state or emotion_rest_mouth_state(brow_state)
        cache_rest_state = rest_state if shape == REST_VISEME else ""
        resolved_brow_angle = (
            self.brow_angle(state)
            if brow_angle_deg is None
            else float(brow_angle_deg)
        )
        cache_key = (
            shape,
            int(eye_state),
            state,
            round(resolved_brow_angle, 2),
            cache_rest_state,
            angle,
        )
        cached = self._articulated_head_cache.get(cache_key)
        if cached is not None:
            self._articulated_head_cache.move_to_end(cache_key)
            return cached

        head = (
            self._rest_head(rest_state)
            if shape == REST_VISEME
            else self._head_stack[shape]
        )
        if eye_state in (1, 2):
            head = _alpha_composite_rgba(
                head,
                self._eye_overlay[eye_state],
                self._eye_bbox[eye_state],
            )
        brow = self._brow_overlay(
            state,
            brow_emphasized,
            resolved_brow_angle,
        )
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
        new_w = max(1, int(w * factor))
        new_h = max(1, int(h * factor))
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
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
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
    "REST_MOUTH_LAYER_KEYS",
    "REST_MOUTH_STATES",
    "VISEME_LAYER_KEYS",
    "emotion_brow_angles",
    "emotion_brow_state",
    "emotion_rest_mouth_state",
    "rest_mouth_layer_key",
    "viseme_layer_key",
]
