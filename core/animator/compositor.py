# -*- coding: utf-8 -*-
"""Shot-reverse-shot cinematic director.

Canvas: 1080x1920 vertical with a full-bleed cinematic portrait:

===========  ==================  ====================================
Band         Pixels (y)          Content
===========  ==================  ====================================
Top 15%      0 - 288             Clean atmospheric headroom
Middle 60%   288 - 1440          Hero avatar, medium close-up
Bottom 25%   1440 - 1920         Full-bleed torso + karaoke overlay
===========  ==================  ====================================

**Exactly one character is ever rendered.** This is not a split screen with
one side hidden: on a given frame the inactive character's rig is never
composed, never pasted, never transformed — it simply does not participate.
The camera hard-cuts on ``DialogueTurn.start_time``, switching hero
character and camera angle atomically in the same frame.

The artist cels already carry their intended three-quarter perspective.
The camera preserves it with strict uniform scaling—no synthetic yaw warp
or independent-axis stretch is applied.
"""
from __future__ import annotations

import logging
from typing import Iterator, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .puppet import (
    PuppetRig,
    emotion_brow_state,
    emotion_rest_mouth_state,
)
from .types import REST_VISEME, AnalyzedAudio, SpeakerStyle

_LOG = logging.getLogger("animator.compositor")

# -- Band geometry (fractions of canvas height) ----------------------------- #
HUD_BAND_FRAC = 0.0
HERO_BAND_FRAC = 0.75
SUBTITLE_BAND_FRAC = 0.25

# -- Hero framing ----------------------------------------------------------- #
#: Compatibility default for non-production callers that omit target width.
HERO_CONTENT_WIDTH = 870
# Kept as a compatibility export; full-bleed V2 composition uses no HUD.
HERO_ELEVATION_PX = 0
#: Legacy compatibility constant. Artist cels already carry this approximate
#: three-quarter yaw; the compositor no longer applies a second warp.
HERO_YAW_DEG = 25.0
CAMERA_NORMAL = "normal"
CAMERA_TIGHT = "tight"
CAMERA_NORMAL_ZOOM = 0.95
CAMERA_TIGHT_ZOOM = 1.35
GEMINI_LEAD_X = 420
LLAMA_LEAD_X = 660

# -- Idle physics ----------------------------------------------------------- #
_IDLE_GLOW = 0.10

class ShotReverseShotCompositor:
    """Renders one 1080x1920 RGB frame per call from :class:`AnalyzedAudio`.

    ``rigs`` and ``styles`` are keyed by the same ``character_id`` values the
    dialogue ledger's ``speaker`` field uses. The engine stays generic: it
    never learns what those ids mean, only that exactly one of them holds
    the camera at any instant.
    """

    def __init__(
        self,
        *,
        rigs: dict[str, PuppetRig],
        styles: dict[str, SpeakerStyle],
        width: int = 1080,
        height: int = 1920,
        enable_cta: bool = False,
        outro_start_s: float | None = None,
        outro_frame=None,
    ) -> None:
        if not rigs:
            raise ValueError("shot-reverse-shot needs at least one character rig")
        self.rigs = rigs
        self.styles = styles
        self.width = width
        self.height = height

        self.hud_band = (0, int(round(height * HUD_BAND_FRAC)))
        self.hero_band = (self.hud_band[1], int(round(height * (HUD_BAND_FRAC + HERO_BAND_FRAC))))
        self.subtitle_band = (self.hero_band[1], height)
        self._outro_start_s = outro_start_s if enable_cta else None
        self._outro_frame = outro_frame if enable_cta else None

        # Built once as PIL images (gradients, text, glows are expensive but
        # static) then frozen into numpy — the per-frame loop is pure
        # numpy/OpenCV and never touches PIL again.
        self._background = np.asarray(self._build_background(), dtype=np.uint8).copy()
        shared_panorama = self._load_shared_panorama()
        self._backgrounds: dict[str, np.ndarray] = {
            speaker_id: self._prepare_camera_background(
                rig,
                styles.get(speaker_id),
                shared_panorama,
            )
            for speaker_id, rig in rigs.items()
        }
        self._camera: dict[str, _HeroCamera] = {
            speaker_id: _HeroCamera(
                rig,
                styles.get(speaker_id),
                target_width=self.width,
                target_height=self.height,
                zoom=CAMERA_NORMAL_ZOOM,
            )
            for speaker_id, rig in rigs.items()
        }
        self._speaker_base: dict[str, dict[int, np.ndarray]] = {}
        for speaker_id, camera in self._camera.items():
            breathing_states: dict[int, np.ndarray] = {}
            background = self._backgrounds.get(
                speaker_id, self._background
            )
            for y_offset in range(-3, 4):
                frame = background.copy()
                _alpha_blend_paste(
                    frame,
                    camera.static_body,
                    camera.offset_x,
                    camera.offset_y + y_offset,
                )
                breathing_states[y_offset] = frame
            self._speaker_base[speaker_id] = breathing_states

    # -- Static layers --------------------------------------------------- #
    def _load_shared_panorama(self) -> Image.Image | None:
        """Load/generate one V2 arena shared by both reverse-angle cameras."""
        if not any(character_id.endswith("_v2") for character_id in self.rigs):
            return None
        try:
            from .asset_generator import ensure_shared_panorama  # noqa: PLC0415

            root = next(iter(self.rigs.values())).skin.root.parent
            path = ensure_shared_panorama(puppets_dir=root)
            with Image.open(path) as panorama:
                return panorama.convert("RGB")
        except Exception as exc:  # noqa: BLE001 — skin bg remains a valid fallback
            _LOG.warning("shared panorama unavailable (%s); using per-skin background", exc)
            return None

    def _prepare_camera_background(
        self,
        rig: PuppetRig,
        style: SpeakerStyle | None,
        shared_panorama: Image.Image | None,
    ) -> np.ndarray:
        """Prepare a camera-side crop of the shared room or skin ``bg.png``.

        Camera A takes the panorama's left half and receives cyan atmosphere;
        Camera B takes the right half and receives amber atmosphere. Both
        retain exactly the same horizon and floor geometry.
        """
        using_shared_panorama = shared_panorama is not None
        if using_shared_panorama:
            panorama = shared_panorama.resize(
                (self.width * 2, self.height),
                Image.Resampling.LANCZOS,
            )
            facing = (style.facing if style else "right").lower()
            left = 0 if facing == "right" else self.width
            bg = panorama.crop((left, 0, left + self.width, self.height))
        else:
            bg = rig.background.convert("RGB").resize(
                (self.width, self.height),
                Image.Resampling.LANCZOS,
            )
        prepared = np.asarray(bg, dtype=np.uint8).copy()
        if using_shared_panorama:
            prepared = np.clip(
                prepared.astype(np.float32) * 0.80,
                0,
                255,
            ).astype(np.uint8)
        return prepared

    def _build_background(self) -> Image.Image:
        """Dark cyber-gradient arena with a faint grid and vignette."""
        w, h = self.width, self.height
        top = np.array([12, 15, 26], dtype=np.float32)
        bottom = np.array([2, 3, 7], dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
        gradient = top.reshape(1, 1, 3) * (1 - ramp) + bottom.reshape(1, 1, 3) * ramp
        img = Image.fromarray(np.repeat(gradient, w, axis=1).astype(np.uint8), mode="RGB")

        draw = ImageDraw.Draw(img, "RGBA")
        grid = (36, 54, 82, 26)
        step = 90
        for x in range(0, w, step):
            draw.line([(x, 0), (x, h)], fill=grid, width=1)
        for y in range(0, h, step):
            draw.line([(0, y), (w, y)], fill=grid, width=1)

        # Floor plane under the hero, stopping short of the subtitle zone so
        # nothing but karaoke text ever appears in the bottom quarter.
        floor_top = int(h * 0.52)
        floor_bottom = self.subtitle_band[0]
        vanish_x = w // 2
        for i in range(1, 7):
            frac = i / 7.0
            y = floor_top + int((floor_bottom - floor_top) * frac)
            spread = int(w * 0.70 * frac)
            draw.line([(vanish_x - spread, y), (vanish_x + spread, y)], fill=(60, 88, 130, 34), width=1)

        vignette = Image.new("L", (w, h), 0)
        ImageDraw.Draw(vignette).ellipse([-w * 0.30, -h * 0.12, w * 1.30, h * 1.02], fill=255)
        vignette = vignette.filter(ImageFilter.GaussianBlur(120))
        return Image.composite(img, Image.new("RGB", (w, h), (0, 0, 0)), vignette)

    def _tight_view(self, frame: np.ndarray, anchor_x: int) -> np.ndarray:
        """Apply the discrete 1.35x crop centered on the docked face."""
        crop_w = int(round(self.width / CAMERA_TIGHT_ZOOM))
        crop_h = int(round(self.height / CAMERA_TIGHT_ZOOM))
        left = int(np.clip(anchor_x - crop_w // 2, 0, self.width - crop_w))
        focal_y = int(round(self.height * 0.36))
        top = int(np.clip(focal_y - crop_h * 0.34, 0, self.height - crop_h))
        crop = frame[top : top + crop_h, left : left + crop_w]
        return cv2.resize(
            crop,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )

    # -- Frame loop ------------------------------------------------------- #
    def iter_frames(self, analyzed: AnalyzedAudio) -> Iterator[np.ndarray]:
        """Yield RGB24 frames (H, W, 3) uint8, ready for the ffmpeg pipe."""
        camera_speakers: Sequence[str | None] = analyzed.active_speaker
        speech_speakers: Sequence[str | None] = (
            analyzed.speaking_speaker or analyzed.active_speaker
        )
        emphasis_thresholds: dict[str, float] = {}
        brow_thresholds: dict[str, float] = {}
        for speaker_id in self.rigs:
            levels = [
                float(level)
                for level, speaker in zip(analyzed.rms, speech_speakers)
                if speaker == speaker_id and float(level) > 0.0
            ]
            emphasis_thresholds[speaker_id] = (
                float(np.percentile(levels, 70)) if levels else 1.0
            )
            brow_thresholds[speaker_id] = (
                float(np.percentile(levels, 80)) if levels else 1.0
            )
        current = next((s for s in camera_speakers if s in self.rigs), None) or next(iter(self.rigs))
        for index in range(analyzed.n_frames):
            t = analyzed.frame_time(index)
            if (
                self._outro_start_s is not None
                and self._outro_frame is not None
                and t >= self._outro_start_s
            ):
                yield self._outro_frame(t - self._outro_start_s)
                continue
            # Eyelid alpha is stencil-clipped to each lens circle on the
            # puppet canvas before this crop, so a tight or wide frame cannot
            # reveal a lid pixel outside the optic.
            camera_speaker = (
                camera_speakers[index] if index < len(camera_speakers) else None
            )
            speaking = (
                speech_speakers[index] if index < len(speech_speakers) else None
            )
            if camera_speaker in self.rigs:
                # Hard cut: hero and camera angle switch together
                # on this exact frame — no cross-fade, no interpolation.
                current = camera_speaker
            is_speaking = speaking == current

            active_rms = float(_sequence_at(analyzed.rms, index, 0.0))
            emotion = _sequence_at(analyzed.emotion, index, "neutral")
            camera_mode = dramatic_camera_mode(
                camera_tight=bool(
                    _sequence_at(analyzed.camera_tight, index, False)
                )
            )
            camera = self._camera[current]
            breathing_y = camera.breathing_offset(t)
            frame = self._speaker_base[current][breathing_y].copy()

            # Background + breathing body are pre-baked. One cached head crop
            # carries the viseme, eyelid and Ghibli brow sprites together so
            # neck-pivot movement never detaches facial features.
            camera.render_dirty(
                frame,
                t=t,
                viseme=_sequence_at(analyzed.viseme.get(current), index, REST_VISEME) if is_speaking else REST_VISEME,
                eye_state=_sequence_at(analyzed.eye_state.get(current), index, 0),
                rms=active_rms,
                emphasis_threshold=emphasis_thresholds.get(current, 1.0),
                brow_emphasis_threshold=brow_thresholds.get(current, 1.0),
                is_speaking=is_speaking,
                emotion=emotion,
                body_offset_y=breathing_y,
            )
            if camera_mode == CAMERA_TIGHT:
                frame = self._tight_view(frame, camera.lead_anchor_x)

            yield frame


class _HeroCamera:
    """One character's native-aspect close-up camera and idle physics."""

    def __init__(
        self,
        rig: PuppetRig,
        style: SpeakerStyle | None,
        *,
        target_width: int = HERO_CONTENT_WIDTH,
        target_height: int = 1920,
        zoom: float = 1.0,
    ) -> None:
        self.rig = rig
        self.facing = (style.facing if style else "right").lower()
        self._head_angle = 0.0
        self._head_target = 0.0
        self._emphasis_active = False
        self._emphasis_direction = -1.0 if self.facing == "left" else 1.0
        self._last_emphasis_t = -10.0
        self._emotion = "neutral"
        self._previous_emotion = "neutral"
        self._emotion_transition_frame = 3
        self._overlay_cache: dict[
            tuple,
            tuple[np.ndarray, np.ndarray, int, int],
        ] = {}
        self.last_brow_state = "neutral"

        # Head and body were authored on one matching vertical canvas. Keep
        # that entire coordinate space intact: no top crop, no replicated
        # bottom rows, and no independent layer placement.
        crop_w, crop_h = rig.canvas_size
        self.crop = (0, 0, crop_w, crop_h)

        # Exact-size art is a true passthrough. Other artist resolutions are
        # uniformly contained once as a complete canvas.
        scale = (
            min(target_width / float(crop_w), target_height / float(crop_h))
            * float(np.clip(zoom, 0.80, 1.60))
        )
        self.out_w = int(round(crop_w * scale))
        self.out_h = int(round(crop_h * scale))
        scale_x = self.out_w / float(crop_w)
        scale_y = self.out_h / float(crop_h)
        self.pixel_aspect_error = abs(scale_x - scale_y) / scale_x
        if self.pixel_aspect_error > 0.001:
            raise ValueError(
                f"uniform camera scaling violated: x={scale_x:.6f}, y={scale_y:.6f}"
            )
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.lead_anchor_x = lead_anchor_x(self.facing, target_width)
        self.offset_x = int(round(self.lead_anchor_x - (self.out_w / 2.0)))
        self.offset_y = target_height - self.out_h
        self.static_body = cv2.resize(
            self.rig.body_rgba,
            (self.out_w, self.out_h),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def breathing_offset(t: float) -> int:
        return int(np.clip(np.rint(np.sin(float(t) * 2.0) * 2.5), -3, 3))

    def _scaled_overlay(
        self,
        key: tuple,
        overlay: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        cached = self._overlay_cache.get(key)
        if cached is not None:
            return cached
        x0, y0, x1, y1 = bbox
        out_x0 = int(round(x0 * self.scale_x))
        out_y0 = int(round(y0 * self.scale_y))
        out_x1 = int(round(x1 * self.scale_x))
        out_y1 = int(round(y1 * self.scale_y))
        resized = cv2.resize(
            overlay,
            (max(1, out_x1 - out_x0), max(1, out_y1 - out_y0)),
            interpolation=cv2.INTER_AREA,
        )
        alpha = resized[..., 3]
        alpha_3ch = cv2.merge([alpha, alpha, alpha])
        premultiplied = cv2.multiply(
            resized[..., :3],
            alpha_3ch,
            scale=1.0 / 255.0,
        )
        result = (
            premultiplied,
            cv2.bitwise_not(alpha_3ch),
            self.offset_x + out_x0,
            self.offset_y + out_y0,
        )
        self._overlay_cache[key] = result
        return result

    def render_dirty(
        self,
        frame: np.ndarray,
        *,
        t: float,
        viseme: str,
        eye_state: int,
        rms: float,
        emphasis_threshold: float,
        brow_emphasis_threshold: float,
        is_speaking: bool,
        emotion: str = "neutral",
        body_offset_y: int = 0,
    ) -> None:
        """Blit one articulated head crop onto a pre-baked breathing body."""
        self._update_emotion(emotion)
        brow_state = emotion_brow_state(self._emotion)
        brow_emphasized = bool(is_speaking and rms > brow_emphasis_threshold)
        self.last_brow_state = brow_state
        head_angle = self._update_head_angle(
            t=t,
            rms=rms,
            emphasis_threshold=emphasis_threshold,
            is_speaking=is_speaking,
        )
        head, bbox = self.rig.articulated_head_overlay(
            viseme=viseme,
            eye_state=eye_state,
            brow_state=brow_state,
            brow_emphasized=brow_emphasized,
            rest_mouth_state=emotion_rest_mouth_state(self._emotion),
            angle_deg=head_angle,
        )
        angle_key = round(float(head_angle) / 0.3) * 0.3
        rest_state = emotion_rest_mouth_state(self._emotion)
        premultiplied, inv_alpha, x, y = self._scaled_overlay(
            (
                "head",
                (viseme or REST_VISEME).upper()[:1],
                eye_state,
                brow_state,
                rest_state
                if (viseme or REST_VISEME).upper()[:1] == REST_VISEME
                else "",
                angle_key,
            ),
            head,
            bbox,
        )
        _alpha_blend_paste_precomputed(
            frame,
            premultiplied,
            inv_alpha,
            x,
            y + int(body_offset_y),
        )

    def render(
        self,
        *,
        t: float,
        viseme: str,
        eye_state: int,
        rms: float,
        emphasis_threshold: float,
        is_speaking: bool,
        emotion: str = "neutral",
    ) -> np.ndarray:
        """Compose and uniformly scale this character for one frame."""
        glow = _IDLE_GLOW + (max(0.0, rms) - _IDLE_GLOW) * 0.9 if is_speaking else _IDLE_GLOW * 0.7
        head_angle = self._update_head_angle(
            t=t,
            rms=rms,
            emphasis_threshold=emphasis_threshold,
            is_speaking=is_speaking,
        )
        previous_emotion, emotion_mix = self._update_emotion(emotion)
        sprite = self.rig.compose(
            viseme=viseme,
            eye_state=eye_state,
            y_offset=0.0,
            brightness=1.0,
            glow_intensity=float(np.clip(glow, 0.0, 1.0)),
            head_angle=head_angle,
            emotion=self._emotion,
            previous_emotion=previous_emotion,
            emotion_mix=emotion_mix,
        )
        x0, y0, x1, y1 = self.crop
        return cv2.resize(
            sprite[y0:y1, x0:x1],
            (self.out_w, self.out_h),
            interpolation=cv2.INTER_AREA,
        )

    def _update_emotion(self, emotion: str) -> tuple[str, float]:
        """Ease brow posture over exactly three rendered camera frames."""
        target = (emotion or "neutral").lower()
        if target != self._emotion:
            self._previous_emotion = self._emotion
            self._emotion = target
            self._emotion_transition_frame = 0
        if self._emotion_transition_frame < 3:
            self._emotion_transition_frame += 1
        return self._previous_emotion, self._emotion_transition_frame / 3.0

    def _update_head_angle(
        self,
        *,
        t: float,
        rms: float,
        emphasis_threshold: float,
        is_speaking: bool,
    ) -> float:
        """Ease toward brief emphasis impulses; never run a periodic oscillator."""
        if not is_speaking:
            self._head_target = 0.0
            self._emphasis_active = False
            emphasized = False
        else:
            emphasized = rms > emphasis_threshold
            if emphasized:
                if not self._emphasis_active and (t - self._last_emphasis_t) >= 0.35:
                    self._emphasis_direction *= -1.0
                    self._last_emphasis_t = t
                self._head_target = emphasis_head_target(
                    rms,
                    emphasis_threshold,
                    direction=self._emphasis_direction,
                )
            else:
                self._head_target = 0.0
            self._emphasis_active = emphasized

        easing = 0.32 if emphasized else 0.18
        self._head_angle += (self._head_target - self._head_angle) * easing
        self._head_angle = float(np.clip(self._head_angle, -1.2, 1.2))
        if not emphasized and abs(self._head_angle) < 0.01:
            self._head_angle = 0.0
        return self._head_angle


def lead_anchor_x(facing: str, width: int = 1080) -> int:
    """Dock a right-facing hero left, and a left-facing hero right."""
    if (facing or "").lower() == "right":
        return GEMINI_LEAD_X
    if (facing or "").lower() == "left":
        return LLAMA_LEAD_X
    return width // 2


def dramatic_camera_mode(*, camera_tight: bool) -> str:
    """Choose one of two discrete precomputed viewports."""
    return CAMERA_TIGHT if camera_tight else CAMERA_NORMAL


def emphasis_head_target(rms: float, threshold: float, *, direction: float) -> float:
    """One bounded emphasis impulse; ordinary speech produces no rotation."""
    if rms <= threshold:
        return 0.0
    span = max(1e-6, 1.0 - threshold)
    strength = float(np.clip((rms - threshold) / span, 0.0, 1.0))
    return float(np.clip(direction, -1.0, 1.0)) * (0.45 + 0.75 * strength)


def _sequence_at(seq, index: int, default):
    if not seq or index >= len(seq):
        return default
    return seq[index]


def _alpha_blend_paste(dest: np.ndarray, src_rgba: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite ``src_rgba`` onto ``dest`` (RGB) at offset ``(x, y)``,
    clipped to ``dest``'s bounds. cv2-accelerated (SIMD uint8 ops) and only
    ever touches the region it actually overlaps."""
    dh, dw = dest.shape[:2]
    sh, sw = src_rgba.shape[:2]
    dst_x0, dst_y0 = max(0, x), max(0, y)
    dst_x1, dst_y1 = min(dw, x + sw), min(dh, y + sh)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x0, src_y0 = dst_x0 - x, dst_y0 - y
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

    src_region = np.ascontiguousarray(src_rgba[src_y0:src_y1, src_x0:src_x1])
    dst_region = dest[dst_y0:dst_y1, dst_x0:dst_x1]

    alpha = src_region[..., 3]
    inv_alpha = cv2.bitwise_not(alpha)
    # cv2.merge to broadcast a single channel to 3 is ~2-3x faster than
    # np.repeat for this shape/size — measured, not assumed.
    alpha_3ch = cv2.merge([alpha, alpha, alpha])
    inv_alpha_3ch = cv2.merge([inv_alpha, inv_alpha, inv_alpha])

    fg = cv2.multiply(src_region[..., :3], alpha_3ch, scale=1.0 / 255.0)
    bg = cv2.multiply(dst_region, inv_alpha_3ch, scale=1.0 / 255.0)
    dest[dst_y0:dst_y1, dst_x0:dst_x1] = cv2.add(fg, bg)


def _alpha_blend_paste_precomputed(
    dest: np.ndarray,
    premultiplied_rgb: np.ndarray,
    inv_alpha_3ch: np.ndarray,
    x: int,
    y: int,
) -> None:
    """Blend a cached premultiplied sprite with one multiply and one add."""
    dh, dw = dest.shape[:2]
    sh, sw = premultiplied_rgb.shape[:2]
    dst_x0, dst_y0 = max(0, x), max(0, y)
    dst_x1, dst_y1 = min(dw, x + sw), min(dh, y + sh)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x0, src_y0 = dst_x0 - x, dst_y0 - y
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    dst_region = dest[dst_y0:dst_y1, dst_x0:dst_x1]
    bg = cv2.multiply(
        dst_region,
        inv_alpha_3ch[src_y0:src_y1, src_x0:src_x1],
        scale=1.0 / 255.0,
    )
    dest[dst_y0:dst_y1, dst_x0:dst_x1] = cv2.add(
        premultiplied_rgb[src_y0:src_y1, src_x0:src_x1],
        bg,
    )


__all__ = [
    "CAMERA_NORMAL",
    "CAMERA_NORMAL_ZOOM",
    "CAMERA_TIGHT",
    "CAMERA_TIGHT_ZOOM",
    "GEMINI_LEAD_X",
    "LLAMA_LEAD_X",
    "lead_anchor_x",
    "HERO_CONTENT_WIDTH",
    "HERO_ELEVATION_PX",
    "HERO_YAW_DEG",
    "HUD_BAND_FRAC",
    "ShotReverseShotCompositor",
    "dramatic_camera_mode",
    "emphasis_head_target",
]
