# -*- coding: utf-8 -*-
"""
MoviePy assembler — Ken Burns + duotone grading + captions + channel logo.

Reuses slow-zoom framing similar to compile_dynamic_reel without mutating it.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image as PILImage
from utils.pipeline_paths import moviepy_temp_audio_dir
from PIL import ImageEnhance

from core.economic_reel_lofi import config as lofi_cfg
from core.economic_reel_lofi.caption_style_lofi import (
    render_lofi_caption_layer,
    render_lofi_caption_layer_word_fade,
    render_lofi_watermark_layer,
)

_LOG = logging.getLogger(__name__)
_CAPTION_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _visual_only(clip: Any):
    """Strip every audio handle from a visual clip. Echo-safe."""
    silent = clip.without_audio() if hasattr(clip, "without_audio") else clip
    try:
        silent.audio = None
    except Exception:  # noqa: BLE001
        pass
    return silent


def _purge_moviepy_temp_audio() -> list[Path]:
    """Drop leftover MoviePy mux files so a prior encode cannot be remixed."""
    root = Path(moviepy_temp_audio_dir())
    if not root.is_dir():
        return []
    removed: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        low = path.name.lower()
        if "temp_mpy" not in low and path.suffix.lower() not in {
            ".mp3",
            ".m4a",
            ".aac",
            ".wav",
        }:
            continue
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue
    if removed:
        print(f"[LOFI audio] purged moviepy temp audio files={len(removed)}")
    return removed


def _caption_words(text: str) -> list[str]:
    return _CAPTION_WORD_RE.findall(text or "")


def caption_cycle_windows(
    chunks: Sequence[str],
    timings: Sequence[tuple[str, float, float]] | None,
    scene_dur: float,
) -> list[tuple[str, float, float, list[tuple[str, float, float]]]]:
    """Map on-screen caption chunks onto a held still's VO timeline.

    Returns (chunk_text, start_s, end_s, word_timings) covering [0, scene_dur].
    One still stays on screen; captions replace each other in order.
    """
    cleaned = [str(c).strip() for c in (chunks or []) if str(c).strip()]
    if not cleaned:
        return []
    dur = max(0.12, float(scene_dur or 0.0))
    timed = [(str(w), float(s), float(e)) for w, s, e in (timings or [])]
    windows: list[tuple[str, float, float, list[tuple[str, float, float]]]] = []
    cursor = 0
    if timed:
        for i, chunk in enumerate(cleaned):
            n = max(1, len(_caption_words(chunk)))
            slice_t = timed[cursor : cursor + n]
            if not slice_t and timed:
                slice_t = [timed[-1]]
            start = float(slice_t[0][1]) if slice_t else 0.0
            end = float(slice_t[-1][2]) if slice_t else start
            cursor += n
            windows.append((chunk, start, end, slice_t))
    else:
        weights = [max(1, len(_caption_words(c))) for c in cleaned]
        total_w = float(sum(weights)) or 1.0
        t0 = 0.0
        for chunk, w in zip(cleaned, weights):
            span = dur * (w / total_w)
            windows.append((chunk, t0, t0 + span, []))
            t0 += span
    if not windows:
        return []
    windows[0] = (windows[0][0], 0.0, windows[0][2], windows[0][3])
    last = list(windows[-1])
    last[2] = max(float(last[2]), dur)
    windows[-1] = (last[0], last[1], last[2], last[3])
    for i in range(len(windows) - 1):
        text, start, end, loc = windows[i]
        nxt = windows[i + 1][1]
        if end < nxt:
            windows[i] = (text, start, nxt, loc)
    print(
        f"[LOFI caption-cycle] chunks={len(windows)} "
        + " ".join(f"{i + 1}:{w[1]:.2f}-{w[2]:.2f}s" for i, w in enumerate(windows))
    )
    return windows


def active_caption_cycle(
    windows: Sequence[tuple[str, float, float, list[tuple[str, float, float]]]],
    t: float,
) -> tuple[str, Sequence[tuple[str, float, float]] | None, float | None]:
    """Pick the caption chunk visible at scene-local time t."""
    if not windows:
        return "", None, None
    chosen = windows[0]
    for row in windows:
        if t + 1e-6 >= row[1]:
            chosen = row
        else:
            break
    text, start, end, loc = chosen
    if t < start:
        return "", None, max(0.12, end - start)
    return text, loc or None, max(0.12, end - start)


def apply_duotone(
    rgb: np.ndarray,
    *,
    shadow: tuple[int, int, int] | None = None,
    highlight: tuple[int, int, int] | None = None,
    bands: int | None = None,
) -> np.ndarray:
    """
    Poster-style luminance remap (not a smooth gradient wash).

    1. Convert to grayscale via Rec.601 luminance (discards source hue).
    2. Quantize luminance into a few discrete bands (default 4) so skies /
       fills read as solid color blocks with hard-ish tone edges.
    3. Map each band onto the shadow→highlight palette (no continuous lerp).
    """
    img = rgb.astype(np.float32)
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    t = np.clip(lum / 255.0, 0.0, 1.0)
    # Mild contrast before quantize so midtones don't collapse into one band
    t = np.power(t, 0.88)

    n = int(bands if bands is not None else lofi_cfg.DUOTONE_TONAL_BANDS)
    n = max(2, min(8, n))
    # Snap to n discrete levels in [0, 1] (e.g. 0, 1/3, 2/3, 1 for n=4)
    levels = np.round(t * (n - 1)) / float(n - 1)
    levels = np.clip(levels, 0.0, 1.0)

    sh = np.array(shadow if shadow is not None else lofi_cfg.DUOTONE_SHADOW, dtype=np.float32)
    hi = np.array(
        highlight if highlight is not None else lofi_cfg.DUOTONE_HIGHLIGHT,
        dtype=np.float32,
    )
    out = sh + (hi - sh) * levels[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_vignette(rgb: np.ndarray, strength: float | None = None) -> np.ndarray:
    """Dark vintage vignette — corner darken, transparent center."""
    if strength is None:
        if not bool(getattr(lofi_cfg, "ENABLE_VIGNETTE", True)):
            return rgb
        strength = float(getattr(lofi_cfg, "VIGNETTE_STRENGTH", 0.18))
    strength = float(strength)
    if strength <= 0.001:
        return rgb
    strength = max(0.0, min(0.45, strength))
    h, w = rgb.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # Normalized radial distance; corners ≈ 1.0+
    dist = np.sqrt(((x - cx) / max(cx, 1.0)) ** 2 + ((y - cy) / max(cy, 1.0)) ** 2)
    # Soft falloff: center untouched, corners darkened by ``strength``
    edge = np.clip((dist - 0.35) / 0.95, 0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)  # smoothstep
    mask = (1.0 - strength * edge)[..., None]
    return np.clip(rgb.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def _smootherstep01(u: float) -> float:
    """Ken Perlin smootherstep on [0,1] — soft acceleration / deceleration."""
    u = max(0.0, min(1.0, float(u)))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def light_pulse_brightness_factor(t: float, seed: int = 3) -> float:
    """
    Global sine brightness factor, independent of Ken Burns zoom.

    brightness_factor = 1 + AMP * sin(2π t / PERIOD + phase)
    """
    if not bool(getattr(lofi_cfg, "ENABLE_LIGHT_BREATH", True)):
        return 1.0
    amp = float(getattr(lofi_cfg, "LIGHT_BREATH_AMP", 0.065))
    period = float(getattr(lofi_cfg, "LIGHT_BREATH_PERIOD_S", 6.5))
    amp = max(0.0, min(0.12, amp))
    period = max(5.5, min(8.5, period))
    if amp < 0.001:
        return 1.0
    phase = ((int(seed) % 97) * 0.13) % (2.0 * math.pi)
    return 1.0 + amp * math.sin(2.0 * math.pi * (t / period) + phase)


def apply_animated_light_pulse(
    rgb: np.ndarray,
    *,
    t: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Post-process breathing light — independent of baked artwork lighting.

    Uses ``light_pulse_brightness_factor`` then biases the delta toward
    bright sources (streetlamp / window / headlights).
    """
    factor = light_pulse_brightness_factor(t, seed=seed)
    pulse = factor - 1.0
    if abs(pulse) < 0.0005:
        return rgb
    amp = float(getattr(lofi_cfg, "LIGHT_BREATH_AMP", 0.065))
    source_bias = float(getattr(lofi_cfg, "LIGHT_BREATH_SOURCE_BIAS", 0.70))
    bloom = float(getattr(lofi_cfg, "LIGHT_BREATH_BLOOM", 0.040))

    img = rgb.astype(np.float32)
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    source = np.clip((lum - 145.0) / 85.0, 0.0, 1.0)
    source = (
        source
        + np.roll(source, 3, 0)
        + np.roll(source, -3, 0)
        + np.roll(source, 3, 1)
        + np.roll(source, -3, 1)
    ) / 5.0
    source = (
        source
        + np.roll(source, 6, 0)
        + np.roll(source, -6, 0)
        + np.roll(source, 6, 1)
        + np.roll(source, -6, 1)
    ) / 5.0

    global_w = 1.0 - source_bias
    weight = (global_w + source_bias * source)[..., None]
    out = img * (1.0 + pulse * weight)

    if pulse > 0.0 and bloom > 0.0:
        warm = np.array([1.12, 1.04, 0.92], dtype=np.float32)
        bloom_amt = (pulse / max(amp, 1e-6)) * bloom * source[..., None]
        out = out + out * bloom_amt * (warm - 1.0)

    return np.clip(out, 0, 255).astype(np.uint8)


def clamp_baked_highlights(rgb: np.ndarray) -> np.ndarray:
    """
    Soft-knee highlight clamp on the still, before Ken Burns.

    Pulls blown whites / chest-collar bloom back without a LUT or duotone.
    """
    knee = float(getattr(lofi_cfg, "HIGHLIGHT_SOFT_KNEE", 208.0))
    compress = float(getattr(lofi_cfg, "HIGHLIGHT_COMPRESS", 0.48))
    ceiling = float(getattr(lofi_cfg, "HIGHLIGHT_CLAMP_CEILING", 236.0))
    img = rgb.astype(np.float32)
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    over = np.maximum(lum - knee, 0.0)
    if float(np.max(over)) < 0.5:
        return np.clip(img, 0, ceiling).astype(np.uint8)
    pull = 1.0 - compress * (over / (over + 40.0))
    out = img * pull[..., None]
    return np.clip(out, 0, ceiling).astype(np.uint8)


def apply_film_grain(
    rgb: np.ndarray,
    *,
    seed: int = 0,
    t: float = 0.0,
) -> np.ndarray:
    """Animated monochrome film grain. Disabled when using the reference overlay."""
    if not bool(getattr(lofi_cfg, "ENABLE_PROCEDURAL_GRAIN", False)):
        return rgb
    h, w = rgb.shape[:2]
    opacity = float(getattr(lofi_cfg, "GRAIN_OPACITY", 0.13))
    opacity = max(0.0, min(0.20, opacity))
    if opacity < 0.001:
        # Legacy additive fallback
        grain_alpha = lofi_cfg.GRAIN_INTENSITY * (0.85 + 0.15 * math.sin(t * 11.0))
        noise = np.random.RandomState(int(seed)).randint(0, 64, (h, w, 3), dtype=np.uint8)
        return np.clip(
            rgb.astype(np.int16) + (noise * grain_alpha).astype(np.int16),
            0,
            255,
        ).astype(np.uint8)

    # Uncached RNG keyed by frame time — do not reuse arrays across frames
    rng = np.random.default_rng(
        (int(seed) ^ (int(t * 1000.0) * 2654435761)) & 0x7FFFFFFF
    )
    half = bool(getattr(lofi_cfg, "GRAIN_HALF_RES", True))
    nh, nw = (max(1, h // 2), max(1, w // 2)) if half else (h, w)
    mono = rng.normal(loc=0.0, scale=1.0, size=(nh, nw)).astype(np.float32)
    # Mild blur → finer optical grain (not blocky)
    mono = (
        mono
        + np.roll(mono, 1, 0)
        + np.roll(mono, -1, 0)
        + np.roll(mono, 1, 1)
        + np.roll(mono, -1, 1)
    ) / 5.0
    if half:
        mono = np.array(
            PILImage.fromarray(
                np.clip(mono * 32.0 + 128.0, 0, 255).astype(np.uint8), mode="L"
            ).resize((w, h), PILImage.BILINEAR),
            dtype=np.float32,
        )
        mono = (mono - 128.0) / 32.0
    grain = np.clip(0.5 + mono * 0.12, 0.0, 1.0)[..., None]

    base = rgb.astype(np.float32) / 255.0
    # Overlay blend
    overlay = np.where(
        base < 0.5,
        2.0 * base * grain,
        1.0 - 2.0 * (1.0 - base) * (1.0 - grain),
    )
    # Shadow density: more grain in darks
    lum = 0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]
    dark_boost = (1.0 - lum) * 0.55 + 0.45
    local_op = opacity * dark_boost[..., None]
    out = base * (1.0 - local_op) + overlay * local_op
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def apply_caption_film_multiply(
    rgb: np.ndarray,
    *,
    seed: int = 0,
    t: float = 0.0,
) -> np.ndarray:
    """
    Film-stock multiply texture behind captions (depth without a solid box).

    Static plate per scene (t ignored for noise) — avoids TV-static flicker.
    Soft vertical band around caption center; text is composited AFTER this.
    """
    del t  # intentionally unused — plate must not animate per frame
    op = float(getattr(lofi_cfg, "CAPTION_FILM_MULTIPLY_OPACITY", 0.20))
    op = max(0.0, min(0.35, op))
    if op < 0.001:
        return rgb
    h, w = rgb.shape[:2]
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    nh, nw = max(1, h // 2), max(1, w // 2)
    mono = rng.normal(0.0, 1.0, size=(nh, nw)).astype(np.float32)
    mono = (
        mono
        + np.roll(mono, 2, 0)
        + np.roll(mono, -2, 0)
        + np.roll(mono, 2, 1)
        + np.roll(mono, -2, 1)
    ) / 5.0
    plate = np.clip(0.58 + mono * 0.09, 0.32, 0.88)
    plate = np.array(
        PILImage.fromarray(
            np.clip(plate * 255.0, 0, 255).astype(np.uint8), mode="L"
        ).resize((w, h), PILImage.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    plate = plate[..., None]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    band = np.exp(-((y - 0.46) ** 2) / (2.0 * 0.12 ** 2))
    weight = (band * op)[..., None]
    base = rgb.astype(np.float32) / 255.0
    multiplied = base * plate
    out = base * (1.0 - weight) + multiplied * weight
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


_DUST_TILE_CACHE: dict[int, np.ndarray] = {}


def _build_dust_tile(seed: int = 11) -> np.ndarray:
    """Soft particle tile (RGBA float 0–1) drifted upward each frame."""
    if seed in _DUST_TILE_CACHE:
        return _DUST_TILE_CACHE[seed]
    from PIL import ImageDraw as _ImageDraw

    w, h = lofi_cfg.REEL_WIDTH, lofi_cfg.REEL_HEIGHT
    tile = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = _ImageDraw.Draw(tile)
    rng = np.random.default_rng(seed)
    n = int(getattr(lofi_cfg, "DUST_PARTICLE_COUNT", 48))
    base_a = float(getattr(lofi_cfg, "DUST_PARTICLE_OPACITY", 0.12))
    for _ in range(max(8, n)):
        px = int(rng.integers(8, w - 8))
        py = int(rng.integers(8, h - 8))
        pr = int(rng.integers(1, 3))
        pa = int(255 * base_a * float(rng.uniform(0.35, 1.0)))
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(255, 255, 255, pa))
    arr = np.array(tile, dtype=np.float32) / 255.0
    _DUST_TILE_CACHE[seed] = arr
    return arr


def apply_dust_particles(rgb: np.ndarray, *, t: float, seed: int = 11) -> np.ndarray:
    if not bool(getattr(lofi_cfg, "ENABLE_DUST_PARTICLES", False)):
        return rgb
    tile = _build_dust_tile(seed=seed)
    h, w = rgb.shape[:2]
    # Drift upward ~18 px/s
    shift = int((t * 18.0) % h)
    shifted = np.roll(tile, -shift, axis=0)
    if shifted.shape[0] != h or shifted.shape[1] != w:
        shifted = np.array(
            PILImage.fromarray((shifted * 255).astype(np.uint8)).resize(
                (w, h), PILImage.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    base = rgb.astype(np.float32) / 255.0
    a = shifted[..., 3:4]
    rgb_p = shifted[..., :3]
    out = base * (1.0 - a) + np.clip(base + rgb_p * 0.55, 0, 1) * a
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def ensure_dust_overlay_asset(engine_root: Path) -> Path:
    """
    Reference film-grain / dust overlay (black bg, light particles).
    Prefer the configured mp4; never procedurally generate when it exists.
    """
    rel = str(
        getattr(
            lofi_cfg,
            "DUST_OVERLAY_REL",
            "channels_config/wonder_feed/overlays/overlay film grain.mp4",
        )
    )
    path = Path(rel) if Path(rel).is_absolute() else Path(engine_root) / rel
    if path.is_file() and path.stat().st_size > 10_000:
        return path
    overlay_dir = Path(engine_root) / "channels_config" / "wonder_feed" / "overlays"
    for cand in sorted(overlay_dir.glob("overlay*grain*.mp4")):
        if cand.is_file() and cand.stat().st_size > 10_000:
            print(f"[LOFI assemble] grain_overlay fallback path={cand}")
            return cand
    npz_path = path.with_suffix(".npz")
    if bool(getattr(lofi_cfg, "DUST_OVERLAY_PREFER_NPZ", False)) and npz_path.is_file():
        return npz_path
    # Last-resort generate only if the reference file is missing
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import ImageDraw as _ImageDraw

    w, h = lofi_cfg.REEL_WIDTH, lofi_cfg.REEL_HEIGHT
    fps = int(lofi_cfg.REEL_FPS)
    duration_s = 8.0
    n_frames = int(duration_s * fps)
    rng = np.random.default_rng(20260817)

    scratch = PILImage.new("RGB", (w, h), (0, 0, 0))
    sd = _ImageDraw.Draw(scratch)
    for _ in range(40):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        x1 = int(np.clip(x0 + rng.integers(-80, 80), 0, w - 1))
        y1 = int(np.clip(y0 + rng.integers(-400, 400), 0, h - 1))
        val = int(rng.integers(40, 120))
        sd.line([(x0, y0), (x1, y1)], fill=(val, val, val), width=1)
    scratch_arr = np.array(scratch, dtype=np.float32)

    n_p = 90
    px = rng.uniform(0, w, size=n_p)
    py = rng.uniform(0, h, size=n_p)
    pr = rng.uniform(0.8, 2.4, size=n_p)
    pspd = rng.uniform(8.0, 28.0, size=n_p)
    pbright = rng.uniform(90, 220, size=n_p)

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        t = fi / float(fps)
        canvas = scratch_arr.copy()
        for i in range(n_p):
            x = int(px[i]) % w
            y = int((py[i] - pspd[i] * t) % h)
            r = max(1, int(pr[i]))
            b = float(pbright[i])
            y0b, y1b = max(0, y - r), min(h, y + r + 1)
            x0b, x1b = max(0, x - r), min(w, x + r + 1)
            canvas[y0b:y1b, x0b:x1b] = np.maximum(canvas[y0b:y1b, x0b:x1b], b)
        if fi % 17 == 0:
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, max(1, h // 2)))
            yy, xx = np.ogrid[:h, :w]
            blob = np.exp(
                -(((xx - cx) ** 2) / (2 * 90**2) + ((yy - cy) ** 2) / (2 * 60**2))
            )
            canvas = np.maximum(canvas, (blob * 70.0)[..., None])
        frames.append(np.clip(canvas, 0, 255).astype(np.uint8))

    try:
        from moviepy import ImageSequenceClip  # type: ignore

        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(
            str(path),
            fps=fps,
            codec="libx264",
            audio=False,
            preset="ultrafast",
            ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "23"],
            logger=None,
            temp_audiofile_path=moviepy_temp_audio_dir(),
        )
        clip.close()
    except Exception as exc:  # noqa: BLE001
        npz = path.with_suffix(".npz")
        np.savez_compressed(npz, frames=np.stack(frames, axis=0), fps=fps)
        _LOG.warning("dust overlay mp4 failed (%s) — wrote %s", exc, npz)
        return npz

    _LOG.info("Generated dust overlay asset → %s", path)
    print(f"[LOFI assemble] generated dust overlay -> {path}")
    return path


_OVERLAY_FRAME_CACHE: dict[str, np.ndarray] = {}


def _load_overlay_frames(path: Path) -> tuple[np.ndarray, float]:
    """Return (frames[N,H,W,3] uint8, fps)."""
    key = str(path.resolve())
    if key in _OVERLAY_FRAME_CACHE:
        return _OVERLAY_FRAME_CACHE[key], float(lofi_cfg.REEL_FPS)
    if path.suffix.lower() == ".npz":
        data = np.load(str(path))
        frames = data["frames"]
        fps = float(data["fps"]) if "fps" in data else float(lofi_cfg.REEL_FPS)
        _OVERLAY_FRAME_CACHE[key] = frames
        return frames, fps
    from moviepy import VideoFileClip  # type: ignore

    clip = VideoFileClip(str(path))
    fps = float(clip.fps or lofi_cfg.REEL_FPS)
    frames_list = []
    dur = float(clip.duration or 0.1)
    for ft in np.arange(0, dur, 1.0 / max(fps, 1.0)):
        frames_list.append(clip.get_frame(min(ft, dur - 1e-3)))
    clip.close()
    frames = np.stack(frames_list, axis=0).astype(np.uint8)
    _OVERLAY_FRAME_CACHE[key] = frames
    return frames, fps


_GRAIN_PREP_LOGGED = False
_GRAIN_BLEND_LOGS = 0
_GRAIN_BLEND_LOGGED_T: set[int] = set()


def _cover_crop_to_canvas(ov: np.ndarray, w: int, h: int) -> np.ndarray:
    """Center-cover to 9:16 canvas (crop, never stretch)."""
    arr = np.asarray(ov)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    oh, ow = int(arr.shape[0]), int(arr.shape[1])
    scale = max(w / max(ow, 1), h / max(oh, 1))
    nw = max(w, int(round(ow * scale)))
    nh = max(h, int(round(oh * scale)))
    if (ow, oh) != (nw, nh):
        arr = np.array(
            PILImage.fromarray(arr.astype(np.uint8)).resize((nw, nh), PILImage.LANCZOS)
        )
    y0 = max(0, (arr.shape[0] - h) // 2)
    x0 = max(0, (arr.shape[1] - w) // 2)
    cropped = arr[y0 : y0 + h, x0 : x0 + w]
    if cropped.shape[0] != h or cropped.shape[1] != w:
        cropped = np.array(
            PILImage.fromarray(cropped.astype(np.uint8)).resize((w, h), PILImage.LANCZOS)
        )
    return cropped


def _extract_grain_specks(ov: np.ndarray) -> np.ndarray:
    """Lift near-black film-grain plates so Screen/Overlay actually reads."""
    rgb = ov.astype(np.float32)
    luma = rgb.mean(axis=2)
    mean = float(luma.mean())
    std = float(luma.std())
    p99 = float(np.percentile(luma, 99.7))
    thr = mean + 0.25 * std
    denom = max(p99 - thr, 4.0)
    specks = np.clip((luma - thr) / denom, 0.0, 1.0)
    plate = specks * 255.0
    return np.stack([plate, plate, plate], axis=-1)


def _midgray_grain_plate(ov: np.ndarray) -> np.ndarray:
    """Center a dark film-grain plate on mid-gray so overlay/soft-light is identity-neutral."""
    rgb = ov.astype(np.float32)
    luma = rgb.mean(axis=2)
    mean = float(luma.mean())
    p99 = float(np.percentile(luma, 99.5))
    scale = 90.0 / max(p99 - mean, 6.0)
    centered = np.clip(128.0 + (luma - mean) * scale, 0.0, 255.0)
    return np.stack([centered, centered, centered], axis=-1)


def _overlay_chroma(ov: np.ndarray, gain: float) -> np.ndarray:
    """RGB minus luma — the tint authored into the grain file, in 0–255 units."""
    rgb = ov.astype(np.float32)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return (rgb - y[..., None]) * float(gain)


def _blend_overlay(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.where(
        base < 0.5,
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
    )


def _blend_screen(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - base) * (1.0 - blend)


def _blend_soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    # W3C / Photoshop soft-light
    d = np.where(
        base <= 0.25,
        ((16.0 * base - 12.0) * base + 4.0) * base,
        np.sqrt(np.clip(base, 0.0, 1.0)),
    )
    return np.where(
        blend <= 0.5,
        base - (1.0 - 2.0 * blend) * base * (1.0 - base),
        base + (2.0 * blend - 1.0) * (d - base),
    )


def apply_dust_overlay_screen(
    rgb: np.ndarray,
    *,
    t: float,
    overlay_frames: np.ndarray | None = None,
    overlay_getter=None,
    start_offset_s: float = 0.0,
    fps: float = 30.0,
    overlay_duration_s: float | None = None,
    overlay_source: str | None = None,
) -> np.ndarray:
    """Single-pass grain: screen the overlay file onto the picture. No chroma pass."""
    del overlay_duration_s
    if not bool(getattr(lofi_cfg, "ENABLE_DUST_OVERLAY", True)):
        return rgb
    op = float(getattr(lofi_cfg, "DUST_OVERLAY_OPACITY", 1.0))
    cap = float(getattr(lofi_cfg, "DUST_OVERLAY_OPACITY_CAP", 1.0))
    op = max(0.0, min(cap, op))
    if op < 0.001:
        return rgb
    ov = None
    t_ov = t + start_offset_s
    if overlay_getter is not None:
        ov = overlay_getter(t_ov)
    elif overlay_frames is not None and len(overlay_frames) > 0:
        n = len(overlay_frames)
        idx = int((t_ov * fps) % n)
        ov = overlay_frames[idx]
    if ov is None:
        return rgb
    ov = np.asarray(ov)
    h, w = rgb.shape[:2]
    native_h, native_w = int(ov.shape[0]), int(ov.shape[1])
    ov = _cover_crop_to_canvas(ov, w, h)
    gain = float(getattr(lofi_cfg, "DUST_OVERLAY_GAIN", 1.0))
    pre_mean = float(ov.astype(np.float32).mean())
    pre_max = float(ov.max())
    if gain != 1.0:
        ov = np.clip(ov.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    blend = str(getattr(lofi_cfg, "DUST_OVERLAY_BLEND", "screen")).lower()
    a = rgb.astype(np.float32) / 255.0
    g = ov.astype(np.float32) / 255.0
    if g.ndim == 2:
        g = np.stack([g, g, g], axis=-1)
    elif g.shape[-1] == 4:
        g = g[..., :3]
    global _GRAIN_BLEND_LOGGED_T
    t_key = int(round(float(t)))
    if (
        t_key in (0, 3, 6)
        and t_key not in _GRAIN_BLEND_LOGGED_T
        and abs(float(t) - t_key) < 0.05
    ):
        _GRAIN_BLEND_LOGGED_T.add(t_key)
        src = overlay_source or str(getattr(lofi_cfg, "DUST_OVERLAY_REL", ""))
        print(
            f"[LOFI grain BLEND] t={float(t):.3f} gain={gain:.2f} "
            f"opacity={op:.2f} cap={cap:.2f} blend={blend} "
            f"overlay={src} pre_mean={pre_mean:.1f} pre_max={pre_max:.0f} "
            f"post_mean={float(ov.mean()):.1f} post_max={float(ov.max()):.0f}"
        )
    if blend == "soft-light" or blend == "softlight":
        mixed = _blend_soft_light(a, g)
    elif blend == "overlay":
        mixed = _blend_overlay(a, g)
    else:
        mixed = _blend_screen(a, g)
    out = mixed if op >= 0.999 else (a * (1.0 - op) + mixed * op)
    global _GRAIN_PREP_LOGGED
    if not _GRAIN_PREP_LOGGED:
        _GRAIN_PREP_LOGGED = True
        print(
            f"[LOFI assemble] grain_prep native={native_w}x{native_h} "
            f"cover_crop={w}x{h} blend={blend} op={op:.2f} "
            f"gain={gain:.2f} pass=single "
            f"src_mean={float(np.asarray(ov).mean()):.1f} loop=True"
        )
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


_BGM_DUMP_NAME_RE = re.compile(
    str(getattr(lofi_cfg, "BGM_DENY_NAME_RE", r"^\d{3,8}(\(\d+\))?$")),
    re.I,
)


def is_verified_instrumental_bgm(path: Path | str) -> bool:
    """True only for pure instrumental library beds — never reference-reel rips."""
    p = Path(path)
    if p.suffix.lower() != ".mp3":
        return False
    parts_l = {part.lower() for part in p.parts}
    quarantine = str(
        getattr(lofi_cfg, "BGM_QUARANTINE_DIRNAME", "_quarantine_vocals")
    ).lower()
    if quarantine in parts_l:
        return False
    stem = p.stem
    if _BGM_DUMP_NAME_RE.match(stem):
        return False
    low = p.name.lower()
    exclude = tuple(getattr(lofi_cfg, "BGM_EXCLUDE_PREFIXES", ("lofi_bed",)))
    if any(low.startswith(pref.lower()) for pref in exclude):
        return False
    denied = tuple(getattr(lofi_cfg, "BGM_DENY_SUBSTRINGS", ()))
    if any(token in low for token in denied):
        return False
    return True


def list_library_bgm_tracks(engine_root: Path) -> list[Path]:
    """Library BGM from wonder_feed/audio/bgm — instrumentals only."""
    bgm_dir = Path(engine_root) / str(
        getattr(lofi_cfg, "BGM_DIR_REL", "channels_config/wonder_feed/audio/bgm")
    )
    if not bgm_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(bgm_dir.glob("*.mp3")):
        if not is_verified_instrumental_bgm(p):
            continue
        try:
            if not p.is_file() or p.stat().st_size < 1000:
                continue
        except OSError:
            continue
        out.append(p)
    return out


def pick_library_bgm(engine_root: Path, *, seed: int | None = None) -> Path | None:
    tracks = list_library_bgm_tracks(engine_root)
    if not tracks:
        return None
    rng = np.random.default_rng(
        int(seed if seed is not None else np.random.randint(0, 2**31 - 1))
    )
    return Path(tracks[int(rng.integers(0, len(tracks)))])


_UPSCALE_SCALE_EPS = 1e-4
_UPSCALE_LOGGED: set[str] = set()


def uniform_upscale_to_reel(im: PILImage.Image, *, src_name: str = "") -> PILImage.Image:
    """
    Pure uniform scale of a still onto the 1080×1920 delivery canvas.

    720×1280 → 1.5× / 1.5×. Any leftover 768×1344 (or other mismatch) raises
    so assemble cannot silently crop or stretch.
    """
    src_w, src_h = im.size
    tw, th = int(lofi_cfg.REEL_WIDTH), int(lofi_cfg.REEL_HEIGHT)
    sx = tw / max(src_w, 1)
    sy = th / max(src_h, 1)
    uniform = abs(sx - sy) <= _UPSCALE_SCALE_EPS
    key = f"{src_w}x{src_h}->{tw}x{th}"
    if key not in _UPSCALE_LOGGED:
        print(
            f"[LOFI upscale] src={src_w}x{src_h} dst={tw}x{th} "
            f"scale_x={sx:.6f} scale_y={sy:.6f} uniform={int(uniform)} "
            f"aspect_src={src_w / max(src_h, 1):.6f} aspect_dst={tw / th:.6f}"
            + (f" file={src_name}" if src_name else "")
        )
        _UPSCALE_LOGGED.add(key)
    if not uniform:
        raise RuntimeError(
            f"[LOFI upscale] REFUSE non-uniform scale: "
            f"scale_x={sx:.6f} scale_y={sy:.6f} "
            f"src={src_w}x{src_h} dst={tw}x{th} "
            f"— leftover aspect mismatch (no crop, no stretch)"
        )
    if (src_w, src_h) == (tw, th):
        return im
    return im.resize((tw, th), PILImage.LANCZOS)


def prep_base_frame(
    image_path: Path,
    *,
    shadow: tuple[int, int, int] | None = None,
    highlight: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """
    Load + uniform-scale to reel canvas (no cover-crop, no stretch).

    When ``LOFI_APPLY_GRADING`` is False (default), preserves Flux/riso palette —
    no duotone/LUT. Legacy grading path kept behind the flag only.
    """
    im = PILImage.open(image_path).convert("RGB")
    im = uniform_upscale_to_reel(im, src_name=Path(image_path).name)
    arr = np.array(im)
    if bool(getattr(lofi_cfg, "ENABLE_HIGHLIGHT_CLAMP", True)):
        arr = clamp_baked_highlights(arr)
    if not bool(getattr(lofi_cfg, "LOFI_APPLY_GRADING", False)):
        return arr
    arr = apply_duotone(arr, shadow=shadow, highlight=highlight)
    pil = PILImage.fromarray(arr)
    pil = ImageEnhance.Contrast(pil).enhance(1.08)
    return apply_vignette(np.array(pil), strength=lofi_cfg.VIGNETTE_STRENGTH)


def grade_still_frame(
    image_path: Path,
    *,
    grain_seed: int = 42,
    shadow: tuple[int, int, int] | None = None,
    highlight: tuple[int, int, int] | None = None,
    mood_id: str | None = None,
) -> np.ndarray:
    """
    Production grading for a single still: duotone + contrast + vignette + grain.

    Shared by ``test_preview`` and the video assembler so approved stills match ship.
    Pass mood-matched ``shadow``/``highlight`` (or ``mood_id``) for color variety.
    """
    if mood_id and (shadow is None or highlight is None):
        mood = lofi_cfg.lighting_mood_by_id(mood_id)
        if shadow is None:
            shadow = tuple(mood["shadow"])
        if highlight is None:
            highlight = tuple(mood["highlight"])
    sh = tuple(shadow) if shadow is not None else tuple(lofi_cfg.DUOTONE_SHADOW)
    hi = tuple(highlight) if highlight is not None else tuple(lofi_cfg.DUOTONE_HIGHLIGHT)
    print(
        f"[LOFI grade] apply_duotone={'ACTIVE' if getattr(lofi_cfg, 'LOFI_APPLY_GRADING', False) else 'SKIPPED'} "
        f"(bands={lofi_cfg.DUOTONE_TONAL_BANDS}) | "
        f"mood={mood_id or 'custom'} | "
        f"shadow={sh} highlight={hi} | vignette={lofi_cfg.VIGNETTE_STRENGTH} | "
        f"source={image_path}"
    )
    _LOG.info(
        "LOFI grade_still_frame | grading=%s | mood=%s | %s",
        getattr(lofi_cfg, "LOFI_APPLY_GRADING", False),
        mood_id,
        image_path,
    )
    graded = prep_base_frame(image_path, shadow=sh, highlight=hi)
    return apply_film_grain(graded, seed=grain_seed, t=0.5)


def render_logo_layer(
    logo_path: Path | None,
    *,
    opacity: float,
    scale: float,
    bottom_px: int = 48,
    size_multiplier: float = 1.0,
    trim_transparent_padding: bool = False,
) -> np.ndarray | None:
    """
    Composite the channel PNG on the final reel canvas.

    ``scale`` remains accepted for caller compatibility. ``size_multiplier``
    supports explicit per-channel calibration relative to the PNG's own size.
    """
    if not logo_path or not Path(logo_path).is_file():
        return None
    try:
        logo = PILImage.open(logo_path).convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("logo load failed: %s", exc)
        return None
    # Knock out near-black *background* (flood from corners) so a black-plate
    # PNG does not stamp a rectangle over the reel. Letterforms on the badge stay.
    arr = np.array(logo)
    h0, w0 = arr.shape[:2]
    lum = arr[..., :3].min(axis=2)
    is_bg = lum <= 22
    vis = np.zeros((h0, w0), dtype=bool)
    stack = []
    for y, x in ((0, 0), (0, w0 - 1), (h0 - 1, 0), (h0 - 1, w0 - 1)):
        if is_bg[y, x]:
            stack.append((y, x))
            vis[y, x] = True
    while stack:
        y, x = stack.pop()
        arr[y, x, 3] = 0
        if y > 0 and not vis[y - 1, x] and is_bg[y - 1, x]:
            vis[y - 1, x] = True
            stack.append((y - 1, x))
        if y + 1 < h0 and not vis[y + 1, x] and is_bg[y + 1, x]:
            vis[y + 1, x] = True
            stack.append((y + 1, x))
        if x > 0 and not vis[y, x - 1] and is_bg[y, x - 1]:
            vis[y, x - 1] = True
            stack.append((y, x - 1))
        if x + 1 < w0 and not vis[y, x + 1] and is_bg[y, x + 1]:
            vis[y, x + 1] = True
            stack.append((y, x + 1))
    logo = PILImage.fromarray(arr)
    # Trim transparent canvas padding before calibration so multiplier and
    # bottom inset operate on the visible badge, not invisible PNG margins.
    if trim_transparent_padding:
        visible_bbox = logo.getchannel("A").getbbox()
        if visible_bbox:
            logo = logo.crop(visible_bbox)
    # Scale only through an explicit channel calibration, never relative to
    # frame width. This keeps replacement logo assets predictable.
    _ = scale
    size_multiplier = max(0.1, float(size_multiplier))
    if abs(size_multiplier - 1.0) > 1e-6:
        resized = (
            max(1, round(logo.width * size_multiplier)),
            max(1, round(logo.height * size_multiplier)),
        )
        logo = logo.resize(resized, PILImage.Resampling.LANCZOS)
    native_w, native_h = logo.size
    r, g, b, a = logo.split()
    a = a.point(lambda v: int(v * max(0.05, min(1.0, opacity))))
    logo = PILImage.merge("RGBA", (r, g, b, a))
    canvas = PILImage.new("RGBA", (lofi_cfg.REEL_WIDTH, lofi_cfg.REEL_HEIGHT), (0, 0, 0, 0))
    x = (lofi_cfg.REEL_WIDTH - native_w) // 2
    inset = max(8, int(bottom_px))
    y = lofi_cfg.REEL_HEIGHT - native_h - inset
    canvas.paste(logo, (x, y), logo)
    print(
        f"[LOFI assemble] logo={Path(logo_path).name} rendered={native_w}x{native_h} "
        f"size_multiplier={size_multiplier:.2f} pos=bottom_center y={y} inset={inset} "
        f"center_from_bottom={inset + native_h / 2:.0f} every_scene=True"
    )
    return np.array(canvas)


def audit_watermark_native_size(
    logo_layer: np.ndarray | None,
    *,
    logo_path: Path | None,
    use_text: bool,
    expected_scale: float = 1.0,
) -> dict[str, Any]:
    """
    Confirm the watermark lives on the 1080×1920 canvas at file-native pixels.

    Fail if the layer is the still size (would ride the 1.5× upscale) or if
    the painted badge is larger than the PNG on disk (scaled-up watermark).
    """
    tw, th = int(lofi_cfg.REEL_WIDTH), int(lofi_cfg.REEL_HEIGHT)
    rec: dict[str, Any] = {
        "watermark_native_size": 0,
        "canvas_w": 0,
        "canvas_h": 0,
        "logo_file_w": 0,
        "logo_file_h": 0,
        "painted_w": 0,
        "painted_h": 0,
        "composited_on_reel_canvas": 0,
        "not_baked_into_still": 1,
        "use_text_watermark": int(bool(use_text)),
        "expected_logo_scale": float(expected_scale),
        "reason": "",
    }
    if logo_layer is None:
        rec["reason"] = "no_logo_layer"
        print("[LOFI assemble] watermark_native_size=0 reason=no_logo_layer")
        return rec
    rec["canvas_h"] = int(logo_layer.shape[0])
    rec["canvas_w"] = int(logo_layer.shape[1])
    on_reel = rec["canvas_w"] == tw and rec["canvas_h"] == th
    rec["composited_on_reel_canvas"] = int(on_reel)
    if not on_reel:
        rec["reason"] = (
            f"layer {rec['canvas_w']}x{rec['canvas_h']} != reel {tw}x{th} "
            "(watermark must be painted after upscale)"
        )
        print(f"[LOFI assemble] watermark_native_size=0 reason={rec['reason']}")
        return rec
    if use_text:
        rec["watermark_native_size"] = 1
        rec["reason"] = "text_handle_on_reel_canvas"
        print("[LOFI assemble] watermark_native_size=1 (text handle on 1080x1920)")
        return rec
    if not logo_path or not Path(logo_path).is_file():
        rec["reason"] = "logo_file_missing"
        print("[LOFI assemble] watermark_native_size=0 reason=logo_file_missing")
        return rec
    try:
        with PILImage.open(logo_path) as logo_im:
            rec["logo_file_w"], rec["logo_file_h"] = logo_im.size
    except Exception as exc:  # noqa: BLE001
        rec["reason"] = f"logo_read_failed:{exc}"
        print(f"[LOFI assemble] watermark_native_size=0 reason={rec['reason']}")
        return rec
    alpha = logo_layer[..., 3] if logo_layer.ndim == 3 and logo_layer.shape[2] >= 4 else None
    if alpha is None:
        rec["reason"] = "layer_has_no_alpha"
        print("[LOFI assemble] watermark_native_size=0 reason=layer_has_no_alpha")
        return rec
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        rec["reason"] = "empty_alpha"
        print("[LOFI assemble] watermark_native_size=0 reason=empty_alpha")
        return rec
    rec["painted_w"] = int(xs.max() - xs.min() + 1)
    rec["painted_h"] = int(ys.max() - ys.min() + 1)
    # Flood-fill knockout can shrink the bbox a few px vs the calibrated size.
    expected_w = rec["logo_file_w"] * max(0.1, float(expected_scale))
    expected_h = rec["logo_file_h"] * max(0.1, float(expected_scale))
    w_ok = rec["painted_w"] <= round(expected_w) + 2
    h_ok = rec["painted_h"] <= round(expected_h) + 2
    not_overscaled = rec["painted_w"] < int(expected_w * 1.08) + 2
    if w_ok and h_ok and not_overscaled:
        rec["watermark_native_size"] = 1
        rec["reason"] = "png_calibrated_on_reel_canvas"
    else:
        rec["reason"] = (
            f"painted {rec['painted_w']}x{rec['painted_h']} vs file "
            f"{rec['logo_file_w']}x{rec['logo_file_h']} — looks scaled"
        )
    print(
        f"[LOFI assemble] watermark_native_size={rec['watermark_native_size']} "
        f"file={rec['logo_file_w']}x{rec['logo_file_h']} "
        f"painted={rec['painted_w']}x{rec['painted_h']} "
        f"canvas={rec['canvas_w']}x{rec['canvas_h']} "
        f"reason={rec['reason']}"
    )
    return rec


def compute_caption_scene_duration_s(
    word_timings: Sequence[tuple[str, float, float]] | None,
    voice_path: Path | None,
    *,
    base_s: float | None = None,
    fade_s: float | None = None,
    hold_s: float | None = None,
) -> tuple[float, bool, dict[str, float]]:
    """
    Scene length must cover full caption reveal + a readable hold.

    Never shorter than *base_s* (default 3s). If speech needs more time,
    extend the scene instead of cutting mid-clause.
    """
    base = float(base_s if base_s is not None else lofi_cfg.SCENE_DURATION_S)
    fade = float(
        fade_s if fade_s is not None else getattr(lofi_cfg, "CAPTION_WORD_FADE_S", 0.20)
    )
    hold = float(
        hold_s if hold_s is not None else getattr(lofi_cfg, "CAPTION_HOLD_S", 0.45)
    )
    last_end = 0.0
    last_start = 0.0
    if word_timings:
        last_end = max(float(e) for _w, _s, e in word_timings)
        last_start = max(float(s) for _w, s, _e in word_timings)
    vo_dur = 0.0
    if voice_path is not None and Path(voice_path).is_file():
        try:
            from agents.media.audio_engine import _audio_file_duration_s

            vo_dur = float(_audio_file_duration_s(Path(voice_path)))
        except Exception:  # noqa: BLE001
            vo_dur = 0.0
    speech_end = max(last_end, last_start + fade, vo_dur)
    needed = speech_end + hold
    dur = max(base, needed)
    extended = dur > base + 0.04
    meta = {
        "base_s": base,
        "vo_dur": round(vo_dur, 3),
        "last_word_start": round(last_start, 3),
        "last_word_end": round(last_end, 3),
        "needed_s": round(needed, 3),
        "duration_s": round(dur, 3),
    }
    return round(dur, 3), extended, meta


def _silence_audio_clip(duration_s: float, fps: int = 44100):
    from moviepy.audio.AudioClip import AudioClip as _AC  # type: ignore

    dur = max(0.05, float(duration_s))
    return _AC(lambda t: 0, duration=dur, fps=fps)


# Match ffmpeg silencedetect noise=-30dB so line-boundary hush is not inherited.
_VO_EDGE_TRIM_DB: float = -30.0
_VO_EDGE_TRIM_SR: int = 44100


def _pcm_peak_envelope(path: Path, sr: int = _VO_EDGE_TRIM_SR) -> tuple[np.ndarray, float]:
    """Mono peak envelope + file duration. Prefers MoviePy; ffmpeg PCM fallback."""
    path = Path(path)
    try:
        from moviepy import AudioFileClip  # type: ignore

        clip = AudioFileClip(str(path))
        try:
            dur = float(clip.duration or 0.0)
            arr = np.asarray(clip.to_soundarray(fps=sr), dtype=np.float32)
        finally:
            clip.close()
        if arr.ndim > 1:
            env = np.max(np.abs(arr), axis=1)
        else:
            env = np.abs(arr)
        return env, dur
    except Exception:
        import subprocess

        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        raw = subprocess.check_output(
            [
                exe,
                "-v",
                "error",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-f",
                "f32le",
                "-",
            ],
            stderr=subprocess.DEVNULL,
        )
        env = np.abs(np.frombuffer(raw, dtype=np.float32))
        dur = float(len(env)) / float(sr) if sr else 0.0
        return env, dur


def _load_vo_samples(path: Path, sr: int = _VO_EDGE_TRIM_SR) -> tuple[np.ndarray, int]:
    """Return float32 samples shaped (n, channels)."""
    path = Path(path)
    from moviepy import AudioFileClip  # type: ignore

    clip = AudioFileClip(str(path))
    try:
        arr = np.asarray(clip.to_soundarray(fps=sr), dtype=np.float32)
    finally:
        clip.close()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, sr


def normalize_vo_pcm(
    path: Path | str,
    *,
    gap_s: float | None = None,
    thresh_db: float = _VO_EDGE_TRIM_DB,
    min_run_s: float = 0.15,
    sr: int = _VO_EDGE_TRIM_SR,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """
    Strip leading/trailing hush and collapse every internal hush run (>= min_run_s)
    to exactly ``gap_s`` (default VO_INTERLINE_SILENCE_S). Concat then inserts
    that same flat gap between lines.
    """
    path = Path(path)
    target = float(
        lofi_cfg.VO_INTERLINE_SILENCE_S if gap_s is None else gap_s
    )
    arr, sr = _load_vo_samples(path, sr=sr)
    n = int(arr.shape[0])
    if n < 8:
        return arr, sr, {"duration_s": n / float(sr), "lead_s": 0.0, "map": []}
    env = np.max(np.abs(arr), axis=1)
    thresh = float(10.0 ** (float(thresh_db) / 20.0))
    silent = env <= thresh
    runs: list[tuple[int, int, bool]] = []
    i = 0
    while i < n:
        flag = bool(silent[i])
        j = i + 1
        while j < n and bool(silent[j]) == flag:
            j += 1
        runs.append((i, j, flag))
        i = j
    min_run = max(1, int(round(min_run_s * sr)))
    gap_n = max(1, int(round(target * sr)))
    trim_edges = bool(getattr(lofi_cfg, "VO_TRIM_TTS_EDGES", True))
    pieces: list[np.ndarray] = []
    time_map: list[tuple[float, float]] = []
    new_i = 0
    n_runs = len(runs)
    for idx, (a, b, is_sil) in enumerate(runs):
        orig_a_s = a / float(sr)
        if is_sil:
            run_n = b - a
            if idx == 0 or idx == n_runs - 1:
                if trim_edges:
                    time_map.append((orig_a_s, new_i / float(sr)))
                    continue
                pieces.append(arr[a:b])
                time_map.append((orig_a_s, new_i / float(sr)))
                new_i += run_n
                continue
            if run_n < min_run:
                pieces.append(arr[a:b])
                time_map.append((orig_a_s, new_i / float(sr)))
                new_i += run_n
                continue
            zeros = np.zeros((gap_n, arr.shape[1]), dtype=np.float32)
            pieces.append(zeros)
            time_map.append((orig_a_s, new_i / float(sr)))
            new_i += gap_n
        else:
            pieces.append(arr[a:b])
            time_map.append((orig_a_s, new_i / float(sr)))
            new_i += b - a
    if not pieces:
        return arr, sr, {"duration_s": n / float(sr), "lead_s": 0.0, "map": []}
    out = np.concatenate(pieces, axis=0)
    lead_s = 0.0
    if trim_edges and runs and runs[0][2]:
        lead_s = (runs[0][1] - runs[0][0]) / float(sr)
    meta = {
        "duration_s": float(out.shape[0]) / float(sr),
        "lead_s": round(lead_s, 4),
        "file_duration_s": n / float(sr),
        "map": time_map,
        "internal_flat": sum(
            1
            for i, (a, b, sil) in enumerate(runs)
            if sil and i not in {0, n_runs - 1} and (b - a) >= min_run
        ),
    }
    return out, sr, meta


def remap_word_timings(
    timings: Sequence[tuple[str, float, float]] | None,
    time_map: list[tuple[float, float]],
    lead_s: float = 0.0,
) -> list[tuple[str, float, float]] | None:
    """Map original TTS word times onto normalized (edge-stripped) audio."""
    if not timings:
        return None if timings is None else []
    if not time_map:
        return shift_word_timings(timings, lead_s)

    xs = np.array([p[0] for p in time_map], dtype=np.float64)
    ys = np.array([p[1] for p in time_map], dtype=np.float64)

    def _map(t: float) -> float:
        tt = float(t)
        if xs.size == 1:
            return max(0.0, tt - float(lead_s or 0.0))
        return float(np.interp(tt, xs, ys))

    out: list[tuple[str, float, float]] = []
    for w, s, e in timings:
        out.append(
            (
                str(w),
                round(max(0.0, _map(float(s))), 4),
                round(max(0.0, _map(float(e))), 4),
            )
        )
    return out


def _audio_clip_from_samples(arr: np.ndarray, sr: int):
    from moviepy.audio.AudioClip import AudioArrayClip  # type: ignore

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return AudioArrayClip(arr.astype(np.float32), fps=int(sr))


def measure_vo_speech_span(
    path: Path | str,
    *,
    thresh_db: float = _VO_EDGE_TRIM_DB,
    sr: int = _VO_EDGE_TRIM_SR,
) -> tuple[float, float, float]:
    """
    Return (speech_start_s, speech_end_s, file_duration_s).

    Strips leading/trailing hush so concat can insert a flat inter-line gap
    instead of inheriting whatever silence each TTS clip has at its edges.
    """
    path = Path(path)
    if not path.is_file():
        return 0.0, 0.0, 0.0
    env, file_dur = _pcm_peak_envelope(path, sr=sr)
    file_dur = max(float(file_dur or 0.0), 0.0)
    if env.size == 0 or file_dur <= 0.0:
        return 0.0, file_dur, file_dur
    thresh = float(10.0 ** (float(thresh_db) / 20.0))
    above = np.flatnonzero(env > thresh)
    if above.size == 0:
        return 0.0, file_dur, file_dur
    start = float(above[0]) / float(sr)
    end = float(above[-1] + 1) / float(sr)
    start = max(0.0, min(start, file_dur))
    end = max(start, min(end, file_dur))
    if end - start < 0.12:
        return 0.0, file_dur, file_dur
    return round(start, 4), round(end, 4), round(file_dur, 4)


def measure_vo_speech_duration(path: Path | str) -> float:
    """Spoken duration after edge-trim and internal hush flattened to 300ms."""
    try:
        _arr, sr, meta = normalize_vo_pcm(path)
        dur = float(meta.get("duration_s") or 0.0)
        if dur > 0.05:
            return dur
    except Exception:  # noqa: BLE001
        pass
    start, end, file_dur = measure_vo_speech_span(path)
    spoken = end - start
    if spoken <= 0.05:
        return float(file_dur or 0.0)
    return float(spoken)


def shift_word_timings(
    timings: Sequence[tuple[str, float, float]] | None,
    lead_s: float,
) -> list[tuple[str, float, float]] | None:
    """Rebase ElevenLabs word times after leading hush is trimmed."""
    if not timings:
        return None if timings is None else []
    lead = max(0.0, float(lead_s or 0.0))
    if lead < 0.001:
        return [(str(w), float(s), float(e)) for w, s, e in timings]
    out: list[tuple[str, float, float]] = []
    for w, s, e in timings:
        out.append(
            (
                str(w),
                round(max(0.0, float(s) - lead), 4),
                round(max(0.0, float(e) - lead), 4),
            )
        )
    return out


def _clip_samples(clip, sr: int = 22050) -> np.ndarray:
    arr = np.asarray(clip.to_soundarray(fps=int(sr)), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _rms_in_windows(arr: np.ndarray, windows: list[tuple[float, float]], sr: int) -> float:
    acc = []
    n = int(arr.shape[0])
    for a, b in windows:
        i0 = max(0, int(round(float(a) * sr)))
        i1 = min(n, int(round(float(b) * sr)))
        if i1 - i0 < 8:
            continue
        sl = arr[i0:i1]
        acc.append(float(np.sqrt(np.mean(np.square(sl)))))
    if not acc:
        return 0.0
    return float(np.mean(acc))


def _speech_windows_from_gaps(
    gaps: list[tuple[float, float]],
    total_s: float,
    *,
    inset_s: float = 0.35,
) -> list[tuple[float, float]]:
    bounds = [0.0]
    for a, b in gaps:
        bounds.extend([float(a), float(b)])
    bounds.append(float(total_s))
    out: list[tuple[float, float]] = []
    for i in range(0, len(bounds) - 1, 2):
        s = bounds[i] + float(inset_s)
        e = bounds[i + 1] - float(inset_s)
        if e - s >= 0.35:
            out.append((s, e))
    return out


def _crossfade_bgm_gaps(
    bac,
    windows: list[tuple[float, float]],
    *,
    center_factor: float | Sequence[float],
):
    """Cosine breathe in each gap: bed at the edges, center_factor at the midpoint.

    ``center_factor`` is relative to the already-applied under-VO BGM fader.
    1.0 = no extra duck (gap is the same bed that sits under dialogue).
    """
    if not windows or bac is None:
        return bac
    if isinstance(center_factor, (int, float)):
        factors = [float(center_factor)] * len(windows)
    else:
        factors = [float(x) for x in center_factor]
    if len(factors) < len(windows):
        factors.extend([factors[-1] if factors else 1.0] * (len(windows) - len(factors)))

    def _env(t):
        arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.ones(arr.shape, dtype=np.float32)
        for i, (a, b) in enumerate(windows):
            w = float(b) - float(a)
            if w <= 1e-6:
                continue
            mask = (arr >= float(a)) & (arr < float(b))
            if not np.any(mask):
                continue
            tt = arr[mask]
            factor = float(factors[i])
            ramp = min(0.08, 0.35 * w)
            env = np.full(tt.shape, factor, dtype=np.float32)
            left = tt < (float(a) + ramp)
            right = tt > (float(b) - ramp)
            if np.any(left):
                u = np.clip((tt[left] - float(a)) / max(ramp, 1e-6), 0.0, 1.0)
                fade = 0.5 - 0.5 * np.cos(np.pi * u)
                env[left] = (1.0 + (factor - 1.0) * fade).astype(np.float32)
            if np.any(right):
                u = np.clip((float(b) - tt[right]) / max(ramp, 1e-6), 0.0, 1.0)
                fade = 0.5 - 0.5 * np.cos(np.pi * u)
                env[right] = (1.0 + (factor - 1.0) * fade).astype(np.float32)
            out[mask] = env
        if np.isscalar(t) or (getattr(t, "ndim", 1) == 0):
            return float(out.flat[0])
        return out

    def _xform(gf, t):
        frame = np.asarray(gf(t), dtype=np.float32)
        env = _env(t)
        if frame.ndim == 1:
            return frame * env
        if np.ndim(env) == 0:
            return frame * float(env)
        return frame * env.reshape(-1, 1)

    try:
        return bac.transform(_xform)
    except Exception:  # noqa: BLE001
        try:
            return bac.fl(_xform)
        except Exception:  # noqa: BLE001
            _LOG.warning("BGM gap crossfade failed — mixing unducked")
            return bac


def assemble_lofi_reel(
    scene_images: Sequence[Path],
    captions: Sequence[str],
    output_mp4: Path,
    *,
    engine_root: Path,
    page_id: str,
    scene_duration_s: float = lofi_cfg.SCENE_DURATION_S,
    scene_durations: Sequence[float] | None = None,
    moods: Sequence[dict[str, Any] | str | None] | None = None,
    caption_style: str | None = None,
    bgm_path: Path | None = None,
    voice_paths: Sequence[Path | None] | None = None,
    word_timings_per_scene: Sequence[Sequence[tuple[str, float, float]] | None] | None = None,
    caption_beats_per_scene: Sequence[Sequence[str] | None] | None = None,
    sidecar_dir: Path | None = None,
    vo_sidecar_dir: Path | None = None,
    audit_out: dict[str, Any] | None = None,
) -> Path:
    """
    Compile Wonder Feed LOFI relationship/reflection reels.

    Stack: base (no crushing LUT) → light pulse → uniform grain → dust →
    film multiply (behind text) → word-fade caption → logo.
    Mixes per-scene VO + library BGM. Does not affect Ancient Knowledge.

    One still per written line. ``caption_beats_per_scene`` cycles shorter
    on-screen captions over that held still, timed to the line's VO.
    """
    try:
        from moviepy import (  # type: ignore
            AudioFileClip,
            CompositeAudioClip,
            VideoClip,
            concatenate_audioclips,
            concatenate_videoclips,
        )
    except ImportError as exc:
        raise RuntimeError(
            "assembler requires moviepy>=2.0, numpy, Pillow.\n"
            f"Original error: {exc}"
        ) from exc

    global _GRAIN_PREP_LOGGED, _GRAIN_BLEND_LOGS, _GRAIN_BLEND_LOGGED_T, _UPSCALE_LOGGED
    _GRAIN_PREP_LOGGED = False
    _GRAIN_BLEND_LOGS = 0
    _GRAIN_BLEND_LOGGED_T = set()
    _UPSCALE_LOGGED = set()

    if len(scene_images) != len(captions):
        raise ValueError("scene_images and captions length mismatch")
    if not scene_images:
        raise ValueError("no scenes to assemble")

    from core.economic_reel_lofi.ship_gates import beat_integrity_blockers

    _beat_blockers = beat_integrity_blockers(
        script=None,
        scene_images=list(scene_images),
        captions=list(captions),
        voice_paths=list(voice_paths) if voice_paths is not None else None,
        require_voiceover=bool(getattr(lofi_cfg, "REQUIRE_VOICEOVER", True))
        and voice_paths is not None,
        expected_beats=len(captions),
    )
    if _beat_blockers:
        raise ValueError("; ".join(_beat_blockers))

    n_scenes = len(scene_images)
    scene_durs: list[float] = []
    vo_pcm: list[tuple[np.ndarray, int] | None] = []
    caption_timings: list[Sequence[tuple[str, float, float]] | None] = []
    lock_beat = bool(getattr(lofi_cfg, "LOCK_FIXED_BEAT_DURATION", True))
    for i in range(n_scenes):
        preset = None
        if scene_durations is not None and i < len(scene_durations):
            preset = float(scene_durations[i])
        vp = None
        if voice_paths is not None and i < len(voice_paths):
            vp = voice_paths[i]
        timings_i = None
        if word_timings_per_scene is not None and i < len(word_timings_per_scene):
            timings_i = word_timings_per_scene[i]
        pcm: tuple[np.ndarray, int] | None = None
        vo_dur = 0.0
        lead_s = 0.0
        if vp and Path(vp).is_file():
            try:
                arr, sr, meta = normalize_vo_pcm(Path(vp))
                pcm = (arr, sr)
                vo_dur = float(meta.get("duration_s") or 0.0)
                lead_s = float(meta.get("lead_s") or 0.0)
                timings_i = remap_word_timings(
                    timings_i,
                    list(meta.get("map") or []),
                    lead_s=lead_s,
                )
                print(
                    f"[LOFI assemble] scene {i + 1} VO normalized "
                    f"{float(meta.get('file_duration_s') or 0):.3f}s -> "
                    f"{vo_dur:.3f}s internal_flat={meta.get('internal_flat')}"
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("VO normalize failed for %s (%s)", vp, exc)
                pcm = None
                vo_dur = measure_vo_speech_duration(Path(vp))
                timings_i = shift_word_timings(timings_i, 0.0)
        vo_pcm.append(pcm)
        caption_timings.append(timings_i)
        declared_s = float(preset if preset is not None else scene_duration_s)
        trail = (
            float(getattr(lofi_cfg, "VO_SLOT_PAD_S", 0.12))
            if i >= n_scenes - 1
            else float(getattr(lofi_cfg, "VO_INTERLINE_SILENCE_S", 0.30))
        )
        if preset is not None and preset > 0:
            dur_i = round(declared_s, 3)
            extended_i = dur_i > float(scene_duration_s) + 0.04
            dur_meta = {
                "vo_dur": round(vo_dur, 3),
                "last_word_end": 0.0,
                "needed_s": dur_i,
                "duration_s": dur_i,
                "speech_start": round(lead_s, 3),
                "speech_end": round(lead_s + vo_dur, 3),
            }
            print(
                f"[LOFI assemble] scene {i + 1} elastic preset "
                f"vo={vo_dur:.3f}s slot={dur_i:.3f}s"
            )
        elif vo_dur > 0.05:
            dur_i, extended_i = lofi_cfg.slot_duration_for_vo(
                vo_dur, base_s=0.0, trailing_silence_s=trail
            )
            dur_meta = {
                "vo_dur": round(vo_dur, 3),
                "last_word_end": 0.0,
                "needed_s": dur_i,
                "duration_s": dur_i,
                "speech_start": round(lead_s, 3),
                "speech_end": round(lead_s + vo_dur, 3),
            }
            print(
                f"[LOFI assemble] scene {i + 1} audio-driven "
                f"vo={vo_dur:.3f}s slot={dur_i:.3f}s "
                f"ignored_estimate={declared_s:.3f}s"
            )
        elif lock_beat:
            dur_i, extended_i = lofi_cfg.slot_duration_for_vo(
                vo_dur, base_s=declared_s, trailing_silence_s=trail
            )
            dur_meta = {
                "vo_dur": round(vo_dur, 3),
                "last_word_end": 0.0,
                "needed_s": dur_i,
                "duration_s": dur_i,
                "speech_start": round(lead_s, 3),
                "speech_end": round(lead_s + vo_dur, 3),
            }
        else:
            dur_i, extended_i, dur_meta = compute_caption_scene_duration_s(
                timings_i,
                Path(vp) if vp else None,
                base_s=scene_duration_s,
            )
        scene_durs.append(dur_i)
        cap_preview = str(captions[i] if i < len(captions) else "")
        flag = ""
        if (not lock_beat) and (
            extended_i
            or (preset is not None and preset > scene_duration_s + 0.04)
        ):
            flag = " EXTENDED"
        print(
            f"[LOFI caption-timing] scene={i + 1}{flag} dur={dur_i:.2f}s "
            f"vo={dur_meta['vo_dur']:.2f}s last_word_end={dur_meta['last_word_end']:.2f}s "
            f"needed={dur_meta['needed_s']:.2f}s text={cap_preview!r}"
        )
        if flag:
            print(
                f"[LOFI caption-timing] FLAG scene {i + 1} extended past "
                f"{scene_duration_s:.1f}s so the line can fully display + hold"
            )

    cfg = lofi_cfg.channel_assembly_cfg(page_id)
    style = caption_style or lofi_cfg.DEFAULT_CAPTION_STYLE
    if cfg.get("use_text_watermark", True):
        logo_layer = np.array(
            render_lofi_watermark_layer(
                str(cfg.get("watermark_handle") or ""),
                engine_root=engine_root,
                opacity=float(cfg.get("logo_opacity", 0.55)),
                style=style,
            )
        )
    else:
        logo_path = lofi_cfg.resolve_logo_path(page_id, engine_root)
        logo_layer = render_logo_layer(
            logo_path,
            opacity=float(cfg.get("logo_opacity", 0.85)),
            scale=float(cfg.get("logo_scale", 0.14)),
            bottom_px=int(cfg.get("logo_bottom_px", 48)),
            size_multiplier=float(cfg.get("logo_size_multiplier", 1.0)),
            trim_transparent_padding=bool(
                cfg.get("trim_logo_transparent_padding", False)
            ),
        )
        if logo_layer is None:
            print(f"[LOFI assemble] WARN logo missing path={logo_path}")

    wm_audit = audit_watermark_native_size(
        logo_layer,
        logo_path=lofi_cfg.resolve_logo_path(page_id, engine_root)
        if not cfg.get("use_text_watermark", True)
        else None,
        use_text=bool(cfg.get("use_text_watermark", True)),
        expected_scale=float(cfg.get("logo_size_multiplier", 1.0)),
    )
    if audit_out is not None:
        audit_out.update(wm_audit)

    clips = []
    z0 = float(lofi_cfg.KEN_BURNS_ZOOM_START)
    z_cap = float(getattr(lofi_cfg, "KEN_BURNS_ZOOM_CAP", 1.05))
    z1 = min(z_cap, float(lofi_cfg.KEN_BURNS_ZOOM_END))
    pan_amp_x = float(getattr(lofi_cfg, "KEN_BURNS_PAN_AMP_X", 5.0))
    pan_amp_y = float(getattr(lofi_cfg, "KEN_BURNS_PAN_AMP_Y", 3.0))
    enable_pan = bool(getattr(lofi_cfg, "ENABLE_KINETIC_PAN", True))
    grading_on = bool(getattr(lofi_cfg, "LOFI_APPLY_GRADING", False))

    # Reference film-grain overlay — loop/trim to final duration; Screen below caption
    overlay_getter = None
    overlay_clip = None
    overlay_start = 0.0
    overlay_fps = float(lofi_cfg.REEL_FPS)
    overlay_source_path = ""
    if bool(getattr(lofi_cfg, "ENABLE_DUST_OVERLAY", True)):
        try:
            from moviepy import VideoFileClip as _VFC  # type: ignore

            ov_path = ensure_dust_overlay_asset(Path(engine_root))
            overlay_source_path = str(ov_path)
            overlay_clip = _VFC(str(ov_path), audio=False)
            ov_dur = float(overlay_clip.duration or 1.0)
            overlay_fps = float(overlay_clip.fps or lofi_cfg.REEL_FPS)
            overlay_start = 0.0
            def overlay_getter(t, _clip=overlay_clip, _dur=ov_dur):
                u = float(t) % max(_dur, 0.001)
                u = min(max(0.0, u), _dur - 1e-3)
                return _clip.get_frame(u)
            print(
                f"[LOFI assemble] grain_overlay={ov_path} "
                f"native={int(overlay_clip.w)}x{int(overlay_clip.h)} "
                f"src_dur={ov_dur:.2f}s loop_or_trim=True blend="
                f"{getattr(lofi_cfg, 'DUST_OVERLAY_BLEND', 'screen')} "
                f"op={getattr(lofi_cfg, 'DUST_OVERLAY_OPACITY', 0.6)} "
                f"gain={getattr(lofi_cfg, 'DUST_OVERLAY_GAIN', 1.0)} "
                f"below_caption=True below_logo=True"
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("grain overlay load failed: %s", exc)
            print(f"[LOFI assemble] WARN grain overlay failed: {exc}")
            overlay_clip = None
            overlay_getter = None

    # Brightness-factor debug table (independent of zoom) at 0.5s steps
    n_est = len(scene_images)
    total_est = float(sum(scene_durs))
    pulse_rows: list[dict[str, float]] = []
    if bool(getattr(lofi_cfg, "LIGHT_BREATH_DEBUG", True)):
        print("[LOFI pulse] brightness_factor vs time (independent of Ken Burns zoom)")
        t_s = 0.0
        while t_s <= total_est + 0.001:
            bf = light_pulse_brightness_factor(t_s, seed=3)
            pulse_rows.append({"t": round(t_s, 3), "brightness_factor": round(bf, 5)})
            print(f"  t={t_s:5.2f}s  brightness_factor={bf:.4f}")
            t_s += 0.5

    print(
        f"[LOFI assemble] grading={grading_on} | "
        f"light_pulse={lofi_cfg.ENABLE_LIGHT_BREATH} amp={lofi_cfg.LIGHT_BREATH_AMP} | "
        f"procedural_grain={getattr(lofi_cfg, 'ENABLE_PROCEDURAL_GRAIN', False)} | "
        f"multiply={getattr(lofi_cfg, 'CAPTION_FILM_MULTIPLY_OPACITY', 0):.2f} | "
        f"vignette={getattr(lofi_cfg, 'ENABLE_VIGNETTE', True)}:"
        f"{getattr(lofi_cfg, 'VIGNETTE_STRENGTH', 0):.2f} | "
        f"kenburns={z0:.3f}->{z1:.3f} ease | pan={enable_pan} "
        f"amp=({pan_amp_x:.0f},{pan_amp_y:.0f}) | "
        f"highlight_clamp={getattr(lofi_cfg, 'ENABLE_HIGHLIGHT_CLAMP', True)} | "
        f"caption={style} size_frac={getattr(lofi_cfg, 'CAPTION_LINE_HEIGHT_FRAC', 0)} "
        f"size_px={int(lofi_cfg.REEL_HEIGHT * float(getattr(lofi_cfg, 'CAPTION_LINE_HEIGHT_FRAC', 0.034)))} "
        f"word_fade=True | voice={lofi_cfg.tts_voice_id()} "
        f"speed={lofi_cfg.tts_speed()}"
    )

    for idx, (img_path, caption) in enumerate(zip(scene_images, captions)):
        mood_raw: Any = None
        if moods is not None and idx < len(moods):
            mood_raw = moods[idx]
        if isinstance(mood_raw, dict):
            mood = mood_raw
        elif isinstance(mood_raw, str) and mood_raw.strip():
            mood = lofi_cfg.lighting_mood_by_id(mood_raw)
        else:
            mood = lofi_cfg.select_lighting_mood(key=idx)
        sh = tuple(mood.get("shadow") or lofi_cfg.DUOTONE_SHADOW)
        hi = tuple(mood.get("highlight") or lofi_cfg.DUOTONE_HIGHLIGHT)
        base = prep_base_frame(Path(img_path), shadow=sh, highlight=hi)
        timings: Sequence[tuple[str, float, float]] | None = None
        if idx < len(caption_timings):
            timings = caption_timings[idx]
        h, w = base.shape[:2]
        this_dur = float(scene_durs[idx])
        t_offset = float(sum(scene_durs[:idx]))
        raw_beats = None
        if caption_beats_per_scene is not None and idx < len(caption_beats_per_scene):
            raw_beats = caption_beats_per_scene[idx]
        cycle_chunks = [str(c).strip() for c in (raw_beats or []) if str(c).strip()]
        if len(cycle_chunks) <= 1:
            cycle_windows = []
        else:
            cycle_windows = caption_cycle_windows(cycle_chunks, timings, this_dur)
        # Alternate pan/zoom direction per scene (still slow ease)
        pan_sign = 1.0 if (idx % 2 == 0) else -1.0
        zoom_out = idx % 2 == 1

        def _make_frame(
            t,
            _base=base,
            _logo=logo_layer,
            _idx=idx,
            _caption=str(caption),
            _timings=timings,
            _cycle=cycle_windows,
            _t_offset=t_offset,
            _dur=this_dur,
            _pan_sign=pan_sign,
            _zoom_out=zoom_out,
            _ov_get=overlay_getter,
            _ov_start=overlay_start,
            _ov_src=overlay_source_path,
        ):
            # Ease-in-out across FULL scene duration — slow drift
            u = max(0.0, min(1.0, t / max(_dur, 0.001)))
            ease = _smootherstep01(u)
            if _zoom_out:
                zoom = z1 - (z1 - z0) * ease
            else:
                zoom = z0 + (z1 - z0) * ease
            zoom = max(1.0, min(z_cap, zoom))
            crop_w = int(w / zoom)
            crop_h = int(h / zoom)
            # Kinetic pan — same ease, subtle amplitude
            if enable_pan:
                pan_e = math.sin(ease * math.pi)
                ox = int(_pan_sign * pan_amp_x * pan_e)
                oy = int((-_pan_sign) * pan_amp_y * pan_e * 0.6)
            else:
                ox = oy = 0
            x0 = (w - crop_w) // 2 + ox
            y0 = (h - crop_h) // 2 + oy
            x0 = max(0, min(w - crop_w, x0))
            y0 = max(0, min(h - crop_h, y0))
            cropped = _base[y0 : y0 + crop_h, x0 : x0 + crop_w]
            frame = np.array(
                PILImage.fromarray(cropped).resize((w, h), PILImage.LANCZOS)
            )
            t_global = _t_offset + float(t)
            # Picture stack only — overlay never after captions/logo
            frame = apply_animated_light_pulse(frame, t=t_global, seed=3)
            if bool(getattr(lofi_cfg, "ENABLE_PROCEDURAL_GRAIN", False)):
                frame = apply_film_grain(frame, seed=17, t=t_global)
            frame = apply_vignette(frame)
            if bool(getattr(lofi_cfg, "ENABLE_DUST_PARTICLES", False)):
                frame = apply_dust_particles(frame, t=t_global, seed=11)
            frame = apply_caption_film_multiply(frame, seed=41, t=0.0)
            frame = apply_dust_overlay_screen(
                frame,
                t=t_global,
                overlay_getter=_ov_get,
                start_offset_s=_ov_start,
                fps=overlay_fps,
                overlay_source=_ov_src,
            )
            # UI last: captions then logo, always above the grain overlay

            cap_text = _caption
            cap_timings = _timings
            if _cycle:
                cap_text, cap_timings, _span = active_caption_cycle(_cycle, float(t))
            cap = render_lofi_caption_layer_word_fade(
                cap_text,
                cap_timings,
                float(t),
                engine_root=engine_root,
                style=style,
                scene_duration_s=_dur,
            )
            rgba = PILImage.fromarray(frame).convert("RGBA")
            rgba = PILImage.alpha_composite(rgba, cap)
            if _logo is not None:
                rgba = PILImage.alpha_composite(rgba, PILImage.fromarray(_logo))
            return np.array(rgba.convert("RGB"))

        clip = VideoClip(frame_function=_make_frame, duration=float(this_dur))
        # Scene clips are visual-only. Narration is attached exactly once as
        # the master VO mix after concatenation.
        clip = _visual_only(clip.with_fps(lofi_cfg.REEL_FPS))
        clips.append(clip)

    processed_image_clips = [_visual_only(clip) for clip in clips]
    clips = processed_image_clips
    final = _visual_only(concatenate_videoclips(processed_image_clips, method="compose"))
    total_dur = float(final.duration or sum(scene_durs))

    # ── Audio: current-run scene VO only + one BGM layer, mixed once ─────────
    # Never glob *.mp3 — only the ordered voice_paths from this run.
    audio_clips_to_close: list[Any] = []
    mixed = None
    vo_parts: list[Any] = []
    gap_windows: list[tuple[float, float]] = []
    scene_voice_paths = [
        Path(vp) if vp is not None and Path(vp).is_file() else None
        for vp in (voice_paths or [])
    ]
    if any(vp is not None for vp in scene_voice_paths):
        present = [i for i, vp in enumerate(scene_voice_paths) if vp is not None]
        last_present = present[-1] if present else -1
        t_cursor = 0.0
        for idx, vp in enumerate(scene_voice_paths):
            if vp is None or not Path(vp).is_file():
                slot = float(
                    scene_durs[idx] if idx < len(scene_durs) else scene_duration_s
                )
                try:
                    silence = _silence_audio_clip(slot)
                    vo_parts.append(silence)
                    audio_clips_to_close.append(silence)
                    t_cursor += float(silence.duration or 0)
                except Exception:  # noqa: BLE001
                    continue
                continue
            pcm = vo_pcm[idx] if idx < len(vo_pcm) else None
            if pcm is None:
                vac = AudioFileClip(str(vp))
            else:
                vac = _audio_clip_from_samples(pcm[0], pcm[1])
            audio_clips_to_close.append(vac)
            vdur = float(vac.duration or 0.0)
            slot = float(scene_durs[idx] if idx < len(scene_durs) else scene_duration_s)
            never_trim = bool(getattr(lofi_cfg, "NEVER_TRIM_VOICEOVER", True))
            if vdur > slot + 0.05:
                print(
                    f"[LOFI assemble] scene {idx + 1} VO {vdur:.2f}s > "
                    f"slot {slot:.2f}s — keeping full speech "
                    f"({'never trim' if never_trim else 'no trim'})"
                )
            vac = vac.with_volume_scaled(
                float(getattr(lofi_cfg, "LOFI_VOICE_VOLUME", 1.0))
            )
            vo_parts.append(vac)
            t_cursor += vdur
            visual_slot = float(
                scene_durs[idx] if idx < len(scene_durs) else scene_duration_s
            )
            gap_s = max(0.0, visual_slot - vdur)
            if idx < last_present and gap_s > 0.001:
                silence = _silence_audio_clip(gap_s)
                vo_parts.append(silence)
                audio_clips_to_close.append(silence)
                gap_windows.append(
                    (
                        round(t_cursor, 3),
                        round(t_cursor + gap_s, 3),
                    )
                )
                t_cursor += gap_s
                print(
                    f"[LOFI assemble] interline silence {gap_s:.3f}s "
                    f"after scene {idx + 1} duck={gap_windows[-1]}"
                )

    bgm = bgm_path or pick_library_bgm(
        engine_root,
        seed=int(hashlib.md5(str(output_mp4).encode("utf-8")).hexdigest()[:8], 16),
    )
    if bgm is not None and not is_verified_instrumental_bgm(bgm):
        print(
            f"[LOFI assemble] REJECT contaminated BGM {Path(bgm).name} "
            "— picking verified instrumental"
        )
        bgm = pick_library_bgm(
            engine_root,
            seed=int(hashlib.md5(str(output_mp4).encode("utf-8")).hexdigest()[:8], 16)
            ^ 0xA5A5,
        )
    bac = None
    vo_full = None
    if vo_parts:
        vo_full = concatenate_audioclips(vo_parts)
        audio_clips_to_close.append(vo_full)
    bgm_full_vol = float(getattr(lofi_cfg, "BGM_VOLUME", 0.38))
    if bgm is None or not Path(bgm).is_file():
        msg = (
            f"LOFI BGM required but no library .mp3 found under "
            f"{Path(engine_root) / lofi_cfg.BGM_DIR_REL}"
        )
        if bool(getattr(lofi_cfg, "REQUIRE_BGM", True)):
            raise RuntimeError(msg)
        _LOG.warning(msg)
    else:
        bac = AudioFileClip(str(bgm))
        audio_clips_to_close.append(bac)
        bgm_dur = float(bac.duration or 0.0)
        if bgm_dur > total_dur + 0.05:
            max_start = max(0.0, bgm_dur - total_dur)
            start = float(
                np.random.default_rng(
                    int(
                        hashlib.md5(
                            f"{output_mp4}|{Path(bgm).name}".encode("utf-8")
                        ).hexdigest()[:8],
                        16,
                    )
                ).random()
                * max_start
            )
            bac = bac.subclipped(start, start + total_dur)
        elif bgm_dur < total_dur - 0.05:
            loops = int(math.ceil(total_dur / max(bgm_dur, 0.1))) + 1
            bac = concatenate_audioclips([bac] * max(1, loops))
            bac = bac.subclipped(0, total_dur)
        else:
            bac = bac.subclipped(0, total_dur)
        bac = bac.with_volume_scaled(bgm_full_vol)
        if gap_windows:
            center = 1.0
            try:
                sr_m = 22050
                bgm_arr = _clip_samples(bac, sr_m)
                if vo_full is not None:
                    vo_arr = _clip_samples(vo_full, sr_m)
                    n = min(int(bgm_arr.shape[0]), int(vo_arr.shape[0]))
                    bgm_arr = bgm_arr[:n]
                    vo_arr = vo_arr[:n]
                    if vo_arr.shape[1] != bgm_arr.shape[1]:
                        if vo_arr.shape[1] == 1:
                            vo_arr = np.repeat(vo_arr, bgm_arr.shape[1], axis=1)
                        elif bgm_arr.shape[1] == 1:
                            bgm_arr = np.repeat(bgm_arr, vo_arr.shape[1], axis=1)
                    mix_est = vo_arr.astype(np.float32) + bgm_arr.astype(np.float32)
                    speech_w = _speech_windows_from_gaps(
                        gap_windows, n / float(sr_m)
                    )
                    mid_mix = _rms_in_windows(mix_est, speech_w, sr_m)
                    bed = _rms_in_windows(bgm_arr, speech_w, sr_m)
                    vs_db = float(getattr(lofi_cfg, "BGM_GAP_VS_SPEECH_DB", 10.0))
                    target = mid_mix * (10.0 ** (-vs_db / 20.0))
                    lo = float(getattr(lofi_cfg, "BGM_GAP_CENTER_FACTOR_MIN", 0.75))
                    hi = float(getattr(lofi_cfg, "BGM_GAP_CENTER_FACTOR_MAX", 2.20))
                    factors: list[float] = []
                    for ga, gb in gap_windows:
                        unducked = _rms_in_windows(bgm_arr, [(ga, gb)], sr_m)
                        if unducked > 1e-8 and target > 1e-8:
                            f = target / unducked
                        else:
                            f = 1.0
                        factors.append(min(hi, max(lo, float(f))))
                    center = factors
                    print(
                        f"[LOFI assemble] BGM bed={20.0 * math.log10(max(bed, 1e-9)):.1f}dB "
                        f"mid-speech mix={20.0 * math.log10(max(mid_mix, 1e-9)):.1f}dB "
                        f"target gap={20.0 * math.log10(max(target, 1e-9)):.1f}dB "
                        f"center_factors={[round(x, 2) for x in factors]} "
                        f"(per-window cosine vs live BGM in that gap, not the fader)"
                    )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("BGM gap measure failed (%s) — bed unchanged in pauses", exc)
                print(f"[LOFI assemble] BGM gap measure failed ({exc}) — no extra duck")
                center = 1.0
            bac = _crossfade_bgm_gaps(
                bac, gap_windows, center_factor=center
            )
            shown = center if isinstance(center, (int, float)) else [round(x, 2) for x in center]
            print(
                f"[LOFI assemble] BGM gap crossfade {len(gap_windows)} windows "
                f"center_factor={shown}"
            )
        print(
            f"[LOFI assemble] BGM={Path(bgm).name} "
            f"vol={bgm_full_vol} trimmed={total_dur:.1f}s"
        )

    audio_attached = False
    try:
        if vo_parts:
            if vo_full is None:
                vo_full = concatenate_audioclips(vo_parts)
                audio_clips_to_close.append(vo_full)
            vo_len = float(vo_full.duration or 0)
            mix_dur = float(total_dur)
            if vo_len > total_dur + 0.05:
                print(
                    f"[LOFI assemble] VO concat {vo_len:.2f}s > video {total_dur:.2f}s "
                    "— keeping full VO and extending mix (never trim)"
                )
                mix_dur = vo_len
            vo_dir = Path(vo_sidecar_dir) if vo_sidecar_dir else (
                Path(output_mp4).parent.parent / "assets"
                if Path(output_mp4).parent.name == "clips"
                else Path(output_mp4).parent
            )
            vo_dir.mkdir(parents=True, exist_ok=True)
            vo_sidecar = vo_dir / f"{Path(output_mp4).stem}_vo_concat.mp3"
            try:
                vo_full.write_audiofile(str(vo_sidecar), logger=None)
                print(f"[LOFI assemble] raw VO concat -> {vo_sidecar}")
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("raw VO concat sidecar failed: %s", exc)
            if isinstance(audit_out, dict):
                audit_out["vo_concat_path"] = str(vo_sidecar)
                audit_out["interline_gaps"] = gap_windows
            layers = [vo_full]
            if bac is not None:
                layers.append(bac)
            mixed = CompositeAudioClip(layers).with_duration(mix_dur)
            if mix_dur > float(final.duration or 0) + 0.05:
                try:
                    final = final.with_duration(mix_dur)
                except Exception:  # noqa: BLE001
                    pass
            final = final.with_audio(mixed)
            audio_attached = True
            print(
                f"[LOFI assemble] single master VO files="
                f"{len(present) if voice_paths else 0} "
                f"+ {len(gap_windows)} interline silences + BGM mixed once"
            )
        elif bac is not None:
            final = final.with_audio(bac)
            audio_attached = True
            print("[LOFI assemble] BGM only (no VO paths)")
        elif bool(getattr(lofi_cfg, "REQUIRE_VOICEOVER", False)):
            raise RuntimeError("LOFI voiceover required but no VO clips were provided")
    except Exception as exc:  # noqa: BLE001
        if bool(getattr(lofi_cfg, "REQUIRE_BGM", True)) or bool(
            getattr(lofi_cfg, "REQUIRE_VOICEOVER", False)
        ):
            raise RuntimeError(f"LOFI audio mix failed: {exc}") from exc
        _LOG.warning("audio mix failed: %s", exc)

    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    _purge_moviepy_temp_audio()
    try:
        final.write_videofile(
            str(output_mp4),
            fps=lofi_cfg.REEL_FPS,
            codec="libx264",
            audio=audio_attached,
            audio_codec="aac" if audio_attached else None,
            preset="fast",
            ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p"],
            logger=None,
            temp_audiofile_path=moviepy_temp_audio_dir(),
        )
    finally:
        try:
            final.close()
        except Exception:  # noqa: BLE001
            pass
        for c in clips:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        for ac in audio_clips_to_close:
            try:
                ac.close()
            except Exception:  # noqa: BLE001
                pass
        if mixed is not None:
            try:
                mixed.close()
            except Exception:  # noqa: BLE001
                pass
        if overlay_clip is not None:
            try:
                overlay_clip.close()
            except Exception:  # noqa: BLE001
                pass

    if pulse_rows:
        try:
            import json as _json

            json_dir = Path(sidecar_dir) if sidecar_dir else (
                Path(output_mp4).parent.parent / "metadata"
                if Path(output_mp4).parent.name == "clips"
                else Path(output_mp4).parent
            )
            json_dir.mkdir(parents=True, exist_ok=True)
            dbg = json_dir / f"{Path(output_mp4).stem}_pulse_debug.json"
            dbg.write_text(
                _json.dumps(
                    {
                        "amp": float(lofi_cfg.LIGHT_BREATH_AMP),
                        "period_s": float(lofi_cfg.LIGHT_BREATH_PERIOD_S),
                        "note": "brightness_factor is independent of Ken Burns zoom",
                        "samples": pulse_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[LOFI pulse] debug written -> {dbg}")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("pulse debug write failed: %s", exc)

    _LOG.info("LOFI reel assembled -> %s (%.1fs)", output_mp4, total_dur)
    print(f"[LOFI assemble] wrote {output_mp4}")
    return output_mp4
