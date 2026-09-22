# -*- coding: utf-8 -*-
"""
Channel context loader for the Unified Multi-Page Factory.

Bootstrap for ``channels_config/{channel_id}/`` (CLI flag is ``--channel``, alias ``--page``).
Resolves paths and pipeline flags from ``--channel`` (``--page``), ``--avatar``, and ``--format``.
Each channel lives in an isolated directory and carries its own:

  - master_dna.json      — persona data, environments, voice, CTAs
  - persona_dna.py       — Python interface over master_dna.json
  - page_config.py       — channel-level overrides (aspect ratio, atmosphere style, etc.)
  - avatar_reference/    — optional: avatar.png for likeness-locked generation
  - voice_reference/     — optional: F5-TTS ref wav + matching transcript .txt
  - product_reference/   — optional: PDF corpus for the research brain

Supported channels
---------------
  anna_protocol   Holistic Legacy — ancestral wellness, natural remedies, avatar ON
  master_mei      SUPER channel — Stoic financial freedom / wealth mindset (US), avatar ON
  wonder_feed     Emotional intelligence, attachment science, avatar OFF (default)
  down_dirty      Matrix escape, financial sovereignty, raw mindset, avatar OFF (default)
  ancient_knowledge  Ancient history, conspiracies, mysteries, photorealistic style, avatar OFF
  momma_circle    Parenting / PARENTAL_CONTENTS — reference-clip reels, warm lullaby audio, avatar OFF
  principles_of_wealth_finance_economics
                  Educational / Financial Curation — library ingest of the
                  Principles of Wealth production drive (wealth_main.py)
  endless_summer_paradise
                  1950s surreal architecture library ingest — local source/
                  + factory_settings.json (esp_main.py; duration > 40s)

Usage
-----
    from channel_loader import load_page_context, PageContext

    ctx = load_page_context("anna_protocol", avatar_mode="ON", post_format="IMAGE_AVATAR")
    print(ctx.page_dir)
    print(ctx.avatar_on)
    print(ctx.atmosphere_style)
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MODULE 2 — Gemini Vision style-reference extraction. Sent ONCE per page
# per run; the extracted ~40-word anchor is cached on the PageContext
# instance and prepended to every image prompt generated for that run.
# ---------------------------------------------------------------------------
_STYLE_VISION_PROMPT: str = (
    "Analyze these reference images to extract the exact style rules. "
    "Focus on: dark 80s biomechanical cyberpunk, practical body horror FX, "
    "H.R. Giger aesthetics, flesh pierced by cables/tubes with scarred entry wounds, "
    "exposed copper wiring stitched into pale distressed skin, dirty raw metal, "
    "gritty 35mm film grain. Do NOT describe rubber aprons, chestplates, or tactical vests. "
    "Vary tech mods — do not put VR goggles on every face."
)


def _extract_style_anchor_via_gemini_vision(image_paths: "list[Path]", page_id: str) -> str:
    """
    Send ``image_paths`` ONCE to gemini-2.5-flash and extract a dense ~40-word
    visual style anchor string. Returns "" on ANY failure (missing API key,
    unreadable image, API error) so callers can gracefully fall back to a
    static style anchor — this call must never crash the pipeline.
    """
    if not image_paths:
        return ""
    try:
        from PIL import Image

        import config as app_config
        from core.google_guardrail import guarded_generate_content, make_guarded_gemini_client
        api_key = getattr(app_config, "GEMINI_API_KEY", None)
        if not api_key:
            _LOG.warning(
                "DYNAMIC_STYLE_ANCHOR | GEMINI_API_KEY missing — skipping vision "
                "extraction for page=%s; falling back to static style anchor.",
                page_id,
            )
            return ""

        contents: list[Any] = []
        for p in image_paths:
            try:
                contents.append(Image.open(p))
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("DYNAMIC_STYLE_ANCHOR | could not open %s (%s)", p, exc)
        if not contents:
            return ""
        contents.append(_STYLE_VISION_PROMPT)

        client = make_guarded_gemini_client(api_key)
        response = guarded_generate_content(
            client,
            model="models/gemini-2.5-flash",
            contents=contents,
            source="channel_loader._extract_dynamic_style_anchor",
        )
        text = (getattr(response, "text", "") or "").strip()
        if text:
            _LOG.info(
                "DYNAMIC_STYLE_ANCHOR extracted | page=%s | %d word(s): %s",
                page_id, len(text.split()), text[:200],
            )
        return text
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "DYNAMIC_STYLE_ANCHOR | vision extraction failed for page=%s (%s) — "
            "falling back to static style anchor.",
            page_id, exc,
        )
        return ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PAGES: tuple[str, ...] = (
    "anna_protocol",
    "master_mei",
    "wonder_feed",
    "down_dirty",
    "ancient_knowledge",
    "momma_circle",
    "principles_of_wealth_finance_economics",
    "endless_summer_paradise",
)

VALID_AVATAR_MODES: tuple[str, ...] = ("ON", "OFF")

VALID_FORMATS: tuple[str, ...] = (
    "IMAGE_AVATAR",       # standard portrait image ± avatar
    "IMAGE_QUOTE",        # Gemini image + text overlay (legacy alias)
    "IMAGE_BACKGROUND",   # hyper-literal Gemini background + text overlay (SMART_BAIT default)
    "HYBRID_VIDEO",       # 7-second Ken Burns zoom loop from generated image
    "TEXT_QUOTE",         # brand-colour solid backdrop + text only (no Gemini image call)
    "DYNAMIC_REEL",       # ECONOMIC_REEL: single image → MP4 via video_engine
    "SEQUENCE_REEL",      # multi-image 80-second reel via core.reel_sequence_engine
    "REFERENCE_BASED_REELS",  # raw footage clip + hook overlay + lullaby audio (momma_circle)
)

_ENGINE_ROOT: Path = Path(__file__).resolve().parent
_CHANNELS_CONFIG_ROOT: Path = _ENGINE_ROOT / "channels_config"
_LEGACY_PAGES_CONFIG_ROOT: Path = _ENGINE_ROOT / "pages_config"


def _resolve_channels_config_root() -> Path:
    """Prefer channels_config/; fall back to historic pages_config/."""
    if _CHANNELS_CONFIG_ROOT.is_dir():
        return _CHANNELS_CONFIG_ROOT
    return _LEGACY_PAGES_CONFIG_ROOT


# Back-compat alias for callers that imported the private constant.
_channels_config_ROOT = _CHANNELS_CONFIG_ROOT


# ---------------------------------------------------------------------------
# PageContext dataclass
# ---------------------------------------------------------------------------

@dataclass
class PageContext:
    """
    All page-specific runtime parameters resolved from --page, --avatar, --format.

    Attributes
    ----------
    page_id:
        Slug identifying the active page (e.g. 'anna_protocol').
    avatar_mode:
        'ON' — include human subject + reference image in image generation.
        'OFF' — bypass avatar pipeline; generate purely atmospheric imagery.
    post_format:
        'IMAGE_AVATAR'      — standard portrait image with (or without) avatar.
        'IMAGE_QUOTE'       — Gemini image + text overlay (legacy alias for IMAGE_BACKGROUND).
        'IMAGE_BACKGROUND'  — hyper-literal Gemini background + text overlay (SMART_BAIT default).
        'HYBRID_VIDEO'      — 7-second Ken Burns zoom loop from generated image.
        'TEXT_QUOTE'        — brand-colour solid backdrop + text only (no Gemini image call).
    page_dir:
        Absolute path to channels_config/{page_id}/.
    persona_dna_path:
        Absolute path to channels_config/{page_id}/persona_dna.py.
    master_dna_path:
        Absolute path to channels_config/{page_id}/master_dna.json.
    avatar_reference_dir:
        Absolute path to channels_config/{page_id}/avatar_reference/ (auto-created).
    voice_reference_dir:
        Absolute path to channels_config/{page_id}/voice_reference/ (auto-created).
        Drop ``{page_id}_voice_ref_10s.wav`` + matching ``.txt`` transcript for F5-TTS.
    logo_dir:
        Absolute path to channels_config/{page_id}/logo/ (auto-created).
        Drop a transparent PNG here to activate the logo watermark layer.
    product_reference_dir:
        Absolute path to channels_config/{page_id}/product_reference/ (may not exist).
    outputs_dir:
        Absolute path to {OUTPUT_PATH}/{page_id}/ for all page-namespaced artifacts.
    store_dir:
        Permanent local state: ``channels_config/{page_id}/store/``.
    page_cfg:
        Dict of values exported from page_config.py (atmosphere_style, aspect_ratio, etc.).
    """

    page_id: str
    avatar_mode: str
    post_format: str
    page_dir: Path
    persona_dna_path: Path
    master_dna_path: Path
    avatar_reference_dir: Path
    voice_reference_dir: Path
    logo_dir: Path
    product_reference_dir: Path
    outputs_dir: Path
    page_cfg: dict[str, Any] = field(default_factory=dict)
    store_dir: Path | None = None
    _dynamic_style_anchor_cache: "str | None" = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def avatar_on(self) -> bool:
        """True when --avatar ON is active."""
        return self.avatar_mode == "ON"

    @property
    def is_hybrid_video(self) -> bool:
        """True when --format HYBRID_VIDEO is active."""
        return self.post_format == "HYBRID_VIDEO"

    @property
    def is_image_quote(self) -> bool:
        """True when --format IMAGE_QUOTE is active."""
        return self.post_format == "IMAGE_QUOTE"

    @property
    def display_name(self) -> str:
        return self.page_cfg.get("PAGE_DISPLAY_NAME", self.page_id)

    @property
    def atmosphere_style(self) -> str:
        """Atmospheric visual style string used when --avatar OFF."""
        return self.page_cfg.get(
            "ATMOSPHERE_STYLE",
            "Cinematic environmental photography. Moody, high-fidelity, no human subjects.",
        )

    @property
    def image_aspect_ratio(self) -> str:
        """Aspect ratio override from page_config.py; falls back to global config."""
        return self.page_cfg.get("IMAGE_ASPECT_RATIO", "")

    @property
    def content_niche(self) -> str:
        return self.page_cfg.get("CONTENT_NICHE", "")

    @property
    def avatar_reference_png(self) -> Path:
        """Resolved path to avatar.png inside avatar_reference/."""
        return self.avatar_reference_dir / "avatar.png"

    @property
    def avatar_reference_exists(self) -> bool:
        return self.avatar_reference_png.is_file()

    @property
    def uses_avatar_reference(self) -> bool:
        """Whether the page was designed to use a human likeness reference."""
        return bool(self.page_cfg.get("USES_AVATAR_REFERENCE", False))

    @property
    def voice_reference_wav(self) -> Path | None:
        """
        Preferred F5-TTS reference wav under ``voice_reference/``, if present.

        Resolution order:
          1. ``VOICE_REFERENCE_AUDIO`` from page_config.py (filename or absolute path)
          2. ``{page_id}_voice_ref_10s.wav``
          3. First ``*_voice_ref*.wav`` / sole ``.wav`` in the folder
        """
        from core.remote_gpu_manager import resolve_page_voice_reference  # noqa: PLC0415

        audio, _text = resolve_page_voice_reference(self.page_id, page_dir=self.page_dir)
        return audio

    @property
    def voice_reference_text(self) -> str | None:
        """Exact transcript for the page voice-reference clip (F5 ``sample_text``)."""
        from core.remote_gpu_manager import resolve_page_voice_reference  # noqa: PLC0415

        _audio, text = resolve_page_voice_reference(self.page_id, page_dir=self.page_dir)
        return text

    @property
    def voice_reference_ready(self) -> bool:
        """True when both the reference wav and its matching transcript exist."""
        from core.remote_gpu_manager import resolve_page_voice_reference  # noqa: PLC0415

        audio, text = resolve_page_voice_reference(self.page_id, page_dir=self.page_dir)
        return audio is not None and bool((text or "").strip())

    @property
    def remote_gpu_lora_enabled(self) -> bool:
        """True when this channel has a Flux LoRA configured for remote GPU."""
        from core.remote_gpu_manager import resolve_page_lora_config  # noqa: PLC0415

        return bool(
            resolve_page_lora_config(self.page_id, page_dir=self.page_dir).get("enabled")
        )

    @property
    def remote_gpu_lora_name(self) -> str | None:
        from core.remote_gpu_manager import resolve_page_lora_config  # noqa: PLC0415

        return resolve_page_lora_config(self.page_id, page_dir=self.page_dir).get("lora_name")

    @property
    def remote_gpu_lora_trigger(self) -> str:
        from core.remote_gpu_manager import resolve_page_lora_config  # noqa: PLC0415

        return str(
            resolve_page_lora_config(self.page_id, page_dir=self.page_dir).get("trigger") or ""
        )

    @property
    def remote_gpu_lora_strength(self) -> float:
        from core.remote_gpu_manager import resolve_page_lora_config  # noqa: PLC0415

        try:
            return float(
                resolve_page_lora_config(self.page_id, page_dir=self.page_dir).get("strength")
                or 0.8
            )
        except (TypeError, ValueError):
            return 0.8

    @property
    def logo_png(self) -> Path | None:
        """
        Returns the first .png found in logo_dir, or None if the folder is empty.

        The logo PNG should have a transparent background (RGBA).  It is applied
        as a watermark on all final image outputs and HYBRID_VIDEO frames.
        """
        if not self.logo_dir.is_dir():
            return None
        for candidate in sorted(self.logo_dir.iterdir()):
            if candidate.suffix.lower() == ".png" and candidate.is_file():
                return candidate
        return None

    @property
    def logo_exists(self) -> bool:
        """True when at least one .png is present in logo_dir."""
        return self.logo_png is not None

    @property
    def logo_size_scale(self) -> float:
        """
        Logo width as a fraction of canvas width (e.g. 0.30 = 30 %).
        Sourced from LOGO_SIZE_SCALE in page_config.py; defaults to 0.30.
        """
        raw = self.page_cfg.get("LOGO_SIZE_SCALE", 0.30)
        try:
            val = float(raw)
            return max(0.05, min(val, 0.50))   # clamp to 5–50 %
        except (TypeError, ValueError):
            return 0.18

    @property
    def caption_signature(self) -> str:
        """
        Brand copyright line appended to every LONG_CAPTION_IMAGE caption.
        Not used for CTA_CAPTION_IMAGE (original Anna CTA posts have no © footer).

        Sources (in priority order):
          1. CAPTION_SIGNATURE in page_config.py  (explicit override; an empty
             string DISABLES the footer — the signature is intentionally omitted)
          2. ``© {display_name} | by MediaUpScale`` derived from display_name
          3. ``© MediaUpScale`` ultimate fallback
        """
        if "CAPTION_SIGNATURE" in self.page_cfg:
            return str(self.page_cfg.get("CAPTION_SIGNATURE") or "").strip()
        name = self.display_name
        if name:
            return f"© {name} | by MediaUpScale"
        return "© MediaUpScale"

    @property
    def logo_position(self) -> str:
        """
        Placement for the logo watermark.
        One of: 'top_left', 'top_right', 'bottom_left', 'bottom_right',
                'bottom_center', 'top_center'.
        Sourced from LOGO_POSITION in page_config.py; defaults to 'bottom_right'.
        """
        raw = str(self.page_cfg.get("LOGO_POSITION", "bottom_right")).lower().strip()
        valid = {
            "top_left", "top_right", "bottom_left", "bottom_right",
            "bottom_center", "top_center",
        }
        return raw if raw in valid else "bottom_right"

    @property
    def illustration_style(self) -> str:
        """
        Visual style directive used for LONG_CAPTION_IMAGE and as a supplementary
        modifier for SMART_BAIT image prompts.
        Sourced from ILLUSTRATION_STYLE in page_config.py; falls back to empty string.
        """
        return str(self.page_cfg.get("ILLUSTRATION_STYLE", "")).strip()

    @property
    def font_path(self) -> str:
        """
        Relative path (from engine root) to the preferred .ttf font file.
        Sourced from FONT_PATH in page_config.py; falls back to empty string.
        The caller is responsible for resolving to an absolute path.
        """
        return str(self.page_cfg.get("FONT_PATH", "")).strip()

    @property
    def font_size_scale(self) -> float:
        """
        Font size expressed as a fraction of canvas width (e.g. 0.08 = 8 %).
        Sourced from FONT_SIZE_SCALE in page_config.py; defaults to 0.08.
        """
        raw = self.page_cfg.get("FONT_SIZE_SCALE", 0.08)
        try:
            return max(0.02, min(float(raw), 0.30))
        except (TypeError, ValueError):
            return 0.08

    @property
    def elevenlabs_voice_id(self) -> str:
        """ElevenLabs voice ID for ECONOMIC_REEL voiceover generation."""
        return str(self.page_cfg.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")).strip()

    @property
    def elevenlabs_model(self) -> str:
        """ElevenLabs TTS model for ECONOMIC_REEL voiceover generation."""
        default = (
            "eleven_v3"
            if (self.page_id or "").lower() == "master_mei"
            else "eleven_multilingual_v2"
        )
        return str(self.page_cfg.get("ELEVENLABS_MODEL", default)).strip() or default

    @property
    def tts_expressive_mode(self) -> bool:
        """Prefer expressive eleven_v3 delivery when True (Master Mei default)."""
        raw = self.page_cfg.get("TTS_EXPRESSIVE_MODE", None)
        if raw is None:
            return (self.page_id or "").lower() == "master_mei"
        return bool(raw)

    @property
    def tts_voice_preference(self) -> str:
        """
        Human-readable voice label for documentation and logging.
        Sourced from TTS_VOICE_PREFERENCE in page_config.py; defaults to empty string.
        This is a descriptive label only — the actual API call uses elevenlabs_voice_id.
        """
        return str(self.page_cfg.get("TTS_VOICE_PREFERENCE", "")).strip()

    @property
    def reel_duration(self) -> float:
        """Target ECONOMIC_REEL duration in seconds (fallback if no audio).

        ``VIDEO_LENGTH_OVERRIDE`` (CLI ``--video-length``) always wins and is
        not clamped into the channel window.
        """
        override = self.page_cfg.get("VIDEO_LENGTH_OVERRIDE", None)
        if override is not None:
            try:
                return max(5.0, float(override))
            except (TypeError, ValueError):
                pass
        try:
            dur = max(5.0, float(self.page_cfg.get("REEL_DURATION", 30.0)))
        except (TypeError, ValueError):
            dur = 30.0
        # Clamp into optional ECONOMIC_REEL window (independent of WAN_REEL).
        lo = self.reel_duration_target_min
        hi = self.reel_duration_target_max
        if lo and hi and hi >= lo:
            dur = max(lo, min(hi, dur))
        return dur

    @property
    def scene_duration(self) -> str:
        """
        Scene pacing spec for ``plan_scenes`` (CLI ``--scene-duration`` wins via
        ``SCENE_DURATION`` override). Factory default ``equal`` preserves legacy
        equal-split behaviour for channels that do not set this.
        """
        raw = self.page_cfg.get("SCENE_DURATION", None)
        if raw is None or str(raw).strip() == "":
            return "equal"
        return str(raw).strip()

    @property
    def scene_progressive_start_s(self) -> float:
        try:
            return max(0.5, float(self.page_cfg.get("SCENE_PROGRESSIVE_START_S", 4.0)))
        except (TypeError, ValueError):
            return 4.0

    @property
    def scene_progressive_step_every(self) -> int:
        try:
            return max(1, int(self.page_cfg.get("SCENE_PROGRESSIVE_STEP_EVERY", 3)))
        except (TypeError, ValueError):
            return 3

    @property
    def scene_progressive_step_s(self) -> float:
        try:
            return float(self.page_cfg.get("SCENE_PROGRESSIVE_STEP_S", 1.0))
        except (TypeError, ValueError):
            return 1.0

    @property
    def scene_progressive_cap_s(self) -> float:
        try:
            return max(
                self.scene_progressive_start_s,
                float(self.page_cfg.get("SCENE_PROGRESSIVE_CAP_S", 7.5)),
            )
        except (TypeError, ValueError):
            return 7.5

    @property
    def wan_scene_duration(self) -> str:
        """WAN_REEL pacing spec (independent of ECONOMIC_REEL SCENE_DURATION)."""
        raw = self.page_cfg.get("WAN_SCENE_DURATION", None)
        if raw is None or str(raw).strip() == "":
            return "fixed:7"
        return str(raw).strip()

    @property
    def reel_duration_target_min(self) -> float:
        """ECONOMIC_REEL duration floor (default 70). Not used for WAN_REEL."""
        try:
            return max(5.0, float(self.page_cfg.get("REEL_DURATION_TARGET_MIN", 70.0)))
        except (TypeError, ValueError):
            return 70.0

    @property
    def reel_duration_target_max(self) -> float:
        """ECONOMIC_REEL duration ceiling (default 90). Not used for WAN_REEL."""
        try:
            return max(
                self.reel_duration_target_min,
                float(self.page_cfg.get("REEL_DURATION_TARGET_MAX", 90.0)),
            )
        except (TypeError, ValueError):
            return 90.0

    @property
    def pacing_sequence(self) -> "list[float] | None":
        """Exact per-still hold curve; None → engine default (AK stepped 3→7s)."""
        raw = self.page_cfg.get("PACING_SEQUENCE", None)
        if not raw:
            return None
        try:
            seq = [float(x) for x in raw]
        except (TypeError, ValueError):
            return None
        return seq if seq else None

    @property
    def scene_length(self) -> "float | None":
        """Uniform per-still hold override (CLI ``--scene-length`` / SCENE_LENGTH)."""
        raw = self.page_cfg.get("SCENE_LENGTH", None)
        if raw is None or raw == "":
            return None
        try:
            val = float(raw)
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None

    @property
    def encoding_preset(self) -> str:
        return str(self.page_cfg.get("ENCODING_PRESET", "medium") or "medium").strip() or "medium"

    @property
    def enable_subtitle_padding(self) -> bool:
        return bool(self.page_cfg.get("ENABLE_SUBTITLE_PADDING", True))

    @property
    def reuse_existing_images(self) -> bool:
        return bool(self.page_cfg.get("REUSE_EXISTING_IMAGES", False))

    @property
    def wan_reel_duration_target_min(self) -> float:
        """WAN_REEL duration floor (default 80). Independent of ECONOMIC_REEL."""
        try:
            return max(5.0, float(self.page_cfg.get("WAN_REEL_DURATION_TARGET_MIN", 80.0)))
        except (TypeError, ValueError):
            return 80.0

    @property
    def wan_reel_duration_target_max(self) -> float:
        """WAN_REEL duration ceiling (default 120). Independent of ECONOMIC_REEL."""
        try:
            return max(
                self.wan_reel_duration_target_min,
                float(self.page_cfg.get("WAN_REEL_DURATION_TARGET_MAX", 120.0)),
            )
        except (TypeError, ValueError):
            return 120.0

    @property
    def wan_reel_duration(self) -> float:
        """WAN_REEL planning target, clamped to its own 80–120 s window."""
        try:
            dur = max(5.0, float(self.page_cfg.get("WAN_REEL_DURATION", 100.0)))
        except (TypeError, ValueError):
            dur = 100.0
        return max(
            self.wan_reel_duration_target_min,
            min(self.wan_reel_duration_target_max, dur),
        )

    @property
    def reel_hook_hold_s(self) -> float:
        """First-act still hold (ECONOMIC_REEL pacing). Default 5 s."""
        try:
            return max(1.0, float(self.page_cfg.get("REEL_HOOK_HOLD_S", 5.0)))
        except (TypeError, ValueError):
            return 5.0

    @property
    def reel_body_hold_s(self) -> float:
        """Subsequent-act still hold (ECONOMIC_REEL). Default 7.5 s (7–8 band)."""
        try:
            return max(3.0, float(self.page_cfg.get("REEL_BODY_HOLD_S", 7.5)))
        except (TypeError, ValueError):
            return 7.5

    @property
    def reel_overlay_opacity(self) -> float:
        """Dark vignette opacity (0-1) applied over the graphite base in ECONOMIC_REEL."""
        try:
            val = float(self.page_cfg.get("REEL_OVERLAY_OPACITY", 0.35))
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.35

    @property
    def subtitle_words_per_phrase(self) -> int:
        from channels_config.shared.mei_chrome import SUBTITLE_WORDS_PER_PHRASE

        return max(3, min(5, int(SUBTITLE_WORDS_PER_PHRASE)))

    @property
    def subtitle_fill(self) -> tuple:
        from channels_config.shared.mei_chrome import SUBTITLE_FILL

        return tuple(int(c) for c in SUBTITLE_FILL)

    @property
    def subtitle_stroke_fill(self) -> "tuple | None":
        from channels_config.shared.mei_chrome import SUBTITLE_STROKE_FILL

        return tuple(int(c) for c in SUBTITLE_STROKE_FILL)

    @property
    def subtitle_stroke_width(self) -> int:
        from channels_config.shared.mei_chrome import SUBTITLE_STROKE_WIDTH

        return max(0, int(SUBTITLE_STROKE_WIDTH))

    @property
    def subtitle_fontsize(self) -> int:
        """ECONOMIC_REEL subtitle size — inherited from master_mei chrome."""
        from channels_config.shared.mei_chrome import SUBTITLE_FONTSIZE

        return max(20, int(SUBTITLE_FONTSIZE))

    @property
    def subtitle_y_position(self) -> "int | None":
        """Lower-third Y — inherited from master_mei chrome."""
        from channels_config.shared.mei_chrome import SUBTITLE_Y_POSITION

        return max(0, int(SUBTITLE_Y_POSITION))

    @property
    def cta_subtitle_y_position(self) -> "int | None":
        """Follow/CTA overlay Y — inherited from master_mei chrome."""
        from channels_config.shared.mei_chrome import CTA_SUBTITLE_Y_POSITION

        return max(0, int(CTA_SUBTITLE_Y_POSITION))

    @property
    def logo_width_px(self) -> int:
        """Absolute pixel width to resize the logo PNG in ECONOMIC_REEL.
        Overrides the fractional 18%-of-canvas-width default."""
        try:
            return max(40, int(self.page_cfg.get("LOGO_WIDTH", 160)))
        except (TypeError, ValueError):
            return 160

    @property
    def logo_y_offset_px(self) -> int:
        """Pixels from the bottom canvas edge to the bottom of the logo in ECONOMIC_REEL.
        Reads LOGO_BOTTOM_MARGIN first (canonical key); falls back to LOGO_Y_OFFSET for
        pages that still use the older key name."""
        try:
            val = (
                self.page_cfg.get("LOGO_BOTTOM_MARGIN")
                or self.page_cfg.get("LOGO_Y_OFFSET", 90)
            )
            return max(10, int(val))
        except (TypeError, ValueError):
            return 90

    @property
    def logo_opacity(self) -> float:
        """Logo PNG opacity (0.0–1.0) applied during ECONOMIC_REEL pre-render.
        60% = humble, authentic blend; 70% = prominent."""
        try:
            val = float(self.page_cfg.get("LOGO_OPACITY", 0.70))
            return max(0.05, min(1.0, val))
        except (TypeError, ValueError):
            return 0.70

    @property
    def logo_max_height_px(self) -> "int | None":
        """Hard pixel cap on logo height in ECONOMIC_REEL.
        Returns None when unset, allowing free aspect-ratio scaling from logo_width_px."""
        raw = self.page_cfg.get("LOGO_MAX_HEIGHT", None)
        if raw is None:
            return None
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return None

    @property
    def hook_y_frac(self) -> float:
        """Vertical centre of the hook/bait text as a fraction of canvas height.
        0.30 = upper-middle Zone A (~576 px on a 1920 px canvas).
        Defaults to 0.55 (legacy centre) when not set."""
        try:
            val = float(self.page_cfg.get("HOOK_Y_FRAC", 0.55))
            return max(0.05, min(0.95, val))
        except (TypeError, ValueError):
            return 0.55

    @property
    def topic_pool(self) -> "list[str]":
        """Rotating topic seeds used when no --topic flag is supplied.
        Prevents the static 'Holistic vitality protocol' fallback from driving
        identical LLM output on every run.  Returns an empty list when unset."""
        raw = self.page_cfg.get("TOPIC_POOL", [])
        if not isinstance(raw, list):
            return []
        return [str(t).strip() for t in raw if t and str(t).strip()]

    @property
    def base_graphite_prompt(self) -> str:
        """Single permanent visual style string sent to Gemini for all post formats."""
        return str(self.page_cfg.get("BASE_GRAPHITE_PROMPT", "")).strip()

    @property
    def enable_sketch_style(self) -> bool:
        """True = use fine-art sketch pipeline; False = photorealistic fallback."""
        return bool(self.page_cfg.get("ENABLE_SKETCH_STYLE", False))

    @property
    def enable_horror_transformations(self) -> bool:
        """True = inject dark surrealist mask/monster transformation directives."""
        return bool(self.page_cfg.get("ENABLE_HORROR_TRANSFORMATIONS", False))

    @property
    def sketch_style_prompt(self) -> str:
        """Fine-art graphite-sketch style directive for SMART_BAIT image generation."""
        return str(self.page_cfg.get("SKETCH_STYLE_PROMPT", "")).strip()

    @property
    def raw_graphite_horror_prompt(self) -> str:
        """Dark psychological surrealism / horror-transformation style directive."""
        return str(self.page_cfg.get("RAW_GRAPHITE_HORROR_PROMPT", "")).strip()

    @property
    def use_style_reference(self) -> bool:
        """Toggle: if True, image prompt includes recurring style_characters personas."""
        return bool(self.page_cfg.get("USE_STYLE_REFERENCE", False))

    @property
    def style_characters(self) -> str:
        """Recurring persona description injected into the image prompt when use_style_reference=True."""
        return str(self.page_cfg.get("STYLE_CHARACTERS", "")).strip()

    @property
    def text_outline_width(self) -> int:
        """
        PIL stroke_width for text overlay (0 = no outline).
        Sourced from TEXT_OUTLINE_WIDTH in page_config.py; defaults to 0.
        """
        try:
            return max(0, int(self.page_cfg.get("TEXT_OUTLINE_WIDTH", 0)))
        except (TypeError, ValueError):
            return 0

    @property
    def style_reference_dir(self) -> str:
        """
        Relative path (from engine root) to the aesthetic reference image directory.
        Sourced from STYLE_REFERENCE_DIR in page_config.py; falls back to empty string.
        """
        return str(self.page_cfg.get("STYLE_REFERENCE_DIR", "")).strip()

    @property
    def style_reference_weight(self) -> float:
        """
        IP-Adapter / style-ref strength (0.65–0.80 recommended) — enforce local
        visual style while still allowing prompt compliance. Sourced from
        STYLE_REFERENCE_WEIGHT in page_config.py; defaults to 0.72.
        """
        try:
            w = float(self.page_cfg.get("STYLE_REFERENCE_WEIGHT", 0.72))
        except (TypeError, ValueError):
            w = 0.72
        return max(0.0, min(1.0, w))

    @property
    def style_reference_max_images(self) -> int:
        """Max style reference images loaded per generation run (2–3 recommended)."""
        try:
            return max(1, int(self.page_cfg.get("STYLE_REFERENCE_MAX_IMAGES", 3)))
        except (TypeError, ValueError):
            return 3

    @property
    def master_style_anchor(self) -> str:
        """
        Style-anchor text appended VERBATIM to every generated image prompt.
        Sourced from MASTER_STYLE_ANCHOR in page_config.py; defaults to "".
        """
        return str(self.page_cfg.get("MASTER_STYLE_ANCHOR", "")).strip()

    def resolve_style_reference_images(self, max_images: "int | None" = None) -> list[Path]:
        """
        MODULE 3 — dynamic per-page style reference resolution.

        Base directory pattern: ``channels_config/<page_id>/style_reference/``
        (or the explicit ``STYLE_REFERENCE_DIR`` override when set). Loads all
        valid ``*.png`` / ``*.jpg`` / ``*.jpeg`` assets, sorted by filename,
        capped at ``max_images`` (defaults to ``style_reference_max_images``,
        2–3 recommended).
        """
        limit = max_images if max_images is not None else self.style_reference_max_images
        override = self.style_reference_dir
        sref_dir = (
            (_ENGINE_ROOT / override) if override else (self.page_dir / "style_reference")
        )
        if not sref_dir.is_dir():
            return []
        found: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            found.extend(sorted(sref_dir.glob(ext)))
        # De-dupe while preserving order, then cap.
        seen: set[str] = set()
        ordered: list[Path] = []
        for p in sorted(found, key=lambda x: x.name.lower()):
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                ordered.append(p)
        return ordered[: max(1, limit)]

    def resolve_dynamic_style_anchor(self, *, force_refresh: bool = False) -> str:
        """
        MODULE 2 — Gemini Vision style-reference extraction.

        Loads ``channels_config/<page_id>/style_reference/`` images (via
        ``resolve_style_reference_images``), sends them ONCE to
        ``gemini-2.5-flash`` to extract a dense ~40-word visual style anchor,
        and caches the result (``DYNAMIC_STYLE_ANCHOR``) on this PageContext
        instance for the remainder of the run. Falls back to the static
        ``MASTER_STYLE_ANCHOR`` (page_config.py) when no reference images
        exist or vision extraction fails for any reason — never blocks or
        crashes the pipeline.
        """
        if self._dynamic_style_anchor_cache is not None and not force_refresh:
            return self._dynamic_style_anchor_cache

        images = self.resolve_style_reference_images()
        print(
            f"[DEBUG] Loaded {len(images)} style reference assets for {self.page_id}: "
            f"{[p.name for p in images]}"
        )
        anchor = _extract_style_anchor_via_gemini_vision(images, self.page_id) if images else ""
        resolved = anchor or self.master_style_anchor
        self._dynamic_style_anchor_cache = resolved
        return resolved

    @property
    def font_color(self) -> tuple[int, int, int]:
        """
        RGB text colour tuple (e.g. (255, 255, 255) for white).
        Sourced from FONT_COLOR in page_config.py; defaults to white.
        """
        raw = self.page_cfg.get("FONT_COLOR", (255, 255, 255))
        try:
            r, g, b = int(raw[0]), int(raw[1]), int(raw[2])
            return (r, g, b)
        except Exception:
            return (255, 255, 255)

    # ------------------------------------------------------------------
    # Core-engine modular properties (new — all pages may define these)
    # ------------------------------------------------------------------

    @property
    def cost_tier(self) -> str:
        """
        Cost tier for this page: 'nano' | 'economic' | 'premium'.
        Drives CostTracker pricing keys and model selection.
        Sourced from COST_TIER in page_config.py; defaults to 'economic'.
        """
        raw = str(self.page_cfg.get("COST_TIER", "economic")).lower().strip()
        return raw if raw in ("nano", "economic", "premium") else "economic"

    @property
    def enable_cost_tracking(self) -> bool:
        """True = write cost telemetry JSON (cost_*.json) after each variant."""
        return bool(self.page_cfg.get("ENABLE_COST_TRACKING", False))

    @property
    def enable_sequence_reel(self) -> bool:
        """
        True = use core.reel_sequence_engine (4-image 80s reel).
        False = use agents.media.video_engine single-image DYNAMIC_REEL.
        Sourced from ENABLE_SEQUENCE_REEL in page_config.py; defaults to False.
        """
        return bool(self.page_cfg.get("ENABLE_SEQUENCE_REEL", False))

    @property
    def reel_image_count(self) -> int:
        """
        Max distinct images in a SEQUENCE_REEL.

        Defaults to 11 (ECONOMIC_REEL paced); dense pages may still set 18.

        When USE_TWO_TIER_PACING is True (AK), the page_cfg REEL_IMAGE_COUNT
        entry is deliberately omitted so this property returns a large
        sentinel (9999) that effectively removes the static ceiling — the
        actual per-variant act count is computed at runtime via
        ``core.reel_sequence_engine.compute_two_tier_act_count``
        from ``page_ctx.reel_duration`` (see main.py). This closes the
        Round-6 bug where --video-length 180 silently rendered only 90 s
        of content because the 16-still cap clipped the plan.
        """
        if bool(self.page_cfg.get("USE_TWO_TIER_PACING", False)) and \
                "REEL_IMAGE_COUNT" not in self.page_cfg:
            return 9999
        try:
            return max(2, int(self.page_cfg.get("REEL_IMAGE_COUNT", 11)))
        except (TypeError, ValueError):
            return 11

    @property
    def use_two_tier_pacing(self) -> bool:
        """Opt-in flag for two-tier act pacing (Tier1 fast-cut / Tier2 slower)."""
        return bool(self.page_cfg.get("USE_TWO_TIER_PACING", False))

    @property
    def reel_tier1_max_acts(self) -> int:
        try:
            return max(4, int(self.page_cfg.get("REEL_TIER1_MAX_ACTS", 20)))
        except (TypeError, ValueError):
            return 20

    @property
    def reel_tier1_horizon_s(self) -> float:
        try:
            return max(30.0, float(self.page_cfg.get("REEL_TIER1_HORIZON_S", 90.0)))
        except (TypeError, ValueError):
            return 90.0

    @property
    def reel_tier1_seconds_per_act(self) -> float:
        try:
            return max(3.0, min(6.0, float(self.page_cfg.get("REEL_TIER1_SECONDS_PER_ACT", 4.5))))
        except (TypeError, ValueError):
            return 4.5

    @property
    def reel_tier2_seconds_per_act(self) -> float:
        try:
            return max(6.0, min(14.0, float(self.page_cfg.get("REEL_TIER2_SECONDS_PER_ACT", 10.0))))
        except (TypeError, ValueError):
            return 10.0

    @property
    def reel_seconds_per_act(self) -> float:
        """
        Target seconds per visual act (legacy dense path / body-hold alias).
        Upper clamp raised to 8.5 so ECONOMIC_REEL 7–8 s body holds are kept.
        """
        try:
            spa = float(
                self.page_cfg.get(
                    "REEL_SECONDS_PER_ACT",
                    self.page_cfg.get("REEL_BODY_HOLD_S", 4.0),
                )
            )
        except (TypeError, ValueError):
            spa = 4.0
        return max(3.5, min(8.5, spa))

    @property
    def reel_image_min_count(self) -> int:
        """
        Min distinct images/acts (floor) for scene sync.
        Defaults to 10 for ECONOMIC_REEL paced path.
        """
        try:
            return max(2, int(self.page_cfg.get("REEL_IMAGE_MIN_COUNT", 10)))
        except (TypeError, ValueError):
            return 10

    @property
    def reel_act_duration(self) -> float:
        """
        Per-act clip length when no audio drives the timeline.
        Audio-driven compiles use weighted ``act_durations`` / equal split.
        """
        try:
            return max(3.5, float(self.page_cfg.get("REEL_ACT_DURATION", 7.5)))
        except (TypeError, ValueError):
            return 7.5

    @property
    def reel_use_hook_body_pacing(self) -> bool:
        """
        Legacy hook/body pacing (REEL_HOOK_HOLD_S). Prefer ``scene_duration`` /
        ``plan_scenes`` when SCENE_DURATION is progressive/fixed.
        """
        # Shared engine wins when explicitly configured.
        mode = (self.scene_duration or "equal").strip().lower()
        if mode and mode not in ("equal", "legacy", "default"):
            return False
        raw = self.page_cfg.get("REEL_HOOK_HOLD_S", None)
        return raw is not None and float(raw or 0) > 0

    @property
    def uses_plan_scenes_pacing(self) -> bool:
        """True when SCENE_DURATION is fixed:* or progressive (shared engine)."""
        mode = (self.scene_duration or "equal").strip().lower()
        return bool(mode) and mode not in ("equal", "legacy", "default")

    @property
    def enable_top_hook_text(self) -> bool:
        """
        When False the headline/hook text is NOT burned into the top of the frame.
        Only lower-third word subtitles and the logo remain.
        Sourced from ENABLE_TOP_HOOK_TEXT in page_config.py; defaults to True
        for backward compatibility with wonder_feed and other pages.
        """
        return bool(self.page_cfg.get("ENABLE_TOP_HOOK_TEXT", True))

    @property
    def vignette_strength(self) -> float:
        """
        Vignette darkening applied at the frame corners (0 = off, 1 = full black).
        Sourced from VIGNETTE_STRENGTH in page_config.py; defaults to 0.0.
        """
        try:
            val = float(self.page_cfg.get("VIGNETTE_STRENGTH", 0.0))
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.0

    @property
    def grain_intensity(self) -> float:
        """
        Film grain amplitude in pixel value units (±grain_intensity added to each pixel).
        Sourced from GRAIN_INTENSITY in page_config.py; defaults to 18.0.
        """
        try:
            return max(0.0, float(self.page_cfg.get("GRAIN_INTENSITY", 18.0)))
        except (TypeError, ValueError):
            return 18.0

    @property
    def enable_flicker(self) -> bool:
        """
        Subtle ±5 % brightness oscillation at ~0.10 s random intervals (torch/flame effect).
        Sourced from ENABLE_FLICKER in page_config.py; defaults to False.
        """
        return bool(self.page_cfg.get("ENABLE_FLICKER", False))

    @property
    def enable_light_rays(self) -> bool:
        """
        Animated warm-gold volumetric light beam column sweeping across the canvas.
        Sourced from ENABLE_LIGHT_RAYS in page_config.py; defaults to False.
        """
        return bool(self.page_cfg.get("ENABLE_LIGHT_RAYS", False))

    @property
    def enable_dust_particles(self) -> bool:
        """
        Floating dust/debris particles drifting upward (for ruin/cave environments).
        Sourced from ENABLE_DUST_PARTICLES in page_config.py; defaults to False.
        """
        return bool(self.page_cfg.get("ENABLE_DUST_PARTICLES", False))

    @property
    def enable_light_refraction(self) -> bool:
        """
        Subtle prismatic light-refraction glow (for glass/crystal subjects).
        Sourced from ENABLE_LIGHT_REFRACTION in page_config.py; defaults to False.
        """
        return bool(self.page_cfg.get("ENABLE_LIGHT_REFRACTION", False))

    @property
    def niche_disclaimer(self) -> str:
        """
        Optional niche-specific disclaimer injected into LLM system prompts.
        Sourced from NICHE_DISCLAIMER in page_config.py; defaults to empty string.
        """
        return str(self.page_cfg.get("NICHE_DISCLAIMER", "")).strip()

    @property
    def narrative_mode(self) -> str:
        """
        Script/prompt structure mode for reel narration.
        Sourced from NARRATIVE_MODE in page_config.py.
        Known values: 'investigative' | 'warrior_discipline' | 'psychology' (default).
        """
        raw = str(self.page_cfg.get("NARRATIVE_MODE", "")).strip().lower()
        if raw:
            return raw
        # Infer from known page IDs when unset
        pid = (self.page_id or "").lower()
        if pid == "ancient_knowledge":
            return "investigative"
        if pid == "master_mei":
            return "warrior_discipline"
        return "psychology"

    @property
    def reel_cta_text(self) -> str:
        """
        Spoken CTA line generated as a separate audio block after narration.
        Sourced from REEL_CTA_TEXT in page_config.py; empty = no CTA stitch.
        """
        return str(self.page_cfg.get("REEL_CTA_TEXT", "")).strip()

    @property
    def ambient_sfx_prompt(self) -> str:
        """
        ElevenLabs SFX prompt for ambient underlay.
        Sourced from AMBIENT_SFX_PROMPT in page_config.py; empty = engine default.
        """
        return str(self.page_cfg.get("AMBIENT_SFX_PROMPT", "")).strip()

    @property
    def ambient_volume(self) -> float:
        """
        Ambient BGM bed volume (0–1) mixed under voiceover.
        Sourced from AMBIENT_VOLUME in page_config.py.
        Master Mei music bed locked near 0.32 (clear voice separation).
        """
        try:
            val = float(self.page_cfg.get("AMBIENT_VOLUME", 0.32))
            if (self.page_id or "").lower() == "master_mei":
                # Absolute BGM mix (default 0.24) — under voice for narration clarity (−20% from 0.30)
                return max(0.10, min(0.55, val))
            return max(0.08, min(1.0, val))
        except (TypeError, ValueError):
            return 0.32

    @property
    def impact_sfx_volume(self) -> float:
        """Cinematic impact SFX gain at t=0 (Master Mei default 0.50)."""
        try:
            val = float(self.page_cfg.get("IMPACT_SFX_VOLUME", 0.50))
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.50

    @property
    def use_music_v2_bed(self) -> bool:
        """When True, prefer Eleven Music v2 composition over local ambient pad."""
        raw = self.page_cfg.get("USE_MUSIC_V2_BED", None)
        if raw is None:
            return (self.page_id or "").lower() == "master_mei"
        return bool(raw)

    @property
    def ambient_music_style(self) -> str:
        """``warrior`` (mei) or ``mystery`` (ancient_knowledge) music_prompt profile."""
        raw = str(self.page_cfg.get("AMBIENT_MUSIC_STYLE", "") or "").strip().lower()
        if raw:
            return raw
        return (
            "warrior"
            if (self.page_id or "").lower() == "master_mei"
            else "mystery"
        )

    @property
    def music_prompt_directive_path(self) -> "Path | None":
        """Optional per-page music_prompt directive file (absolute Path)."""
        from pathlib import Path as _P

        rel = str(self.page_cfg.get("MUSIC_PROMPT_DIRECTIVE_RELPATH", "") or "").strip()
        if not rel:
            # Convention: channels_config/<page>/prompts/music_prompt_directive.txt
            cand = (
                _P(__file__).resolve().parent
                / "channels_config"
                / (self.page_id or "")
                / "prompts"
                / "music_prompt_directive.txt"
            )
            return cand if cand.is_file() else None
        p = _P(rel)
        if not p.is_absolute():
            p = _P(__file__).resolve().parent / p
        return p if p.is_file() else None

    @property
    def atmosphere_sfx_volume(self) -> float:
        try:
            return max(
                0.10,
                min(0.60, float(self.page_cfg.get("ATMOSPHERE_SFX_VOLUME", 0.35))),
            )
        except (TypeError, ValueError):
            return 0.35

    @property
    def atmosphere_sfx_fade_in(self) -> float:
        try:
            return max(0.0, float(self.page_cfg.get("ATMOSPHERE_SFX_FADE_IN", 0.2)))
        except (TypeError, ValueError):
            return 0.2

    @property
    def impact_sfx_prompt(self) -> str:
        return str(
            self.page_cfg.get(
                "IMPACT_SFX_PROMPT",
                "Cinematic Braam, Dystopian Sub-Bass Heavy Drop",
            )
        ).strip()

    @property
    def ambient_duck_ratio(self) -> float:
        """Multiply ambient by this while voiceover plays (Master Mei default 0.70)."""
        try:
            val = float(self.page_cfg.get("AMBIENT_DUCK_RATIO", 0.70))
            return max(0.30, min(1.0, val))
        except (TypeError, ValueError):
            return 0.70

    @property
    def master_audio_gain(self) -> float:
        """Overall master mix gain after loudnorm (Master Mei default +15% = 1.15)."""
        try:
            val = float(self.page_cfg.get("MASTER_AUDIO_GAIN", 1.15))
            return max(1.0, min(1.50, val))
        except (TypeError, ValueError):
            return 1.15

    @property
    def voice_volume_gain(self) -> "float | None":
        """Linear VO gain override (e.g. 1.334 ≈ +2.5 dB). None → engine default."""
        raw = self.page_cfg.get("VOICE_VOLUME_GAIN", None)
        if raw is None:
            return None
        try:
            val = float(raw)
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None

    @property
    def ambient_sfx_gain_mul(self) -> "float | None":
        """Ambient/SFX multiplier after bed volume (e.g. ~1.995 for +3 dB over 1.4×)."""
        raw = self.page_cfg.get("AMBIENT_SFX_GAIN_MUL", None)
        if raw is None:
            return None
        try:
            val = float(raw)
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None

    @property
    def bgm_start_time(self) -> float:
        """Seconds before background music enters (Master Mei default 8.0)."""
        try:
            return max(0.0, float(self.page_cfg.get("BGM_START_TIME", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @property
    def bgm_fade_in_duration(self) -> float:
        """BGM fade-in duration in seconds (Master Mei default 2.5)."""
        try:
            return max(0.0, float(self.page_cfg.get("BGM_FADE_IN_DURATION", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @property
    def sfx_volume_gain_db(self) -> float:
        """Global ambient/SFX gain boost in dB (Master Mei default +2.5)."""
        try:
            return float(self.page_cfg.get("SFX_VOLUME_GAIN_DB", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @property
    def hook_environments(self) -> list:
        """
        Scenic environments for Master Mei Act-1 identity hook.
        Hash-picks across PRIMARY+SECONDARY outdoor pools (never a fixed cliff lock).
        """
        primary = self.page_cfg.get("HOOK_ENVIRONMENTS_PRIMARY", [])
        secondary = self.page_cfg.get("HOOK_ENVIRONMENTS_SECONDARY", [])
        pooled: list = []
        if isinstance(primary, list):
            pooled.extend(primary)
        if isinstance(secondary, list):
            pooled.extend(secondary)
        if pooled:
            return [str(t).strip() for t in pooled if t and str(t).strip()]
        raw = self.page_cfg.get("HOOK_ENVIRONMENTS", [])
        if not isinstance(raw, list):
            return []
        return [str(t).strip() for t in raw if t and str(t).strip()]

    @property
    def master_mei_visual_dna(self) -> str:
        """Mandatory elder-sage visual DNA string for Master Mei image prompts."""
        return str(self.page_cfg.get("MASTER_MEI_VISUAL_DNA", "")).strip()

    @property
    def sequence_force_avatar_off(self) -> bool:
        """When True, sequence/ECONOMIC_REEL images ignore avatar likeness refs."""
        return bool(self.page_cfg.get("SEQUENCE_FORCE_AVATAR_OFF", False))

    @property
    def avatar_image_weight(self) -> float:
        """Likeness strength 0–1 injected into Gemini reference prompt (IP-adapter analogue)."""
        try:
            return max(0.1, min(1.0, float(self.page_cfg.get("AVATAR_IMAGE_WEIGHT", 0.75))))
        except (TypeError, ValueError):
            return 0.75

    @property
    def forced_avatar_reference_path(self) -> "Path | None":
        """
        Absolute path to the page's mandatory avatar.png.

        For master_mei: ALWAYS resolves to
        channels_config/master_mei/avatar_reference/avatar.png (zero-hallucination lock).
        """
        page_id = (self.page_id or "").lower()
        if page_id == "master_mei":
            try:
                from agents.media.avatar_engine.mei_visual import resolve_master_mei_avatar_path
                p = resolve_master_mei_avatar_path(_ENGINE_ROOT)
                return p
            except Exception:
                pass
        rel = str(self.page_cfg.get("AVATAR_REFERENCE_PATH", "")).strip()
        if rel:
            p = Path(rel)
            if not p.is_absolute():
                p = _ENGINE_ROOT / rel
            return p
        if self.avatar_reference_png.is_file():
            return self.avatar_reference_png
        return None

    @property
    def reel_force_exact_duration(self) -> bool:
        """When True, sequence reel container is floored at REEL_DURATION (e.g. 80 s)."""
        return bool(self.page_cfg.get("REEL_FORCE_EXACT_DURATION", False))

    def _words_for_duration_runtime(self) -> int:
        """Runtime words_for_duration(self.reel_duration) — used when the
        page_cfg omits an explicit REEL_NARRATION_* value so ``--video-length``
        overrides scale the word budget instead of being pinned to the
        REEL_DURATION_TARGET_MIN import-time constant (the Round-6 bug)."""
        try:
            from config import words_for_duration as _wfd
            return int(_wfd(float(self.reel_duration)))
        except Exception:  # noqa: BLE001
            return 140

    @property
    def reel_narration_words(self) -> int:
        """Target narration word count for sequence TTS.

        Behaviour: if the page_cfg supplies an explicit REEL_NARRATION_WORDS
        value it wins (backwards-compat with pages that still pin a fixed
        word budget). Otherwise fall through to
        ``words_for_duration(self.reel_duration)`` at read time so
        ``--video-length`` (or any runtime duration override) scales the
        Gemini word target correctly.
        """
        raw = self.page_cfg.get("REEL_NARRATION_WORDS", None)
        if raw in (None, ""):
            return self._words_for_duration_runtime()
        try:
            return max(80, int(raw))
        except (TypeError, ValueError):
            return self._words_for_duration_runtime()

    @property
    def reel_narration_min_words(self) -> int:
        """Minimum acceptable narration word count before regenerate/fallback warn.

        Falls through to ``words_for_duration(self.reel_duration)`` when
        the page_cfg omits an explicit value.
        """
        raw = self.page_cfg.get("REEL_NARRATION_MIN_WORDS", None)
        if raw in (None, ""):
            return self._words_for_duration_runtime()
        try:
            return max(60, int(raw))
        except (TypeError, ValueError):
            return self._words_for_duration_runtime()

    @property
    def reel_narration_max_words(self) -> int:
        """Hard ceiling for narration trim.

        Falls through to ``words_for_duration(reel_duration) + 25 %`` of
        the requested duration (in words) when the page_cfg omits an
        explicit REEL_NARRATION_MAX_WORDS value.
        """
        raw = self.page_cfg.get("REEL_NARRATION_MAX_WORDS", None)
        if raw in (None, ""):
            try:
                from config import words_for_duration as _wfd
                dur = float(self.reel_duration)
                return int(_wfd(dur) + max(20.0, dur * 0.25))
            except Exception:  # noqa: BLE001
                return max(240, self.reel_narration_words)
        try:
            return max(
                self.reel_narration_words,
                int(raw),
            )
        except (TypeError, ValueError):
            return max(240, self.reel_narration_words)

    @property
    def strip_audio_tags_before_tts(self) -> bool:
        """Strip comedy bracket tags before ElevenLabs TTS (page-configurable)."""
        if "STRIP_AUDIO_TAGS_BEFORE_TTS" in self.page_cfg:
            return bool(self.page_cfg.get("STRIP_AUDIO_TAGS_BEFORE_TTS"))
        return False

    @property
    def tts_enable_ssml(self) -> bool:
        """Enable ElevenLabs SSML parsing (``<break time="…" />``) when True."""
        if "TTS_ENABLE_SSML" in self.page_cfg:
            return bool(self.page_cfg.get("TTS_ENABLE_SSML"))
        return (self.page_id or "").lower() == "master_mei"

    @property
    def reel_tail_pad_s(self) -> float:
        """
        Extra seconds appended after final audio so CTA/subtitles never clip.
        Sourced from REEL_TAIL_PAD_S; defaults to 1.0 (ancient_knowledge buffer).
        """
        try:
            return max(0.0, float(self.page_cfg.get("REEL_TAIL_PAD_S", 1.0)))
        except (TypeError, ValueError):
            return 1.0

    @property
    def ambient_audio_path(self) -> "Path | None":
        """
        Optional local ambient loop preferred over ElevenLabs SFX.
        Sourced from AMBIENT_AUDIO_RELPATH in page_config.py (relative to engine root).
        """
        rel = str(self.page_cfg.get("AMBIENT_AUDIO_RELPATH", "")).strip()
        if not rel:
            return None
        candidate = _ENGINE_ROOT / rel
        return candidate if candidate.is_file() else candidate  # path for existence check by caller

    @property
    def avatar_assets_dir(self) -> "Path | None":
        """
        Priority directory of cycleable avatar reference portraits.
        Sourced from AVATAR_ASSETS_DIR in page_config.py (relative to engine root).
        Falls back to channels_config/{page}/avatar_reference/ when unset.
        """
        rel = str(self.page_cfg.get("AVATAR_ASSETS_DIR", "")).strip()
        if rel:
            return _ENGINE_ROOT / rel
        return self.avatar_reference_dir

    def list_avatar_references(self) -> list:
        """
        Return sorted portrait paths for likeness cycling.
        Priority: AVATAR_ASSETS_DIR images, then channels_config avatar_reference/.
        """
        found: list = []
        seen: set = set()
        for folder in (self.avatar_assets_dir, self.avatar_reference_dir):
            if folder is None or not folder.is_dir():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                for p in sorted(folder.glob(ext)):
                    if p.is_file() and p.resolve() not in seen:
                        seen.add(p.resolve())
                        found.append(p)
        return found

    def cycle_avatar_reference(self, index: int = 0) -> "Path | None":
        """Pick avatar reference by index (wraps). None if no portraits exist."""
        refs = self.list_avatar_references()
        if not refs:
            return None
        return refs[index % len(refs)]

    @property
    def tts_narration_speed(self) -> float:
        """
        ElevenLabs narration speed multiplier.
        Prefer ELEVENLABS_VOICE_SETTINGS['speed'], else TTS_NARRATION_SPEED.
        Clamped to ElevenLabs practical range ~0.70–1.20 for short-form.
        """
        try:
            vs = self.page_cfg.get("ELEVENLABS_VOICE_SETTINGS", None)
            if isinstance(vs, dict) and vs.get("speed") is not None:
                val = float(vs["speed"])
            else:
                val = float(self.page_cfg.get("TTS_NARRATION_SPEED", 1.05))
            return max(0.70, min(1.20, val))
        except (TypeError, ValueError):
            return 1.05

    @property
    def elevenlabs_voice_settings(self) -> dict:
        """
        Optional ElevenLabs VoiceSettings overrides from page_config.
        Sourced from ELEVENLABS_VOICE_SETTINGS; empty dict = engine defaults.
        Keys: stability, similarity_boost, style, use_speaker_boost, speed.
        """
        raw = self.page_cfg.get("ELEVENLABS_VOICE_SETTINGS", None)
        if not isinstance(raw, dict):
            return {}
        out: dict = {}
        for key in ("stability", "similarity_boost", "style", "speed"):
            if key in raw:
                try:
                    out[key] = float(raw[key])
                except (TypeError, ValueError):
                    pass
        if "use_speaker_boost" in raw:
            out["use_speaker_boost"] = bool(raw["use_speaker_boost"])
        # Keep speed aligned with TTS_NARRATION_SPEED when only one is set
        if "speed" not in out:
            try:
                out["speed"] = float(self.page_cfg.get("TTS_NARRATION_SPEED", 1.05))
            except (TypeError, ValueError):
                pass
        return out

    @property
    def audio_behavior_tags(self) -> list:
        """
        Bracketed ElevenLabs behavioral tags for this page's voice
        (e.g. [chuckles], [cackles]). Sourced from AUDIO_BEHAVIOR_TAGS.
        """
        raw = self.page_cfg.get("AUDIO_BEHAVIOR_TAGS", [])
        if not isinstance(raw, list):
            return []
        return [str(t).strip() for t in raw if t and str(t).strip()]

    @property
    def image_model_override(self) -> "str | None":
        """
        Explicit image model ID override sourced from IMAGE_MODEL_OVERRIDE in
        page_config.py.  When set, this takes highest priority in main.py's
        img_model_id resolution — overrides both the nano-tier constant and the
        global economic flag.  Returns None when not configured.
        Invalid flash-lite SKUs are remapped to a live Imagen model.
        """
        val = (
            self.page_cfg.get("IMAGE_PRIMARY_CLI")
            or self.page_cfg.get("IMAGE_MODEL_OVERRIDE")
            or self.page_cfg.get("IMAGE_PRIMARY")
        )
        if not val:
            return None
        raw = str(val).strip()
        if not raw:
            return None
        try:
            import config as _cfg
            return _cfg.normalize_image_model_id(raw)
        except Exception:  # noqa: BLE001
            return raw

    @property
    def prompt_negative_terms(self) -> list:
        """
        List of words / phrases to strip from inherited atmosphere prompts.
        Used to prevent cross-page style contamination (e.g. 'graphite' leaking
        into ancient_knowledge's photorealistic prompts).
        Sourced from PROMPT_NEGATIVE_TERMS in page_config.py; defaults to [].
        """
        raw = self.page_cfg.get("PROMPT_NEGATIVE_TERMS", [])
        if not isinstance(raw, list):
            return []
        return [str(t).strip() for t in raw if t and str(t).strip()]

    @property
    def page_economic_brain_mode(self) -> "bool | None":
        """
        Page-level override for economic brain mode.
        Returns True / False if ECONOMIC_BRAIN_MODE is explicitly set in
        page_config.py, otherwise None (no override — CLI flag decides).
        """
        val = self.page_cfg.get("ECONOMIC_BRAIN_MODE", None)
        if val is None:
            return None
        return bool(val)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_page_context(
    page_id: str,
    avatar_mode: str = "ON",
    post_format: str = "IMAGE_AVATAR",
) -> PageContext:
    """
    Build and return a PageContext for the given page slug.

    Validates all three runtime flags and resolves filesystem paths.
    Loads page_config.py from channels_config/{page_id}/ if present.

    Parameters
    ----------
    page_id:
        One of: anna_protocol, master_mei, wonder_feed, down_dirty.
    avatar_mode:
        'ON' or 'OFF'.
    post_format:
        'IMAGE_AVATAR', 'IMAGE_QUOTE', or 'HYBRID_VIDEO'.

    Raises
    ------
    ValueError
        If any of the three arguments are not in their valid sets.
    """
    page_id = page_id.lower().strip()
    avatar_mode = avatar_mode.upper().strip()
    post_format = post_format.upper().strip()

    if page_id not in VALID_PAGES:
        raise ValueError(
            f"Unknown --page '{page_id}'. "
            f"Valid options: {', '.join(VALID_PAGES)}"
        )
    if avatar_mode not in VALID_AVATAR_MODES:
        raise ValueError(
            f"Unknown --avatar '{avatar_mode}'. "
            f"Valid options: {', '.join(VALID_AVATAR_MODES)}"
        )
    if post_format not in VALID_FORMATS:
        raise ValueError(
            f"Unknown --format '{post_format}'. "
            f"Valid options: {', '.join(VALID_FORMATS)}"
        )

    root = _resolve_channels_config_root()
    page_dir = root / page_id
    if not page_dir.is_dir() and _LEGACY_PAGES_CONFIG_ROOT.is_dir():
        legacy = _LEGACY_PAGES_CONFIG_ROOT / page_id
        if legacy.is_dir():
            page_dir = legacy
    from utils.pipeline_paths import (
        channel_store_dir,
        page_assets_dir,
        page_library_dir,
        page_outputs_dir,
    )

    outputs_dir = page_outputs_dir(page_id, create=True)

    # Ensure outputs directory exists at load time (OUTPUT_PATH / page_id).
    page_assets_dir(page_id, create=True)
    page_library_dir(page_id, create=True)
    (outputs_dir / "postplanner").mkdir(parents=True, exist_ok=True)
    store_dir = channel_store_dir(page_id, create=True)
    try:
        from modules.durable_store import restore_channel_state

        restore_channel_state(page_id)
    except Exception as exc:  # noqa: BLE001 — G: offline must not abort a run
        import logging

        logging.getLogger(__name__).warning(
            "Could not restore %s store from G: (%s)", page_id, exc
        )

    # Ensure brand asset subfolders exist inside the page config directory.
    (page_dir / "avatar_reference").mkdir(parents=True, exist_ok=True)
    (page_dir / "voice_reference").mkdir(parents=True, exist_ok=True)
    (page_dir / "logo").mkdir(parents=True, exist_ok=True)

    page_cfg = _load_page_config(page_dir, page_id)

    return PageContext(
        page_id=page_id,
        avatar_mode=avatar_mode,
        post_format=post_format,
        page_dir=page_dir,
        persona_dna_path=page_dir / "persona_dna.py",
        master_dna_path=page_dir / "master_dna.json",
        avatar_reference_dir=page_dir / "avatar_reference",
        voice_reference_dir=page_dir / "voice_reference",
        logo_dir=page_dir / "logo",
        product_reference_dir=page_dir / "product_reference",
        outputs_dir=outputs_dir,
        page_cfg=page_cfg,
        store_dir=store_dir,
    )


def _load_page_config(page_dir: Path, page_id: str) -> dict[str, Any]:
    """
    Dynamically import channels_config/{page_id}/page_config.py and return
    its public symbols as a plain dict. Returns empty dict if file is absent.
    """
    config_py = page_dir / "page_config.py"
    if not config_py.is_file():
        return {}

    module_name = f"channels_config.{page_id}.page_config"
    spec = importlib.util.spec_from_file_location(module_name, config_py)
    if spec is None or spec.loader is None:
        return {}

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "Failed to load page_config for '%s': %s", page_id, exc
        )
        return {}

    return {k: v for k, v in vars(mod).items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Default avatar mode resolution (per-page preference)
# ---------------------------------------------------------------------------

def resolve_default_avatar_mode(page_cfg: dict[str, Any]) -> str:
    """Return the page's preferred avatar mode ('ON' or 'OFF')."""
    raw = str(page_cfg.get("DEFAULT_AVATAR_MODE", "ON")).upper().strip()
    return raw if raw in VALID_AVATAR_MODES else "ON"


def resolve_default_format(page_cfg: dict[str, Any]) -> str:
    """Return the page's preferred post format."""
    raw = str(page_cfg.get("DEFAULT_FORMAT", "IMAGE_AVATAR")).upper().strip()
    return raw if raw in VALID_FORMATS else "IMAGE_AVATAR"
