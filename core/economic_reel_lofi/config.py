# -*- coding: utf-8 -*-
"""ECONOMIC_REEL_LOFI — duration, style, and per-channel assembly config."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

# ── Scene timing ────────────────────────────────────────────────────────────
SCENE_DURATION_S: float = 3.0
# Thematic default: 9 beats × 3.0s = 27.0s (loss baseline). Writer may emit 8–9;
# anything above 9 is clamped unless --duration is explicitly longer than 27s.
LOCK_FIXED_BEAT_DURATION: bool = False
# Never hard-trim finished VO. If a beat's VO would crowd the 3.0s slot,
# extend instead of compressing speech. Inter-line breath is real silence
# in the VO concat (not MoviePy slot padding that BGM fills).
NEVER_TRIM_VOICEOVER: bool = True
VO_SLOT_PAD_S: float = 0.12
VO_SLOT_BREATH_S: float = 0.35  # retired as the gap mechanism; kept for logs
VO_INTERLINE_SILENCE_S: float = 0.60  # mandatory music-only breath between beats
MIN_SCENE_SLOT_S: float = 2.5
MAX_SCENE_SLOT_S: float = 4.0
# Keep encoder pad on each TTS file so the timeline segment equals file
# duration (speed=0.80 line 1 ≈ 2.93s). Trimming edges was collapsing that
# to ~2.62s and hiding the 0.80 take behind a "speech-only" slot.
VO_TRIM_TTS_EDGES: bool = False
# BGM pause duck is relative to BGM_VOLUME (see BGM_GAP_DUCK_DB). Do not use a
# near-silence absolute floor — that reads as a hole even when timing is exact.
DEFAULT_DURATION_S: int = 27  # thematic default total; writer target = 9 scenes
MIN_DURATION_S: int = 15
MAX_DURATION_S: int = 90
MIN_SCENES: int = 8
MAX_SCENES: int = 30
THEMATIC_DEFAULT_SCENES: int = 9
THEMATIC_MAX_SCENES: int = 9
# Human hold after Stage 1 and Stage 3. False auto-passes both gates.
# Gate 1 protects TTS spend; Gate 2 protects image spend.
REVIEW_REQUIRED: bool = True

# ── Script / validation ─────────────────────────────────────────────────────
# Fast production writer: one Gemini call, no LLM judge or repair loop.
LOFI_EMOTIONAL_WRITER_MODEL: str = str(
    os.environ.get("LOFI_EMOTIONAL_WRITER_MODEL") or "models/gemini-3.8-flash"
).strip()
LOFI_ENABLE_STRICT_JUDGE: bool = False


def strict_judge_enabled() -> bool:
    """Read the strict legacy writer switch at call time for CLI/env tests."""
    raw = str(os.environ.get("LOFI_ENABLE_STRICT_JUDGE") or "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return bool(LOFI_ENABLE_STRICT_JUDGE)


MAX_CAPTION_CHARS: int = 84  # room for natural micro-philosophical prose
MAX_CAPTION_WORDS: int = 12
# Hard per-beat spoken cap. Direct 9-line compose must satisfy this itself.
THEMATIC_MAX_CAPTION_CHARS: int = 84
THEMATIC_MAX_CAPTION_WORDS: int = 7
# Writer target for beat 1 only — not a still-duration override. The still
# holds whatever the rendered hook VO actually lasts.
HOOK_LINE_TARGET_WORDS: int = 4
BEAT_TARGET_MAX_WORDS: int = 6
HOOK_LINE_TARGET_S: float = 3.0
THEMATIC_ARC_ID: str = "thematic_arc"
# Measured 2026-08-23 from shipped caption_timing at LOFI_VOICE_SPEED=0.80:
# loneliness 60w / 19.40s = 3.09; evening five-script mean ≈ 3.01 w/s.
MEASURED_SPEECH_WPS: float = 3.0
MEASURED_SPEECH_AT_VOICE_SPEED: float = 0.80
MONOLOGUE_DURATION_SAFETY: float = 0.90
# Writer-side spoken pace. Distance-pilot VO at speed 0.80 ran ~2.7–3.1 w/s
# (162–186 wpm) on long literary beats. 150 wpm is slightly conservative so
# TTS has headroom inside the declared beat window.
NARRATION_WPM: float = 120.0
BEAT_WORD_BUDGET_SLACK: int = 1  # 3s target 6, hard ceiling 7 words
TTS_DURATION_TOLERANCE: float = 0.15  # post-TTS vs declared duration_s
LINE_REWRITE_MAX_PASSES: int = 2
# Per-beat validator repair budget. These attempts do not consume
# SCRIPT_MAX_RETRIES, which remains reserved for structural rewrites.
MAX_LINE_REPAIRS: int = 2
# Isolated TTS overrun: auto-bump one beat's duration_s instead of sign-off.
TTS_AUTO_BUMP_MAX_BEATS: int = 1
TTS_AUTO_BUMP_MAX_FRAC: float = 0.35  # overrun ≤ 35% of original declared
TTS_AUTO_BUMP_MAX_ABS_S: float = 1.5  # bump target ≤ +1.5s
TTS_AUTO_BUMP_TOTAL_FRAC: float = 0.10  # total stays within 10% of request
# Wonder Feed + Momma Circle production default (object arcs remain in the bank).
DEFAULT_ARC_BY_MODULE: dict[str, str] = {
    "relationship": "thematic_arc",
    "parenting": "thematic_arc",
}
CAPTION_HOLD_S: float = 0.55  # breath after the last word before a cut
AUTHORITY_QUOTE_PROBABILITY: float = 0.0  # story monologue, not quote-maxims
SCRIPT_MAX_RETRIES: int = 3
DEDUP_SIMILARITY_THRESHOLD: float = 0.85
# Auto-reject if a script copies stored wf1–4 / mc1–3 transcripts.
REFERENCE_OVERLAP_THRESHOLD: float = 0.80
# Total tries per scene = IMAGE_MAX_RETRIES_PER_SCENE + 1.
# HARD CAP: never more than 2 image+critic attempts per beat per run.
IMAGE_MAX_RETRIES_PER_SCENE: int = 1
IMAGE_ATTEMPTS_PER_BEAT: int = 2
IMAGE_ATTEMPTS_PER_SCENE: int = 2
# Regen cycles in manual_overrides history. A 3rd cycle needs force=True.
MAX_REGEN_CYCLES_PER_BEAT: int = 2
# Stop the episode once image calls would exceed this × beat count.
IMAGE_CALL_BUDGET_MULT: float = 2.0


def image_call_budget(n_beats: int) -> int:
    n = max(1, int(n_beats))
    return max(n, int(math.ceil(n * IMAGE_CALL_BUDGET_MULT)))


def clamp_attempt_budget(requested: int | None) -> int:
    """Never more than IMAGE_ATTEMPTS_PER_BEAT, even if a caller asks for more."""
    cap = max(1, int(IMAGE_ATTEMPTS_PER_BEAT))
    if requested is None:
        n = int(IMAGE_ATTEMPTS_PER_SCENE or cap)
    else:
        n = int(requested)
    return max(1, min(n, cap))


def log_image_budget(*, used: int, budget: int, n_beats: int = 0) -> None:
    over = int(used) > int(budget)
    tag = "OVERRUN" if over else "ok"
    print("=" * 64)
    print(
        f"[LOFI budget] image_calls={int(used)}  "
        f"image_call_budget={int(budget)}  beats={int(n_beats)}  {tag}"
    )
    print("=" * 64)


# Core Mode (a) retrieval — not quote/philosopher mode (b)
CORE_DETAIL_COUNT: int = 3
REQUIRE_ANCHOR_OBJECT: bool = True
REQUIRE_BEAT_CONCRETENESS: bool = True
THEMATIC_HOOK_TYPES: frozenset[str] = frozenset(
    {"bold_claim", "question", "statistic", "definition", "rhetorical_question"}
)


def is_thematic_arc(arc_id: str | None) -> bool:
    return str(arc_id or "").strip() == THEMATIC_ARC_ID


def caption_limits(arc_id: str | None = None) -> tuple[int, int]:
    """Return (max_words, max_chars) for this arc."""
    if is_thematic_arc(arc_id):
        return thematic_caption_limits()
    return MAX_CAPTION_WORDS, MAX_CAPTION_CHARS


def tts_voice_id() -> str:
    return str(TTS_VOICE_ID or LOFI_VOICE_ID or "").strip()


def tts_model() -> str:
    return str(TTS_MODEL or LOFI_TTS_MODEL or "eleven_multilingual_v2").strip()


def tts_speed() -> float:
    """Production ElevenLabs speed. Valid range 0.7–1.2. Read at call time."""
    return float(TTS_SPEED if TTS_SPEED is not None else LOFI_VOICE_SPEED)


def bgm_gap_vs_speech_db() -> float:
    """Target mix-RMS distance: gap should sit this far below mid-speech mix."""
    return float(getattr(globals(), "BGM_GAP_VS_SPEECH_DB", 10.0))


def speech_words_per_sec(voice_speed: float | None = None) -> float:
    """Spoken words/sec at the current (or given) ElevenLabs voice_speed."""
    speed = float(voice_speed if voice_speed is not None else tts_speed())
    ref = float(MEASURED_SPEECH_AT_VOICE_SPEED) or 0.80
    return float(MEASURED_SPEECH_WPS) * (speed / ref)


def monologue_duration_s(
    *,
    duration_s: int | float | None = None,
    scene_count: int | None = None,
) -> float:
    """Writer runtime budget: explicit duration, else scenes × beat length."""
    if duration_s is not None:
        return float(duration_s)
    n = int(scene_count if scene_count is not None else THEMATIC_DEFAULT_SCENES)
    return float(n) * float(SCENE_DURATION_S)


def monologue_word_budget(
    *,
    duration_s: int | float | None = None,
    scene_count: int | None = None,
    voice_speed: float | None = None,
) -> dict[str, Any]:
    """
    Spoken-word target for one reel: duration × measured rate × safety.

    Default 27s × 3.0 w/s × 0.90 ≈ 73 words (band 65–75).
    """
    n = int(scene_count if scene_count is not None else THEMATIC_DEFAULT_SCENES)
    dur = monologue_duration_s(duration_s=duration_s, scene_count=n)
    rate = speech_words_per_sec(voice_speed)
    target = dur * rate * float(MONOLOGUE_DURATION_SAFETY)
    lo = max(1, int(round(target * 0.89)))
    hi = max(lo, int(round(target * 1.03)))
    return {
        "duration_s": round(dur, 3),
        "scene_count": n,
        "voice_speed": float(
            voice_speed if voice_speed is not None else tts_speed()
        ),
        "words_per_sec": round(rate, 3),
        "safety": float(MONOLOGUE_DURATION_SAFETY),
        "target": int(round(target)),
        "min_words": lo,
        "max_words": hi,
    }


def monologue_word_range(
    *,
    duration_s: int | float | None = None,
    scene_count: int | None = None,
    voice_speed: float | None = None,
) -> tuple[int, int]:
    rec = monologue_word_budget(
        duration_s=duration_s,
        scene_count=scene_count,
        voice_speed=voice_speed,
    )
    return int(rec["min_words"]), int(rec["max_words"])


def estimated_spoken_duration_s(
    word_count: int,
    *,
    voice_speed: float | None = None,
) -> float:
    rate = speech_words_per_sec(voice_speed)
    if rate <= 0:
        return 0.0
    return round(float(word_count) / rate, 2)


def narration_wpm() -> float:
    return float(NARRATION_WPM)


def narration_wps() -> float:
    return float(NARRATION_WPM) / 60.0


def hook_line_brevity_clause(*, paraphrase: bool = False) -> str:
    """Shared writer constraint: compact opening, not a fixed still hold."""
    words = int(HOOK_LINE_TARGET_WORDS)
    max_words = int(BEAT_TARGET_MAX_WORDS)
    secs = float(HOOK_LINE_TARGET_S)
    base = (
        f"HOOK LINE (beat 1): target {words}–{max_words} words / roughly "
        f"{secs:.0f}s of natural spoken pace. Open with emotional tension, "
        "clarity, and a recognisable human truth. This is a spoken-line target "
        "only; still duration follows the rendered VO, not this estimate."
    )
    if paraphrase:
        return (
            base
            + " Prefer a shorter first spoken sentence when that does not "
            "break a protected anaphora/refrain or the light-reword contract."
        )
    return base


def beat_word_budget(duration_s: float | None = None) -> int:
    """Floor words that fit ``duration_s`` at NARRATION_WPM. 3.0s → 7."""
    dur = float(duration_s if duration_s is not None else SCENE_DURATION_S)
    return max(1, math.floor(dur / 60.0 * float(NARRATION_WPM)))


def beat_word_ceiling(duration_s: float | None = None) -> int:
    """Hard per-beat spoken cap: budget + slack. 3.0s → 9."""
    return beat_word_budget(duration_s) + max(0, int(BEAT_WORD_BUDGET_SLACK))


def max_beats_for_duration(duration_s: float | None = None) -> int:
    """How many SCENE_DURATION_S beats fit the requested total. Default 27s → 9."""
    dur = float(duration_s if duration_s is not None else DEFAULT_DURATION_S)
    n = int(math.floor(dur / float(SCENE_DURATION_S) + 1e-9))
    return max(1, min(int(MAX_SCENES), n))


def declared_duration_s(*, scene_count: int | None = None) -> float:
    n = int(scene_count if scene_count is not None else THEMATIC_DEFAULT_SCENES)
    return round(float(n) * float(SCENE_DURATION_S), 3)


def vo_duration_overrun(
    vo_dur_s: float,
    *,
    duration_s: float | None = None,
    tolerance: float | None = None,
) -> bool:
    """True when rendered speech exceeds declared beat length by more than tolerance."""
    declared = float(duration_s if duration_s is not None else SCENE_DURATION_S)
    tol = float(TTS_DURATION_TOLERANCE if tolerance is None else tolerance)
    return float(vo_dur_s or 0.0) > declared * (1.0 + max(0.0, tol))


def ceil_half_second(duration_s: float) -> float:
    """Next 0.5s at or above ``duration_s``. 3.84 → 4.0."""
    return round(math.ceil(float(duration_s) * 2.0 - 1e-9) / 2.0, 3)


def apply_isolated_tts_duration_bumps(
    overruns: list[dict[str, Any]],
    *,
    requested_total_s: float,
    n_beats: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """Auto-bump a single small TTS overrun after the word-budget gate.

    ``overruns`` items: index (0-based), vo_dur, duration_s, tightened.
    On success, each applied note includes ``bumped_s`` and the item's
    ``duration_s`` is updated. Returns (applied, stop_reason).
    """
    if not overruns:
        return [], None
    if len(overruns) > int(TTS_AUTO_BUMP_MAX_BEATS):
        bits = ", ".join(
            f"beat {int(r.get('index', 0)) + 1} {float(r.get('vo_dur') or 0):.2f}s/"
            f"{float(r.get('duration_s') or SCENE_DURATION_S):.1f}s"
            for r in overruns
        )
        return [], (
            f"{len(overruns)} beats need a TTS duration bump ({bits}) — "
            "not an isolated edge case; sign-off required"
        )
    rec = overruns[0]
    if not rec.get("tightened"):
        return [], (
            f"beat {int(rec.get('index', 0)) + 1} TTS-overran before a tightening "
            "pass — rewrite first; auto-bump is not a shortcut around the word gate"
        )
    original = float(rec.get("duration_s") or SCENE_DURATION_S)
    vo = float(rec.get("vo_dur") or 0.0)
    if original <= 0:
        return [], f"beat {int(rec.get('index', 0)) + 1} has no declared duration"
    frac = (vo / original) - 1.0
    bumped = ceil_half_second(max(vo, original))
    if bumped < vo:
        bumped = round(vo + 0.05, 3)
    abs_delta = bumped - original
    if frac > float(TTS_AUTO_BUMP_MAX_FRAC) + 1e-9:
        return [], (
            f"beat {int(rec.get('index', 0)) + 1} TTS {vo:.2f}s is "
            f"{frac * 100:.0f}% over {original:.1f}s (auto-bump cap "
            f"{float(TTS_AUTO_BUMP_MAX_FRAC) * 100:.0f}%) — sign-off required"
        )
    if abs_delta > float(TTS_AUTO_BUMP_MAX_ABS_S) + 1e-9:
        return [], (
            f"beat {int(rec.get('index', 0)) + 1} bump {original:.1f}s → {bumped:.1f}s "
            f"exceeds +{float(TTS_AUTO_BUMP_MAX_ABS_S):.1f}s — sign-off required"
        )
    n = max(1, int(n_beats))
    new_total = float(requested_total_s) - original + bumped
    cap_total = float(requested_total_s) * (1.0 + float(TTS_AUTO_BUMP_TOTAL_FRAC))
    if new_total > cap_total + 1e-9:
        return [], (
            f"beat {int(rec.get('index', 0)) + 1} bump would make total {new_total:.1f}s "
            f"vs requested {float(requested_total_s):.1f}s "
            f"(cap {cap_total:.1f}s) — sign-off required"
        )
    rec["duration_s"] = bumped
    rec["bumped_s"] = bumped
    rec["original_s"] = original
    note = {
        "beat": int(rec.get("index", 0)) + 1,
        "index": int(rec.get("index", 0)),
        "original_s": original,
        "bumped_s": bumped,
        "vo_dur_s": round(vo, 3),
        "requested_total_s": float(requested_total_s),
        "new_total_s": round(new_total, 3),
        "n_beats": n,
        "reason": "isolated TTS overrun after tightening; auto-approved duration bump",
    }
    print(
        f"[LOFI duration-bump] AUTO beat={note['beat']} {original:.1f}s → {bumped:.1f}s "
        f"(VO {vo:.2f}s) new_total={new_total:.1f}s vs requested {requested_total_s:.1f}s"
    )
    return [note], None


def thematic_caption_limits(
    *,
    duration_s: int | float | None = None,
    scene_count: int | None = None,
) -> tuple[int, int]:
    """Hard per-beat spoken cap: 12 words / 84 characters."""
    _ = (duration_s, scene_count)
    return THEMATIC_MAX_CAPTION_WORDS, THEMATIC_MAX_CAPTION_CHARS


def max_sentence_words(
    *,
    duration_s: int | float | None = None,
    scene_count: int | None = None,
) -> int:
    """Hard max words per spoken sentence: two caption beats."""
    max_w, _ = thematic_caption_limits(
        duration_s=duration_s, scene_count=scene_count
    )
    return int(max_w) * 2


VALID_MODULES: frozenset[str] = frozenset({"relationship", "parenting"})
MODULE_PAGE_GATES: dict[str, frozenset[str]] = {
    "relationship": frozenset({"momma_circle", "wonder_feed"}),
    "parenting": frozenset({"momma_circle", "wonder_feed"}),
}
VALID_PAGES: frozenset[str] = frozenset({"momma_circle", "wonder_feed"})

# Riso library is the primary prompt source (verbatim). This base is fallback only
# when a prompt is missing style language — never used to override library palettes.
LOFI_STYLE_BASE: str = (
    "hand-inked gouache illustration, rich saturated nostalgic color palette, "
    "bold flat color blocks with fine hand-inked linework, grainy halftone texture, "
    "hard-edged color shading, vertical 9:16 composition"
)
# Verbatim riso prompts — do not prepend mood lighting that remaps palette.
USE_RISO_PROMPT_LIBRARY: bool = True
RISO_PROMPT_LIBRARY_REL: str = "core/economic_reel_lofi/store/riso_prompt_library_v2.json"
# Parallel v2 identity bank (does NOT overwrite the live riso library file).
USE_VISUAL_IDENTITY_V2: bool = True
VISUAL_IDENTITY_V2_REL: str = (
    "core/economic_reel_lofi/store/visual_identity_bank_v2.json"
)
# OFF — duotone/LUT crushed scene palettes (blue wash / TV-static first frames).
LOFI_APPLY_GRADING: bool = False

# Per-scene palette rotation + matching grading duotone pairs.
# Avoid "light source" / glow / backlight wording — those bias Flux toward radial skies.
# Keep each palette phrase short so base+palette+scene stays under ~300 chars.
LIGHTING_MOODS: tuple[dict[str, Any], ...] = (
    {
        "id": "amber_dusk",
        "lighting": "flat warm amber and teal palette",
        "shadow": (28, 72, 88),
        "highlight": (250, 185, 105),
    },
    {
        "id": "moonlit",
        "lighting": "flat cool blue and navy palette",
        "shadow": (18, 32, 78),
        "highlight": (190, 210, 235),
    },
    {
        "id": "overcast",
        "lighting": "flat sage grey and cream palette",
        "shadow": (42, 48, 58),
        "highlight": (220, 218, 210),
    },
    {
        "id": "golden_hour",
        "lighting": "flat gold and plum palette",
        "shadow": (55, 42, 62),
        "highlight": (255, 205, 95),
    },
    {
        "id": "indigo_night",
        "lighting": "flat indigo and pale lavender palette",
        "shadow": (16, 18, 58),
        "highlight": (165, 175, 225),
    },
)

# Backward-compat alias: base + first mood lighting (not used for generation).
LOFI_STYLE_PREFIX: str = f"{LOFI_STYLE_BASE}, {LIGHTING_MOODS[0]['lighting']}"

LOFI_NEGATIVE_PROMPT: str = (
    "photorealistic, photograph, 3d render, cgi, smooth vector art, "
    "radial glow, volumetric lighting, soft atmospheric haze, gradient sky wash, "
    "blown highlights, white bloom, halo, overexposed whites, blown-out chest, "
    "collar bloom, vignette bleed, overbright center wash, "
    "painterly blended lighting, monochromatic, grayscale, silhouette only, no linework, "
    "flat graphic, solid color field, empty poster, no outlines, abstract blob, "
    "text, watermark, logo, typography, subtitles, ui, letters, words, "
    "readable text, cursive, signage, poster type, newspaper, captions, "
    "deformed hands, extra fingers, garbled text, nsfw, nude, explicit"
)
# Style-owned retry guards (riso_retro_flat_v4). Line quality only —
# never "detailed interior objects" or other clutter content.
from core.economic_reel_lofi.style_modules import get_active_style as _get_active_style

_ACTIVE_STYLE = _get_active_style()
LOFI_PROMPT_EXPOSURE_GUARD: str = _ACTIVE_STYLE.exposure_guard
LOFI_PROMPT_LINEWORK_GUARD: str = _ACTIVE_STYLE.linework_guard

# ── Active FLUX Dev backend ─────────────────────────────────────────────────
LOFI_DEV_IMAGE_MODEL: str = _ACTIVE_STYLE.model
LOFI_DEV_IMAGE_STEPS: int = int(_ACTIVE_STYLE.steps)
LOFI_DEV_GUIDANCE_SCALE: float = float(_ACTIVE_STYLE.guidance_scale)
DEFAULT_VISUAL_IDENTITY_PROFILE: str = _ACTIVE_STYLE.profile_name
# FLUX.1-dev is the production default. flux2-dev remains an explicit override.
LOFI_FLUX_BACKEND: str = str(os.environ.get("LOFI_FLUX_BACKEND") or "dev").strip()
LOFI_FLUX2_DEV_MODEL: str = "black-forest-labs/FLUX.2-dev"
LOFI_FLUX2_DEV_STEPS: int = 24
LOFI_FLUX2_DEV_GUIDANCE_SCALE: float = 5.5


def flux_backend() -> str:
    return str(
        os.environ.get("LOFI_FLUX_BACKEND") or LOFI_FLUX_BACKEND or "dev"
    ).strip().lower()


def image_provider() -> str:
    provider = str(os.environ.get("LOFI_IMAGE_PROVIDER") or "together").strip().lower()
    return provider if provider in {"together", "gemini"} else "together"


def uses_flux2_dev() -> bool:
    return flux_backend() in {
        "flux2",
        "flux2-dev",
        "flux.2-dev",
        "flux2dev",
        "deepinfra_flux2dev",
    }


def uses_flux_dev() -> bool:
    if uses_flux2_dev():
        return True
    return flux_backend() in {"dev", "flux_dev", "flux.1-dev"}


# Exact 9:16 (0.5625), both axes ÷16, 0.92 MP — under Flux Dev's ~1 MP native
# budget. 1080×1920 delivery is a uniform 1.5× scale (no crop, no stretch).
# Previous 768×1344 was 0.5714 and forced ImageOps.fit to crop ~1.6%.
LOFI_IMAGE_WIDTH: int = 720
LOFI_IMAGE_HEIGHT: int = 1280
LOFI_IMAGE_WIDTH_LEGACY: int = 768
LOFI_IMAGE_HEIGHT_LEGACY: int = 1344
# Mean wall-clock s/image observed on DeepInfra Dev @ 768×1344 × 20 steps
# (2026-08-22 hope run, ~11 calls / ~200 s). Used only to scale a time guess.
LOFI_DEV_SEC_PER_IMAGE_LEGACY: float = 18.0

# Bump when the locked look changes (silhouette, negatives, profile).
STILL_STYLE_VERSION: str = _ACTIVE_STYLE.still_style_version


def current_still_style_tag() -> str:
    """Identity stamped on every still this process writes."""
    if image_provider() == "gemini":
        return f"gemini_flash/gemini/{STILL_STYLE_VERSION}"
    if uses_flux2_dev():
        return f"{DEFAULT_VISUAL_IDENTITY_PROFILE}/flux2-dev/{STILL_STYLE_VERSION}"
    if uses_flux_dev():
        return (
            f"{DEFAULT_VISUAL_IDENTITY_PROFILE}/dev/{STILL_STYLE_VERSION}"
        )
    return f"schnell_live/schnell/{STILL_STYLE_VERSION}"


def still_style_sidecar_path(image_path: Path) -> Path:
    p = Path(image_path)
    return p.with_name(f"{p.stem}.style.json")


def write_still_style_sidecar(
    image_path: Path,
    *,
    run_id: str = "",
    reused: bool = False,
    style_tag: str | None = None,
) -> dict[str, Any]:
    rec = {
        "style_tag": str(style_tag or current_still_style_tag()),
        "visual_identity_profile": (
            DEFAULT_VISUAL_IDENTITY_PROFILE if uses_flux_dev() else "schnell_live"
        ),
        "flux_backend": (
            "flux2-dev" if uses_flux2_dev() else ("dev" if uses_flux_dev() else "schnell")
        ),
        "style_version": STILL_STYLE_VERSION,
        "pipeline_run": run_id or Path(image_path).parent.name,
        "reused": bool(reused),
        "gen_width": LOFI_IMAGE_WIDTH,
        "gen_height": LOFI_IMAGE_HEIGHT,
        "aspect": round(LOFI_IMAGE_WIDTH / LOFI_IMAGE_HEIGHT, 6),
    }
    src = Path(image_path)
    if src.is_file():
        try:
            from PIL import Image as PILImage

            with PILImage.open(src) as im:
                rec["gen_width"], rec["gen_height"] = int(im.size[0]), int(im.size[1])
                if rec["gen_height"]:
                    rec["aspect"] = round(rec["gen_width"] / rec["gen_height"], 6)
        except Exception:  # noqa: BLE001
            pass
    side = still_style_sidecar_path(image_path)
    side.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def read_still_style_sidecar(image_path: Path) -> dict[str, Any] | None:
    side = still_style_sidecar_path(image_path)
    if not side.is_file():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def still_style_tag_of(image_path: Path) -> str | None:
    rec = read_still_style_sidecar(image_path)
    if not rec:
        return None
    tag = str(rec.get("style_tag") or "").strip()
    return tag or None


def allow_mixed_era_assemble() -> bool:
    return str(os.environ.get("LOFI_ALLOW_MIXED_ERA") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def lofi_image_cost_per_call_usd(
    width: int | None = None,
    height: int | None = None,
    schnell_steps: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Per-image USD for the backend this process will actually POST.

    Dev live path bills DeepInfra FLUX-1-dev
    ($0.009 × w/1024 × h/1024 × steps/25), not Together's $0.025 flat
    (Together serverless Dev is unavailable). Schnell uses DeepInfra's
    $0.0005 × megapixel × steps formula.
    """
    from agents.media.providers.together_image import (
        estimate_deepinfra_dev_cost_usd,
        estimate_deepinfra_schnell_cost_usd,
        estimate_together_image_cost,
    )

    width = int(width if width is not None else LOFI_IMAGE_WIDTH)
    height = int(height if height is not None else LOFI_IMAGE_HEIGHT)

    if image_provider() == "gemini":
        try:
            import app_config

            model = "models/gemini-3.1-flash-image"
            usd = float(app_config.estimate_gemini_image_usd(model))
        except Exception:
            model = "models/gemini-3.1-flash-image"
            usd = 0.005
        return usd, {
            "backend": "flash-image",
            "provider": "google",
            "model": model,
            "steps": None,
            "usd_per_image": usd,
            "formula": "configured Gemini Flash image estimate",
        }

    if uses_flux2_dev():
        steps = int(LOFI_FLUX2_DEV_STEPS)
        usd = estimate_together_image_cost(LOFI_FLUX2_DEV_MODEL)
        return usd, {
            "backend": "flux2-dev",
            "provider": "together",
            "model": LOFI_FLUX2_DEV_MODEL,
            "steps": steps,
            "guidance_scale": LOFI_FLUX2_DEV_GUIDANCE_SCALE,
            "allow_lora": False,
            "usd_per_image": usd,
            "formula": "Together serverless flat per-image price",
            "note": "Together.ai FLUX.2-dev direct request.",
        }
    if uses_flux_dev():
        steps = int(LOFI_DEV_IMAGE_STEPS)
        usd = estimate_deepinfra_dev_cost_usd(width, height, steps)
        return usd, {
            "backend": "dev",
            "provider": "deepinfra",
            "model": LOFI_DEV_IMAGE_MODEL,
            "steps": steps,
            "guidance_scale": LOFI_DEV_GUIDANCE_SCALE,
            "allow_lora": False,
            "usd_per_image": usd,
            "formula": "0.009*(w/1024)*(h/1024)*(steps/25)",
            "together_flat_usd": estimate_together_image_cost(LOFI_DEV_IMAGE_MODEL),
            "note": (
                "Together serverless Flux Dev is unavailable; "
                "billed DeepInfra FLUX-1-dev formula."
            ),
        }
    steps = int(schnell_steps or 4)
    usd = estimate_deepinfra_schnell_cost_usd(width, height, steps)
    return usd, {
        "backend": "schnell",
        "provider": "deepinfra",
        "model": "black-forest-labs/FLUX-1-schnell",
        "steps": steps,
        "usd_per_image": usd,
        "formula": "0.0005*(w/1024)*(h/1024)*steps",
        "note": "DeepInfra Schnell dashboard formula.",
    }


def lofi_image_cost_delta() -> dict[str, Any]:
    """Old 768×1344 vs current 720×1280, same backend/steps. For run-log delta."""
    old_usd, old_meta = lofi_image_cost_per_call_usd(
        LOFI_IMAGE_WIDTH_LEGACY, LOFI_IMAGE_HEIGHT_LEGACY
    )
    new_usd, new_meta = lofi_image_cost_per_call_usd(
        LOFI_IMAGE_WIDTH, LOFI_IMAGE_HEIGHT
    )
    px_old = LOFI_IMAGE_WIDTH_LEGACY * LOFI_IMAGE_HEIGHT_LEGACY
    px_new = LOFI_IMAGE_WIDTH * LOFI_IMAGE_HEIGHT
    px_ratio = px_new / px_old
    cost_ratio = new_usd / old_usd if old_usd else 0.0
    drift = abs(cost_ratio - px_ratio)
    scaling = (
        "linear_with_pixels"
        if drift < 0.01
        else ("better_than_linear" if cost_ratio < px_ratio else "worse_than_linear")
    )
    time_est = (
        LOFI_DEV_SEC_PER_IMAGE_LEGACY * px_ratio
        if uses_flux_dev()
        else None
    )
    return {
        "old_size": f"{LOFI_IMAGE_WIDTH_LEGACY}x{LOFI_IMAGE_HEIGHT_LEGACY}",
        "new_size": f"{LOFI_IMAGE_WIDTH}x{LOFI_IMAGE_HEIGHT}",
        "old_mp": round(px_old / 1e6, 4),
        "new_mp": round(px_new / 1e6, 4),
        "old_usd_per_image": round(old_usd, 6),
        "new_usd_per_image": round(new_usd, 6),
        "usd_delta": round(new_usd - old_usd, 6),
        "pixel_ratio": round(px_ratio, 4),
        "cost_ratio": round(cost_ratio, 4),
        "scaling": scaling,
        "steps": new_meta.get("steps"),
        "formula": new_meta.get("formula"),
        "time_est_s_linear": None if time_est is None else round(time_est, 2),
        "time_baseline_s": LOFI_DEV_SEC_PER_IMAGE_LEGACY if uses_flux_dev() else None,
        "time_baseline_note": (
            "legacy 768x1344 Dev ~18s/image; time guess assumes attention-bound "
            "linear scaling with pixels"
            if uses_flux_dev()
            else None
        ),
        "old_meta": old_meta,
        "new_meta": new_meta,
    }


# Structural-only Dev negative. Scene-object terms are never appended.
LOFI_DEV_NEGATIVE_PROMPT: str = _ACTIVE_STYLE.style_negative
CLIP_NEGATIVE_TOKEN_CAP: int = 60
_CLIP_TOKENIZER = None


def clip_token_count(text: str) -> int:
    """CLIP ViT-L/14 BPE count (Flux CLIP-L). Excludes BOS/EOS."""
    global _CLIP_TOKENIZER
    if _CLIP_TOKENIZER is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        path = hf_hub_download(
            repo_id="openai/clip-vit-large-patch14",
            filename="tokenizer.json",
        )
        _CLIP_TOKENIZER = Tokenizer.from_file(path)
    enc = _CLIP_TOKENIZER.encode(text or "")
    special = {"<|startoftext|>", "<|endoftext|>"}
    return sum(1 for tok in enc.tokens if tok not in special)


def _cap_clip_phrases(text: str, cap: int = CLIP_NEGATIVE_TOKEN_CAP) -> str:
    parts = [p.strip() for p in str(text or "").split(",") if p.strip()]
    kept: list[str] = []
    for part in parts:
        trial = ", ".join(kept + [part])
        if clip_token_count(trial) > cap:
            break
        kept.append(part)
    return ", ".join(kept)


def noun_number_variants(noun: str) -> list[str]:
    """Singular and plural of a noun (last word inflected). Uncountables stay."""
    n = str(noun or "").strip().lower()
    if not n:
        return []
    irregular = {
        "furniture": ["furniture"],
        "people": ["people", "person"],
        "person": ["person", "people"],
        "shelf": ["shelf", "shelves"],
        "shelves": ["shelves", "shelf"],
        "dish": ["dish", "dishes"],
        "dishes": ["dishes", "dish"],
        "vase": ["vase", "vases"],
        "vases": ["vases", "vase"],
        "knife": ["knife", "knives"],
        "knives": ["knives", "knife"],
    }
    words = n.split()
    last = words[-1]
    head = words[:-1]

    def _pack(word: str) -> str:
        return " ".join(head + [word]) if head else word

    if last in irregular and not head:
        return list(irregular[last])
    if last in irregular:
        return [_pack(w) for w in irregular[last]]

    variants = [n]
    if last.endswith("ies") and len(last) > 3:
        variants.append(_pack(last[:-3] + "y"))
    elif last.endswith(("ches", "shes", "xes", "zes")):
        variants.append(_pack(last[:-2]))
    elif last.endswith("ves") and len(last) > 3:
        variants.append(_pack(last[:-3] + "f"))
        variants.append(_pack(last[:-3] + "fe"))
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 2:
        variants.append(_pack(last[:-1]))
    else:
        if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
            variants.append(_pack(last[:-1] + "ies"))
        elif last.endswith(("ch", "sh")):
            variants.append(_pack(last + "es"))
        elif last.endswith(("s", "x", "z")):
            variants.append(_pack(last + "es"))
        else:
            variants.append(_pack(last + "s"))
    out: list[str] = []
    for v in variants:
        if v and v not in out:
            out.append(v)
    return out


def compose_beat_negative(
    licensed_object: str = "",
    not_in_frame: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Anatomy / photoreal / nsfw / text only. Ignores scene-object lists."""
    del licensed_object, not_in_frame
    parts = [
        p.strip().lower()
        for p in str(LOFI_DEV_NEGATIVE_PROMPT or "").split(",")
        if p.strip()
    ]
    seen: list[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    capped = _cap_clip_phrases(", ".join(seen), CLIP_NEGATIVE_TOKEN_CAP)
    n_tok = clip_token_count(capped)
    if not getattr(compose_beat_negative, "_logged", False):
        print(
            f"[LOFI negative] clip_tokens={n_tok} cap={CLIP_NEGATIVE_TOKEN_CAP} "
            f"phrases={capped}"
        )
        compose_beat_negative._logged = True  # type: ignore[attr-defined]
    return capped


def compose_dev_negative(
    licensed_object: str = "",
    not_in_frame: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Backward-compatible alias. Scene-object terms are ignored."""
    return compose_beat_negative(
        licensed_object=licensed_object,
        not_in_frame=not_in_frame,
    )

# Soft highlight clamp in prep_base_frame (before Ken Burns) — kills baked bloom.
ENABLE_HIGHLIGHT_CLAMP: bool = True
HIGHLIGHT_SOFT_KNEE: float = 208.0
HIGHLIGHT_COMPRESS: float = 0.48
HIGHLIGHT_CLAMP_CEILING: float = 236.0

# ── Video canvas / motion ───────────────────────────────────────────────────
REEL_WIDTH: int = 1080
REEL_HEIGHT: int = 1920
REEL_FPS: int = 30
# Ken Burns — barely-perceptible drift (3–5% scale), ease-in-out, no parallax
KEN_BURNS_ZOOM_START: float = 1.00
KEN_BURNS_ZOOM_END: float = 1.04  # 4% across the scene
KEN_BURNS_ZOOM_CAP: float = 1.05  # hard ceiling
KEN_BURNS_EASE: str = "smootherstep"  # ease-in-out
# Tiny residual drift only — reference is near-static
ENABLE_KINETIC_PAN: bool = True
KEN_BURNS_PAN_AMP_X: float = 5.0
KEN_BURNS_PAN_AMP_Y: float = 3.0
# Animated lighting pulse (measurable luminance delta ~0.5–1s apart)
ENABLE_LIGHT_BREATH: bool = True
LIGHT_BREATH_AMP: float = 0.065  # more perceptible than 0.045
LIGHT_BREATH_PERIOD_S: float = 6.5
LIGHT_BREATH_SOURCE_BIAS: float = 0.70  # bias toward bright sources
LIGHT_BREATH_BLOOM: float = 0.012  # keep off baked chest/collar bloom
LIGHT_BREATH_DEBUG: bool = True  # log brightness_factor independently of zoom
# Procedural numpy grain — OFF; replaced by reference overlay film grain.mp4
ENABLE_PROCEDURAL_GRAIN: bool = False
GRAIN_OPACITY: float = 0.0
GRAIN_HALF_RES: bool = True
# Film-stock multiply BEHIND caption only
CAPTION_FILM_MULTIPLY_OPACITY: float = 0.18
# Reference film-grain overlay (Screen), below caption; loop/trim to video duration
ENABLE_DUST_OVERLAY: bool = True
DUST_OVERLAY_REL: str = "channels_config/wonder_feed/overlays/overlay film grain.mp4"
DUST_OVERLAY_PREFER_NPZ: bool = False
DUST_OVERLAY_OPACITY: float = 1.0  # single-pass screen
DUST_OVERLAY_BLEND: str = "screen"
DUST_OVERLAY_OPACITY_CAP: float = 1.0
DUST_OVERLAY_GAIN: float = 2.0  # multiply overlay pixels before blend, clip 255
# Two-pass chroma restore — OFF
DUST_OVERLAY_COLOR_OPACITY: float = 0.0
DUST_OVERLAY_CHROMA_GAIN: float = 0.0
# Dark vintage vignette (independent of particles)
ENABLE_VIGNETTE: bool = True
VIGNETTE_STRENGTH: float = 0.18  # ~15–20% corner darken
# Legacy procedural dust (disabled when overlay asset is used)
ENABLE_DUST_PARTICLES: bool = False
DUST_PARTICLE_COUNT: int = 48
DUST_PARTICLE_OPACITY: float = 0.12
# Library BGM — random pick, trimmed to clip length (no generated beds)
BGM_DIR_REL: str = "channels_config/wonder_feed/audio/bgm"
BGM_QUARANTINE_DIRNAME: str = "_quarantine_vocals"
# Dump-named reference rips (0816.MP3) and leftover beds stay out of the pool.
BGM_DENY_NAME_RE: str = r"^\d{3,8}(\(\d+\))?$"
BGM_DENY_SUBSTRINGS: tuple[str, ...] = (
    "reference",
    "voiceover",
    "narration",
    "reel_full",
    "let_them",
)
BGM_VOLUME: float = 0.12  # instrumental bed only; never a second voice layer
# Gap gain is measured from that live bed vs mid-speech mix RMS — not from
# this fader. Target: gap mix sits ~10 dB below mid-speech mix (band 8–12).
BGM_GAP_VS_SPEECH_DB: float = 10.0
BGM_GAP_CENTER_FACTOR_MIN: float = 0.25
BGM_GAP_CENTER_FACTOR_MAX: float = 2.20
BGM_GAP_RAMP_S: float = 0.08
BGM_EXCLUDE_PREFIXES: tuple[str, ...] = ("lofi_bed",)
REQUIRE_BGM: bool = True
# Voiceover (ElevenLabs) — first-class TTS knobs (change these, not call sites).
# Valid TTS_SPEED range: 0.7–1.2 (API floor 0.70, default 1.0). See docs/elevenlabs_tts.md.
ENABLE_VOICEOVER: bool = True
TTS_VOICE_ID: str = "hNtG3AcS155nfu8sfWXk"
TTS_MODEL: str = "eleven_multilingual_v2"  # v3 ignores voice_settings.speed; v2 applies it
TTS_SPEED: float = 0.69
LOFI_VOICE_ID = TTS_VOICE_ID
LOFI_TTS_MODEL = TTS_MODEL
LOFI_VOICE_SPEED = TTS_SPEED
LOFI_VOICE_VOLUME: float = 1.0
REQUIRE_VOICEOVER: bool = True
# Defaults = amber_dusk pair (legacy grading path only; LOFI_APPLY_GRADING=False)
DUOTONE_SHADOW: tuple[int, int, int] = (28, 72, 88)
DUOTONE_HIGHLIGHT: tuple[int, int, int] = (250, 185, 105)
DUOTONE_TONAL_BANDS: int = 4
GRAIN_INTENSITY: float = 0.028  # legacy still-grade additive

# Text-handle watermark height as fraction of frame
WATERMARK_SIZE_FRAC: float = 0.018

# Caption typography — Edu NSW ACT Foundation Bold, Whispers-small, tight tracking
DEFAULT_CAPTION_STYLE: str = "edu_nsw"
CAPTION_LINE_HEIGHT_FRAC: float = 0.033  # ~63px at 1920h; one point down from 0.034
CAPTION_MIN_LINE_HEIGHT_FRAC: float = 0.031
CAPTION_LETTER_SPACING_PX: float = -1.5  # tighter tracking
CAPTION_WORD_FADE_S: float = 0.20  # soft per-word fade-in
CAPTION_STYLES: frozenset[str] = frozenset(
    {
        "edu_nsw",
        "caveat",
        "playwrite_nz_basic",
        "more_sugar_thin",
        "lora_italic",
        "rounded_hand",
    }
)

# ── Per-channel watermark / brand ───────────────────────────────────────────
CHANNEL_ASSEMBLY: dict[str, dict[str, Any]] = {
    "momma_circle": {
        "logo_candidates": [
            "channels_config/momma_circle/logo/logo.png",
            "assets/logos/momma_circle.png",
            "assets/logos/momma_circle_watermark.png",
        ],
        "watermark_handle": "@Momma Circle",
        "logo_position": "bottom_center",
        "logo_opacity": 0.90,
        "logo_scale": 0.18,
        # The canonical source is ~4x larger than the former derived asset.
        "logo_size_multiplier": 0.337,
        "logo_bottom_px": 252,  # raised another 40px
        "trim_logo_transparent_padding": True,
        "caption_color": (255, 255, 255),
        "use_text_watermark": False,
    },
    "wonder_feed": {
        "logo_candidates": [
            "channels_config/wonder_feed/logo/logo.png",
        ],
        "watermark_handle": "@Wonder Feed",
        "logo_position": "bottom_center",
        "logo_opacity": 0.95,
        "logo_scale": 0.224,  # +5% vs 0.213
        "logo_bottom_px": 389,  # center ≈ 412px from bottom (was ~262; +150)
        "caption_color": (255, 255, 255),
        # Use channel PNG logo on LOFI stills/reels (not text handle).
        "use_text_watermark": False,
    },
}

_PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = _PACKAGE_DIR / "data"
STORE_DIR = _PACKAGE_DIR / "store"  # JSON-backed RAG store (Chroma optional)


def lighting_mood_by_id(mood_id: str | None) -> dict[str, Any]:
    """Resolve a mood dict by id; default to amber_dusk."""
    want = (mood_id or "").strip().lower()
    for row in LIGHTING_MOODS:
        if str(row["id"]) == want:
            return dict(row)
    return dict(LIGHTING_MOODS[0])


def select_lighting_mood(
    key: str | int | None = None,
    *,
    mood_id: str | None = None,
) -> dict[str, Any]:
    """
    Pick a lighting/mood variant.

    - Explicit ``mood_id`` wins when provided.
    - Else stable hash of ``key`` (theme/scene/index) rotates through the bank.
    - Else first mood (amber_dusk).
    """
    if mood_id:
        return lighting_mood_by_id(mood_id)
    if key is None or key == "":
        return dict(LIGHTING_MOODS[0])
    raw = str(key).encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    idx = int(digest[:8], 16) % len(LIGHTING_MOODS)
    return dict(LIGHTING_MOODS[idx])


def build_style_prefix(mood: dict[str, Any] | None = None) -> str:
    """Short positive style prefix with per-scene lighting descriptor swapped in."""
    m = mood or LIGHTING_MOODS[0]
    lighting = str(m.get("lighting") or "").strip()
    if lighting:
        return f"{LOFI_STYLE_BASE}, {lighting}"
    return LOFI_STYLE_BASE


def thematic_max_scenes(duration_s: int | float | None = None) -> int:
    """Hard cap for thematic_arc. Default 9; only rises if duration > 27s."""
    cap = int(THEMATIC_MAX_SCENES)
    if duration_s is None:
        return cap
    d = float(duration_s)
    if d > float(THEMATIC_DEFAULT_SCENES) * float(SCENE_DURATION_S) + 0.05:
        n = int(round(d / SCENE_DURATION_S))
        return max(cap, min(MAX_SCENES, n))
    return cap


def scene_count_for_duration(
    duration_s: int | float,
    *,
    thematic: bool = True,
) -> int:
    """Writer target scene count. Thematic default is 9, not round(24/3)=8."""
    if thematic:
        d = float(duration_s)
        if d > float(THEMATIC_DEFAULT_SCENES) * float(SCENE_DURATION_S) + 0.05:
            n = int(round(d / SCENE_DURATION_S))
            return max(THEMATIC_DEFAULT_SCENES, min(MAX_SCENES, n))
        return int(THEMATIC_DEFAULT_SCENES)
    d = float(duration_s)
    n = int(round(d / SCENE_DURATION_S))
    return max(MIN_SCENES, min(MAX_SCENES, n))


def slot_duration_for_vo(
    vo_dur: float,
    *,
    base_s: float | None = None,
    trailing_silence_s: float | None = None,
) -> tuple[float, bool]:
    """Return (slot_s, audio_driven).

    After TTS, the still slot is the measured VO length plus the inter-line
    hush (or a last-beat pad). Pre-TTS ``duration_s`` estimates are writer
    targets only — they must not floor or stretch a still past the speech.
    """
    vo = max(0.0, float(vo_dur or 0.0))
    trail = float(
        VO_INTERLINE_SILENCE_S if trailing_silence_s is None else trailing_silence_s
    )
    if vo > 0.05:
        raw = vo + trail
        # Keep a calm minimum hold. The upper target is 4s, but narration is
        # never cut: callers must rewrite an overlong beat before assembly.
        if raw <= float(MAX_SCENE_SLOT_S):
            raw = max(float(MIN_SCENE_SLOT_S), raw)
        return round(raw, 3), True
    base = float(base_s if base_s is not None else SCENE_DURATION_S)
    return round(base, 3), False


def duration_for_beat_count(n_beats: int) -> float:
    """Nominal total if every beat is exactly SCENE_DURATION_S (VO-extend can add a little)."""
    n = max(1, int(n_beats))
    return round(float(n) * float(SCENE_DURATION_S), 3)


def beat_duration_s() -> float:
    return float(SCENE_DURATION_S)


def validate_duration(duration_s: int | float) -> int:
    d = int(round(float(duration_s)))
    if d < MIN_DURATION_S or d > MAX_DURATION_S:
        raise ValueError(
            f"--duration must be {MIN_DURATION_S}–{MAX_DURATION_S} seconds "
            f"(got {d})"
        )
    return d


def validate_module_for_page(module: str, page_id: str) -> str:
    mod = (module or "relationship").strip().lower()
    page = (page_id or "").strip().lower()
    if mod not in VALID_MODULES:
        raise ValueError(
            f"--module must be one of {sorted(VALID_MODULES)} (got {module!r})"
        )
    allowed = MODULE_PAGE_GATES.get(mod, frozenset())
    if page not in allowed:
        raise ValueError(
            f"--module {mod!r} is not allowed for --page {page!r}. "
            f"Allowed pages for this module: {sorted(allowed)}"
        )
    if page not in VALID_PAGES:
        raise ValueError(
            f"ECONOMIC_REEL_LOFI supports pages {sorted(VALID_PAGES)} "
            f"(got {page!r})"
        )
    return mod


def resolve_logo_path(page_id: str, engine_root: Path) -> Path | None:
    cfg = CHANNEL_ASSEMBLY.get(page_id.lower()) or {}
    for rel in cfg.get("logo_candidates") or []:
        p = Path(rel)
        if not p.is_absolute():
            p = engine_root / p
        if p.is_file():
            return p
    return None


def channel_assembly_cfg(page_id: str) -> dict[str, Any]:
    return dict(CHANNEL_ASSEMBLY.get(page_id.lower()) or CHANNEL_ASSEMBLY["wonder_feed"])
