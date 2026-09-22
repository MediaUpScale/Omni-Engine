# -*- coding: utf-8 -*-
"""
Central paths and credentials — Unified Multi-Page Factory edition.

Environment variables load from `.env` in the project root.

Supported keys
--------------
    GEMINI_API_KEY (or GOOGLE_API_KEY), ANTHROPIC_API_KEY,
    GEMINI_IMAGE_MODEL, GEMINI_IMAGE_ASPECT_RATIO,
    GEMINI_RESEARCH_MODEL, CLAUDE_MODEL,
    GEMINI_ECONOMIC_BRAIN_MODEL, ECONOMIC_BRAIN_MODE (true/false),
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL (optional secondary fallback),
    TEXT_LLM_PRIMARY (default: gemini),
    REFERENCE_IMAGE_PATH, DIGITAL_PRODUCTS_PATH, OUTPUT_PATH, OUTPUTS_DIR,
    ASSETS_PATH, PDF_CHUNK_CHAR_LIMIT,
    IMGBB_API_KEY, ANTHROPIC_API_VERSION,
    PUBLISHING_SCHEDULE (e.g. "3h" or "90m" spacing between variant posts)

Page-aware paths
----------------
Path variables that are per-page (PERSONA_DNA_PATH, MASTER_DNA_PATH,
REFERENCE_IMAGE_PATH, DIGITAL_PRODUCTS_PATH, PAGE_OUTPUTS_DIR, ASSETS_DIR,
LIBRARY_DIR, CONTENT_LIBRARY_PATH, POST_PLANNER_XLSX) resolve dynamically
based on the ACTIVE_PAGE environment variable, which is set by main.py before
any module-level import. Defaults to 'anna_protocol' for full backward
compatibility.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from utils.pipeline_paths import (
    assets_root,
    channel_store_dir,
    outputs_root,
    page_assets_dir,
    page_outputs_dir,
)

logger = logging.getLogger(__name__)

ENGINE_ROOT: Path = Path(__file__).resolve().parent
DOTENV_PATH: Path = ENGINE_ROOT / ".env"

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def _load_project_dotenv() -> tuple[Path, bool]:
    resolved = DOTENV_PATH.expanduser().resolve()
    # CLI / Phase-0 preparse wins over .env for the active channel.
    preserved_page = (os.environ.get("ACTIVE_PAGE") or "").strip()
    if resolved.is_file():
        loaded = bool(load_dotenv(dotenv_path=resolved, override=True, encoding="utf-8-sig"))
        if preserved_page:
            os.environ["ACTIVE_PAGE"] = preserved_page
        return resolved, loaded
    return resolved, False


_DOTENV_RESOLVED_PATH, DOTENV_LOADED_FROM_FILE = _load_project_dotenv()


def print_dotenv_bootstrap() -> None:
    if DOTENV_LOADED_FROM_FILE:
        print(f"[bootstrap] .env loaded: {_DOTENV_RESOLVED_PATH}")
    else:
        print(f"[bootstrap] .env not loaded from {_DOTENV_RESOLVED_PATH}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _resolve_path(value: str | None, default: Path) -> Path:
    return Path((value or str(default))).expanduser()


def _parse_schedule_minutes(raw: str | None) -> int | None:
    """Parse '3h', '90m', '120' (bare minutes) → integer minutes; None if unset."""
    if not raw:
        return None
    raw = raw.strip().lower()
    m = re.fullmatch(r"(\d+)\s*h(?:ours?)?", raw)
    if m:
        return int(m.group(1)) * 60
    m = re.fullmatch(r"(\d+)\s*m(?:in(?:utes?)?)?", raw)
    if m:
        return int(m.group(1))
    if raw.isdigit():
        return int(raw)
    return None


# ---------------------------------------------------------------------------
# Safe fallback model IDs
# Text: Gemini Flash. Factory-wide image default: Together FLUX.1-schnell.
# Anna Protocol (and --image-primary) may still request native Gemini image SKUs.
# ---------------------------------------------------------------------------
SAFE_GEMINI_TEXT_MODEL: str = "models/gemini-2.5-flash"
SAFE_GEMINI_IMAGE_MODEL: str = "black-forest-labs/FLUX.1-schnell"  # Together FLUX (name legacy)
SAFE_GEMINI_IMAGE_FALLBACK_2: str = "black-forest-labs/FLUX.1-schnell"
SAFE_GEMINI_IMAGE_FALLBACK_3: str = "black-forest-labs/FLUX.1-schnell"
GEMINI_PRO_IMAGE_MODEL: str = "models/gemini-3-pro-image-preview"
GEMINI_FLASH_IMAGE_MODEL: str = "models/gemini-2.5-flash-image"
# 1K is the billing tier (not a forced 1:1 crop). 2K is opt-in only — it bills ~$0.134/img.
GEMINI_IMAGE_SIZE: str = (os.getenv("GEMINI_IMAGE_SIZE") or "1K").strip().upper() or "1K"
if GEMINI_IMAGE_SIZE not in {"1K", "2K"}:
    GEMINI_IMAGE_SIZE = "1K"
# Official 1K list prices used for pre-flight logs + guardrail estimates.
GEMINI_FLASH_IMAGE_1K_USD: float = 0.005
GEMINI_PRO_IMAGE_1K_USD: float = 0.03
GEMINI_PRO_IMAGE_2K_USD: float = 0.134
SAFE_CLAUDE_MODEL: str = "claude-3-5-sonnet-latest"

# CLI --image-model aliases → native Gemini image SKUs
IMAGE_MODEL_ALIASES: dict[str, str] = {
    "flash": GEMINI_FLASH_IMAGE_MODEL,
    "gemini-flash": GEMINI_FLASH_IMAGE_MODEL,
    "gemini-2.5-flash": GEMINI_FLASH_IMAGE_MODEL,
    "gemini-pro": GEMINI_PRO_IMAGE_MODEL,
    "pro": GEMINI_PRO_IMAGE_MODEL,
}

# ---------------------------------------------------------------------------
# Model Router — cost-first by default; premium ONLY when explicitly enabled
# ---------------------------------------------------------------------------
# USE_PREMIUM_MODEL=true  OR  MODEL_TIER=premium  → flagship Pro SKUs
# Otherwise always cheap: gemini-2.5-flash / gemini-*-flash-image
USE_PREMIUM_MODEL: bool = _bool_env("USE_PREMIUM_MODEL", False)
MODEL_TIER: str = (os.getenv("MODEL_TIER") or ("premium" if USE_PREMIUM_MODEL else "cheap")).strip().lower()
USE_OPENROUTER_AUTO: bool = _bool_env("USE_OPENROUTER_AUTO", False)
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY") or None
OPENROUTER_BASE_URL: str = (
    os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
).strip()
# Visual Control Agent — modular VLM evaluator (OpenRouter default).
# Swap via VISUAL_EVAL_PROVIDER=openrouter|gemini|auto and VISUAL_EVAL_MODEL.
VISUAL_EVAL_PROVIDER: str = (
    os.getenv("VISUAL_EVAL_PROVIDER") or "openrouter"
).strip().lower()
VISUAL_EVAL_MODEL: str = (
    os.getenv("VISUAL_EVAL_MODEL") or "qwen/qwen3.5-flash-02-23"
).strip()
VISUAL_EVAL_MODEL_FALLBACK: str = (
    os.getenv("VISUAL_EVAL_MODEL_FALLBACK") or "qwen/qwen3.5-9b"
).strip()
# Hard cap for Visual Inspector OpenRouter calls. On timeout/empty/error the
# inspector fail-opens so TTS / MoviePy are never blocked.
VISUAL_EVAL_TIMEOUT_S: float = float(os.getenv("VISUAL_EVAL_TIMEOUT_S") or "10")
ENGINE_DEBUG: bool = _bool_env("ENGINE_DEBUG", False)

# Hard timeout for every image API call (prevents terminal hangs)
IMAGE_API_TIMEOUT_S: float = float(os.getenv("IMAGE_API_TIMEOUT_S") or "25")

# ---------------------------------------------------------------------------
# Google / Gemini emergency billing guardrails (see core/google_guardrail.py)
# ---------------------------------------------------------------------------
# ALLOW_GOOGLE_API=false → every Gemini/Imagen call aborts before the HTTP request.
# Unset defaults to True so tests/CI stay unblocked; the factory .env may force false.
ALLOW_GOOGLE_API: bool = _bool_env("ALLOW_GOOGLE_API", True)
MAX_GOOGLE_COST_PER_RUN_USD: float = float(os.getenv("MAX_GOOGLE_COST_PER_RUN_USD") or "0.50")
MAX_GOOGLE_RETRIES: int = max(0, int(os.getenv("MAX_GOOGLE_RETRIES") or "2"))

# ---------------------------------------------------------------------------
# API keys & versioning
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# Together AI — FLUX.1-dev / LoRA image backend (Schnell moved to DeepInfra)
TOGETHER_API_KEY: str | None = os.getenv("TOGETHER_API_KEY") or None
TOGETHER_IMAGE_MODEL: str = (
    os.getenv("TOGETHER_IMAGE_MODEL") or "black-forest-labs/FLUX.1-schnell"
).strip()
TOGETHER_IMAGE_STEPS: int = int(os.getenv("TOGETHER_IMAGE_STEPS") or "4")
# DeepInfra — FLUX.1-schnell via OpenAI-compatible images API
DEEPINFRA_API_KEY: str | None = os.getenv("DEEPINFRA_API_KEY") or None
DEEPINFRA_OPENAI_BASE_URL: str = (
    os.getenv("DEEPINFRA_OPENAI_BASE_URL") or "https://api.deepinfra.com/v1/openai"
).strip()
DEEPINFRA_FLUX_SCHNELL_MODEL: str = (
    os.getenv("DEEPINFRA_FLUX_SCHNELL_MODEL") or "black-forest-labs/FLUX-1-schnell"
).strip()
# Estimated USD — dynamic per model via together_image.estimate_together_image_cost()
# Schnell default; override model with TOGETHER_IMAGE_MODEL without changing this floor.
TOGETHER_IMAGE_COST_USD: float = float(os.getenv("TOGETHER_IMAGE_COST_USD") or "0.003")
# Global Together AI rate limiter (process-wide, thread-safe): forces strictly
# sequential image requests (max 1 in-flight at any time across ALL variant
# workers) with a small micro-delay between calls, to avoid 429 storms when
# multiple bulk-production workers run concurrently. Only gates image calls —
# LLM research, TTS, video compile, and uploads stay fully concurrent. This is
# the sync/threading equivalent of an ``asyncio.Semaphore(1)`` — the pipeline
# itself is fully synchronous (threading-based), so a real asyncio primitive
# would require converting main.py's ThreadPoolExecutor orchestration to an
# event loop; this threading.Semaphore delivers an identical hard guarantee
# (exactly 1 in-flight Together request, process-wide) without that rewrite.
TOGETHER_MIN_CALL_INTERVAL_S: float = float(os.getenv("TOGETHER_MIN_CALL_INTERVAL_S") or "0.2")
TOGETHER_IMAGE_MAX_RETRIES: int = int(os.getenv("TOGETHER_IMAGE_MAX_RETRIES") or "4")
# 429 back-off is short and precise (server-hinted Retry-After when present,
# else this fixed short ladder) — never the old long exponential chains.
TOGETHER_429_BACKOFF_MIN_S: float = float(os.getenv("TOGETHER_429_BACKOFF_MIN_S") or "1.0")
TOGETHER_429_BACKOFF_MAX_S: float = float(os.getenv("TOGETHER_429_BACKOFF_MAX_S") or "2.5")

# Native portrait for all Together/FLUX outputs (test + production)
DRAFT_IMAGE_SIZE: tuple[int, int] = (768, 1344)
PRODUCTION_IMAGE_SIZE: tuple[int, int] = (768, 1344)

# VisualQA_Agent critic calibration (also mirrored in quality/VisualQA_Agent/config.py)
MAX_RETRIES: int = 5
QUALITY_THRESHOLD: float = 6.0

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_VERSION: str = (os.getenv("ANTHROPIC_API_VERSION") or "2023-06-01").strip()
IMGBB_API_KEY: str | None = os.getenv("IMGBB_API_KEY")
IMAGE_PROVIDER: str = (os.getenv("IMAGE_PROVIDER") or "together").strip().lower()

# ---------------------------------------------------------------------------
# Remote GPU (ComfyUI / RunPod) — opt-in adapter; legacy paths unchanged when false
# ---------------------------------------------------------------------------
# When ENABLE_REMOTE_GPU_WORKFLOWS=true, thin routers in get_image_adapter() and
# generate_voiceover() delegate to core.remote_gpu_manager. When false
# (default), Together / ElevenLabs / MoviePy continue exactly as before.
ENABLE_REMOTE_GPU_WORKFLOWS: bool = _bool_env("ENABLE_REMOTE_GPU_WORKFLOWS", False)
# Default: serverless (always on-demand). Pod (comfyui) is the explicit exception.
REMOTE_GPU_MODE: str = (os.getenv("REMOTE_GPU_MODE") or "runpod").strip().lower()


def _strip_url_fragment(url: str) -> str:
    """Drop browser ``#workflow-uuid`` fragments from ComfyUI proxy URLs."""
    raw = (url or "").strip().split("#", 1)[0].strip().rstrip("/")
    return raw


REMOTE_GPU_BASE_URL: str = _strip_url_fragment(
    os.getenv("REMOTE_GPU_BASE_URL") or os.getenv("COMFYUI_BASE_URL") or ""
)
REMOTE_GPU_WORKFLOWS_DIR: str = (
    os.getenv("REMOTE_GPU_WORKFLOWS_DIR") or "infra/runpod/workflows"
).strip()
REMOTE_GPU_POLL_INTERVAL_S: float = float(os.getenv("REMOTE_GPU_POLL_INTERVAL_S") or "2.0")
REMOTE_GPU_TIMEOUT_S: float = float(os.getenv("REMOTE_GPU_TIMEOUT_S") or "600")
REMOTE_GPU_DEFAULT_REF_AUDIO: str = (os.getenv("REMOTE_GPU_DEFAULT_REF_AUDIO") or "").strip()
# Optional exact transcript for the global F5 reference clip (or use sibling .txt)
REMOTE_GPU_DEFAULT_REF_TEXT: str = (os.getenv("REMOTE_GPU_DEFAULT_REF_TEXT") or "").strip()
# Empty = auto-select per channel (LoRA graph when REMOTE_GPU_LORA_NAME set)
REMOTE_GPU_FLUX_WORKFLOW: str = (os.getenv("REMOTE_GPU_FLUX_WORKFLOW") or "").strip()
RUNPOD_API_KEY: str | None = os.getenv("RUNPOD_API_KEY") or None
RUNPOD_ENDPOINT_ID: str = (os.getenv("RUNPOD_ENDPOINT_ID") or "").strip()
RUNPOD_ENDPOINT_URL: str = _strip_url_fragment(os.getenv("RUNPOD_ENDPOINT_URL") or "")
# Max concurrent remote-GPU jobs (serverless workers / Comfy queue depth).
# Not hardcoded — set to match endpoint worker count (1, 5, 10, …).
REMOTE_GPU_MAX_PARALLEL: int = max(
    1, int(os.getenv("REMOTE_GPU_MAX_PARALLEL") or os.getenv("REMOTE_GPU_WORKERS") or "5")
)
# Pricing tier hints for CostTracker GPU line items
# Pod: community (~$0.34/hr RTX 4090) | secure (~$0.69/hr)
RUNPOD_CLOUD_TYPE: str = (os.getenv("RUNPOD_CLOUD_TYPE") or "community").strip().lower()
# Serverless: flex (full rate) | active (−40% always-on workers)
RUNPOD_SERVERLESS_WORKER_TYPE: str = (
    os.getenv("RUNPOD_SERVERLESS_WORKER_TYPE") or "flex"
).strip().lower()
# Optional overrides (USD); 0 / empty → built-in table in cost_tracker
RUNPOD_POD_RTX4090_USD_PER_HOUR: float = float(
    os.getenv("RUNPOD_POD_RTX4090_USD_PER_HOUR") or "0"
)
RUNPOD_SERVERLESS_RTX4090_USD_PER_SEC: float = float(
    os.getenv("RUNPOD_SERVERLESS_RTX4090_USD_PER_SEC") or "0"
)

# ---------------------------------------------------------------------------
# DeepSeek — OPTIONAL secondary fallback only (Gemini is the primary text brain)
# Models (2026): deepseek-v4-flash | deepseek-v4-pro
# Legacy ``deepseek-chat`` is rejected by the API (HTTP 400) — never use it.
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY: str | None = os.getenv("DEEPSEEK_API_KEY") or None
DEEPSEEK_BASE_URL: str = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1").strip()


def _normalize_deepseek_model(raw: str, *, default: str) -> str:
    name = (raw or default).strip() or default
    if name.lower() in {"deepseek-chat", "deepseek-coder", "deepseek-chat-v3"}:
        return default
    return name


DEEPSEEK_FLASH_MODEL: str = _normalize_deepseek_model(
    os.getenv("DEEPSEEK_FLASH_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash",
    default="deepseek-v4-flash",
)
DEEPSEEK_PRO_MODEL: str = _normalize_deepseek_model(
    os.getenv("DEEPSEEK_PRO_MODEL") or "deepseek-v4-pro",
    default="deepseek-v4-pro",
)
DEEPSEEK_MODEL: str = DEEPSEEK_FLASH_MODEL

# ---------------------------------------------------------------------------
# Text LLM routing — Gemini is ALWAYS the primary provider for all pages
# ---------------------------------------------------------------------------
# "gemini" (default) | "deepseek" (legacy override — not recommended)
TEXT_LLM_PRIMARY: str = (os.getenv("TEXT_LLM_PRIMARY") or "gemini").strip().lower()

# SEED-ONLY (Round 7 — Measure-Then-Correct 2026-08-15). DO NOT use this as
# a duration gate anywhere in the pipeline. Every calibration bug this
# project has hit (2.25 vs 1.70 vs 3.15 vs 1.77, across ElevenLabs / F5-TTS,
# across voice-preset changes) traces back to the same anti-pattern:
# PREDICTING how long N words will take to speak with a stored constant
# BEFORE any real audio exists. That number is always eventually wrong
# for some voice/engine/update.
#
# The pipeline now uses this constant purely as an INITIAL word-count SEED
# for the first Gemini script request. Downstream,
# ``main.py::_synthesize_sequence_voice_track`` synthesizes the narration
# ONCE, MEASURES the actual audio duration + observed WPS live for this
# voice + engine + speed, and if the resulting total is outside ±15 % of
# the requested video duration it does EXACTLY ONE corrective script
# regeneration using the just-measured WPS (not this constant). No third
# try. No infinite loop.
#
# → Bottom line: this value only affects the first draft's rough length.
#   Actual duration accuracy is a downstream measurement, not an assumption.
#   Never re-introduce logic that gates acceptance/rejection against it.
NARRATION_WORDS_PER_SECOND: float = 1.77  # SEED ONLY — not a gate.

# Episode-level Gemini generateContent counter (successful + attempted calls).
GEMINI_FLASH_CALLS: int = 0


def words_for_duration(seconds: float, safety_margin: float = 1.08) -> int:
    """
    SEED-ONLY (Round 7 — Measure-Then-Correct). Convert a target spoken
    duration into a rough word count for use as the FIRST-DRAFT script
    request only.

    This is NOT a correctness gate. The actual per-run WPS is measured
    live in ``main.py::_synthesize_sequence_voice_track`` after TTS, and
    a corrective regeneration (if needed) uses that observed WPS instead
    of this constant. Never build acceptance/rejection logic on top of
    this function — it will be silently wrong for any voice/engine/
    speed the ``NARRATION_WORDS_PER_SECOND`` constant hasn't been
    hand-tuned for, which is every case we haven't measured yet.

    The 1.08 safety margin was a historical bias-toward-overshoot; kept
    for backwards compatibility of callers that consume the seed
    directly (e.g. ``channel_loader.PageContext.reel_narration_words``). The
    measure-then-correct path in ``_synthesize_sequence_voice_track``
    does not care about this margin — it derives the corrective word
    count from the observed WPS.
    """
    return round(float(seconds) * NARRATION_WORDS_PER_SECOND * float(safety_margin))


def note_gemini_flash_call(task: str = "") -> int:
    """Increment and log the per-process Gemini Flash call counter."""
    global GEMINI_FLASH_CALLS
    GEMINI_FLASH_CALLS += 1
    logger.info("GEMINI_FLASH_CALL | n=%d task=%s", GEMINI_FLASH_CALLS, task)
    return GEMINI_FLASH_CALLS


# Minimum words accepted from sequence voiceover before Gemini retry/fallback.
#
# DEPRECATED (Final Round 2026-08-15): the caption_engine word-floor gate now
# reads ``words_for_duration(duration_s)`` directly from the runtime
# ``duration_s`` argument, so this module-level constant is no longer used by
# the AK pipeline. The scalar is retained for backwards compatibility with any
# external ops script that imports ``app_config.SEQUENCE_VOICEOVER_MIN_WORDS``
# — it now resolves to the words-for-80 s baseline. Prefer calling
# ``sequence_voiceover_min_words(duration_s)`` (below) instead when a caller
# needs a duration-scaled floor.
_env_vo_min = (os.getenv("SEQUENCE_VOICEOVER_MIN_WORDS") or "").strip()
SEQUENCE_VOICEOVER_MIN_WORDS: int = (
    int(_env_vo_min) if _env_vo_min else words_for_duration(80.0)
)


def sequence_voiceover_min_words(duration_s: float) -> int:
    """Minimum accepted narration words for a reel of ``duration_s`` seconds.

    Duration-proportional replacement for the ``SEQUENCE_VOICEOVER_MIN_WORDS``
    module-level constant. Honours the ``SEQUENCE_VOICEOVER_MIN_WORDS`` env
    var override when set (returned as-is regardless of ``duration_s``);
    otherwise returns ``words_for_duration(duration_s)`` so longer
    ``--video-length`` overrides scale the floor correctly.
    """
    env = (os.getenv("SEQUENCE_VOICEOVER_MIN_WORDS") or "").strip()
    if env:
        try:
            return int(env)
        except (TypeError, ValueError):
            pass
    return int(words_for_duration(float(duration_s)))

# ---------------------------------------------------------------------------
# ElevenLabs — voiceover TTS + ambient SFX for ECONOMIC_REEL
# ---------------------------------------------------------------------------
ELEVENLABS_API_KEY: str | None = os.getenv("ELEVENLABS_API_KEY") or None


def elevenlabs_api_key_is_secret_format(key: str | None = None) -> bool:
    """
    True when *key* looks like an ElevenLabs **secret** API key (``sk_…``).

    ElevenLabs distinguishes Key ID vs secret: sending a Key ID yields HTTP 400
    ``api_key_id_used_as_api_key``. A 64-char hex blob without ``sk_`` is almost
    always the ID, not the secret.
    """
    raw = (key if key is not None else ELEVENLABS_API_KEY) or ""
    return str(raw).strip().startswith("sk_")


def assert_elevenlabs_api_key_usable(key: str | None = None) -> None:
    """Raise with a clear fix hint when the env value is a Key ID, not a secret."""
    raw = (key if key is not None else ELEVENLABS_API_KEY) or ""
    if not str(raw).strip():
        raise RuntimeError(
            "ELEVENLABS_API_KEY is empty. Add the secret key from the ElevenLabs "
            "dashboard (it must start with 'sk_')."
        )
    if not elevenlabs_api_key_is_secret_format(raw):
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not a secret key (must start with 'sk_'). "
            "ElevenLabs error api_key_id_used_as_api_key means this value is a "
            "Key ID (or other non-secret), not the API secret. Open the ElevenLabs "
            "dashboard → API Keys → copy the secret shown at creation/rotation "
            "(sk_…), paste it into .env as ELEVENLABS_API_KEY, and re-run. "
            "model_api_flows does not remap this variable — the .env value itself "
            "is what gets sent as xi-api-key."
        )

# ---------------------------------------------------------------------------
# YouTube Data API v3 — OAuth2 upload
# ---------------------------------------------------------------------------
# Set ENABLE_YOUTUBE_UPLOAD=true in .env to automatically publish every
# compiled reel.  CLI flags --publish-youtube / --upload-youtube override this.
ENABLE_YOUTUBE_UPLOAD: bool = os.getenv("ENABLE_YOUTUBE_UPLOAD", "false").strip().lower() in (
    "true", "1", "yes",
)
YOUTUBE_PRIVACY_STATUS: str = os.getenv("YOUTUBE_PRIVACY_STATUS", "unlisted").strip().lower()
# Path to Google OAuth2 desktop client-secrets file (never commit this file).
YOUTUBE_CLIENT_SECRETS: str = os.getenv(
    "YOUTUBE_CLIENT_SECRETS",
    str(Path(__file__).resolve().parent / "client_secret.json"),
)
# Per-page OAuth refresh tokens: credentials/tokens/youtube_token_{page}.json
YOUTUBE_TOKEN_DIR: str = os.getenv(
    "YOUTUBE_TOKEN_DIR",
    str(Path(__file__).resolve().parent / "credentials" / "tokens"),
)
# Daily upload-quota (~20 videos/channel/day) pending queue — populated when
# YouTubeQuotaExceededError is caught; replayed via --resume-youtube-queue.
YOUTUBE_PENDING_QUEUE_PATH: str = os.getenv(
    "YOUTUBE_PENDING_QUEUE_PATH",
    str(Path(__file__).resolve().parent / "credentials" / "pending_youtube_uploads.json"),
)

# ---------------------------------------------------------------------------
# Model IDs — Gemini 2.5 Flash is the default primary text engine for ALL pages
# ---------------------------------------------------------------------------
GEMINI_RESEARCH_MODEL: str = os.getenv("GEMINI_RESEARCH_MODEL", SAFE_GEMINI_TEXT_MODEL)
GEMINI_ECONOMIC_BRAIN_MODEL: str = os.getenv(
    "GEMINI_ECONOMIC_BRAIN_MODEL",
    SAFE_GEMINI_TEXT_MODEL,  # models/gemini-2.5-flash
)
GEMINI_ECONOMIC_IMAGE_MODEL: str = os.getenv(
    "GEMINI_ECONOMIC_IMAGE_MODEL",
    SAFE_GEMINI_IMAGE_MODEL,  # models/gemini-3.1-flash-image
)
GEMINI_NANO_IMAGE_MODEL: str = os.getenv(
    "GEMINI_NANO_IMAGE_MODEL",
    SAFE_GEMINI_IMAGE_MODEL,
)
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", SAFE_CLAUDE_MODEL)
GEMINI_IMAGE_MODEL: str = os.getenv("GEMINI_IMAGE_MODEL", SAFE_GEMINI_IMAGE_MODEL)
GEMINI_IMAGE_ASPECT_RATIO: str = os.getenv("GEMINI_IMAGE_ASPECT_RATIO", "3:4")
ECONOMIC_BRAIN_MODE: bool = _bool_env("ECONOMIC_BRAIN_MODE", False)
GEMINI_IMAGE_MODEL_PREFERENCE: str = SAFE_GEMINI_IMAGE_MODEL

# Ordered live Gemini image chain — CHEAP tier only (no pro drift)
IMAGE_MODEL_FALLBACK_CHAIN: list[str] = [
    SAFE_GEMINI_IMAGE_MODEL,        # models/gemini-3.1-flash-image
    SAFE_GEMINI_IMAGE_FALLBACK_2,   # models/gemini-2.5-flash-image
]
# Premium image SKU — only appended when USE_PREMIUM_MODEL / MODEL_TIER=premium
PREMIUM_IMAGE_MODEL: str = SAFE_GEMINI_IMAGE_FALLBACK_3
PREMIUM_TEXT_MODEL: str = "models/gemini-2.5-pro"


def resolve_image_model_alias(raw: str | None) -> str | None:
    """Map ``flash`` / ``gemini-pro`` (and raw SKUs) to a canonical image model id."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    key = text.lower().removeprefix("models/")
    aliased = IMAGE_MODEL_ALIASES.get(key)
    if aliased:
        return aliased
    return normalize_image_model_id(text)


def estimate_gemini_image_usd(model_id: str | None, *, image_size: str | None = None) -> float:
    """USD / image at the configured size tier (default 1K)."""
    size = (image_size or GEMINI_IMAGE_SIZE or "1K").strip().upper()
    low = (model_id or "").lower()
    if "pro" in low or "imagen" in low:
        return GEMINI_PRO_IMAGE_2K_USD if size == "2K" else GEMINI_PRO_IMAGE_1K_USD
    if "flash" in low and "image" in low:
        return GEMINI_FLASH_IMAGE_1K_USD
    if is_gemini_image_model(model_id):
        return GEMINI_FLASH_IMAGE_1K_USD
    return 0.0


def is_gemini_image_model(raw: str | None) -> bool:
    """True for native Gemini image SKUs (pro-image-preview, flash-image, …)."""
    low = (raw or "").strip().lower()
    return "gemini" in low and "image" in low


def normalize_image_model_id(raw: str | None) -> str:
    """
    Normalize image model IDs.

    Together / FLUX IDs pass through. Explicit Gemini image SKUs stay Gemini
    (used by anna_protocol IMAGE_PRIMARY and ``--image-primary``). Imagen IDs
    still remap to FLUX because this project cannot call Imagen.
    """
    flux = "black-forest-labs/FLUX.1-schnell"
    name = (raw or flux).strip() or flux
    low = name.lower().removeprefix("models/")

    # Together Juggernaut Lightning Flux — never remap to black-forest-labs
    if "juggernaut" in low:
        return "Rundiffusion/Juggernaut-Lightning-Flux"

    if "flux.2-dev" in low or "flux2-dev" in low or "flux2dev" in low:
        return "black-forest-labs/FLUX.2-dev"

    # Together / FLUX — keep as-is
    if "flux" in low or "black-forest-labs" in low:
        return name if "/" in name else f"black-forest-labs/{low}"

    if is_gemini_image_model(name):
        return name if name.lower().startswith("models/") else f"models/{low}"

    # Imagen SKUs 404 on this API project
    if low.startswith("imagen"):
        logger.info("Image SKU '%s' remapped → Together %s", name, flux)
        return flux

    return name


# Sanitize env overrides (Gemini image SKUs stay Gemini; Flux stays Flux)
GEMINI_ECONOMIC_IMAGE_MODEL = normalize_image_model_id(GEMINI_ECONOMIC_IMAGE_MODEL)
GEMINI_NANO_IMAGE_MODEL = normalize_image_model_id(GEMINI_NANO_IMAGE_MODEL)
GEMINI_IMAGE_MODEL = normalize_image_model_id(GEMINI_IMAGE_MODEL)
TOGETHER_IMAGE_MODEL = normalize_image_model_id(TOGETHER_IMAGE_MODEL)

# ---------------------------------------------------------------------------
# Engagement-format defaults (CLI-overridable via --cta and --post-type)
# ---------------------------------------------------------------------------
CTA_ENABLED: bool = _bool_env("CTA_ENABLED", True)
POST_TYPE: str = os.getenv("POST_TYPE", "STANDARD_QUOTE").strip().upper()

# ---------------------------------------------------------------------------
# Active page — set by main.py via ACTIVE_PAGE env var before any import.
# Defaults to 'anna_protocol' for full backward compatibility.
# ---------------------------------------------------------------------------
ACTIVE_PAGE: str = os.getenv("ACTIVE_PAGE", "anna_protocol")
CHANNELS_CONFIG_ROOT: Path = ENGINE_ROOT / "channels_config"
_LEGACY_PAGES_CONFIG_ROOT: Path = ENGINE_ROOT / "pages_config"
# Prefer channels_config/; fall back to historic pages_config/ if still present.
PAGES_CONFIG_ROOT: Path = (
    CHANNELS_CONFIG_ROOT
    if CHANNELS_CONFIG_ROOT.is_dir()
    else _LEGACY_PAGES_CONFIG_ROOT
)
ACTIVE_PAGE_DIR: Path = (
    (CHANNELS_CONFIG_ROOT / ACTIVE_PAGE)
    if (CHANNELS_CONFIG_ROOT / ACTIVE_PAGE).is_dir()
    else (_LEGACY_PAGES_CONFIG_ROOT / ACTIVE_PAGE)
)

# ---------------------------------------------------------------------------
# Page-aware persona paths
# ---------------------------------------------------------------------------
PERSONA_DNA_PATH: Path = ACTIVE_PAGE_DIR / "persona_dna.py"
MASTER_DNA_PATH: Path = ACTIVE_PAGE_DIR / "master_dna.json"

# Fallback: legacy agents.media/master_dna.json for anna_protocol
# if channels_config hasn't been set up yet.
if not MASTER_DNA_PATH.is_file() and ACTIVE_PAGE == "anna_protocol":
    MASTER_DNA_PATH = ENGINE_ROOT / "agents.media" / "master_dna.json"
if not PERSONA_DNA_PATH.is_file() and ACTIVE_PAGE == "anna_protocol":
    PERSONA_DNA_PATH = ENGINE_ROOT / "agents.media" / "persona_dna.py"

# ---------------------------------------------------------------------------
# Page-aware asset paths
# ---------------------------------------------------------------------------

# Reference avatar: prefer channels_config/{page}/avatar_reference/avatar.png,
# then fall back to the legacy hardcoded Drive path for anna_protocol.
_page_ref_avatar: Path = ACTIVE_PAGE_DIR / "avatar_reference" / "avatar.png"
_REFERENCE_AVATAR_LEGACY = Path(
    r"G:\My Drive\Z sosFiles\Z_act\@ NETWORK\@_Content 2026\The Holistic Legacy - Anna's Protocol"
    r"\Anna's Automated Image Posts Engine\avatar_reference\avatar.png",
)
_ref_avatar_default: Path = (
    _page_ref_avatar if _page_ref_avatar.parent.is_dir() else _REFERENCE_AVATAR_LEGACY
)
REFERENCE_IMAGE_PATH: Path = _resolve_path(
    os.getenv("REFERENCE_IMAGE_PATH"), _ref_avatar_default
)

# Digital products (PDF corpus): prefer channels_config/{page}/product_reference/
_page_digital_products: Path = ACTIVE_PAGE_DIR / "product_reference"
_DEFAULT_DIGITAL_PRODUCTS = ENGINE_ROOT / "product_reference" / "Digital Products"
DIGITAL_PRODUCTS_PATH: Path = _resolve_path(
    os.getenv("DIGITAL_PRODUCTS_PATH"),
    _page_digital_products if _page_digital_products.is_dir() else _DEFAULT_DIGITAL_PRODUCTS,
)

PDF_CHUNK_CHAR_LIMIT: int = int(os.getenv("PDF_CHUNK_CHAR_LIMIT", "48000"))

# ---------------------------------------------------------------------------
# Output paths — page-namespaced under {OUTPUT_PATH}/{page}/
# Prefers OUTPUT_PATH / ASSETS_PATH from .env (OUTPUTS_DIR is a legacy alias).
# ---------------------------------------------------------------------------
OUTPUTS_DIR: Path = outputs_root()
OUTPUT_PATH: Path = OUTPUTS_DIR
FACTORY_ASSETS_DIR: Path = assets_root()
ASSETS_PATH: Path = FACTORY_ASSETS_DIR

# Page-namespaced output root.  All per-page artifacts (images, library JSON,
# Excel planners) land here so pages never share or collide on outputs.
PAGE_OUTPUTS_DIR: Path = page_outputs_dir(ACTIVE_PAGE)
ASSETS_DIR: Path = page_assets_dir(ACTIVE_PAGE)
LIBRARY_DIR: Path = PAGE_OUTPUTS_DIR / "library"
CHANNEL_STORE_DIR: Path = channel_store_dir(ACTIVE_PAGE)
CONTENT_LIBRARY_PATH: Path = CHANNEL_STORE_DIR / "content_library.json"
SESSION_HOOKS_CACHE_PATH: Path = CHANNEL_STORE_DIR / "session_hooks_cache.json"

_SAMPLE_BULK_V3: Path = ENGINE_ROOT / "sample_bulk_posts_import_3.xlsx"
_SAMPLE_BULK_LEGACY: Path = ENGINE_ROOT / "sample_bulk_posts_import.xlsx"
BULK_POSTS_TEMPLATE_XLSX: Path = _SAMPLE_BULK_V3 if _SAMPLE_BULK_V3.is_file() else _SAMPLE_BULK_LEGACY
POST_PLANNER_XLSX: Path = PAGE_OUTPUTS_DIR / "automated_bulk_posts_import.xlsx"

_LEGACY_PLANNER_XLSX: Path = ENGINE_ROOT / "automated_bulk_posts_import.xlsx"

# ---------------------------------------------------------------------------
# Publishing schedule (Instagram / PostPlanner)
# ---------------------------------------------------------------------------
PUBLISHING_SCHEDULE: str | None = os.getenv("PUBLISHING_SCHEDULE") or None
PUBLISHING_INTERVAL_MINUTES: int | None = _parse_schedule_minutes(PUBLISHING_SCHEDULE)

# ---------------------------------------------------------------------------
# Pinterest safe-drip interval
# ---------------------------------------------------------------------------
def _parse_interval_hours(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


MIN_INTERVAL_HOURS: float = _parse_interval_hours("MIN_INTERVAL_HOURS", 3.0)
MAX_INTERVAL_HOURS: float = _parse_interval_hours("MAX_INTERVAL_HOURS", 6.0)
PINTEREST_PINS_PER_DAY: int = int(os.getenv("PINTEREST_PINS_PER_DAY", "4"))

# ---------------------------------------------------------------------------
# Directory bootstrap (create if missing)
# ---------------------------------------------------------------------------
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PAGE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

if _LEGACY_PLANNER_XLSX.is_file() and not POST_PLANNER_XLSX.exists():
    import shutil
    try:
        shutil.copy2(_LEGACY_PLANNER_XLSX, POST_PLANNER_XLSX)
    except OSError:
        logger.debug("Legacy planner copy skipped.", exc_info=True)


def bind_active_page(page: str) -> str:
    """Rebind page-namespaced output paths after a late ``--channel`` resolve.

    ``config`` may have been imported before ``sys.argv`` was rewritten
    (E2E harnesses, CaptionEngine preloads). Call this once the CLI channel
    is known so clips land under ``{OUTPUT_PATH}/{page}/clips``.
    """
    global ACTIVE_PAGE, ACTIVE_PAGE_DIR, PERSONA_DNA_PATH, MASTER_DNA_PATH
    global PAGE_OUTPUTS_DIR, ASSETS_DIR, LIBRARY_DIR, CHANNEL_STORE_DIR
    global CONTENT_LIBRARY_PATH, SESSION_HOOKS_CACHE_PATH, POST_PLANNER_XLSX

    slug = (page or "").strip().lower() or "anna_protocol"
    os.environ["ACTIVE_PAGE"] = slug
    ACTIVE_PAGE = slug
    ACTIVE_PAGE_DIR = (
        (CHANNELS_CONFIG_ROOT / ACTIVE_PAGE)
        if (CHANNELS_CONFIG_ROOT / ACTIVE_PAGE).is_dir()
        else (_LEGACY_PAGES_CONFIG_ROOT / ACTIVE_PAGE)
    )
    PERSONA_DNA_PATH = ACTIVE_PAGE_DIR / "persona_dna.py"
    MASTER_DNA_PATH = ACTIVE_PAGE_DIR / "master_dna.json"
    PAGE_OUTPUTS_DIR = page_outputs_dir(ACTIVE_PAGE)
    ASSETS_DIR = page_assets_dir(ACTIVE_PAGE)
    LIBRARY_DIR = PAGE_OUTPUTS_DIR / "library"
    CHANNEL_STORE_DIR = channel_store_dir(ACTIVE_PAGE)
    CONTENT_LIBRARY_PATH = CHANNEL_STORE_DIR / "content_library.json"
    SESSION_HOOKS_CACHE_PATH = CHANNEL_STORE_DIR / "session_hooks_cache.json"
    POST_PLANNER_XLSX = PAGE_OUTPUTS_DIR / "automated_bulk_posts_import.xlsx"
    PAGE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    (PAGE_OUTPUTS_DIR / "clips").mkdir(parents=True, exist_ok=True)
    logger.info(
        "ACTIVE_PAGE bound | page=%s outputs=%s clips=%s",
        ACTIVE_PAGE, PAGE_OUTPUTS_DIR, PAGE_OUTPUTS_DIR / "clips",
    )
    return ACTIVE_PAGE


# ---------------------------------------------------------------------------
# Dynamic model discovery helpers (called by CaptionEngine / adapters)
# ---------------------------------------------------------------------------

def get_best_claude_model(anthropic_client: object | None = None) -> str:
    """
    Query the Anthropic Models API and return the best available conversational
    model. Falls back to SAFE_CLAUDE_MODEL if the API call fails.
    """
    if anthropic_client is None:
        return CLAUDE_MODEL or SAFE_CLAUDE_MODEL
    try:
        page = anthropic_client.models.list()  # type: ignore[union-attr]
        models = list(getattr(page, "data", None) or page)
        for priority in ("sonnet", "haiku"):
            for m in models:
                mid = str(getattr(m, "id", "") or "").lower()
                if priority in mid and "claude" in mid:
                    logger.debug("Dynamic Claude model resolved: %s", mid)
                    return mid
        for m in models:
            mid = str(getattr(m, "id", "") or "")
            if "claude" in mid.lower():
                return mid
    except Exception as exc:  # noqa: BLE001
        logger.debug("Claude model discovery failed (%s); using configured fallback.", exc)
    return CLAUDE_MODEL or SAFE_CLAUDE_MODEL


def get_best_gemini_text_model(client: object | None = None) -> str:  # type: ignore[type-arg]
    """
    Query Gemini models.list() and return the highest-scoring GA text model.
    Falls back to SAFE_GEMINI_TEXT_MODEL if discovery fails.
    """
    if client is None:
        return GEMINI_RESEARCH_MODEL or SAFE_GEMINI_TEXT_MODEL
    try:
        from agents.media.providers.gemini_utils import (  # avoid circular at module load
            _list_models,
            _parse_version_score,
            _strip_model_id,
            _supports_generate_content,
        )
        candidates = []
        for m in _list_models(client):
            mid = _strip_model_id(getattr(m, "name", None))
            if not mid:
                continue
            low = mid.lower()
            if not any(k in low for k in ("flash", "pro")):
                continue
            if "image" in low or "vision" in low or "embed" in low:
                continue
            if not _supports_generate_content(m):
                continue
            candidates.append((mid, _parse_version_score(mid)))
        if candidates:
            best = max(candidates, key=lambda x: x[1])[0]
            logger.debug("Dynamic Gemini text model resolved: %s", best)
            return best
    except Exception as exc:  # noqa: BLE001
        logger.debug("Gemini model discovery failed (%s); using fallback.", exc)
    return GEMINI_RESEARCH_MODEL or SAFE_GEMINI_TEXT_MODEL


# ---------------------------------------------------------------------------
# Cost-first router bootstrap (cheap unless USE_PREMIUM_MODEL / MODEL_TIER)
# ---------------------------------------------------------------------------
try:
    from agents.media.providers.model_router import (
        resolve_tier as _resolve_model_tier,
        route_model,
        sync_config_defaults as _sync_model_router_defaults,
    )

    _sync_model_router_defaults()
    # If premium explicitly enabled, expand image fallback chain to include Pro
    if _resolve_model_tier() == "premium":
        IMAGE_MODEL_FALLBACK_CHAIN = [
            PREMIUM_IMAGE_MODEL,
            SAFE_GEMINI_IMAGE_MODEL,
            SAFE_GEMINI_IMAGE_FALLBACK_2,
        ]
        if not os.getenv("GEMINI_IMAGE_MODEL"):
            GEMINI_IMAGE_MODEL = normalize_image_model_id(PREMIUM_IMAGE_MODEL)
        if not os.getenv("GEMINI_ECONOMIC_BRAIN_MODEL"):
            GEMINI_ECONOMIC_BRAIN_MODEL = PREMIUM_TEXT_MODEL
        if not os.getenv("GEMINI_RESEARCH_MODEL"):
            GEMINI_RESEARCH_MODEL = PREMIUM_TEXT_MODEL
except Exception as _router_exc:  # noqa: BLE001
    logger.debug("Model router bootstrap skipped (%s)", _router_exc)


# ---------------------------------------------------------------------------
# Avatar helpers
# ---------------------------------------------------------------------------

def reference_avatar_resolved_path() -> Path:
    return REFERENCE_IMAGE_PATH.resolve()


def reference_avatar_exists() -> bool:
    return REFERENCE_IMAGE_PATH.is_file()


def warn_if_reference_avatar_missing() -> None:
    if reference_avatar_exists():
        return
    logger.warning(
        "Reference likeness file not found at %s. Image generation falls back to text-only prompting.",
        reference_avatar_resolved_path(),
    )
