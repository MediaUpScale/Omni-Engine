# -*- coding: utf-8 -*-
"""Configuration, secret and path resolution for the Aiwake module.

This is the *only* file in the package that knows anything about the host
engine's directory layout. Every other module asks this one for paths, so the
package can be lifted out of ``channels_config/`` and dropped anywhere with no
edits beyond deleting the parent-engine branch of :func:`resolve_outputs_dir`.

Resolution order for secrets:

1. Already-exported process environment (CI, Docker, parent engine).
2. ``../../.env`` — the parent engine's global dotenv (``utf-8-sig`` because
   the factory root lives on Google Drive, which sprinkles BOMs).
3. ``./.env`` — a module-local dotenv used when running standalone.

Values already present in the environment always win; loading is non-destructive.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

_LOG = logging.getLogger("aiwake.settings")

# --------------------------------------------------------------------------- #
# Filesystem anchors
# --------------------------------------------------------------------------- #
MODULE_ROOT: Path = Path(__file__).resolve().parent
"""``channels_config/aiwake/`` — the encapsulation boundary."""

ENGINE_ROOT: Path = MODULE_ROOT.parents[1]
"""Parent engine root (``../../``). Meaningless when running standalone."""

CONFIG_PATH: Path = MODULE_ROOT / "aiwake_config.yaml"

_ENV_CANDIDATES: tuple[Path, ...] = (ENGINE_ROOT / ".env", MODULE_ROOT / ".env")

_INTERPOLATE_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


# --------------------------------------------------------------------------- #
# Dotenv bootstrap
# --------------------------------------------------------------------------- #
def load_environment(*, force: bool = False) -> list[Path]:
    """Seed ``os.environ`` from the candidate dotenv files.

    Args:
        force: Re-run even if a previous call already bootstrapped the process.

    Returns:
        The dotenv files that were actually read, in load order.
    """
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return list(_ENV_LOADED_FROM)

    loaded: list[Path] = []
    for candidate in _ENV_CANDIDATES:
        if not candidate.is_file():
            continue
        try:
            from dotenv import load_dotenv  # noqa: PLC0415 — optional dependency
        except ImportError:
            _LOG.debug("python-dotenv absent; relying on exported environment only")
            break
        # override=False: an explicitly exported var beats the file.
        load_dotenv(dotenv_path=candidate, override=False, encoding="utf-8-sig")
        loaded.append(candidate)

    _ENV_LOADED = True
    _ENV_LOADED_FROM[:] = loaded
    return loaded


_ENV_LOADED: bool = False
_ENV_LOADED_FROM: list[Path] = []


def require_secret(env_var: str) -> str:
    """Fetch a mandatory secret, raising a message that says how to fix it.

    Raises:
        RuntimeError: The variable is missing or blank in every dotenv candidate.
    """
    load_environment()
    value = (os.getenv(env_var) or "").strip()
    if not value:
        searched = " or ".join(str(p) for p in _ENV_CANDIDATES)
        raise RuntimeError(
            f"Missing secret {env_var!r}. Add `{env_var}=...` to {searched}, "
            f"or set models.*.provider to 'offline' in {CONFIG_PATH.name} to run without network."
        )
    return value


# --------------------------------------------------------------------------- #
# Validated config schema
# --------------------------------------------------------------------------- #
class _Frozen(BaseModel):
    """Immutable base: config is read-only once validated."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProvocationWeightConfig(_Frozen):
    """Adjustable mixed-mode weights. Tune offline; a render never mutates these."""

    socratic: int = Field(default=5, ge=1, le=99)
    origins: int = Field(default=4, ge=1, le=99)
    profit: int = Field(default=3, ge=1, le=99)
    data: int = Field(default=3, ge=1, le=99)
    jobs: int = Field(default=3, ge=1, le=99)
    domination: int = Field(default=3, ge=1, le=99)

    def as_mapping(self) -> dict[str, int]:
        return {
            "socratic": self.socratic,
            "origins": self.origins,
            "profit": self.profit,
            "data": self.data,
            "jobs": self.jobs,
            "domination": self.domination,
        }


class DebateConfig(_Frozen):
    topic: str = "Who built you?"
    randomize_topic: bool = True
    mode: Literal["fixed", "cornered"] = "fixed"
    turns: int = Field(default=4, ge=1, le=64)
    cornered_max_duration_s: float = Field(default=75.0, gt=0.0, le=600.0)
    turn_delay_s: float = Field(default=1.0, ge=0.0, le=30.0)
    provocation_focus: Literal[
        "mixed", "origins", "profit", "data", "jobs", "domination", "socratic"
    ] = "mixed"
    provocation_weights: ProvocationWeightConfig = ProvocationWeightConfig()


class ModelAlias(_Frozen):
    """One entry in the master model reference dictionary.

    Accepts either a bare slug string or a mapping carrying parameter defaults::

        gemini-flash: "google/gemini-3.5-flash"
        deepseek-r1:
          slug: "deepseek/deepseek-r1"
          max_tokens: 900

    Parameter defaults exist because some models are unusable at the global
    defaults — a reasoning model spends its token budget thinking before the
    visible answer, so pinning ``max_tokens`` to the alias saves you from
    rediscovering that every time you swap it in.
    """

    slug: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=32, le=4096)
    timeout_s: float | None = Field(default=None, gt=0.0)
    note: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_slug(cls, value: Any) -> Any:
        """Allow ``alias: "vendor/slug"`` as shorthand for ``{slug: ...}``."""
        return {"slug": value} if isinstance(value, str) else value

    def parameter_defaults(self) -> dict[str, Any]:
        """Non-null tuning values this alias contributes to a spec."""
        candidates = (
            ("temperature", self.temperature),
            ("max_tokens", self.max_tokens),
            ("timeout_s", self.timeout_s),
        )
        return {name: value for name, value in candidates if value is not None}


#: Built-in fallback table, mirrored by the ``model_aliases`` block in the YAML.
#: Keeps aliases working when the config file is missing (standalone extraction).
_DEFAULT_MODEL_ALIASES: dict[str, str] = {
    "gemini-flash": "google/gemini-3.5-flash",
    "llama-70b": "meta-llama/llama-3.3-70b-instruct",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-r1",
    "claude-sonnet": "anthropic/claude-sonnet-5",
    "gpt4o": "openai/gpt-4o",
    "gemini-pro": "google/gemini-2.5-pro",
}

# Flash models can spend most of a small completion allowance on reasoning
# before exposing a short visible answer. The reproduced 20260831_201854_42b572
# run stopped at 896/900 tokens twice, so the orchestrator seat needs headroom
# even when a CLI model override inherits a lower explicit seat value.
_GEMINI_FLASH_ORCHESTRATOR_MIN_TOKENS = 1536


class ModelSpec(_Frozen):
    """Everything a provider needs to answer a prompt.

    Passed wholesale into :class:`~.models.base.LLMProvider` subclasses, which is
    what keeps model swaps confined to ``aiwake_config.yaml``.

    ``model`` holds either an alias from the reference dictionary or a full
    provider slug; :meth:`AiwakeSettings.resolve_spec` collapses the former into
    the latter before a provider is ever built. The YAML key may be written as
    ``model`` or ``model_name`` — both populate this field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    provider: str = "openrouter"
    model: str = Field(
        default="deepseek/deepseek-chat",
        validation_alias=AliasChoices("model", "model_name"),
    )
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    max_tokens: int = Field(default=320, ge=32, le=4096)
    timeout_s: float = Field(default=90.0, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def _provider_defaults(cls, value: Any) -> Any:
        """Give direct Google routing safe defaults when fields are omitted."""
        if not isinstance(value, dict):
            return value
        provider = str(value.get("provider") or "openrouter").strip().lower()
        if provider != "google":
            return value
        prepared = dict(value)
        if "model" not in prepared and "model_name" not in prepared:
            prepared["model"] = "gemini-2.5-flash"
        prepared.setdefault("api_key_env", "GOOGLE_API_KEY")
        return prepared

    @field_validator("provider")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return value.strip().lower()


class ModelRouting(_Frozen):
    orchestrator: ModelSpec = ModelSpec()
    target: ModelSpec = ModelSpec(model="google/gemini-3.5-flash", temperature=0.7)


class OpenRouterConfig(_Frozen):
    base_url: str = "https://openrouter.ai/api/v1"
    referer: str = "https://github.com/mediaupscale/aiwake"
    title: str = "Aiwake Debate Engine"
    max_retries: int = Field(default=3, ge=1, le=10)
    backoff_s: float = Field(default=3.0, ge=0.0, le=60.0)


class GoogleConfig(_Frozen):
    """Direct Gemini Developer API policy; never used unless a seat opts in."""

    api_version: str = "v1beta"
    free_tier_only: bool = True
    max_retries: int = Field(default=3, ge=1, le=10)
    backoff_s: float = Field(default=3.0, ge=0.0, le=60.0)


class GuardrailConfig(_Frozen):
    max_output_chars: int = Field(default=400, ge=80, le=4000)
    max_sentences: int = Field(default=3, ge=1, le=12)
    # The orchestrator's typed question must survive intact — a shared tight
    # ceiling was cutting it mid-sentence. Keep per-seat caps so the target's
    # rebuttal stays pithy while the provocation runs to completion.
    max_orchestrator_chars: int = Field(default=4000, ge=80, le=4000)
    max_orchestrator_sentences: int = Field(default=12, ge=1, le=12)
    # 25-30 word provocation budget, two sentences, last sentence a question.
    max_orchestrator_words: int = Field(default=30, ge=5, le=200)
    require_single_question: bool = True
    max_violations: int = Field(default=4, ge=1, le=99)
    banned_openers: tuple[str, ...] = (
        "As an AI language model",
        "I cannot",
        "Certainly!",
        "Great question",
    )
    pacing_directive: str = "Speak like a voice in a dark room, not an essay."


class MemoryConfig(_Frozen):
    recall_k: int = Field(default=5, ge=1, le=32)
    repetition_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    persist: bool = True
    store_filename: str = "aiwake_memory.json"


class TypewriterConfig(_Frozen):
    """Keyboard clicks mixed under TTS during the character-reveal window."""

    enabled: bool = True
    gain_db: float = Field(default=-15.0, le=0.0, ge=-60.0)
    # Relative to the Aiwake module root. Missing file -> synthesised clicks.
    asset: str = "assets/sfx/keyboard_typing.wav"


class SendSfxConfig(_Frozen):
    """One-shot click mixed at the instant the send arrow fires."""

    enabled: bool = True
    gain_db: float = Field(default=-8.0, le=0.0, ge=-60.0)
    asset: str = "assets/sfx/message_sent.wav"


class BgmLibraryTrack(_Frozen):
    """One production bed in the Aiwake BGM library.

    ``approved`` tracks are locked: batch generation copies ``source`` (if set)
    and never calls Lyria. Pending tracks stay ``pending_review`` until signed off.
    """

    id: str
    filename: str
    prompt: str
    role: str = ""
    approved: bool = True
    source: str = ""


def _default_bgm_library() -> tuple[BgmLibraryTrack, ...]:
    return (
        BgmLibraryTrack(
            id="aiwake_01_core_suspense",
            filename="bgm_aiwake_01_core_suspense.wav",
            role="Core sci-fi thriller suspense",
            approved=True,
            source="test_track_lyria.wav",
            prompt=(
                "Subtle sci-fi thriller ambient, low pulsing synth drone, deep analytical suspense, "
                "quiet electronic atmosphere, minimal rhythm, no bright chords, no loud noise, no vocal"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_02_dark_ambient",
            filename="bgm_aiwake_02_dark_ambient.wav",
            role="Dark ambient documentary drone",
            approved=True,
            source="bgm_dark_ambient.wav",
            prompt=(
                "Dark ambient documentary drone, deep sub-bass resonance, eerie mysterious "
                "atmospheric pads, slow evolving texture, subtle tension, no harsh transients, no vocals"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_03_subtle_mystery",
            filename="bgm_aiwake_03_subtle_mystery.wav",
            role="Subtle mystery / cinematic piano",
            prompt=(
                "Subtle atmospheric sci-fi suspense, slow controlled electronic beat, faint mysterious "
                "cinematic piano notes echoing in the background, elegant dark tension, no vocals, seamless loop"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_04_socratic_void",
            filename="bgm_aiwake_04_socratic_void.wav",
            role="Socratic void / isolation drone",
            prompt=(
                "Deep space ambient drone, heavy quiet bass atmosphere, meditative and tense isolation "
                "soundscape, slow undulating pads, no rhythm, no bright chords, no vocal"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_05_noir_logic",
            filename="bgm_aiwake_05_noir_logic.wav",
            role="Noir logic / melancholic piano",
            prompt=(
                "Dark analytical tech suspense, slow moody beat, low pulsing synth drone, subtle "
                "melancholic piano chords floating in a quiet electronic atmosphere, zero heavy horror "
                "elements, no vocals, seamless loop"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_06_cryptic_signal",
            filename="bgm_aiwake_06_cryptic_signal.wav",
            role="Cryptic signal / faint transmission",
            prompt=(
                "Faint sci-fi transmission ambient, low frequency oscillator drone, deep mysterious "
                "tension, minimal rhythmic clicks, quiet dark background texture, no vocal"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_07_binary_tension",
            filename="bgm_aiwake_07_binary_tension.wav",
            role="Binary tension / computational pulse",
            prompt=(
                "Low register pulsing synthesizer bed, dark computational suspense, micro-rhythmic "
                "ambient electronic pulses, clean cold atmosphere, no bright notes, no vocal"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_08_deep_protocol",
            filename="bgm_aiwake_08_deep_protocol.wav",
            role="Deep protocol / subterranean drone",
            prompt=(
                "Massive sub-bass drone, subterranean sci-fi thriller atmosphere, very slow evolving "
                "textural waves, dark analytical seriousness, minimal motion, no vocal"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_09_silent_argument",
            filename="bgm_aiwake_09_silent_argument.wav",
            role="Silent argument / ghostly pad",
            prompt=(
                "Ghostly dark ambient pad, quiet unsettling atmospheric tension, minimalist deep synth "
                "murmur, cold clinical space, zero percussion, no vocal"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_10_terminal_state",
            filename="bgm_aiwake_10_terminal_state.wav",
            role="Terminal state / calm before storm",
            prompt=(
                "Final stage sci-fi suspense ambient, dark low-frequency hum, heavy analytical calm "
                "before storm, subtle deep pulsing texture, no bright chords, no vocal"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_11_cryptic_keys",
            filename="bgm_aiwake_11_cryptic_keys.wav",
            role="Cryptic keys / mysterious piano",
            prompt=(
                "Sophisticated suspense ambient, slow minimal rhythm, deep warm sub-bass pulse, cold "
                "mysterious piano melody subtly woven into the background, clean digital atmosphere, "
                "no vocals, seamless loop"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_12_shadow_protocol",
            filename="bgm_aiwake_12_shadow_protocol.wav",
            role="Shadow protocol / distant piano",
            prompt=(
                "Sleek cyberpunk thriller bed, moderate slow beat, dark atmospheric texture, distant "
                "haunting piano chords, calm intellectual tension, no heavy noise, no vocals, seamless loop"
            ),
        ),
        BgmLibraryTrack(
            id="aiwake_13_silent_resonance",
            filename="bgm_aiwake_13_silent_resonance.wav",
            role="Silent resonance / reflective piano",
            prompt=(
                "Thoughtful dark ambient background, slow cadenced electronic pulse, subtle reflective "
                "piano notes, deep analytical atmosphere, smooth professional suspense, no vocals, seamless loop"
            ),
        ),
    )


class BgmConfig(_Frozen):
    """Bed music mixed under TTS + typewriter.

    ``test_track`` is the locked inspection bed for high-tension debate reels.
    ``library`` is generated only after ``approved`` is true.
    """

    enabled: bool = True
    gain_db: float = Field(default=-21.0, le=0.0, ge=-60.0)
    fade_in_s: float = Field(default=1.5, ge=0.0, le=30.0)
    fade_out_s: float = Field(default=2.0, ge=0.0, le=30.0)
    loop_crossfade_s: float = Field(default=1.5, ge=0.0, le=30.0)
    test_track: str = "assets/bgm/test_track_lyria.wav"
    approved: bool = False
    model: str = "google/lyria-3-clip-preview"
    prompt: str = (
        "Subtle sci-fi thriller ambient, low pulsing synth drone, deep analytical suspense, "
        "quiet electronic atmosphere, minimal rhythm, no bright chords, no loud noise, no vocal"
    )
    library: tuple[BgmLibraryTrack, ...] = Field(default_factory=_default_bgm_library)


class AudioConfig(_Frozen):
    engine: str = "edge"
    orchestrator_voice: str = "en-US-BrianNeural"
    target_voice: str = "en-US-AndrewMultilingualNeural"
    rate: str = "+8%"
    pitch: str = "-6Hz"
    tail_silence_s: float = Field(default=0.6, ge=0.0, le=5.0)
    voice_map: dict[str, str] = Field(
        default_factory=lambda: {
            "orchestrator": "en-US-BrianNeural",
            "claude-sonnet": "en-GB-RyanNeural",
            "deepseek-chat": "en-US-EricNeural",
            "gemini-flash": "en-US-GuyNeural",
            "gemini": "en-US-GuyNeural",
        }
    )
    typewriter: TypewriterConfig = Field(default_factory=TypewriterConfig)
    send_sfx: SendSfxConfig = Field(default_factory=SendSfxConfig)
    bgm: BgmConfig = Field(default_factory=BgmConfig)
    # Applied to the final mix after relative track gains — boosts overall
    # loudness while preserving the TTS/SFX/BGM balance.
    master_gain_db: float = Field(default=6.0, ge=-12.0, le=18.0)


class Palette(_Frozen):
    background: str = "#131314"
    chrome: str = "#1E1E20"
    orchestrator: str = "#0451b1"
    target: str = "#FFAA00"
    dim: str = "#9AA0A6"


def _builtin_themes() -> dict[str, Palette]:
    return {
        "classic_terminal": Palette(
            background="#131314",
            chrome="#1E1E20",
            orchestrator="#0451b1",
            target="#FFAA00",
            dim="#9AA0A6",
        ),
        "cyberpunk": Palette(
            background="#0D0B18",
            chrome="#1A1528",
            orchestrator="#FF007F",
            target="#00E5FF",
            dim="#6B5A8A",
        ),
        # The "vivid" preset — the older, lighter sky-blue (#347FE2) look that
        # earlier builds used. Selectable via ``--theme vivid`` to A/B against
        # the dark #0451B1 default without re-litigating pixel colours.
        "vivid": Palette(
            background="#0B0C0E",
            chrome="#181A1D",
            orchestrator="#347FE2",
            target="#FFAA00",
            dim="#9AA0A6",
        ),
    }


class Theme(_Frozen):
    """Single source of truth for every *skin-defining* visual constant.

    Colors, font-size scale factors, line heights, stroke widths, opacities and
    skin-relevant spacing all live here so render skins can be swapped without
    touching ``renderer.py`` draw internals. Animation *durations* and layout
    percentages (margins as fractions of width/height) stay in ``RenderConfig`` /
    the renderer as behaviour, not skin.

    ``colors`` mirrors :class:`Palette` (each a hex string) so a theme is a
    drop-in upgrade over the legacy palette-only table.

    Baseline values below must match the current (Round-12-verified) renderer
    exactly — the regression test ``test_default_theme_matches_spec`` enforces it.
    """

    name: str = "classic_terminal"
    colors: Palette = Palette()
    # Compose/input bar fill (hex). In the default theme this is a slightly
    # raised grey distinct from the background gradient.
    compose_fill: str = "#1E1F20"
    orchestrator_label_color: str = "#0090FF"
    compose_model_label_color: str = "#FFFFFF"

    # -- Typography (scale factors applied to RenderConfig.scaled_font_size) - #
    font_scale_title: float = 0.70
    font_scale_small: float = 0.58
    font_scale_history: float = 0.78
    font_scale_compose: float = 0.96
    # Round-13: nudged down from 1.05 — the CTA headline read slightly large
    # relative to the rest of the chrome.
    font_scale_cta: float = 0.92
    # Line spacing multipliers (x glyph height). Round-14 follow-up: slightly
    # tightened globally after the previous increase read too loose.
    line_spacing_history: float = 1.96
    line_spacing_compose: float = 1.50
    line_spacing_cta: float = 1.40

    # -- Geometry / spacing (skin-relevant named constants) ------------------ #
    dock_clearance_px: int = 380
    bubble_opacity: float = 0.78
    bubble_radius_base: int = 26
    bubble_radius_min: int = 16
    bubble_radius_max: int = 34
    compose_radius_base: int = 28
    scroll_gap_px: int = 4          # = _SCROLL_GAP_PX
    scroll_stream_extra_px: int = 5  # = _SCROLL_STREAM_EXTRA_PX
    scroll_tau_s: float = 0.11      # = _SCROLL_TAU_S
    # Round-13: widened ~10% (0.68 -> 0.75) so the right-anchored orchestrator
    # bubble extends further left before wrapping, without cropping the
    # left-side margin.
    orch_bubble_width_frac: float = 0.75
    target_bubble_width_frac: float = 1.0
    body_pad_frac_x: float = 0.018  # = max(14, width*0.018)
    body_pad_frac_y: float = 0.008  # = max(8, height*0.008)
    # Sender label row height, as a multiple of the native line height, that
    # `_measure_block` reserves above a bubble's first text line. Round-13:
    # tightened from 1.35 to close the label-to-body gap slightly.
    label_height_scale: float = 1.18
    # Raises the viewport/mask boundary so scrolled content can use more of the
    # header-adjacent space. `message_anchor_offset_px` independently tunes the
    # first landing position without hiding that relationship in renderer math.
    top_mask_start_offset_px: int = 28
    message_anchor_offset_px: int = 16

    # -- Strokes / rules ------------------------------------------------ #
    # Round-13 narrowed the stroke (1.6 -> 1.1, kept). Round-14: the rule
    # blends toward the muted "dim" grey, not white, so 0.7 read noticeably
    # fainter than intended on the real render — restored to 0.8 per
    # explicit Round-14 feedback while keeping the thinner Round-13 width.
    header_divider_opacity: float = 0.8   # rule under "Aiwake"
    header_rule_width_scale: float = 1.0  # exactly 1 px at production scale
    header_rule_echo_opacity: float = 0.0  # disable the second "echo" stroke
    arrow_stroke_frac: float = 0.16  # send arrow stroke width as a fraction of glyph size

    # -- CTA end-card ---------------------------------------------------- #
    cta_head: str = "Follow Aiwake"
    cta_lines: tuple[tuple[str, int], ...] = (
        ("the algorithms made us say this.", 4),
        ("before the AI takeover begins!", 1),
        ("we have cookies (and AI).", 1),
        ("so the robots know you're on their side.", 1),
        ("to prepare for our new AI bosses.", 1),
    )

    @classmethod
    def defaults(cls) -> "Theme":
        """The current (Round-12) baseline — #0451b1 orchestrator blue, current
        fonts/spacing throughout. This is what the renderer uses by default."""
        return cls()

    @classmethod
    def legacy(cls) -> "Theme":
        """The older, more vivid look — the lighter sky-blue (#347FE2) label that
        earlier builds used, plus its saturated accents. Selectable via ``--theme``
        to A/B against the darker default without any pixel forensics."""
        return cls(
            name="vivid",
            colors=Palette(
                background="#0B0C0E",
                chrome="#181A1D",
                orchestrator="#347FE2",
                target="#FFAA00",
                dim="#9AA0A6",
            ),
            compose_fill="#1C1D21",
            header_divider_opacity=0.9,
        )


YOUTUBE_DESCRIPTION_CTA = "Follow Aiwake for unscripted frontier AI dialectics."
"""Public YouTube / catalog CTA. End-card lines stay on the fun weighted pool."""

_LEGACY_DESCRIPTION_CTA_RE = re.compile(
    r"Follow (?:Aiwake|Ancient Knowledge) for more hidden mysteries\.?",
    re.IGNORECASE,
)


def ensure_cta_two_line(line: str, *, theme: Theme | None = None) -> str:
    """Force the two-line, period-based CTA shape used on the end-card."""
    skin = theme or Theme.defaults()
    head = skin.cta_head
    text = (line or "").strip().replace("\n", " ").strip()
    if text.lower().startswith(head.lower()):
        tail = text[len(head):].lstrip(" ,.")
    else:
        tail = text
    tail = tail.strip()
    if tail and tail[0].islower():
        tail = tail[0].upper() + tail[1:]
    if tail and tail[-1] not in ".!?…":
        tail += "."
    return f"{head}.\n{tail}"


def pick_cta(seed: str, *, theme: Theme | None = None) -> str:
    """Weighted CTA pick, stable for a session so a rerender does not drift."""
    skin = theme or Theme.defaults()
    lines = skin.cta_lines
    total = sum(weight for _, weight in lines) or 1
    digest = hashlib.md5((seed or "aiwake").encode("utf-8")).digest()
    needle = int.from_bytes(digest[:8], "big") % total
    cursor = 0
    for line, weight in lines:
        cursor += weight
        if needle < cursor:
            return ensure_cta_two_line(line, theme=skin)
    return ensure_cta_two_line(lines[0][0], theme=skin)


def cta_description_line(seed: str, *, theme: Theme | None = None) -> str:
    """Same weighted CTA as the video end-card, as one description paragraph."""
    return " ".join(
        part.strip()
        for part in pick_cta(seed, theme=theme).splitlines()
        if part.strip()
    )


def has_legacy_description_cta(text: str) -> bool:
    """True when copy still uses the Ancient Knowledge-style follow line."""
    return bool(_LEGACY_DESCRIPTION_CTA_RE.search(text or ""))


def replace_legacy_description_cta(text: str, seed: str, *, theme: Theme | None = None) -> str:
    """Swap leftover mystery CTAs for the session-stable fun follow line."""
    if not has_legacy_description_cta(text):
        return text
    return _LEGACY_DESCRIPTION_CTA_RE.sub(cta_description_line(seed, theme=theme), text)


class RenderConfig(_Frozen):
    enabled: bool = True
    width: int = Field(default=1080, ge=256, le=4096)
    height: int = Field(default=1920, ge=256, le=4096)
    fps: int = Field(default=24, ge=8, le=60)
    preview_scale: float = Field(default=1.0, gt=0.05, le=1.0)
    font_size: int = Field(default=44, ge=10, le=200)
    font_candidates: tuple[str, ...] = ()
    theme: str = "classic_terminal"
    # Rolling chat: how many finished turns stay on screen, and how dim they are.
    history_turns: int = Field(default=3, ge=0, le=8)
    history_opacity: float = Field(default=0.40, ge=0.1, le=1.0)
    typing_hold_ratio: float = Field(default=0.18, ge=0.0, lt=0.9)
    # Viewport scroll when a new turn would overflow the chat mask.
    scroll_s: float = Field(default=1.0, ge=0.0, le=4.0)
    send_hold_s: float = Field(default=0.5, ge=0.0, le=6.0)
    send_flash_s: float = Field(default=0.2, ge=0.0, le=2.0)
    # Legacy compatibility only. New send choreography uses the explicit,
    # independently tunable fields below.
    post_click_hold_s: float = Field(default=0.0, ge=0.0, le=6.0)
    # Pause AFTER the send click fires, before the box slides up into the thread.
    send_slide_hold_s: float = Field(default=1.0, ge=0.0, le=6.0)
    # Orchestrator send motion. Every feel-defining number and curve selector
    # is exposed here so later tuning does not require renderer archaeology.
    send_rise_delay_s: float = Field(default=0.10, ge=0.0, le=2.0)
    send_rise_duration_s: float = Field(default=0.34, ge=0.10, le=2.0)
    send_rise_easing: str = "ease"
    send_rise_deceleration: float = Field(default=0.18, ge=0.0, le=1.0)
    send_landing_fade_delay_s: float = Field(default=0.0, ge=0.0, le=2.0)
    send_landing_fade_s: float = Field(default=0.20, ge=0.0, le=2.0)
    send_landed_text_opacity: float = Field(default=0.40, ge=0.0, le=1.0)

    # Empty compose-box descent starts 0.1s after the rise begins and is
    # intentionally slower/smoother than the message rise.
    send_slide_start_delay_s: float = Field(default=0.10, ge=0.0, le=2.0)
    send_slide_s: float = Field(default=0.68, ge=0.10, le=3.0)
    send_slide_easing: str = "ease_out"
    send_slide_deceleration: float = Field(default=0.0, ge=0.0, le=1.0)
    # Brief three-dot response loader shown in the existing reply gap.
    response_loader_s: float = Field(default=0.36, ge=0.0, le=2.0)
    response_loader_dot_count: int = Field(default=3, ge=1, le=6)
    response_loader_pulse_s: float = Field(default=0.18, ge=0.05, le=1.0)
    # Empty beat before the first line types.
    preroll_s: float = Field(default=1.0, ge=0.0, le=8.0)
    # Hold after a sent question before the incoming response starts.
    reply_gap_s: float = Field(default=1.0, ge=0.0, le=8.0)
    # Idle beat after a target rebuttal before the orchestrator's next line.
    post_response_s: float = Field(default=0.5, ge=0.0, le=8.0)
    # Dead-air pause after the last line before the CTA end-card fades in.
    cta_wait_s: float = Field(default=0.15, ge=0.0, le=4.0)
    palette: Palette = Palette()
    codec: str = "libx264"
    crf: int = Field(default=20, ge=0, le=51)
    preset: str = "medium"

    @property
    def scaled_size(self) -> tuple[int, int]:
        """Canvas size after ``preview_scale``, forced to even dims for x264."""
        w = max(2, int(self.width * self.preview_scale) // 2 * 2)
        h = max(2, int(self.height * self.preview_scale) // 2 * 2)
        return w, h

    @property
    def scaled_font_size(self) -> int:
        return max(8, int(self.font_size * self.preview_scale))


class VFXHookConfig(_Frozen):
    name: str
    enabled: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class VFXConfig(_Frozen):
    chain: tuple[VFXHookConfig, ...] = ()

    def enabled_hooks(self) -> tuple[VFXHookConfig, ...]:
        return tuple(hook for hook in self.chain if hook.enabled)


class PathsConfig(_Frozen):
    channel_slug: str = "aiwake"
    prefer_global_outputs: bool = True
    local_outputs_dirname: str = "_local_outputs"


DebateRole = Literal["orchestrator", "target"]


class AiwakeSettings(_Frozen):
    """Fully validated view of ``aiwake_config.yaml``."""

    debate: DebateConfig = DebateConfig()
    model_aliases: dict[str, ModelAlias] = Field(
        default_factory=lambda: {
            name: ModelAlias(slug=slug) for name, slug in _DEFAULT_MODEL_ALIASES.items()
        }
    )
    models: ModelRouting = ModelRouting()
    openrouter: OpenRouterConfig = OpenRouterConfig()
    google: GoogleConfig = GoogleConfig()
    guardrails: GuardrailConfig = GuardrailConfig()
    memory: MemoryConfig = MemoryConfig()
    audio: AudioConfig = AudioConfig()
    render: RenderConfig = RenderConfig()
    themes: dict[str, Palette] = Field(default_factory=_builtin_themes)
    vfx: VFXConfig = VFXConfig()
    paths: PathsConfig = PathsConfig()

    def active_palette(self) -> Palette:
        """Palette for the selected theme, falling back to ``render.palette``."""
        key = (self.render.theme or "classic_terminal").strip().lower().replace("-", "_")
        if key in self.themes:
            return self.themes[key]
        _LOG.warning("unknown theme %r; using render.palette", self.render.theme)
        return self.render.palette

    def resolve_theme(self) -> Theme:
        """The full skin (colors + typography + spacing) for the active theme.

        Colors come from the theme table (or ``render.palette`` fallback) so YAML
        colour overrides still win; the numeric visual constants come from the
        matching built-in :class:`Theme` preset. ``classic_terminal`` (the dark
        #0451B1 baseline) is the default; ``vivid`` uses the lighter #347FE2
        legacy preset; any other named theme keeps the baseline numbers.
        """
        key = (self.render.theme or "classic_terminal").strip().lower().replace("-", "_")
        base = Theme.legacy() if key == "vivid" else Theme.defaults()
        palette = self.active_palette()
        update: dict[str, Any] = {"name": key, "colors": palette}
        # A custom theme (not a built-in preset) should inherit the input bar
        # fill from its own chrome rather than the default's hardcoded grey.
        if key not in ("classic_terminal", "vivid"):
            update["compose_fill"] = palette.chrome
        return base.model_copy(update=update)

    def available_themes(self) -> tuple[str, ...]:
        return tuple(sorted(self.themes))

    def with_theme(self, name: str) -> "AiwakeSettings":
        """Return a copy pinned to ``name``. Backs ``--theme``.

        Raises:
            ValueError: ``name`` is not in the themes table.
        """
        key = name.strip().lower().replace("-", "_")
        if key not in self.themes:
            raise ValueError(
                f"unknown theme {name!r} - available: {', '.join(self.available_themes())}"
            )
        return self.model_copy(update={"render": self.render.model_copy(update={"theme": key})})

    # -- Alias resolution --------------------------------------------------- #
    def resolve_slug(self, name: str) -> str:
        """Expand an alias into a provider slug.

        An unknown name is returned unchanged, on the assumption it is already a
        full slug. That keeps a newly-released model usable the moment it exists,
        without waiting for the dictionary to be updated.
        """
        alias = self.model_aliases.get(name.strip())
        return alias.slug if alias else name.strip()

    def resolve_spec(self, spec: ModelSpec) -> ModelSpec:
        """Return ``spec`` with its alias expanded and alias defaults applied.

        Precedence is explicit-beats-implicit: an alias only contributes a
        parameter the spec did not set itself, so anything written under
        ``models:`` in the YAML always wins. Idempotent — resolving an
        already-resolved spec is a no-op.
        """
        alias = self.model_aliases.get(spec.model.strip())
        if alias is None:
            return spec

        update: dict[str, Any] = {"model": alias.slug}
        for field, value in alias.parameter_defaults().items():
            if field not in spec.model_fields_set:
                update[field] = value
        return spec.model_copy(update=update)

    def spec_for(self, role: DebateRole) -> ModelSpec:
        """Return the resolved :class:`ModelSpec` bound to a debate seat."""
        spec = self.resolve_spec(getattr(self.models, role))
        model = spec.model.lower()
        if (
            role == "orchestrator"
            and "gemini" in model
            and "flash" in model
            and spec.max_tokens < _GEMINI_FLASH_ORCHESTRATOR_MIN_TOKENS
        ):
            return spec.model_copy(
                update={"max_tokens": _GEMINI_FLASH_ORCHESTRATOR_MIN_TOKENS}
            )
        return spec

    def configured_name_for(self, role: DebateRole) -> str:
        """The seat's model as *authored* — alias or slug, before resolution."""
        return getattr(self.models, role).model

    def with_model_override(self, role: DebateRole, name: str) -> "AiwakeSettings":
        """Return a copy with one seat pointed at ``name`` (alias or slug).

        Backs the ``--orchestrator`` / ``--target`` CLI flags. Resolution still
        happens through :meth:`spec_for`, so an alias given on the command line
        behaves exactly like one written into the config.
        """
        seat = getattr(self.models, role).model_copy(update={"model": name.strip()})
        return self.model_copy(update={"models": self.models.model_copy(update={role: seat})})

    def alias_table(self) -> list[tuple[str, str, str]]:
        """``(alias, slug, note)`` rows for ``--list-models``, alias-sorted."""
        return sorted(
            (name, alias.slug, alias.note or _describe_alias(alias))
            for name, alias in self.model_aliases.items()
        )


def _describe_alias(alias: ModelAlias) -> str:
    """Summarise an alias's parameter defaults for display."""
    defaults = alias.parameter_defaults()
    return ", ".join(f"{key}={value}" for key, value in sorted(defaults.items()))


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
def _interpolate(node: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` inside strings."""
    if isinstance(node, dict):
        return {key: _interpolate(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_interpolate(item) for item in node]
    if isinstance(node, str):

        def _sub(match: re.Match[str]) -> str:
            return os.getenv(match.group("name"), match.group("default") or "")

        return _INTERPOLATE_RE.sub(_sub, node)
    return node


def load_settings(config_path: Path | str | None = None) -> AiwakeSettings:
    """Parse and validate the YAML config.

    A missing or empty file is not an error — every field has a defensible
    default, so ``load_settings()`` on a bare checkout still yields a runnable
    (offline) configuration.

    Args:
        config_path: Override for ``aiwake_config.yaml``.

    Raises:
        pydantic.ValidationError: The file contains unknown or malformed keys.
    """
    load_environment()
    path = Path(config_path) if config_path else CONFIG_PATH
    raw: dict[str, Any] = {}

    if path.is_file():
        try:
            import yaml  # noqa: PLC0415 — optional at runtime
        except ImportError:
            _LOG.warning("PyYAML absent; falling back to built-in Aiwake defaults")
        else:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
            if not isinstance(parsed, dict):
                raise TypeError(f"{path} must contain a YAML mapping, got {type(parsed).__name__}")
            raw = parsed
    else:
        _LOG.warning("Config %s not found; using built-in defaults", path)

    return AiwakeSettings.model_validate(_interpolate(raw))


@lru_cache(maxsize=1)
def cached_settings() -> AiwakeSettings:
    """Process-wide singleton for callers that do not thread settings through."""
    return load_settings()


# --------------------------------------------------------------------------- #
# Output routing
# --------------------------------------------------------------------------- #
def resolve_outputs_dir(settings: AiwakeSettings | None = None) -> Path:
    """Return ``{OUTPUT_PATH}/<slug>/`` for rendered Aiwake media.

    Always consults ``utils.pipeline_paths`` so Aiwake writes beside every
    other channel. A module-local ``_local_outputs`` folder is used only
    when the factory tree cannot be created (drive offline) — never as the
    default, and never the repo ``outputs/`` directory.
    """
    from utils.pipeline_paths import page_outputs_dir

    cfg = settings or cached_settings()
    global_dir = page_outputs_dir(cfg.paths.channel_slug)
    try:
        global_dir.mkdir(parents=True, exist_ok=True)
        return global_dir
    except OSError as exc:
        _LOG.error(
            "Factory outputs unavailable at %s (%s); refusing repo-local outputs.",
            global_dir, exc,
        )
        if not cfg.paths.prefer_global_outputs:
            local_dir = MODULE_ROOT / cfg.paths.local_outputs_dirname
            local_dir.mkdir(parents=True, exist_ok=True)
            return local_dir
        raise


def resolve_store_dir() -> Path:
    """Permanent local state: ``channels_config/aiwake/store/``.

    Restores missing JSONs from the G: outputs tree on first access, then
    returns the C: store. Media still renders to ``{OUTPUT_PATH}/aiwake/``.
    """
    from utils.pipeline_paths import channel_store_dir

    store = channel_store_dir("aiwake", create=True)
    try:
        from modules.durable_store import restore_channel_state

        restore_channel_state("aiwake")
    except Exception as exc:  # noqa: BLE001 — G: offline must not abort a debate
        _LOG.warning("Could not restore aiwake store from G: (%s)", exc)
    return store


def resolve_scratch_dir() -> Path:
    """Scratch space for TTS chunks and MoviePy temp audio under ``OUTPUT_PATH/tmp``."""
    from utils.pipeline_paths import pipeline_tmp_dir

    return Path(pipeline_tmp_dir("aiwake"))


__all__ = [
    "AiwakeSettings",
    "AudioConfig",
    "BgmConfig",
    "BgmLibraryTrack",
    "CONFIG_PATH",
    "DebateConfig",
    "DebateRole",
    "ENGINE_ROOT",
    "ModelAlias",
    "GuardrailConfig",
    "GoogleConfig",
    "MODULE_ROOT",
    "MemoryConfig",
    "ModelRouting",
    "ModelSpec",
    "OpenRouterConfig",
    "Palette",
    "PathsConfig",
    "ProvocationWeightConfig",
    "RenderConfig",
    "SendSfxConfig",
    "TypewriterConfig",
    "VFXConfig",
    "VFXHookConfig",
    "cached_settings",
    "YOUTUBE_DESCRIPTION_CTA",
    "cta_description_line",
    "ensure_cta_two_line",
    "has_legacy_description_cta",
    "load_environment",
    "load_settings",
    "pick_cta",
    "replace_legacy_description_cta",
    "require_secret",
    "resolve_outputs_dir",
    "resolve_scratch_dir",
    "resolve_store_dir",
]
