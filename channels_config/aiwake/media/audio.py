# -*- coding: utf-8 -*-
"""TTS abstraction, model-seat voice mapping, typewriter SFX, and BGM.

:class:`TTSEngine` is the swap point. ``edge-tts`` ships as the pilot engine
because it needs no key and no quota; an ElevenLabs or Piper engine is a
subclass plus one line in :func:`build_engine`.

Duration measurement matters more than it looks: the renderer syncs its typing
effect to the *real* audio length, so a wrong duration desyncs the whole video.
:func:`probe_duration` tries ffprobe, then MoviePy, then falls back to a
word-rate estimate, and never raises.

Voice assignment is a lookup, not a pair of seat fields: :func:`resolve_voice`
matches the speaking model's alias or slug against ``audio.voice_map`` (and the
persona ``SEAT_VOICES`` fallback). The Gemini orchestrator is pinned to its
canonical Christopher voice.

Typewriter clicks ride under AIWAKE.CORE (orchestrator) typing only, at
``gain_db`` (default -15 dB), for the character-reveal window — the same
``1 - typing_hold_ratio`` slice the renderer uses to type the line. Target
replies stay dry. Text is globally sanitized to plain speech before synthesis;
SSML/XML is never sent to the engine (ellipses become a period pause).

BGM uses a locked Lyria 3 inspection file (``test_track_lyria.wav``) as the
default debate bed. After ``audio.bgm.approved``, ``--generate-bgm-batch``
writes the production library. Mix gain (-21 dB vs TTS) is applied at mix
time; files are looped with a 1.5 s equal-power crossfade.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

try:
    from ..contracts import SpeakerRole, Utterance
    from ..personas import SEAT_VOICES
    from ..settings import (
        MODULE_ROOT,
        AudioConfig,
        BgmConfig,
        SendSfxConfig,
        TypewriterConfig,
        resolve_scratch_dir,
    )
except ImportError:  # pragma: no cover — standalone extraction
    from contracts import SpeakerRole, Utterance  # type: ignore[no-redef]
    from personas import SEAT_VOICES  # type: ignore[no-redef]
    from settings import (  # type: ignore[no-redef]
        MODULE_ROOT,
        AudioConfig,
        BgmConfig,
        SendSfxConfig,
        TypewriterConfig,
        resolve_scratch_dir,
    )

_LOG = logging.getLogger("aiwake.media.audio")

# Average narration rate for the fallback duration estimate (words/minute).
_FALLBACK_WPM = 165.0

_SAFE_NAME_RE = re.compile(r"[^a-z0-9]+")
_ELLIPSIS_RE = re.compile(r"\.{3,}|…+")
_DRAMATIC_BREAK_S = 1.5
_CORE_ROLE = "orchestrator"
GEMINI_CANONICAL_VOICE = "en-US-BrianNeural"
DEEPSEEK_CANONICAL_VOICE = "en-US-ChristopherNeural"
CTA_VOICE = GEMINI_CANONICAL_VOICE
_AIWAKE_WORD = re.compile(r"\bAiwake\b", re.IGNORECASE)

# Tokens that do not distinguish one model family from another.
_GENERIC_ALIAS_TOKENS = frozenset({"chat", "instruct", "preview", "latest"})

_DEFAULT_HOLD_RATIO = 0.18

TEST_BGM_RELATIVE = "assets/bgm/test_track_lyria.wav"
TEST_BGM_FILENAME = "test_track_lyria.wav"
LYRIA_MODEL = "google/lyria-3-clip-preview"
LYRIA_TEST_PROMPT = (
    "Subtle sci-fi thriller ambient, low pulsing synth drone, deep analytical suspense, "
    "quiet electronic atmosphere, minimal rhythm, no bright chords, no loud noise, no vocal"
)
BGM_APPROVAL_NOTICE = (
    "Using single test BGM: test_track_lyria.wav "
    "(Awaiting manual quality approval for full batch generation)"
)
BGM_APPROVED_NOTICE = (
    "Using approved inspection BGM: test_track_lyria.wav "
    "(Sci-Fi Suspense, -21 dB mix, 1.5s equal-power loop crossfade)"
)
_BGM_NOTICE_PRINTED = False
_LYRIA_TIMEOUT_S = 180.0
_MIN_AUDIO_BYTES = 44
_PRODUCTION_TARGET_S = 60.0
BGM_MANIFEST_FILENAME = "library_manifest.json"
LOCKED_BGM_FILENAMES = frozenset(
    {
        TEST_BGM_FILENAME,
        "bgm_dark_ambient.wav",
        "bgm_aiwake_01_core_suspense.wav",
        "bgm_aiwake_02_dark_ambient.wav",
        "bgm_aiwake_03_subtle_mystery.wav",
        "bgm_aiwake_04_socratic_void.wav",
        "bgm_aiwake_05_noir_logic.wav",
        "bgm_aiwake_06_cryptic_signal.wav",
        "bgm_aiwake_07_binary_tension.wav",
        "bgm_aiwake_08_deep_protocol.wav",
        "bgm_aiwake_09_silent_argument.wav",
        "bgm_aiwake_10_terminal_state.wav",
        "bgm_aiwake_11_cryptic_keys.wav",
        "bgm_aiwake_12_shadow_protocol.wav",
        "bgm_aiwake_13_silent_resonance.wav",
    }
)


class TTSError(RuntimeError):
    """Raised when synthesis fails irrecoverably."""


class BgmError(RuntimeError):
    """Raised when Lyria generation or BGM decode fails."""


@dataclass(slots=True)
class AudioAsset:
    """A synthesised voice track.

    Attributes:
        path: MP3 on disk.
        duration_s: Measured length; the renderer's timing source of truth.
        voice: Voice id used, for reproducibility.
        estimated: True when :func:`probe_duration` had to guess.
        char_count: Spoken character count — drives typewriter click density.
        role: Seat that produced the line (``orchestrator`` / ``target``).
            Typewriter SFX is mixed only for the orchestrator.
    """

    path: Path
    duration_s: float
    voice: str
    estimated: bool = False
    char_count: int = 0
    role: str = ""


# --------------------------------------------------------------------------- #
# Voice assignment
# --------------------------------------------------------------------------- #
def resolve_voice(
    config: AudioConfig,
    role: SpeakerRole,
    model_slug: str = "",
) -> str:
    """Pick an edge-tts voice for a seat + model.

    The Gemini orchestrator is hard-pinned to the first approved V2 voice so stale channel
    configuration cannot silently replace the persona voice. Everyone else
    is matched against alias keys in the map — exact substring first,
    then every meaningful token of the key present in the slug — so a live
    remap like ``google/gemini-3.5-flash`` still hits ``gemini-flash``.
    """
    mapping = {**SEAT_VOICES, **dict(config.voice_map)}
    if role is SpeakerRole.ORCHESTRATOR:
        return GEMINI_CANONICAL_VOICE

    needle = (model_slug or "").strip().lower()
    if needle in mapping:
        return mapping[needle]

    scored: list[tuple[int, str]] = []
    for key, voice in mapping.items():
        if key == "orchestrator":
            continue
        key_l = key.strip().lower()
        if not key_l:
            continue
        if key_l in needle:
            scored.append((len(key_l) + 50, voice))
            continue
        tokens = [
            part
            for part in key_l.replace("_", "-").split("-")
            if part and part not in _GENERIC_ALIAS_TOKENS
        ]
        if tokens and all(part in needle for part in tokens):
            scored.append((sum(len(part) for part in tokens), voice))
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]
    return config.target_voice


def resolve_cta_voice(config: AudioConfig | None = None) -> str:
    """CTA uses the orchestrator seat voice, never a third neural."""
    return GEMINI_CANONICAL_VOICE


# --------------------------------------------------------------------------- #
# Orchestrator SSML pauses
# --------------------------------------------------------------------------- #
def count_dramatic_pauses(text: str) -> int:
    """How many ellipsis beats ``text`` contains."""
    return len(_ELLIPSIS_RE.findall(text or ""))


def apply_dramatic_pauses(text: str, *, break_s: float = _DRAMATIC_BREAK_S) -> str:
    """Turn ellipses into a plain-text beat. SSML is never emitted.

    ``break_s`` is accepted for call-site compatibility; pauses are realized as
    a period so a neural voice waits without reading markup.
    """
    _ = break_s
    return _ELLIPSIS_RE.sub(". ", text or "")


def prepare_tts_text(text: str, role: SpeakerRole | str) -> str:
    """Return the exact string a TTS engine may speak: plain text only."""
    _ = role
    try:
        from ..contracts import sanitize_tts_input  # noqa: PLC0415
    except ImportError:  # pragma: no cover — standalone extraction
        from contracts import sanitize_tts_input  # type: ignore[no-redef]

    return sanitize_tts_input(text)


# --------------------------------------------------------------------------- #
# Duration probing
# --------------------------------------------------------------------------- #
def _ffprobe_duration(path: Path) -> float | None:
    """Read duration via ffprobe. Returns None when unavailable."""
    binary = shutil.which("ffprobe")
    if not binary:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 — fixed binary, no shell
            [
                binary,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        value = json.loads(completed.stdout)["format"]["duration"]
        return float(value)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("ffprobe failed on %s: %s", path.name, exc)
        return None


def _moviepy_duration(path: Path) -> float | None:
    """Read duration via MoviePy (which shells out to ffmpeg itself)."""
    try:
        from moviepy import AudioFileClip  # noqa: PLC0415

        with AudioFileClip(str(path)) as clip:
            return float(clip.duration or 0.0) or None
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("moviepy duration probe failed on %s: %s", path.name, exc)
        return None


def estimate_duration(text: str, *, wpm: float = _FALLBACK_WPM) -> float:
    """Word-rate duration estimate, floored at 1.2s so no clip is degenerate."""
    words = max(1, len(text.split()))
    return max(1.2, words / wpm * 60.0)


def probe_duration(path: Path, *, fallback_text: str = "") -> tuple[float, bool]:
    """Measure an audio file's length.

    Returns:
        ``(duration_s, estimated)``. ``estimated`` is True when both real probes
        failed and the value came from :func:`estimate_duration`.
    """
    for probe in (_ffprobe_duration, _moviepy_duration):
        duration = probe(path)
        if duration and duration > 0:
            return duration, False
    _LOG.warning("no probe could read %s — estimating from text length", path.name)
    return estimate_duration(fallback_text), True


# --------------------------------------------------------------------------- #
# End-of-video CTA voice
# --------------------------------------------------------------------------- #
def pronounce_cta_text(text: str) -> str:
    """Spoken CTA: 'Aiwake' becomes 'A.I. wake' so the voice says A-I, then wake."""
    return _AIWAKE_WORD.sub("A.I. wake", text or "")


_CTA_NO_PAUSE_RE = re.compile(r"[,!?;:]|…+")
_CTA_PAUSE_RUN_RE = re.compile(r"\s{2,}")
_CTA_AI_WAKE_SPOKEN = "A.I. wake"


def flatten_cta_spoken(text: str) -> str:
    """One seamless phrase: strip internal punctuation so TTS reads it fluidly.

    The CTA is a single energetic line — dropping commas, periods, and ellipses
    removes every artificial micro-pause while keeping the "A.I. wake"
    pronunciation intact (the spelling dots are preserved so the voice still
    reads A-I, wake).
    """
    spoken = pronounce_cta_text(text or "")
    spoken = spoken.replace(_CTA_AI_WAKE_SPOKEN, "\x00")
    spoken = _CTA_NO_PAUSE_RE.sub(" ", spoken)
    spoken = spoken.replace("\x00", _CTA_AI_WAKE_SPOKEN)
    return _CTA_PAUSE_RUN_RE.sub(" ", spoken).strip()


def synthesize_cta_line(
    text: str,
    destination: Path,
    *,
    config: AudioConfig | None = None,
) -> AudioAsset:
    """Speak the end-card CTA with the orchestrator neural voice.

    Silent engines and missing ``edge-tts`` fall back to an estimated duration
    so tests never hit the network. Live debate renders synthesise a real MP3.
    """
    try:
        from ..contracts import sanitize_tts_input  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from contracts import sanitize_tts_input  # type: ignore[no-redef]

    spoken = sanitize_tts_input(text) or (text or "").strip()
    duration_est = estimate_duration(spoken)
    engine_name = (config.engine if config is not None else "edge").strip().lower()
    voice = resolve_cta_voice(config)
    if engine_name == "silent" or not spoken:
        return AudioAsset(
            path=destination,
            duration_s=duration_est,
            voice="silent",
            estimated=True,
            char_count=len(spoken),
            role="cta",
        )
    try:
        import edge_tts  # noqa: PLC0415
    except ImportError:
        return AudioAsset(
            path=destination,
            duration_s=duration_est,
            voice="silent",
            estimated=True,
            char_count=len(spoken),
            role="cta",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    phrase = flatten_cta_spoken(spoken)

    async def _run() -> None:
        # Faster spoken CTA so the end-card VO keeps pace with the quicker typewriter.
        communicate = edge_tts.Communicate(phrase, voice=voice, rate="+28%")
        await communicate.save(str(destination))

    def _synthesise() -> None:
        asyncio.run(_run())

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            _synthesise()
        else:  # pragma: no cover
            import threading

            error: list[BaseException] = []

            def _worker() -> None:
                try:
                    _synthesise()
                except BaseException as exc:  # noqa: BLE001
                    error.append(exc)

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            thread.join()
            if error:
                raise error[0]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("CTA TTS failed (%s); using estimated duration", exc)
        return AudioAsset(
            path=destination,
            duration_s=duration_est,
            voice="silent",
            estimated=True,
            char_count=len(spoken),
            role="cta",
        )

    if not destination.is_file() or destination.stat().st_size < 512:
        return AudioAsset(
            path=destination,
            duration_s=duration_est,
            voice="silent",
            estimated=True,
            char_count=len(spoken),
            role="cta",
        )
    duration, estimated = probe_duration(destination, fallback_text=spoken)
    return AudioAsset(
        path=destination,
        duration_s=duration,
        voice=voice,
        estimated=estimated,
        char_count=len(spoken),
        role="cta",
    )


# --------------------------------------------------------------------------- #
# Typewriter SFX
# --------------------------------------------------------------------------- #
def _gain_linear(gain_db: float) -> float:
    return float(10.0 ** (gain_db / 20.0))


def synthesize_typewriter_clicks(
    duration_s: float,
    n_clicks: int,
    *,
    fps: int = 44100,
    gain_db: float = -20.0,
    seed: int = 7,
) -> "object":
    """Return a stereo float32 array of keyboard clicks spanning ``duration_s``.

    Clicks are spaced across the typing window (one per revealed character)
    with a hair of jitter so they read as typing rather than a metronome.
    """
    import numpy as np  # noqa: PLC0415

    n_samples = max(1, int(round(duration_s * fps)))
    buf = np.zeros((n_samples, 2), dtype=np.float32)
    clicks = max(1, int(n_clicks))
    if duration_s <= 0:
        return buf

    rng = np.random.default_rng(seed)
    click_len = max(12, int(fps * 0.004))
    interval = n_samples / clicks
    envelope = np.linspace(1.0, 0.0, click_len, dtype=np.float32) ** 2.4

    for index in range(clicks):
        jitter = 0.0 if clicks == 1 else rng.uniform(-0.08, 0.08) * interval
        start = int(index * interval + jitter)
        if start < 0 or start + click_len > n_samples:
            continue
        noise = rng.standard_normal(click_len).astype(np.float32)
        # Differentiate to brighten — closer to a mechanical click than rumble.
        click = np.diff(noise, prepend=noise[:1]) * envelope * 0.55
        buf[start : start + click_len, 0] += click
        buf[start : start + click_len, 1] += click

    peak = float(np.max(np.abs(buf))) or 1.0
    buf *= _gain_linear(gain_db) / peak
    return buf


def _loop_wav_clicks(
    path: Path,
    duration_s: float,
    *,
    fps: int,
    gain_db: float,
) -> "object | None":
    """Tile ``path`` to ``duration_s``. Returns None when the file cannot be read."""
    if not path.is_file():
        return None
    try:
        import numpy as np  # noqa: PLC0415
        from moviepy import AudioFileClip  # noqa: PLC0415
    except ImportError:
        return None

    try:
        with AudioFileClip(str(path)) as clip:
            arr = clip.to_soundarray(fps=fps)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("typewriter wav load failed (%s): %s", path.name, exc)
        return None

    import numpy as np  # noqa: PLC0415

    samples = np.asarray(arr, dtype=np.float32)
    if samples.ndim == 1:
        samples = np.stack([samples, samples], axis=1)
    elif samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)

    n_needed = max(1, int(round(duration_s * fps)))
    if samples.shape[0] == 0:
        return None
    reps = int(np.ceil(n_needed / samples.shape[0]))
    tiled = np.tile(samples, (reps, 1))[:n_needed]
    peak = float(np.max(np.abs(tiled))) or 1.0
    tiled *= _gain_linear(gain_db) / peak
    return tiled


def keyboard_typing_path(config: TypewriterConfig, *, module_root: Path | None = None) -> Path:
    """Absolute path of the looping keyboard WAV."""
    root = module_root or MODULE_ROOT
    asset = Path(config.asset)
    return asset if asset.is_absolute() else (root / asset)


def write_keyboard_typing_wav(path: Path, *, duration_s: float = 1.2, fps: int = 44100) -> Path:
    """Synthesise a short mechanical-click loop and write it as 16-bit PCM WAV."""
    samples = synthesize_typewriter_clicks(duration_s, n_clicks=26, fps=fps, gain_db=0.0, seed=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav_pcm(path, samples, fps)
    return path


def ensure_keyboard_typing_asset(
    config: TypewriterConfig | None = None,
    *,
    module_root: Path | None = None,
) -> Path:
    """Return the keyboard WAV, generating it when the file is missing."""
    cfg = config or TypewriterConfig()
    path = keyboard_typing_path(cfg, module_root=module_root)
    if not path.is_file() or path.stat().st_size < 64:
        write_keyboard_typing_wav(path)
        _LOG.info("wrote keyboard typing SFX %s", path)
    return path


def typewriter_samples(
    duration_s: float,
    n_clicks: int,
    config: TypewriterConfig,
    *,
    fps: int = 44100,
    module_root: Path | None = None,
) -> "object":
    """Clicks from ``config.asset`` if it exists, otherwise a synthesised bed."""
    asset = keyboard_typing_path(config, module_root=module_root)
    looped = _loop_wav_clicks(asset, duration_s, fps=fps, gain_db=config.gain_db)
    if looped is not None:
        return looped
    return synthesize_typewriter_clicks(
        duration_s,
        n_clicks,
        fps=fps,
        gain_db=config.gain_db,
    )


_SEND_CLICK_S = 0.12


def synthesize_send_click(
    *,
    fps: int = 44100,
    gain_db: float = -12.0,
) -> "object":
    """Crisp two-tone UI send tick (stereo float32)."""
    import numpy as np  # noqa: PLC0415

    n_samples = max(8, int(round(_SEND_CLICK_S * fps)))
    t = np.arange(n_samples, dtype=np.float32) / float(fps)
    envelope = np.exp(-t * 42.0).astype(np.float32)
    click = (
        0.62 * np.sin(2.0 * np.pi * 2650.0 * t)
        + 0.38 * np.sin(2.0 * np.pi * 1880.0 * t)
    ) * envelope
    stereo = np.stack([click, click], axis=1).astype(np.float32)
    peak = float(np.max(np.abs(stereo))) or 1.0
    stereo *= _gain_linear(gain_db) / peak
    return stereo


def send_sfx_path(config: object | None = None, *, module_root: Path | None = None) -> Path:
    """Absolute path of the message-sent WAV."""
    root = module_root or MODULE_ROOT
    relative = "assets/sfx/message_sent.wav"
    if config is not None:
        relative = str(getattr(config, "asset", relative) or relative)
    asset = Path(relative)
    return asset if asset.is_absolute() else (root / asset)


def write_send_sfx_wav(path: Path, *, fps: int = 44100) -> Path:
    """Write the synthesised send click as 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav_pcm(path, synthesize_send_click(fps=fps, gain_db=0.0), fps)
    return path


def ensure_send_sfx_asset(
    config: object | None = None,
    *,
    module_root: Path | None = None,
) -> Path:
    """Return the send WAV, generating it when the file is missing."""
    path = send_sfx_path(config, module_root=module_root)
    if not path.is_file() or path.stat().st_size < 64:
        write_send_sfx_wav(path)
        _LOG.info("wrote message-sent SFX %s", path)
    return path


def send_click_on_timeline(
    duration_s: float,
    offset_s: float,
    config: object,
    *,
    fps: int = 44100,
    module_root: Path | None = None,
) -> "object":
    """Place the send click at ``offset_s`` on a ``duration_s`` stereo bed."""
    import numpy as np  # noqa: PLC0415

    n_needed = max(1, int(round(max(0.0, duration_s) * fps)))
    buf = np.zeros((n_needed, 2), dtype=np.float32)
    gain_db = float(getattr(config, "gain_db", -12.0))
    click = None
    asset = send_sfx_path(config, module_root=module_root)
    if asset.is_file():
        try:
            from moviepy import AudioFileClip  # noqa: PLC0415

            with AudioFileClip(str(asset)) as clip:
                click = np.asarray(clip.to_soundarray(fps=fps), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("send sfx wav load failed: %s", exc)
    if click is None:
        click = synthesize_send_click(fps=fps, gain_db=0.0)
    if click.ndim == 1:
        click = np.stack([click, click], axis=1)
    elif click.shape[1] == 1:
        click = np.repeat(click, 2, axis=1)
    peak = float(np.max(np.abs(click))) or 1.0
    click = click * (_gain_linear(gain_db) / peak)
    start = min(n_needed - 1, max(0, int(round(offset_s * fps))))
    take = min(click.shape[0], n_needed - start)
    if take > 0:
        buf[start : start + take] += click[:take]
    return buf


# --------------------------------------------------------------------------- #
# Engine interface
# --------------------------------------------------------------------------- #
class TTSEngine(ABC):
    """Swappable voice backend.

    Subclasses implement :meth:`_synthesize`; retry, duration probing and
    filename hygiene are handled here.
    """

    engine_name: str = "abstract"

    def __init__(self, config: AudioConfig | None = None, *, output_dir: Path | None = None) -> None:
        self.config = config or AudioConfig()
        self.output_dir = output_dir or (resolve_scratch_dir() / "voice")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _synthesize(self, text: str, voice: str, destination: Path) -> None:
        """Write ``text`` spoken by ``voice`` to ``destination`` as MP3."""

    def voice_for(self, role: SpeakerRole, model_slug: str = "") -> str:
        """Map a debate seat (and optional model slug) onto a voice id."""
        return resolve_voice(self.config, role, model_slug)

    def speak(self, utterance: Utterance, *, session_id: str = "session") -> AudioAsset:
        """Synthesise one utterance, reusing an existing file when present.

        Raises:
            TTSError: The backend produced no usable audio.
        """
        voice = self.voice_for(utterance.role, utterance.model_slug)
        destination = self.output_dir / self._filename(utterance, session_id)

        try:
            from ..contracts import sanitize_tts_input  # noqa: PLC0415
        except ImportError:  # pragma: no cover — standalone extraction
            from contracts import sanitize_tts_input  # type: ignore[no-redef]

        spoken = sanitize_tts_input(utterance.text) or (utterance.text or "").strip()
        tts_text = prepare_tts_text(spoken, utterance.role)

        if destination.is_file() and destination.stat().st_size > 1024:
            _LOG.debug("reusing cached voice track %s", destination.name)
        else:
            self._synthesize(tts_text, voice, destination)

        if not destination.is_file() or destination.stat().st_size < 512:
            raise TTSError(f"{self.engine_name} produced no audio for turn {utterance.turn_index}")

        duration, estimated = probe_duration(destination, fallback_text=spoken)
        if estimated and utterance.role is SpeakerRole.ORCHESTRATOR:
            duration += count_dramatic_pauses(spoken) * _DRAMATIC_BREAK_S
        return AudioAsset(
            path=destination,
            duration_s=duration + self.config.tail_silence_s,
            voice=voice,
            estimated=estimated,
            char_count=len(spoken),
            role=utterance.role.value,
        )

    @staticmethod
    def _filename(utterance: Utterance, session_id: str) -> str:
        stem = _SAFE_NAME_RE.sub("_", utterance.text[:28].lower()).strip("_") or "line"
        pauses = count_dramatic_pauses(utterance.text) if utterance.role is SpeakerRole.ORCHESTRATOR else 0
        pause_tag = f"_br{pauses}" if pauses else ""
        return f"{session_id}_{utterance.turn_index:02d}_{utterance.role.value}{pause_tag}_{stem}.mp3"


class EdgeTTSEngine(TTSEngine):
    """Microsoft Edge neural voices via the free ``edge-tts`` package.

    Rate and pitch come from the config as SSML-style deltas (``"+8%"``,
    ``"-6Hz"``), which is what edge-tts expects verbatim.
    """

    engine_name = "edge"

    def _synthesize(self, text: str, voice: str, destination: Path) -> None:
        """Run the async edge-tts client from sync code.

        Raises:
            TTSError: The package is missing or the service call failed.
        """
        try:
            import edge_tts  # noqa: PLC0415
        except ImportError as exc:
            raise TTSError("edge-tts is not installed — `pip install edge-tts`") from exc

        async def _run() -> None:
            communicate = edge_tts.Communicate(
                prepare_tts_text(text, "orchestrator"),
                voice=voice,
                rate=self.config.rate,
                pitch=self.config.pitch,
            )
            await communicate.save(str(destination))

        try:
            # asyncio.run is safe here: the debate loop is synchronous. If Aiwake
            # is ever driven from inside a running loop, fall back to a thread.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(_run())
            else:  # pragma: no cover — only under an async host
                import threading

                error: list[BaseException] = []

                def _worker() -> None:
                    try:
                        asyncio.run(_run())
                    except BaseException as exc:  # noqa: BLE001
                        error.append(exc)

                thread = threading.Thread(target=_worker, daemon=True)
                thread.start()
                thread.join()
                if error:
                    raise error[0]
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"edge-tts failed for voice {voice!r}: {exc}") from exc

        _LOG.info("synthesised %s (%s)", destination.name, voice)


class SilentTTSEngine(TTSEngine):
    """No-op engine that emits nothing and reports estimated durations.

    Used by ``--no-audio`` and by tests: the renderer still gets a coherent
    timeline, so layout and typing sync can be verified without network calls.
    """

    engine_name = "silent"

    def _synthesize(self, text: str, voice: str, destination: Path) -> None:  # noqa: ARG002
        raise TTSError("silent engine never synthesises")

    def speak(self, utterance: Utterance, *, session_id: str = "session") -> AudioAsset:
        """Return a track-less asset carrying an estimated duration."""
        spoken = utterance.text.strip()
        pauses = count_dramatic_pauses(spoken) if utterance.role is SpeakerRole.ORCHESTRATOR else 0
        return AudioAsset(
            path=self.output_dir / f"{session_id}_{utterance.turn_index:02d}_silent.mp3",
            duration_s=estimate_duration(utterance.text)
            + pauses * _DRAMATIC_BREAK_S
            + self.config.tail_silence_s,
            voice="silent",
            estimated=True,
            char_count=len(spoken),
            role=utterance.role.value,
        )


_ENGINES: dict[str, type[TTSEngine]] = {
    EdgeTTSEngine.engine_name: EdgeTTSEngine,
    SilentTTSEngine.engine_name: SilentTTSEngine,
}


def build_engine(config: AudioConfig, *, output_dir: Path | None = None) -> TTSEngine:
    """Instantiate the engine named by ``config.engine``.

    Raises:
        TTSError: Unknown engine name.
    """
    engine_cls = _ENGINES.get(config.engine.strip().lower())
    if engine_cls is None:
        raise TTSError(f"unknown TTS engine {config.engine!r} — available: {', '.join(sorted(_ENGINES))}")
    return engine_cls(config, output_dir=output_dir)


# --------------------------------------------------------------------------- #
# BGM — locked inspection bed + production library
# --------------------------------------------------------------------------- #
def test_bgm_path(*, module_root: Path | None = None, relative: str = TEST_BGM_RELATIVE) -> Path:
    """Absolute path of the locked inspection WAV."""
    root = module_root or MODULE_ROOT
    path = Path(relative)
    return path if path.is_absolute() else (root / path)


def bgm_library_dir(*, module_root: Path | None = None, relative: str = TEST_BGM_RELATIVE) -> Path:
    """Directory that holds the inspection bed and production library WAVs."""
    return test_bgm_path(module_root=module_root, relative=relative).parent


def resolve_bgm_track(
    config: BgmConfig | None,
    *,
    module_root: Path | None = None,
    announce: bool = True,
    seed: int | None = None,
) -> Path | None:
    """Return the ambient bed for a render; never the same track twice in a row.

    Reads ``library_manifest.json`` next to the WAVs and picks an approved,
    on-disk production track at random via a non-repeating shuffle queue. When
    the manifest is missing or the library is empty, falls back to the locked
    inspection bed (``test_track_lyria.wav``).
    """
    import random as _random  # noqa: PLC0415

    if config is None or not config.enabled:
        return None
    rng = _random.Random(seed) if seed is not None else _random
    dest_dir = bgm_library_dir(
        module_root=module_root,
        relative=getattr(config, "test_track", None) or TEST_BGM_RELATIVE,
    )
    candidates = _pick_library_tracks(_read_library_manifest(dest_dir), dest_dir, rng=rng)
    picked = _next_bgm_filename(candidates, rng=rng)
    if picked is not None:
        path = dest_dir / picked
        if path.is_file():
            _LOG.info("[BGM Engine] Selected track: %s", path.name)
            print(f"[BGM Engine] Selected track: {path.name}")
            return path

    path = test_bgm_path(module_root=module_root, relative=config.test_track or TEST_BGM_RELATIVE)
    if not path.is_file() or path.stat().st_size < _MIN_AUDIO_BYTES:
        _LOG.info(
            "BGM enabled but %s is missing; run --test-bgm to generate the inspection track",
            TEST_BGM_FILENAME,
        )
        return None
    if announce:
        _announce_test_bgm(approved=bool(getattr(config, "approved", False)))
    return path


def _announce_test_bgm(*, approved: bool = False) -> None:
    global _BGM_NOTICE_PRINTED
    if _BGM_NOTICE_PRINTED:
        return
    _BGM_NOTICE_PRINTED = True
    notice = BGM_APPROVED_NOTICE if approved else BGM_APPROVAL_NOTICE
    print(notice)
    _LOG.info(notice)


#: Process-global non-repeating shuffle queue so consecutive renders in one
#: batch never pick the same ambient track twice in a row.
_BGM_SHUFFLE_QUEUE: list[str] = []


def _read_library_manifest(dest_dir: Path) -> dict:
    """Return the parsed ``library_manifest.json`` payload, or ``{}`` if absent."""
    dest = dest_dir / BGM_MANIFEST_FILENAME
    if not dest.is_file():
        _LOG.info("no BGM library manifest at %s; falling back to inspection bed", dest)
        return {}
    try:
        return json.loads(dest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _LOG.warning("could not read BGM library manifest %s: %s", dest, exc)
        return {}


def _pick_library_tracks(manifest: dict, dest_dir: Path, *, rng) -> list[str]:
    """Approved, on-disk library filenames from the manifest (stable order)."""
    candidates: list[str] = []
    for row in manifest.get("library", ()) or ():
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename", "") or "")
        if not filename or not row.get("approved"):
            continue
        path = dest_dir / filename
        if path.is_file() and path.stat().st_size >= _MIN_AUDIO_BYTES:
            candidates.append(filename)
    if not candidates:
        # Fall back to whatever approved-production WAVs exist even without rows.
        for filename in sorted(LOCKED_BGM_FILENAMES):
            path = dest_dir / filename
            if (
                path.is_file()
                and path.stat().st_size >= _MIN_AUDIO_BYTES
                and filename != TEST_BGM_FILENAME
            ):
                candidates.append(filename)
    return candidates


def _next_bgm_filename(candidates: list[str], *, rng) -> str | None:
    """Pull the next track from a non-repeating shuffle queue."""
    if not candidates:
        return None
    global _BGM_SHUFFLE_QUEUE
    if not _BGM_SHUFFLE_QUEUE:
        _BGM_SHUFFLE_QUEUE = list(candidates)
        rng.shuffle(_BGM_SHUFFLE_QUEUE)
    filename = _BGM_SHUFFLE_QUEUE.pop(0)
    if filename not in candidates:
        _BGM_SHUFFLE_QUEUE = list(candidates)
        rng.shuffle(_BGM_SHUFFLE_QUEUE)
        filename = _BGM_SHUFFLE_QUEUE.pop(0)
    return filename


def bgm_fade_envelope(
    n_samples: int,
    fps: int,
    fade_in_s: float,
    fade_out_s: float,
) -> "object":
    """Linear fade-in / fade-out covering the full bed duration."""
    import numpy as np  # noqa: PLC0415

    envelope = np.ones(max(1, n_samples), dtype=np.float32)
    duration_s = n_samples / max(1, fps)
    in_s = max(0.0, fade_in_s)
    out_s = max(0.0, fade_out_s)
    if in_s + out_s > duration_s > 0:
        scale = duration_s / (in_s + out_s)
        in_s *= scale
        out_s *= scale
    fade_in = min(n_samples, int(round(in_s * fps)))
    fade_out = min(n_samples, int(round(out_s * fps)))
    if fade_in:
        envelope[:fade_in] = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out:
        envelope[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return envelope


def _as_stereo(samples: "object") -> "object":
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        return np.stack([arr, arr], axis=1)
    if arr.shape[1] == 1:
        return np.repeat(arr, 2, axis=1)
    return arr


def crossfade_tile(
    samples: "object",
    *,
    fps: int,
    duration_s: float,
    overlap_s: float,
) -> "object":
    """Repeat ``samples`` to ``duration_s`` with an equal-power loop crossfade."""
    import numpy as np  # noqa: PLC0415

    arr = _as_stereo(samples)
    n_loop = int(arr.shape[0])
    n_needed = max(1, int(round(duration_s * fps)))
    channels = int(arr.shape[1])
    if n_loop == 0:
        return np.zeros((n_needed, 2), dtype=np.float32)
    if n_needed <= n_loop:
        return arr[:n_needed].copy()

    overlap = min(int(round(max(0.0, overlap_s) * fps)), max(0, n_loop // 2))
    if overlap < 1:
        reps = int(np.ceil(n_needed / n_loop))
        return np.tile(arr, (reps, 1))[:n_needed]

    fade_in = np.sin(np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32))
    fade_out = np.cos(np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32))
    out = np.zeros((n_needed, channels), dtype=np.float32)
    take = min(n_loop, n_needed)
    out[:take] = arr[:take]
    cursor = n_loop
    while cursor < n_needed:
        start = cursor - overlap
        head_n = min(overlap, n_needed - start)
        out[start : start + head_n] *= fade_out[:head_n, None]
        out[start : start + head_n] += arr[:head_n] * fade_in[:head_n, None]
        rest_start = start + overlap
        if rest_start >= n_needed:
            break
        rest_n = min(n_loop - overlap, n_needed - rest_start)
        if rest_n <= 0:
            break
        out[rest_start : rest_start + rest_n] = arr[overlap : overlap + rest_n]
        cursor = rest_start + rest_n
    return out


def prepare_bgm_bed(
    samples: "object",
    *,
    fps: int,
    duration_s: float,
    gain_db: float,
    fade_in_s: float,
    fade_out_s: float,
    loop_crossfade_s: float = 1.5,
) -> "object":
    """Tile ``samples`` to ``duration_s`` with loop crossfades, then master fades."""
    import numpy as np  # noqa: PLC0415

    tiled = crossfade_tile(
        samples,
        fps=fps,
        duration_s=duration_s,
        overlap_s=loop_crossfade_s,
    )
    peak = float(np.max(np.abs(tiled))) or 1.0
    tiled = tiled / peak
    envelope = bgm_fade_envelope(tiled.shape[0], fps, fade_in_s, fade_out_s)
    return tiled * _gain_linear(gain_db) * envelope[:, None]


def _decode_b64(payload: str) -> bytes | None:
    text = payload.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        raw = base64.b64decode(text, validate=False)
    except Exception:  # noqa: BLE001
        return None
    return raw if raw and len(raw) >= _MIN_AUDIO_BYTES else None


def extract_audio_bytes(payload: object) -> bytes | None:
    """Pull a base64 audio blob out of an OpenRouter / Gemini-shaped body."""
    if isinstance(payload, dict):
        audio = payload.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            decoded = _decode_b64(str(audio["data"]))
            if decoded:
                return decoded
        mime = str(payload.get("mime_type") or payload.get("media_type") or "")
        if mime.startswith("audio") and payload.get("data"):
            decoded = _decode_b64(str(payload["data"]))
            if decoded:
                return decoded
        for key in ("output_audio", "input_audio"):
            nested = payload.get(key)
            if isinstance(nested, dict) and nested.get("data"):
                decoded = _decode_b64(str(nested["data"]))
                if decoded:
                    return decoded
        kind = str(payload.get("type") or "")
        if "audio" in kind and payload.get("data"):
            decoded = _decode_b64(str(payload["data"]))
            if decoded:
                return decoded
        for value in payload.values():
            found = extract_audio_bytes(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = extract_audio_bytes(item)
            if found:
                return found
    elif isinstance(payload, str) and payload.lstrip().startswith("data:audio"):
        return _decode_b64(payload)
    return None


def _sniff_audio(raw: bytes) -> str:
    if raw.startswith(b"RIFF") and b"WAVE" in raw[:16]:
        return "wav"
    if raw.startswith(b"ID3") or raw[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa"}:
        return "mp3"
    if raw.startswith(b"OggS"):
        return "ogg"
    if raw.startswith(b"fLaC"):
        return "flac"
    return "bin"


def _ffmpeg_to_wav(source: Path, destination: Path) -> bool:
    binary = shutil.which("ffmpeg")
    if not binary:
        return False
    try:
        subprocess.run(  # noqa: S603 — fixed binary, no shell
            [binary, "-y", "-i", str(source), str(destination)],
            capture_output=True,
            timeout=90,
            check=True,
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("ffmpeg wav convert failed: %s", exc)
        return False
    return destination.is_file() and destination.stat().st_size >= _MIN_AUDIO_BYTES


def _ensure_wav(raw: bytes, destination: Path) -> Path:
    """Write ``raw`` as WAV, converting MP3/other containers when needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    kind = _sniff_audio(raw)
    if kind == "wav":
        destination.write_bytes(raw)
        return destination
    tmp = destination.with_suffix(f".src.{kind}")
    tmp.write_bytes(raw)
    try:
        if _ffmpeg_to_wav(tmp, destination):
            return destination
        from moviepy import AudioFileClip  # noqa: PLC0415

        with AudioFileClip(str(tmp)) as clip:
            clip.write_audiofile(
                str(destination),
                codec="pcm_s16le",
                fps=int(getattr(clip, "fps", 44100) or 44100),
                logger=None,
            )
        if destination.is_file() and destination.stat().st_size >= _MIN_AUDIO_BYTES:
            return destination
    except Exception as exc:  # noqa: BLE001
        raise BgmError(f"could not convert Lyria audio to WAV: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    raise BgmError("Lyria returned audio that could not be written as WAV")


def _decode_stream_chunks(chunks: list[str]) -> bytes | None:
    if not chunks:
        return None
    joined = "".join(chunks)
    decoded = _decode_b64(joined)
    if decoded:
        return decoded
    pieces = bytearray()
    for chunk in chunks:
        part = _decode_b64(chunk)
        if part:
            pieces.extend(part)
    return bytes(pieces) if len(pieces) >= _MIN_AUDIO_BYTES else None


def _audio_from_sse_text(text: str) -> bytes | None:
    """Parse an SSE body into audio bytes."""
    chunks: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        choice = (event.get("choices") or [{}])[0]
        message = choice.get("message")
        if message:
            found = extract_audio_bytes(message)
            if found:
                return found
        delta = choice.get("delta") or {}
        audio = delta.get("audio") or {}
        if isinstance(audio, dict) and audio.get("data"):
            chunks.append(str(audio["data"]))
        elif isinstance(audio, str):
            chunks.append(audio)
        content = delta.get("content")
        if content:
            found = extract_audio_bytes(content)
            if found and not chunks:
                return found
    decoded = _decode_stream_chunks(chunks)
    return decoded


def _audio_from_http(response: object) -> bytes | None:
    """Accept either a JSON chat-completions body or an SSE audio stream."""
    text = getattr(response, "text", "") or ""
    content_type = str((getattr(response, "headers", {}) or {}).get("content-type") or "").lower()
    if "text/event-stream" in content_type or text.lstrip().startswith("data:"):
        found = _audio_from_sse_text(text)
        if found:
            return found
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        if isinstance(body.get("error"), dict):
            return None
        found = extract_audio_bytes(body)
        if found:
            return found
    return _audio_from_sse_text(text) if text else None


def _lyria_headers(settings: object) -> dict[str, str]:
    try:
        from ..settings import require_secret  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from settings import require_secret  # type: ignore[no-redef]

    gateway = getattr(settings, "openrouter", None)
    return {
        "Authorization": f"Bearer {require_secret('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": getattr(gateway, "referer", "https://github.com/mediaupscale/aiwake"),
        "X-Title": getattr(gateway, "title", "Aiwake Debate Engine"),
    }


def _lyria_url(settings: object) -> str:
    gateway = getattr(settings, "openrouter", None)
    base = getattr(gateway, "base_url", "https://openrouter.ai/api/v1")
    return f"{base.rstrip('/')}/chat/completions"


def _lyria_payload(model: str, prompt: str, *, stream: bool) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "stream": stream,
    }


def generate_lyria_clip(
    settings: object,
    *,
    prompt: str,
    destination: Path,
    loop: str = "preview",
    allow_inspection_overwrite: bool = False,
) -> Path:
    """Call OpenRouter Lyria 3 and write ``destination``.

    Existing files are replaced only after a usable clip has been decoded, so a
    failed generation leaves the previous file intact.

    ``loop``:
        ``preview`` — bake two copies so the inspection seam is audible.
        ``production`` — tile to ~60 s with the configured equal-power overlap.
        ``none`` — leave the decoded clip as-is.

    Raises:
        BgmError: Missing key, HTTP failure, locked-inspection refusal, or an
            unusable audio payload.
    """
    import requests  # noqa: PLC0415

    if destination.name in LOCKED_BGM_FILENAMES:
        if destination.name != TEST_BGM_FILENAME or not allow_inspection_overwrite:
            raise BgmError(f"refusing to overwrite locked BGM track {destination.name}")

    bgm = getattr(getattr(settings, "audio", None), "bgm", None) or BgmConfig()
    model = (bgm.model or LYRIA_MODEL).strip()
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        raise BgmError("Lyria prompt is empty")
    target = destination
    url = _lyria_url(settings)

    raw: bytes | None = None
    last_error = ""
    # OpenRouter requires stream:true for audio output. Non-stream is a fallback
    # in case a provider returns a complete JSON blob instead of SSE.
    attempts: tuple[dict[str, object], ...] = (
        {"stream": True, "include_audio": True},
        {"stream": True, "include_audio": False},
        {"stream": False, "include_audio": True},
    )
    try:
        for attempt in attempts:
            stream = bool(attempt["stream"])
            payload = _lyria_payload(model, prompt_text, stream=stream)
            if not attempt["include_audio"]:
                payload.pop("audio", None)
            headers = dict(_lyria_headers(settings))
            if stream:
                headers["Accept"] = "text/event-stream"
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload).encode("utf-8"),
                timeout=_LYRIA_TIMEOUT_S,
                stream=stream,
            )
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                _LOG.warning("lyria %s", last_error)
                continue
            raw = _audio_from_http(response)
            if raw:
                break
            preview = (getattr(response, "text", "") or "")[:240].replace("\n", " ")
            last_error = f"response contained no audio payload ({preview!r})"
            _LOG.warning("lyria %s", last_error)
    except requests.RequestException as exc:
        raise BgmError(f"Lyria transport error: {exc}") from exc

    if not raw:
        raise BgmError(last_error or "Lyria returned no audio")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.stem}.partial.wav"
    written = _ensure_wav(raw, staging)
    if target.exists():
        try:
            target.unlink()
        except OSError as exc:
            staging.unlink(missing_ok=True)
            raise BgmError(f"could not overwrite {target}: {exc}") from exc
        _LOG.info("overwrote previous BGM %s", target.name)
    written.replace(target)
    overlap = float(getattr(bgm, "loop_crossfade_s", 1.5))
    if loop == "preview":
        _bake_loop(target, overlap_s=overlap)
    elif loop == "production":
        _bake_loop(target, overlap_s=overlap, target_s=_PRODUCTION_TARGET_S)
    _LOG.info("wrote BGM %s (%.1f KB)", target, target.stat().st_size / 1024)
    return target


def generate_test_bgm(settings: object, *, destination: Path | None = None) -> Path:
    """Call OpenRouter Lyria 3 and force-overwrite the inspection WAV.

    An existing ``test_track_lyria.wav`` is replaced only after a usable clip
    has been decoded, so a failed generation leaves the previous file intact.

    Raises:
        BgmError: Missing key, HTTP failure, or an unusable audio payload.
    """
    bgm = getattr(getattr(settings, "audio", None), "bgm", None) or BgmConfig()
    prompt = (bgm.prompt or LYRIA_TEST_PROMPT).strip()
    target = destination or test_bgm_path(relative=bgm.test_track or TEST_BGM_RELATIVE)
    return generate_lyria_clip(
        settings,
        prompt=prompt,
        destination=target,
        loop="preview",
        allow_inspection_overwrite=True,
    )


def _resolve_track_source(track: object, dest_dir: Path) -> Path | None:
    """Absolute path of a locked source WAV, or None when the track is synthesized."""
    source_name = str(getattr(track, "source", "") or "").strip()
    if not source_name:
        return None
    source = Path(source_name)
    if not source.is_absolute():
        source = dest_dir / source.name
    return source


def _materialize_from_source(
    source: Path,
    dest: Path,
    *,
    overlap_s: float,
    bake_production_loop: bool,
) -> Path:
    """Copy a locked source onto a library filename without mutating the source."""
    if not source.is_file() or source.stat().st_size < _MIN_AUDIO_BYTES:
        raise BgmError(f"locked source missing: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.resolve() == source.resolve():
        return dest
    if dest.exists() and dest.stat().st_size >= _MIN_AUDIO_BYTES:
        _LOG.info("keeping existing approved alias %s", dest.name)
        return dest
    shutil.copy2(source, dest)
    _LOG.info("copied locked source %s -> %s", source.name, dest.name)
    if bake_production_loop:
        _bake_loop(dest, overlap_s=overlap_s, target_s=_PRODUCTION_TARGET_S)
    return dest


def generate_bgm_batch(settings: object) -> list[Path]:
    """Materialize the Aiwake BGM library. Never overwrites locked or existing beds."""
    bgm = getattr(getattr(settings, "audio", None), "bgm", None) or BgmConfig()
    if not bool(getattr(bgm, "approved", False)):
        raise BgmError("inspection track is not approved; batch generation stays locked")
    dest_dir = bgm_library_dir(relative=bgm.test_track or TEST_BGM_RELATIVE)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failures: list[dict[str, str]] = []
    library = tuple(getattr(bgm, "library", ()) or ())
    if not library:
        raise BgmError("BGM library is empty")
    overlap = float(getattr(bgm, "loop_crossfade_s", 1.5))
    for track in library:
        filename = str(getattr(track, "filename", "") or "").strip()
        prompt = str(getattr(track, "prompt", "") or "").strip()
        if not filename:
            failures.append({"filename": "(unnamed)", "error": "missing filename"})
            continue
        dest = dest_dir / Path(filename).name
        if dest.name == TEST_BGM_FILENAME:
            _LOG.info("skipping library slot that would overwrite the inspection bed")
            continue
        approved = bool(getattr(track, "approved", False))
        source = _resolve_track_source(track, dest_dir)
        try:
            if approved or source is not None:
                if source is None:
                    if dest.is_file() and dest.stat().st_size >= _MIN_AUDIO_BYTES:
                        written.append(dest)
                        continue
                    raise BgmError(f"approved track {dest.name} has no source and is missing")
                written.append(
                    _materialize_from_source(
                        source,
                        dest,
                        overlap_s=overlap,
                        bake_production_loop=dest.name == "bgm_aiwake_01_core_suspense.wav",
                    )
                )
                continue
            if not prompt:
                failures.append({"filename": filename, "error": "missing filename or prompt"})
                continue
            if dest.name in LOCKED_BGM_FILENAMES:
                raise BgmError(f"refusing to overwrite locked BGM track {dest.name}")
            if dest.is_file() and dest.stat().st_size >= _MIN_AUDIO_BYTES:
                _LOG.info("keeping existing library track %s", dest.name)
                written.append(dest)
                continue
            written.append(
                generate_lyria_clip(
                    settings,
                    prompt=prompt,
                    destination=dest,
                    loop="production",
                    allow_inspection_overwrite=False,
                )
            )
        except BgmError as exc:
            _LOG.warning("library track %s failed: %s", filename, exc)
            failures.append({"filename": filename, "error": str(exc)})
    _write_library_manifest(settings, written=written, failures=failures)
    if not written:
        detail = "; ".join(f"{item['filename']}: {item['error']}" for item in failures) or "no tracks"
        raise BgmError(f"BGM batch produced no files ({detail})")
    return written


def _write_library_manifest(
    settings: object,
    *,
    written: list[Path],
    failures: list[dict[str, str]],
) -> Path:
    """Log the approved inspection bed plus library status next to the WAVs."""
    from datetime import datetime, timezone  # noqa: PLC0415

    bgm = getattr(getattr(settings, "audio", None), "bgm", None) or BgmConfig()
    dest_dir = bgm_library_dir(relative=getattr(bgm, "test_track", None) or TEST_BGM_RELATIVE)
    dest = dest_dir / BGM_MANIFEST_FILENAME
    written_names = {path.name for path in written}
    failed_names = {item.get("filename", "") for item in failures}
    library_rows = []
    for track in tuple(getattr(bgm, "library", ()) or ()):
        filename = str(getattr(track, "filename", "") or "")
        approved = bool(getattr(track, "approved", False))
        path = dest_dir / filename
        library_rows.append(
            {
                "id": str(getattr(track, "id", "") or ""),
                "filename": filename,
                "role": str(getattr(track, "role", "") or ""),
                "source": str(getattr(track, "source", "") or ""),
                "approved": approved,
                "status": "approved" if approved else "pending_review",
                "present": path.is_file() and path.stat().st_size >= _MIN_AUDIO_BYTES,
                "generated": filename in written_names,
                "failed": filename in failed_names,
            }
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inspection": {
            "file": TEST_BGM_FILENAME,
            "relative": getattr(bgm, "test_track", TEST_BGM_RELATIVE),
            "approved": True,
            "role": "Sci-Fi Suspense inspection bed (Aiwake / Ancient Knowledge)",
            "present": test_bgm_path(
                relative=getattr(bgm, "test_track", None) or TEST_BGM_RELATIVE
            ).is_file(),
        },
        "locked_sources": sorted(LOCKED_BGM_FILENAMES),
        "mix_contract": {
            "gain_db": float(getattr(bgm, "gain_db", -21.0)),
            "loop_crossfade_s": float(getattr(bgm, "loop_crossfade_s", 1.5)),
            "production_target_s": _PRODUCTION_TARGET_S,
            "note": "Gain is applied at mix time versus TTS; WAV files stay full-scale.",
        },
        "library": library_rows,
        "written": [str(path) for path in written],
        "failures": failures,
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _LOG.info("wrote BGM library manifest %s", dest)
    return dest


def _write_wav_pcm(path: Path, samples: object, fps: int) -> None:
    """Write float32 stereo samples as 16-bit PCM WAV."""
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    arr = _as_stereo(samples)
    frames = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(frames.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(fps))
        handle.writeframes(frames.tobytes())


def _bake_loop(path: Path, *, overlap_s: float, target_s: float | None = None) -> None:
    """Bake an equal-power loop into the WAV.

    ``target_s is None``: inspection preview — two copies so the seam is audible.
    Otherwise tile to at least ``target_s`` seconds.
    """
    try:
        from moviepy import AudioFileClip  # noqa: PLC0415
    except ImportError:
        return
    try:
        with AudioFileClip(str(path)) as clip:
            fps = int(getattr(clip, "fps", 44100) or 44100)
            duration = float(clip.duration or 0.0)
            arr = clip.to_soundarray(fps=fps)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("loop bake skipped (load): %s", exc)
        return
    if duration < (overlap_s * 2.0) + 0.25:
        _LOG.info("clip too short for a %.1fs loop crossfade; leaving single pass", overlap_s)
        return
    target_dur = (duration * 2.0) - overlap_s if target_s is None else max(float(target_s), duration)
    looped = crossfade_tile(arr, fps=fps, duration_s=target_dur, overlap_s=overlap_s)
    _write_wav_pcm(path, looped, fps)
    _LOG.info("baked %.1fs equal-power loop into %s (%.1fs)", overlap_s, path.name, target_dur)


def _render_loop_preview(path: Path, *, overlap_s: float) -> None:
    """Bake two crossfaded copies into the inspection WAV so the seam is audible."""
    _bake_loop(path, overlap_s=overlap_s)


def concat_audio(
    assets: list[AudioAsset],
    destination: Path,
    *,
    gap_s: float = 0.0,
    preroll_s: float = 0.0,
    reply_gap_s: float = 0.0,
    post_response_s: float = 0.0,
    postroll_s: float = 0.0,
    send_hold_s: float = 0.0,
    cta_overlay_path: Path | None = None,
    cta_overlay_start_s: float = 0.0,
    typewriter: TypewriterConfig | None = None,
    typing_hold_ratio: float = _DEFAULT_HOLD_RATIO,
    bgm: BgmConfig | None = None,
    send_sfx: SendSfxConfig | None = None,
    master_gain_db: float = 6.0,
) -> Path | None:
    """Stitch voice tracks into one continuous timeline.

    A ``preroll_s`` bed of silence starts the mix. After each CORE (sent)
    question, ``reply_gap_s`` of silence holds before the incoming response;
    after a target rebuttal, ``post_response_s`` holds before the next
    provocation. ``gap_s`` is extra silence between every pair of tracks. A
    ``postroll_s`` bed of silence is appended so the CTA end-card has room.
    When ``cta_overlay_path`` is given it is layered onto the mix beginning at
    ``cta_overlay_start_s`` (the moment the CTA card finishes fading in).

    Per-track tail padding still matches ``asset.duration_s`` so the renderer's
    segment boundaries line up. When ``typewriter`` is enabled, keyboard clicks
    from ``keyboard_typing.wav`` are mixed under AIWAKE.CORE lines for the
    character reveal window. A send click fires at that instant. Target replies
    are left dry. BGM, when approved, rides under the whole timeline.
    When BGM is enabled and the inspection WAV exists, that bed is looped under
    the whole timeline at ``bgm.gain_db`` (-21 dB vs speech by default) with a
    1.5 s loop crossfade, a 1.5 s fade-in and a 2.0 s fade-out.

    Returns:
        The written path, or None when MoviePy is unavailable or no real tracks
        exist (silent engine).
    """
    real = [asset for asset in assets if asset.path.is_file()]
    if not real:
        return None
    try:
        import numpy as np  # noqa: PLC0415
        from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips  # noqa: PLC0415
        from moviepy.audio.AudioClip import AudioArrayClip  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        _LOG.warning("cannot concatenate audio: %s", exc)
        return None

    mix_sfx = typewriter is not None and typewriter.enabled
    mix_send = send_sfx is not None and bool(getattr(send_sfx, "enabled", True))
    typing_window = max(1e-6, 1.0 - typing_hold_ratio)
    if mix_sfx and typewriter is not None:
        try:
            ensure_keyboard_typing_asset(typewriter)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("keyboard_typing.wav ensure skipped: %s", exc)
    if mix_send and send_sfx is not None:
        try:
            ensure_send_sfx_asset(send_sfx)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("message_sent.wav ensure skipped: %s", exc)
    clips = []
    opened = []
    deferred_postroll_click_at: float | None = None
    try:
        lead = AudioFileClip(str(real[0].path))
        opened.append(lead)
        mix_fps = int(getattr(lead, "fps", 44100) or 44100)
        if preroll_s > 0.01:
            clips.append(
                AudioArrayClip(
                    np.zeros((max(1, int(mix_fps * preroll_s)), 2), dtype="float32"),
                    fps=mix_fps,
                )
            )
        for index, asset in enumerate(real):
            pending_gap_click_at: float | None = None
            clip = lead if index == 0 else AudioFileClip(str(asset.path))
            if index != 0:
                opened.append(clip)
            pad = asset.duration_s - float(clip.duration or 0.0)
            pieces = [clip]
            fps = int(getattr(clip, "fps", mix_fps) or mix_fps)
            if pad > 0.01:
                samples = np.zeros((int(fps * pad), 2), dtype="float32")
                pieces.append(AudioArrayClip(samples, fps=fps))
            voice_full = concatenate_audioclips(pieces) if len(pieces) > 1 else clip
            opened.append(voice_full)

            if (
                mix_sfx
                and typewriter is not None
                and asset.char_count > 0
                and asset.role == _CORE_ROLE
            ):
                typing_s = min(float(voice_full.duration or 0.0), asset.duration_s * typing_window)
                try:
                    sfx = AudioArrayClip(
                        typewriter_samples(
                            typing_s,
                            asset.char_count,
                            typewriter,
                            fps=fps,
                        ),
                        fps=fps,
                    )
                    mixed = CompositeAudioClip([voice_full, sfx]).with_duration(asset.duration_s)
                    opened.append(sfx)
                    opened.append(mixed)
                    voice_full = mixed
                except Exception as exc:  # noqa: BLE001
                    _LOG.debug("typewriter mix skipped for %s: %s", asset.path.name, exc)
            if mix_send and send_sfx is not None and asset.role == _CORE_ROLE:
                typing_s = min(float(voice_full.duration or 0.0), asset.duration_s * typing_window)
                # Fire the click at the same instant as the visual send flash:
                # after the read-hold that follows typing completion. When that
                # lands past the voice clip (into reply_gap / postroll), stash
                # the offset so the following silence carries the click.
                click_at = typing_s + max(0.0, float(send_hold_s))
                if click_at + 0.04 <= float(asset.duration_s):
                    try:
                        click = AudioArrayClip(
                            send_click_on_timeline(
                                asset.duration_s,
                                click_at,
                                send_sfx,
                                fps=fps,
                            ),
                            fps=fps,
                        )
                        mixed = CompositeAudioClip([voice_full, click]).with_duration(asset.duration_s)
                        opened.append(click)
                        opened.append(mixed)
                        clips.append(mixed)
                    except Exception as exc:  # noqa: BLE001
                        _LOG.debug("send sfx mix skipped for %s: %s", asset.path.name, exc)
                        clips.append(voice_full)
                else:
                    clips.append(voice_full)
                    pending_gap_click_at = click_at - float(asset.duration_s)
            else:
                clips.append(voice_full)

            extra_gap = 0.0
            if index < len(real) - 1:
                extra_gap = max(float(gap_s), 0.0)
                if asset.role == _CORE_ROLE:
                    extra_gap = max(extra_gap, float(reply_gap_s))
                else:
                    extra_gap = max(extra_gap, float(post_response_s))
            if extra_gap > 0.01:
                if pending_gap_click_at is not None and mix_send and send_sfx is not None:
                    try:
                        gap_buf = send_click_on_timeline(
                            extra_gap,
                            max(0.0, min(float(pending_gap_click_at), max(0.0, extra_gap - 0.05))),
                            send_sfx,
                            fps=fps,
                        )
                        clips.append(AudioArrayClip(gap_buf, fps=fps))
                        pending_gap_click_at = None
                    except Exception as exc:  # noqa: BLE001
                        _LOG.debug("send sfx gap mix skipped for %s: %s", asset.path.name, exc)
                        clips.append(
                            AudioArrayClip(
                                np.zeros((max(1, int(fps * extra_gap)), 2), dtype="float32"),
                                fps=fps,
                            )
                        )
                else:
                    clips.append(
                        AudioArrayClip(
                            np.zeros((max(1, int(fps * extra_gap)), 2), dtype="float32"),
                            fps=fps,
                        )
                    )
            elif pending_gap_click_at is not None:
                # Last CORE line: fold the click into the postroll bed so we do
                # not invent audio-only gap time that desyncs A/V.
                deferred_postroll_click_at = pending_gap_click_at
                pending_gap_click_at = None

        postroll_click_at = deferred_postroll_click_at
        if float(postroll_s) > 0.01:
            if postroll_click_at is not None and mix_send and send_sfx is not None:
                try:
                    post_buf = send_click_on_timeline(
                        float(postroll_s),
                        max(0.0, min(float(postroll_click_at), max(0.0, float(postroll_s) - 0.05))),
                        send_sfx,
                        fps=mix_fps,
                    )
                    clips.append(AudioArrayClip(post_buf, fps=mix_fps))
                except Exception as exc:  # noqa: BLE001
                    _LOG.debug("send sfx postroll mix skipped: %s", exc)
                    clips.append(
                        AudioArrayClip(
                            np.zeros((max(1, int(mix_fps * postroll_s)), 2), dtype="float32"),
                            fps=mix_fps,
                        )
                    )
            else:
                clips.append(
                    AudioArrayClip(
                        np.zeros((max(1, int(mix_fps * postroll_s)), 2), dtype="float32"),
                        fps=mix_fps,
                    )
                )

        stitched = concatenate_audioclips(clips)
        destination.parent.mkdir(parents=True, exist_ok=True)
        mix_target = stitched
        # Overlay the CTA voice at its start offset so the invite lands on the
        # end-card without dragging the rest of the timeline.
        if cta_overlay_path is not None and cta_overlay_path.is_file():
            try:
                cta_start = max(0.0, float(cta_overlay_start_s))
                cta_src = AudioFileClip(str(cta_overlay_path))
                opened.append(cta_src)
                cta_fps = int(getattr(cta_src, "fps", mix_fps) or mix_fps)
                bed = np.zeros((max(1, int(cta_start * cta_fps)), 2), dtype="float32")
                cta_padded = concatenate_audioclips(
                    [AudioArrayClip(bed, fps=cta_fps), cta_src]
                )
                opened.append(cta_padded)
                stitched_len = float(stitched.duration or 0.0)
                cta_padded = cta_padded.with_duration(max(stitched_len, float(cta_padded.duration or 0.0)))
                mix_target = CompositeAudioClip([stitched, cta_padded])
                opened.append(mix_target)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("CTA overlay skipped: %s", exc)
                mix_target = stitched
        bgm_path = resolve_bgm_track(bgm)
        if bgm_path is not None and bgm is not None:
            try:
                duration = float(mix_target.duration or stitched.duration or 0.0)
                bgm_src = AudioFileClip(str(bgm_path))
                opened.append(bgm_src)
                src_fps = int(getattr(bgm_src, "fps", 44100) or 44100)
                bed = prepare_bgm_bed(
                    bgm_src.to_soundarray(fps=src_fps),
                    fps=src_fps,
                    duration_s=duration,
                    gain_db=bgm.gain_db,
                    fade_in_s=bgm.fade_in_s,
                    fade_out_s=bgm.fade_out_s,
                    loop_crossfade_s=bgm.loop_crossfade_s,
                )
                bgm_clip = AudioArrayClip(bed, fps=src_fps).with_duration(duration)
                opened.append(bgm_clip)
                mix_target = CompositeAudioClip([mix_target, bgm_clip]).with_duration(duration)
                opened.append(mix_target)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("BGM overlay skipped: %s", exc)
        # Master loudness boost — preserves relative TTS/SFX/BGM balance.
        master_db = float(master_gain_db or 0.0)
        if abs(master_db) > 1e-6:
            factor = float(10.0 ** (master_db / 20.0))
            try:
                mix_target = mix_target.with_volume_scaled(factor)
                opened.append(mix_target)
            except Exception:  # noqa: BLE001
                try:
                    mix_target = mix_target.volumex(factor)  # type: ignore[attr-defined]
                    opened.append(mix_target)
                except Exception as exc:  # noqa: BLE001
                    _LOG.debug("master gain skipped: %s", exc)
        mix_target.write_audiofile(str(destination), logger=None)
        if mix_target is not stitched:
            mix_target.close()
        stitched.close()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("audio concatenation failed: %s", exc)
        return None
    finally:
        for clip in opened:
            try:
                clip.close()
            except Exception:  # noqa: BLE001, S110
                pass

    return destination


__all__ = [
    "AudioAsset",
    "BGM_APPROVAL_NOTICE",
    "BGM_APPROVED_NOTICE",
    "BgmError",
    "CTA_VOICE",
    "EdgeTTSEngine",
    "LYRIA_MODEL",
    "LYRIA_TEST_PROMPT",
    "LOCKED_BGM_FILENAMES",
    "SilentTTSEngine",
    "TEST_BGM_FILENAME",
    "TTSEngine",
    "TTSError",
    "apply_dramatic_pauses",
    "bgm_fade_envelope",
    "bgm_library_dir",
    "build_engine",
    "concat_audio",
    "count_dramatic_pauses",
    "crossfade_tile",
    "ensure_keyboard_typing_asset",
    "ensure_send_sfx_asset",
    "estimate_duration",
    "extract_audio_bytes",
    "flatten_cta_spoken",
    "generate_bgm_batch",
    "generate_lyria_clip",
    "generate_test_bgm",
    "keyboard_typing_path",
    "prepare_bgm_bed",
    "prepare_tts_text",
    "probe_duration",
    "pronounce_cta_text",
    "resolve_bgm_track",
    "resolve_cta_voice",
    "resolve_voice",
    "synthesize_typewriter_clicks",
    "synthesize_cta_line",
    "synthesize_send_click",
    "test_bgm_path",
    "typewriter_samples",
    "write_keyboard_typing_wav",
]
