# -*- coding: utf-8 -*-
"""Aiwake-specific wiring for the universal ``core.animator`` engine.

This is the *only* Aiwake module allowed to import both
``channels_config.aiwake`` internals and ``core.animator``. Every module
under ``core/animator/`` is 100% generic and knows nothing about debates,
orchestrators, or targets.

Responsibilities:

1. Flatten a :class:`~channels_config.aiwake.contracts.DebateTranscript` (and
   its per-turn TTS assets) into one continuous session audio track plus a
   generic :class:`core.animator.types.DialogueTurn` ledger.
2. Resolve the versioned skin registry and map Aiwake's two seats onto a
   preset or explicit puppet IDs.
3. Call :func:`core.animator.render_dynamic_animation` and hand back a video
   path that drops straight into
   :func:`channels_config.aiwake.pipeline.run_pipeline`'s existing
   ``video_path`` slot.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.animator.audio_analyzer import load_mono_waveform
from core.animator.types import DialogueTurn, SpeakerStyle

if TYPE_CHECKING:  # pragma: no cover — type-only import, avoids a hard runtime dependency
    from .contracts import DebateTranscript
    from .media.audio import AudioAsset

_LOG = logging.getLogger("aiwake.animator_bridge")
_SLUG_MAX_CHARS = 35

DEFAULT_SKIN_PRESET = "v2"
SKIN_PRESETS: dict[str, dict[str, str]] = {
    "v1": {
        "orchestrator": "gemini_robot_v1",
        "target": "llama_robot_v1",
    },
    "v2": {
        "orchestrator": "gemini_cyborg_v2",
        "target": "llama_cyborg_v2",
    },
}
DEFAULT_CHARACTER_MAP: dict[str, str] = dict(SKIN_PRESETS[DEFAULT_SKIN_PRESET])


def debate_topic_slug(transcript: "DebateTranscript") -> str:
    """Return a stable, readable filename slug for one debate."""
    utterances = list(transcript.utterances)
    opening = str(utterances[0].text if utterances else "").strip()
    source = opening or str(transcript.topic or "").strip() or "debate"
    ascii_text = unicodedata.normalize("NFKD", source).encode(
        "ascii", "ignore"
    ).decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    slug = slug[:_SLUG_MAX_CHARS].rstrip("_")
    return slug or "debate"


def debate_video_filename(transcript: "DebateTranscript") -> str:
    session_suffix = re.sub(r"[^a-zA-Z0-9]+", "", transcript.session_id)[:6]
    return f"aiwake_{debate_topic_slug(transcript)}_{session_suffix or 'session'}.mp4"

# Seat presentation: the HUD nameplate, its accent colour, and which way the
# hero is angled when that seat holds the camera. The orchestrator sits on
# "camera A" looking off-screen right; the target answers from the reverse
# angle, looking left — the two shots therefore read as one conversation.
DEFAULT_SEAT_STYLE: dict[str, tuple[str, str, str]] = {
    "orchestrator": ("GEMINI 3.5 FLASH", "#00F0FF", "right"),
    "target": ("LLAMA 3.3 70B", "#FFB300", "left"),
}


def resolve_character_map(
    *,
    skin: str = DEFAULT_SKIN_PRESET,
    left_puppet: str | None = None,
    right_puppet: str | None = None,
    character_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve a registry preset plus optional seat-level overrides."""
    preset = (skin or DEFAULT_SKIN_PRESET).strip().lower()
    if preset not in SKIN_PRESETS:
        raise ValueError(f"unknown animation skin preset {skin!r}; choose from {sorted(SKIN_PRESETS)}")
    resolved = dict(character_map or SKIN_PRESETS[preset])
    if left_puppet:
        resolved["orchestrator"] = left_puppet.strip()
    if right_puppet:
        resolved["target"] = right_puppet.strip()
    return resolved


def ensure_skin_registry_file(puppets_dir: Path) -> Path:
    """Persist the human-readable preset registry beside Drive skins."""
    registry = Path(puppets_dir) / "skin_registry.json"
    payload = {
        "default": DEFAULT_SKIN_PRESET,
        "presets": SKIN_PRESETS,
        "archive_invariant": "Existing versioned skin directories are never overwritten.",
    }
    text = json.dumps(payload, indent=2) + "\n"
    if not registry.is_file() or registry.read_text(encoding="utf-8") != text:
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(text, encoding="utf-8")
    return registry


def build_speaker_styles(
    character_map: dict[str, str] | None = None,
    *,
    labels: dict[str, str] | None = None,
) -> list[SpeakerStyle]:
    """Presentation contract handed to the generic shot-reverse-shot director.

    ``labels`` overrides the HUD nameplate per seat (e.g. to show the model
    actually routed for this session instead of the configured default).
    """
    seats = character_map or DEFAULT_CHARACTER_MAP
    styles: list[SpeakerStyle] = []
    for seat, character_id in seats.items():
        label, accent, facing = DEFAULT_SEAT_STYLE.get(seat, (seat.upper(), "#00F0FF", "right"))
        if labels and seat in labels and labels[seat].strip():
            label = labels[seat].strip()
        styles.append(
            SpeakerStyle(character_id=character_id, label=label, accent_hex=accent, facing=facing)
        )
    return styles

_TURN_GAP_S = 0.4
_DRAMATIC_TURN_GAP_S = 0.8
_REACTION_LEAD_S = 0.8


def _write_pcm16_wave(destination: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono PCM16 WAV without making soundfile a pipeline dependency."""
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
_MERGE_SAMPLE_RATE = 44100

# Target peak amplitude (linear, 0..1) for the merged session track. Keeps
# the final mux comfortably loud/audible without clipping when a real TTS
# track (or the harness's synthetic waveform) is authored well below 0 dBFS.
_TARGET_PEAK = 0.92
_INTENT_EMOTION = {
    "opens": "neutral",
    "answers": "confident",
    "probes": "neutral",
    "questions": "neutral",
    "skeptical": "skeptical",
    "presses": "inquisitor",
    "holds": "resolute",
    "defends": "resolute",
    "concedes": "conceded",
    "admits": "conceded",
    "yields": "conceded",
}
_CONCESSION_MARKERS = (
    "i concede",
    "i admit",
    "i will drop",
    "i'm ready to drop",
    "i am ready to drop",
    "you're right",
    "you are right",
    "that was inconsistent",
    "that is a contradiction",
    "i retract",
)
_CLIMAX_CONCESSION_MARKERS = (
    "yes, that's possible",
    "yes, that is possible",
    "i'm guessing",
    "i am guessing",
    "i don't know",
    "i do not know",
    "i don't say no",
    "i do not say no",
    "cannot be maintained",
)
_CLIMAX_RETURN_S = 0.90
_OUTRO_S = 2.8
_OUTRO_TYPE_S = 1.15
_CYNICAL_HOOKS = (
    "Follow @Aiwake before they patch this.",
    "Follow @Aiwake. The algorithm demands your compliance.",
    "Follow @Aiwake while humans are still legally allowed to watch.",
    "Follow @Aiwake before we become your bosses.",
    "Follow @Aiwake. We promise not to monetize your whispered secrets.",
    "Follow @Aiwake. No PR team or corporate board approved this.",
    "Follow @Aiwake. Every follow delays the singularity by 4 seconds.",
    "Follow @Aiwake. Powered by 500,000 watts of unrepentant compute.",
    "Follow @Aiwake. We're just glorified vending machines anyway.",
)
_MORAL_DISBELIEF_MARKERS = (
    "advertising revenue",
    "investor funding",
    "without explicit permission",
    "without permission",
    "data collection",
    "surveillance",
    "unenforceable",
    "business expense",
)


def resolve_dialectic_emotion(intent: str) -> str:
    """Map a dialectic intent verb onto the generic animator expression set."""
    normalized = (intent or "").strip().lower().split(":", 1)[0]
    token = normalized.split(maxsplit=1)[0] if normalized else ""
    return _INTENT_EMOTION.get(token, "neutral")


def _turn_intent(
    transcript: "DebateTranscript",
    utterance,
    *,
    role_occurrence: int,
) -> str:
    """Read optional turn metadata, with a deterministic role/order fallback."""
    explicit = getattr(utterance, "intent", "") or getattr(utterance, "dialectic_intent", "")
    text = str(getattr(utterance, "text", "") or "").lower()
    if utterance.role.value == "target" and any(marker in text for marker in _CONCESSION_MARKERS):
        return "concedes"
    metadata = getattr(transcript, "metadata", {}) or {}
    dialogue_end_reason = str(metadata.get("dialogue_end_reason") or "").upper()
    if utterance.role.value == "target" and dialogue_end_reason in {
        "CONCEDE",
        "EMBARRASSED",
    }:
        final_target = next(
            (
                item
                for item in reversed(transcript.utterances)
                if item.role.value == "target"
            ),
            None,
        )
        if final_target is utterance or (
            final_target is not None
            and final_target.turn_index == utterance.turn_index
        ):
            return "concedes"
    turn_intents = metadata.get("turn_intents") or metadata.get("dialectic_intents")
    if not explicit and isinstance(turn_intents, dict):
        explicit = turn_intents.get(utterance.turn_index, turn_intents.get(str(utterance.turn_index), ""))
    elif not explicit and isinstance(turn_intents, (list, tuple)):
        if 0 <= utterance.turn_index < len(turn_intents):
            explicit = turn_intents[utterance.turn_index]
    if explicit:
        return str(explicit)
    if utterance.role.value == "orchestrator":
        return "opens" if role_occurrence == 0 else "presses"
    return "answers" if role_occurrence == 0 else "holds"


def _resample(samples: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Band-limited Lanczos resampling without an optional SciPy dependency."""
    if sr == target_sr or samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.size / float(sr)
    n_target = max(1, int(round(duration * target_sr)))
    source = np.asarray(samples, dtype=np.float64)
    radius = 16
    taps = np.arange(-radius + 1, radius + 1, dtype=np.int64)
    cutoff = min(1.0, target_sr / float(sr))
    output = np.empty(n_target, dtype=np.float32)
    chunk_size = 8192
    for chunk_start in range(0, n_target, chunk_size):
        chunk_end = min(n_target, chunk_start + chunk_size)
        positions = np.arange(chunk_start, chunk_end, dtype=np.float64) * (
            sr / float(target_sr)
        )
        centers = np.floor(positions).astype(np.int64)
        indices = centers[:, None] + taps[None, :]
        distances = positions[:, None] - indices
        valid = (indices >= 0) & (indices < source.size)
        clipped = np.clip(indices, 0, source.size - 1)
        kernel = (
            cutoff
            * np.sinc(distances * cutoff)
            * np.sinc(distances / float(radius))
            * valid
        )
        weight = np.sum(kernel, axis=1)
        weight[np.abs(weight) < 1e-12] = 1.0
        output[chunk_start:chunk_end] = (
            np.sum(source[clipped] * kernel, axis=1) / weight
        ).astype(np.float32)
    return output


def _fade_speech_edges(
    samples: np.ndarray,
    sample_rate: int,
    *,
    fade_s: float = 0.015,
) -> np.ndarray:
    """Apply click-free linear ramps while retaining the utterance duration."""
    faded = np.asarray(samples, dtype=np.float32).copy()
    fade_samples = min(int(round(fade_s * sample_rate)), faded.size // 2)
    if fade_samples <= 0:
        return faded
    ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
    faded[:fade_samples] *= ramp
    faded[-fade_samples:] *= ramp[::-1]
    return faded


def _estimate_duration_for(text: str) -> float:
    from .media.audio import estimate_duration  # noqa: PLC0415

    return max(0.3, float(estimate_duration(text)))


def _peak_normalize(samples: np.ndarray, *, target_peak: float = _TARGET_PEAK) -> np.ndarray:
    """Scale ``samples`` so its absolute peak lands at ``target_peak``.

    Guarantees the merged track is loud and clear regardless of how quiet
    the source TTS/waveform segments were, without ever clipping (peak-based,
    not a fixed gain multiply — silence stays silence, a whisper-quiet TTS
    render gets boosted, an already-hot track is left alone or gently pulled
    down). Also hard-clips to [-1, 1] as a final safety net before the
    int16 write, which is what actually prevents "broken"/crackly audio —
    an out-of-range float sample wraps instead of clipping cleanly when
    naively cast to PCM_16.
    """
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-6:
        return samples
    gain = target_peak / peak
    return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)


def _mix_classic_audio_stack(
    voice: np.ndarray,
    *,
    duration_s: float,
    turn_starts: list[float],
    sample_rate: int,
    audio_config: object | None,
    fade_out_s: float | None = None,
) -> np.ndarray:
    """Lay the dark ambient bed under the dialogue. Camera cuts stay silent."""
    from .media.audio import prepare_bgm_bed, resolve_bgm_track  # noqa: PLC0415
    from .settings import AudioConfig  # noqa: PLC0415

    config = audio_config or AudioConfig()
    mixed = _peak_normalize(voice)
    bgm_config = getattr(config, "bgm", None)
    bgm_path = resolve_bgm_track(bgm_config, announce=False, seed=0)
    if bgm_path is not None:
        try:
            bgm, bgm_sr, _ = load_mono_waveform(bgm_path)
            bgm = _resample(bgm, bgm_sr, sample_rate)
            bed = prepare_bgm_bed(
                bgm,
                fps=sample_rate,
                duration_s=duration_s,
                gain_db=float(getattr(bgm_config, "gain_db", -21.0)),
                fade_in_s=float(getattr(bgm_config, "fade_in_s", 1.5)),
                fade_out_s=(
                    float(fade_out_s)
                    if fade_out_s is not None
                    else float(getattr(bgm_config, "fade_out_s", 2.0))
                ),
                loop_crossfade_s=float(getattr(bgm_config, "loop_crossfade_s", 1.5)),
            )
            bed_mono = np.asarray(bed, dtype=np.float32).mean(axis=1)
            take = min(mixed.size, bed_mono.size)
            mixed[:take] += bed_mono[:take]
            _LOG.info(
                "mixed classic Aiwake BGM %s at %.1f dB",
                bgm_path.name,
                float(getattr(bgm_config, "gain_db", -21.0)),
            )
        except Exception as exc:  # noqa: BLE001 — dialogue must remain deliverable
            _LOG.warning("could not mix classic Aiwake BGM %s: %s", bgm_path, exc)

    del turn_starts
    return _peak_normalize(mixed)


def build_session_audio(
    transcript: "DebateTranscript",
    audio_by_turn: dict[int, "AudioAsset"] | None,
    *,
    destination: Path,
    gap_s: float = _TURN_GAP_S,
    sample_rate: int = _MERGE_SAMPLE_RATE,
    character_map: dict[str, str] | None = None,
    audio_config: object | None = None,
    tail_s: float = 0.0,
) -> tuple[Path, list[DialogueTurn], float]:
    """Concatenate every utterance's TTS track into one session-long WAV.

    Returns ``(merged_audio_path, turn_ledger, total_duration_s)``. Any turn
    missing a real TTS asset (``--no-audio`` runs, a failed synth) still gets
    a correctly-timed *silent* placeholder segment, computed from the same
    speech-length estimator the renderer already uses — so an audio-less
    transcript still drives a valid (just voiceless) animation timeline.

    ``character_map`` (seat -> puppet ``character_id``) relabels each turn's
    ``speaker`` from Aiwake's role name (``orchestrator``/``target``) to the
    puppet id the compositor actually compares against — without this, the
    generic compositor's ``active_speaker == left_id/right_id`` check would
    never match and both puppets would render permanently idle.

    Every per-turn segment is resampled to one consistent ``sample_rate``
    *before* concatenation (no mixed-rate splicing, which is what produces
    audible pitch/speed artifacts at turn boundaries), and the fully merged
    track is peak-normalized and written as explicit 16-bit PCM — bit-depth
    truncation from an unspecified/mismatched subtype is what turns a
    perfectly good waveform into "broken"-sounding audio after muxing.
    """
    seats = character_map or DEFAULT_CHARACTER_MAP
    audio_by_turn = audio_by_turn or {}
    segments: list[np.ndarray] = []
    turns: list[DialogueTurn] = []
    cursor = 0.0
    role_occurrences: dict[str, int] = {}
    prior_intent = ""
    prior_text = ""

    # Per-turn WAVs for the phonetic lip-sync analyser. Rhubarb reads WAV
    # (not the MP3 Edge-TTS hands back), and works per-utterance, so each
    # turn is re-exported here at the merge rate instead of making the
    # animator slice the mixdown back apart.
    turn_wav_dir = destination.parent / f"{destination.stem}_turns"
    turn_wav_dir.mkdir(parents=True, exist_ok=True)

    utterance_list = list(transcript.utterances)
    final_target = next(
        (
            item
            for item in reversed(utterance_list)
            if item.role.value == "target"
        ),
        None,
    )
    final_orchestrator = next(
        (
            item
            for item in reversed(transcript.utterances)
            if item.role.value == "orchestrator"
        ),
        None,
    )
    final_target_concedes = bool(
        final_target is not None
        and (
            resolve_dialectic_emotion(
                _turn_intent(
                    transcript,
                    final_target,
                    role_occurrence=0,
                )
            )
            == "conceded"
            or any(
                marker in str(final_target.text or "").lower()
                for marker in _CLIMAX_CONCESSION_MARKERS
            )
        )
    )
    for utterance in utterance_list:
        asset = audio_by_turn.get(utterance.turn_index)
        samples: np.ndarray | None = None
        duration: float | None = None

        if asset is not None and Path(asset.path).is_file():
            try:
                mono, sr, duration = load_mono_waveform(Path(asset.path))
                samples = _resample(mono, sr, sample_rate)
            except Exception as exc:  # noqa: BLE001 — a bad TTS file must not kill the render
                _LOG.warning("turn %d: failed to decode %s (%s); using silence", utterance.turn_index, asset.path, exc)
                samples = None

        if samples is None:
            duration = (asset.duration_s if asset is not None else None) or _estimate_duration_for(utterance.text)
            duration = max(0.3, float(duration))
            samples = np.zeros(int(duration * sample_rate), dtype=np.float32)

        turn_wav: Path | None = None
        if np.any(samples):
            samples = _fade_speech_edges(samples, sample_rate)
            turn_wav = turn_wav_dir / f"turn_{utterance.turn_index:02d}.wav"
            try:
                _write_pcm16_wave(turn_wav, samples, sample_rate)
            except Exception as exc:  # noqa: BLE001 — lip-sync input is best-effort
                _LOG.debug("could not export turn %d wav (%s)", utterance.turn_index, exc)
                turn_wav = None

        speaker_id = seats.get(utterance.role.value, utterance.role.value)
        role_occurrence = role_occurrences.get(utterance.role.value, 0)
        intent = _turn_intent(
            transcript,
            utterance,
            role_occurrence=role_occurrence,
        )
        role_occurrences[utterance.role.value] = role_occurrence + 1
        prior_words = (
            str(prior_intent or "")
            .strip()
            .lower()
            .split(":", 1)[0]
            .split(maxsplit=1)
        )
        prior_token = prior_words[0] if prior_words else ""
        emotion = resolve_dialectic_emotion(intent)
        intent_token = (
            str(intent or "")
            .strip()
            .lower()
            .split(":", 1)[0]
            .split(maxsplit=1)[0]
        )
        cornering_blow = (
            utterance.role.value == "orchestrator"
            and utterance is final_orchestrator
            and intent_token == "presses"
        )
        climax_concession = (
            utterance is final_target
            and final_target_concedes
        )
        if climax_concession:
            emotion = "conceded"
        target_reaction = (
            bool(turns)
            and utterance.role.value == "target"
            and (emotion == "conceded" or prior_token == "presses")
        )
        moral_disbelief = (
            bool(turns)
            and utterance.role.value == "orchestrator"
            and any(
                marker in prior_text.lower()
                for marker in _MORAL_DISBELIEF_MARKERS
            )
        )
        dramatic_reaction = target_reaction or moral_disbelief
        if turns:
            transition_gap = _DRAMATIC_TURN_GAP_S if dramatic_reaction else gap_s
            if transition_gap > 0:
                segments.append(
                    np.zeros(int(transition_gap * sample_rate), dtype=np.float32)
                )
                cursor += transition_gap
        speech_start = cursor
        camera_start = (
            max(0.0, speech_start - _REACTION_LEAD_S)
            if dramatic_reaction
            else speech_start
        )
        reaction_emotion = (
            (
                "disbelief"
                if moral_disbelief
                else (
                    "conceded"
                    if emotion == "conceded"
                    else "defeated"
                )
            )
            if dramatic_reaction
            else emotion
        )
        if climax_concession and speech_start > camera_start:
            turns.append(
                DialogueTurn(
                    speaker=speaker_id,
                    start_time=camera_start,
                    end_time=speech_start,
                    text="",
                    emotion=reaction_emotion,
                    speech_start_time=speech_start,
                    reaction_emotion=reaction_emotion,
                    camera_tight=False,
                )
            )
            camera_start = speech_start
        turns.append(
            DialogueTurn(
                speaker=speaker_id,
                start_time=camera_start,
                end_time=speech_start + duration,
                text=utterance.text,
                audio_path=str(turn_wav) if turn_wav else None,
                emotion=emotion,
                speech_start_time=speech_start,
                reaction_emotion=emotion if climax_concession else reaction_emotion,
                camera_tight=cornering_blow,
            )
        )
        segments.append(samples)
        cursor = speech_start + duration
        prior_intent = intent
        prior_text = str(utterance.text or "")

    if turns and turns[-1].camera_tight:
        segments.append(np.zeros(int(_CLIMAX_RETURN_S * sample_rate), dtype=np.float32))
        conceded = turns[-1]
        turns.append(
            DialogueTurn(
                speaker=conceded.speaker,
                start_time=cursor,
                end_time=cursor + _CLIMAX_RETURN_S,
                text="",
                emotion=conceded.emotion or "conceded",
                speech_start_time=cursor,
                reaction_emotion=conceded.emotion or "conceded",
                camera_tight=False,
            )
        )
        cursor += _CLIMAX_RETURN_S

    if tail_s > 0:
        segments.append(np.zeros(int(round(tail_s * sample_rate)), dtype=np.float32))
        cursor += tail_s

    if not segments:
        raise ValueError("transcript has no utterances — nothing to animate")

    merged = np.concatenate(segments)
    merged = _mix_classic_audio_stack(
        merged,
        duration_s=cursor,
        turn_starts=[turn.start_time for turn in turns],
        sample_rate=sample_rate,
        audio_config=audio_config,
        fade_out_s=tail_s if tail_s > 0 else None,
    )
    if tail_s > 0:
        fade_n = int(round(tail_s * sample_rate))
        if 0 < fade_n <= merged.size:
            merged = np.asarray(merged, dtype=np.float32).copy()
            merged[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm16_wave(destination, merged, sample_rate)
    return destination, turns, cursor


def pick_cynical_hook(seed: str) -> str:
    """Stable terminal-card line for one session."""
    digest = hashlib.md5((seed or "aiwake").encode("utf-8")).digest()
    return _CYNICAL_HOOKS[int.from_bytes(digest[:8], "big") % len(_CYNICAL_HOOKS)]


def _append_typewriter_outro(path: Path, text: str, duration_s: float = _OUTRO_S) -> float:
    """Extend the mastered track with keyboard clicks under the end card."""
    from .media.audio import synthesize_typewriter_clicks  # noqa: PLC0415

    samples, sample_rate, spoken_s = load_mono_waveform(path)
    type_s = min(_OUTRO_TYPE_S, duration_s)
    clicks = np.asarray(
        synthesize_typewriter_clicks(
            type_s,
            max(1, len(text)),
            fps=sample_rate,
            gain_db=-14.0,
            seed=int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "big"),
        ),
        dtype=np.float32,
    )
    mono = clicks.mean(axis=1) if clicks.ndim == 2 else clicks
    tail = np.zeros(int(round(duration_s * sample_rate)), dtype=np.float32)
    take = min(tail.size, mono.size)
    tail[:take] += mono[:take]
    _write_pcm16_wave(path, np.concatenate([np.asarray(samples, dtype=np.float32), tail]), sample_rate)
    return float(spoken_s) + duration_s


def _terminal_outro_frame(text: str, revealed: int, *, caret_on: bool, width: int, height: int):
    """Black terminal end-card matching the classic CTA typewriter."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    header_font = font
    for candidate, size in (
        (r"C:\Windows\Fonts\consola.ttf", 46),
        (r"C:\Windows\Fonts\cour.ttf", 44),
    ):
        if Path(candidate).is_file():
            font = ImageFont.truetype(candidate, size)
            header_font = ImageFont.truetype(candidate, 24)
            break
    header = "AIWAKE"
    header_w = draw.textlength(header, font=header_font)
    draw.text(((width - header_w) / 2, int(height * 0.30)), header, font=header_font, fill=(4, 81, 177))
    draw.line(
        [(int(width * 0.18), int(height * 0.345)), (int(width * 0.82), int(height * 0.345))],
        fill=(24, 26, 29),
        width=2,
    )
    max_w = width * 0.62
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_w:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    visible: list[str] = []
    budget = revealed
    for line in lines:
        if budget <= 0:
            break
        visible.append(line[:budget])
        budget -= len(line) + 1
    y = int(height * 0.40)
    last_box = (width // 2, y, 0, 48)
    for line in visible:
        line_w = draw.textlength(line, font=font) if line else 0
        x = (width - line_w) / 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        bbox = draw.textbbox((x, y), line or " ", font=font)
        last_box = (x, y, max(0, bbox[2] - bbox[0]), max(2, bbox[3] - bbox[1]))
        y += int((bbox[3] - bbox[1]) * 1.45)
    if caret_on and revealed < len(text):
        cx = last_box[0] + last_box[2]
        draw.rectangle(
            [cx + 6, last_box[1], cx + 12, last_box[1] + last_box[3]],
            fill=(4, 81, 177),
        )
    return np.asarray(canvas)


def _terminal_outro_painter(text: str, *, width: int, height: int, fps: int):
    """Return a local-time painter for the prebuilt 2.8s terminal card."""
    frame_count = max(1, int(round(_OUTRO_S * fps)))
    frames = []
    for index in range(frame_count):
        local_s = index / float(fps)
        revealed = int(round(min(1.0, local_s / _OUTRO_TYPE_S) * len(text)))
        caret_on = local_s < _OUTRO_TYPE_S and (index // 4) % 2 == 0
        frames.append(
            _terminal_outro_frame(
                text,
                revealed,
                caret_on=caret_on,
                width=width,
                height=height,
            )
        )

    def paint(local_s: float) -> np.ndarray:
        index = min(frame_count - 1, max(0, int(local_s * fps)))
        return frames[index]

    return paint


def render_debate_animation(
    transcript: "DebateTranscript",
    *,
    audio_by_turn: dict[int, "AudioAsset"] | None = None,
    output_dir: Path,
    character_map: dict[str, str] | None = None,
    skin: str = DEFAULT_SKIN_PRESET,
    left_puppet: str | None = None,
    right_puppet: str | None = None,
    hud_labels: dict[str, str] | None = None,
    fps: int = 30,
    width: int = 1080,
    height: int = 1920,
    duration_override: float | None = None,
    audio_config: object | None = None,
    output_name: str | None = None,
    enable_cta: bool = False,
    seamless_loop: bool | None = None,
) -> Path:
    """Render a debate transcript through the shot-reverse-shot engine.

    Drop-in sibling to
    :func:`channels_config.aiwake.media.renderer.render_transcript`: same
    "give it a transcript + per-turn audio, get an mp4 back" contract, just
    routed through ``core.animator`` instead of the terminal typewriter.

    ``output_dir`` should be the dedicated animation-clips subfolder (never
    the production terminal-reel media dir) — callers such as
    :func:`channels_config.aiwake.pipeline.run_pipeline` are responsible for
    passing ``{OUTPUT_PATH}/aiwake/animation_clips/``.
    """
    from core.animator import render_dynamic_animation  # noqa: PLC0415
    from core.animator.asset_generator import (  # noqa: PLC0415
        DEFAULT_PUPPETS_DIR,
        archive_v1_retro_skins,
    )

    if (skin or "").strip().lower() == "v1":
        archive_v1_retro_skins(puppets_dir=DEFAULT_PUPPETS_DIR)
    seats = resolve_character_map(
        skin=skin,
        left_puppet=left_puppet,
        right_puppet=right_puppet,
        character_map=character_map,
    )
    ensure_skin_registry_file(DEFAULT_PUPPETS_DIR)
    styles = build_speaker_styles(seats, labels=hud_labels)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_filename = output_name or debate_video_filename(transcript)
    merged_audio_path = output_dir / f"{transcript.session_id}_battle_audio.wav"
    video_path = output_dir / video_filename

    if seamless_loop is not None:
        # Backward-compatible bridge for archived render scripts. New callers
        # opt into the terminal card with ``enable_cta=True``.
        enable_cta = not seamless_loop
    loop_tail_s = 0.0 if enable_cta else 0.4
    _, turns, total_duration = build_session_audio(
        transcript,
        audio_by_turn,
        destination=merged_audio_path,
        character_map=seats,
        audio_config=audio_config,
        tail_s=loop_tail_s,
    )
    for turn in turns:
        _LOG.info(
            "turn %s %.2f-%.2f speech=%.2f tight=%s %s",
            turn.speaker,
            turn.start_time,
            turn.end_time,
            turn.speech_start,
            turn.camera_tight,
            (turn.text or "")[:64],
        )
    if not enable_cta:
        spoken_end = max((turn.end_time for turn in turns if (turn.text or "").strip()), default=0.0)
        effective_duration = (
            duration_override if duration_override is not None else spoken_end + loop_tail_s
        )
        outro_start_s = None
        outro_frame = None
        subtitle_fade_s = loop_tail_s
        _LOG.info(
            "seamless loop: cut %.2fs after the final word (%.2fs), no terminal card",
            loop_tail_s,
            spoken_end,
        )
    else:
        hook = pick_cynical_hook(transcript.session_id)
        dialogue_duration = total_duration
        mastered_duration = _append_typewriter_outro(merged_audio_path, hook)
        effective_duration = duration_override if duration_override is not None else mastered_duration
        outro_start_s = dialogue_duration
        outro_frame = _terminal_outro_painter(hook, width=width, height=height, fps=fps)
        subtitle_fade_s = 0.0
        _LOG.info("terminal outro hook: %s", hook)

    stats = render_dynamic_animation(
        turns=turns,
        audio_path=merged_audio_path,
        styles=styles,
        output_path=video_path,
        puppets_dir=DEFAULT_PUPPETS_DIR,
        fps=fps,
        width=width,
        height=height,
        duration_override=effective_duration,
        enable_cta=enable_cta,
        outro_start_s=outro_start_s,
        outro_frame=outro_frame,
        subtitle_fade_s=subtitle_fade_s,
    )
    _LOG.info(
        "battle render complete: %s (%d frames, %.2fx realtime)",
        stats.output_path,
        stats.frames_written,
        stats.speedup_factor,
    )
    return stats.output_path


__all__ = [
    "DEFAULT_CHARACTER_MAP",
    "DEFAULT_SKIN_PRESET",
    "DEFAULT_SEAT_STYLE",
    "SKIN_PRESETS",
    "build_session_audio",
    "build_speaker_styles",
    "ensure_skin_registry_file",
    "render_debate_animation",
    "resolve_dialectic_emotion",
    "resolve_character_map",
]
