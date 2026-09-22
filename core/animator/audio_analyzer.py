# -*- coding: utf-8 -*-
"""Universal audio & dialogue sync.

Accepts ANY session audio (WAV or MP3) plus a generic
:class:`~.types.DialogueTurn` ledger and produces frame-aligned animation
curves at the video's frame rate: phonetic visemes, procedural blinking,
and a restrained idle-breathing float.

Nothing here knows about any specific channel, debates, or robots —
``speaker`` strings in the turn ledger are opaque identifiers the caller
defines.
"""
from __future__ import annotations

import logging
import math
import wave
from pathlib import Path

import numpy as np

from .rhubarb import analyze_visemes, cues_to_frames, visemes_from_envelope
from .types import REST_VISEME, AnalyzedAudio, DialogueTurn

_LOG = logging.getLogger("animator.audio_analyzer")

# --------------------------------------------------------------------------- #
# Secondary-animation constants (per the video-pipeline spec).
# --------------------------------------------------------------------------- #
_BLINK_FIRST_CENTER_S = 1.5
_BLINK_INTERVAL_S = 3.0

_BREATH_HZ = 2.0  # rad/s multiplier: sin(t * 2.0)
_BREATH_AMPLITUDE_PX = 3.0

# RMS quantisation thresholds (fractions of the normalised 0..1 envelope).
_MOUTH_SILENCE_THRESHOLD = 0.06
_MOUTH_WIDE_THRESHOLD = 0.55


def load_mono_waveform(audio_path: Path) -> tuple[np.ndarray, int, float]:
    """Decode any audio file (WAV/MP3/...) into a mono float32 waveform.

    Returns ``(samples, sample_rate, duration_s)``. Uses ``soundfile`` first
    (fast, no subprocess) and falls back to MoviePy's ffmpeg-backed decoder
    for formats soundfile cannot read (some MP3 encodings).
    """
    try:
        import soundfile as sf  # noqa: PLC0415

        data, sr = sf.read(str(audio_path), always_2d=False, dtype="float32")
        mono = data.mean(axis=1) if data.ndim > 1 else data
        duration = len(mono) / float(sr)
        return mono.astype(np.float32), int(sr), duration
    except Exception as exc:  # noqa: BLE001 — soundfile can't read every mp3 encoder
        _LOG.debug("soundfile failed on %s (%s); trying built-in decoders", audio_path, exc)

    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sr = source.getframerate()
                raw = source.readframes(source.getnframes())
            dtype_by_width = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}
            dtype = dtype_by_width[sample_width]
            data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if sample_width == 1:
                data = (data - 128.0) / 128.0
            else:
                data /= float(1 << ((sample_width * 8) - 1))
            if channels > 1:
                data = data.reshape(-1, channels).mean(axis=1)
            return data, int(sr), len(data) / float(sr)
        except (KeyError, wave.Error, ValueError) as exc:
            _LOG.debug("stdlib WAV decode failed on %s (%s); using moviepy", audio_path, exc)

    from moviepy import AudioFileClip  # noqa: PLC0415

    clip = AudioFileClip(str(audio_path))
    try:
        sr = 44100
        arr = clip.to_soundarray(fps=sr)
        mono = arr.mean(axis=1) if arr.ndim > 1 else arr
        duration = float(clip.duration)
        return mono.astype(np.float32), sr, duration
    finally:
        clip.close()


class AudioAnalyzer:
    """Computes per-frame RMS + secondary animation curves for one session."""

    def __init__(self, *, fps: int = 30, seed: int = 0) -> None:
        self.fps = fps
        self.seed = seed

    def analyze(
        self,
        audio_path: str | Path,
        turns: list[DialogueTurn],
        *,
        duration_override: float | None = None,
        use_rhubarb: bool = True,
    ) -> AnalyzedAudio:
        """Build the full :class:`AnalyzedAudio` timeline for ``audio_path``.

        ``turns`` assigns each video frame to at most one active speaker
        (whichever turn's ``[start_time, end_time)`` window contains that
        frame's timestamp); frames outside every window have no active
        speaker (both puppets idle).
        """
        audio_path = Path(audio_path)
        mono, sr, probed_duration = load_mono_waveform(audio_path)
        duration_s = duration_override if duration_override is not None else probed_duration
        duration_s = max(duration_s, 1.0 / self.fps)

        n_frames = max(1, int(math.ceil(duration_s * self.fps)))
        rms = self._frame_rms(mono, sr, n_frames)
        rms_norm = self._normalize(rms)

        active_speaker = active_speaker_lookup(turns, self.fps, n_frames)
        speaking_speaker = speaking_speaker_lookup(turns, self.fps, n_frames)
        emotion = emotion_lookup(turns, self.fps, n_frames)
        camera_tight = camera_tight_lookup(turns, self.fps, n_frames)
        speakers = sorted({t.speaker for t in turns}) or ["speaker"]

        mouth_state: dict[str, list[int]] = {sp: [0] * n_frames for sp in speakers}
        for i in range(n_frames):
            speaker = speaking_speaker[i]
            if speaker is None or speaker not in mouth_state:
                continue
            mouth_state[speaker][i] = self._quantize_mouth(rms_norm[i])

        eye_state: dict[str, list[int]] = {
            sp: self._blink_schedule(n_frames, seed=self.seed)
            for sp in speakers
        }
        self._apply_reaction_blinks(turns, eye_state, n_frames)
        viseme = self._viseme_tracks(turns, rms_norm, n_frames, speakers, use_rhubarb=use_rhubarb)

        return AnalyzedAudio(
            fps=self.fps,
            duration_s=duration_s,
            n_frames=n_frames,
            rms=rms_norm.tolist(),
            active_speaker=active_speaker,
            speaking_speaker=speaking_speaker,
            mouth_state=mouth_state,
            eye_state=eye_state,
            viseme=viseme,
            emotion=emotion,
            camera_tight=camera_tight,
        )

    # -- Phonetic mouth track -------------------------------------------- #
    def _viseme_tracks(
        self,
        turns: list[DialogueTurn],
        rms_norm: np.ndarray,
        n_frames: int,
        speakers: list[str],
        *,
        use_rhubarb: bool,
    ) -> dict[str, list[str]]:
        """One viseme per frame per speaker, Rhubarb-driven where possible.

        Each turn is analysed from its own isolated audio file (phonetic
        recognisers work per-utterance, and the turn text can be supplied as
        a dialog hint), then stamped onto the session timeline at that
        turn's offset. Turns with no usable audio file — or any turn at all
        when Rhubarb is unavailable — fall back to the RMS envelope
        approximation so the mouth still moves with the voice.
        """
        tracks: dict[str, list[str]] = {sp: [REST_VISEME] * n_frames for sp in speakers}
        for turn in turns:
            start = max(0, int(round(turn.speech_start * self.fps)))
            end = min(n_frames, int(round(turn.end_time * self.fps)))
            if end <= start:
                continue
            span = end - start

            cues = None
            if use_rhubarb and turn.audio_path and Path(turn.audio_path).is_file():
                cues = analyze_visemes(Path(turn.audio_path), dialog_text=turn.text)
            if cues is None:
                cues = visemes_from_envelope(rms_norm[start:end], self.fps)

            frames = cues_to_frames(cues, fps=self.fps, n_frames=span)
            track = tracks.setdefault(turn.speaker, [REST_VISEME] * n_frames)
            track[start:end] = frames
        return tracks

    def _apply_reaction_blinks(
        self,
        turns: list[DialogueTurn],
        eye_state: dict[str, list[int]],
        n_frames: int,
    ) -> None:
        """Center one restrained five-frame blink inside each anticipation cut."""
        for turn in turns:
            if turn.speech_start <= turn.start_time + (1.0 / self.fps):
                continue
            states = eye_state.get(turn.speaker)
            if states is None:
                continue
            center_s = turn.start_time + ((turn.speech_start - turn.start_time) * 0.55)
            center = int(round(center_s * self.fps))
            if center - 2 < 0 or center + 2 >= n_frames:
                continue
            states[center - 2 : center + 3] = [0, 1, 2, 1, 0]

    # -- RMS envelope -------------------------------------------------- #
    def _frame_rms(self, mono: np.ndarray, sample_rate: int, n_frames: int) -> np.ndarray:
        samples_per_frame = max(1, int(round(sample_rate / self.fps)))
        out = np.zeros(n_frames, dtype=np.float64)
        total = len(mono)
        for i in range(n_frames):
            start = i * samples_per_frame
            end = min(total, start + samples_per_frame)
            if start >= total:
                break
            window = mono[start:end]
            if window.size:
                out[i] = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
        return out

    @staticmethod
    def _normalize(rms: np.ndarray) -> np.ndarray:
        if rms.size == 0:
            return rms
        nonzero = rms[rms > 1e-6]
        if nonzero.size == 0:
            return np.zeros_like(rms)
        # Robust ceiling: 95th percentile rather than the true max, so a
        # single vocal spike doesn't flatten every other frame toward silence.
        ceiling = float(np.percentile(nonzero, 95)) or float(nonzero.max())
        ceiling = max(ceiling, 1e-6)
        normalized = np.clip(rms / ceiling, 0.0, 1.0)
        return normalized

    @staticmethod
    def _quantize_mouth(level: float) -> int:
        if level < _MOUTH_SILENCE_THRESHOLD:
            return 0
        if level < _MOUTH_WIDE_THRESHOLD:
            return 1
        return 2

    # -- Procedural blinking --------------------------------------------- #
    def _blink_schedule(self, n_frames: int, *, seed: int) -> list[int]:
        del seed  # Deterministic broadcast cadence; retained for API compatibility.
        states = [0] * n_frames
        center = int(round(_BLINK_FIRST_CENTER_S * self.fps))
        interval = int(round(_BLINK_INTERVAL_S * self.fps))
        while center + 2 < n_frames:
            # Five-frame cel cycle centred on the requested proof timestamp:
            # open -> half lid -> closed seam -> half lid -> open.
            states[center - 2] = 0
            states[center - 1] = 1
            states[center] = 2
            states[center + 1] = 1
            states[center + 2] = 0
            center += interval
        return states


def breathing_offset(t: float) -> float:
    """Grounded idle breathing: ``y_offset = sin(t * 2.0) * 3px``."""
    return math.sin(t * _BREATH_HZ) * _BREATH_AMPLITUDE_PX


def active_speaker_lookup(turns: list[DialogueTurn], fps: int, n_frames: int) -> list[str | None]:
    """Frame-indexed active-speaker array built from a turn ledger.

    A frame at time ``t = i / fps`` is "active" for the first turn whose
    ``[start_time, end_time)`` window contains ``t``.
    """
    out: list[str | None] = [None] * n_frames
    if not turns:
        return out
    ordered = sorted(turns, key=lambda turn: turn.start_time)
    cursor = 0
    for i in range(n_frames):
        t = i / float(fps)
        while cursor < len(ordered) and ordered[cursor].end_time <= t:
            cursor += 1
        if cursor < len(ordered) and ordered[cursor].start_time <= t < ordered[cursor].end_time:
            out[i] = ordered[cursor].speaker
    return out


def speaking_speaker_lookup(
    turns: list[DialogueTurn],
    fps: int,
    n_frames: int,
) -> list[str | None]:
    """Frame-indexed voice owner, excluding pre-speech reaction windows."""
    out: list[str | None] = [None] * n_frames
    for turn in turns:
        start = max(0, int(round(turn.speech_start * fps)))
        end = min(n_frames, int(round(turn.end_time * fps)))
        if end > start:
            out[start:end] = [turn.speaker] * (end - start)
    return out


def emotion_lookup(turns: list[DialogueTurn], fps: int, n_frames: int) -> list[str]:
    """Frame-indexed deterministic expression state from the dialogue ledger."""
    out = ["neutral"] * n_frames
    for turn in turns:
        start = max(0, int(round(turn.start_time * fps)))
        speech_start = min(n_frames, int(round(turn.speech_start * fps)))
        end = min(n_frames, int(round(turn.end_time * fps)))
        if speech_start > start:
            reaction = turn.reaction_emotion or turn.emotion or "neutral"
            out[start:speech_start] = [reaction] * (speech_start - start)
        if end > speech_start:
            out[speech_start:end] = [turn.emotion or "neutral"] * (end - speech_start)
    return out


def camera_tight_lookup(
    turns: list[DialogueTurn],
    fps: int,
    n_frames: int,
) -> list[bool]:
    """Frame-indexed explicit camera direction from the dialogue ledger."""
    out = [False] * n_frames
    for turn in turns:
        if not turn.camera_tight:
            continue
        start = max(0, int(round(turn.start_time * fps)))
        end = min(n_frames, int(round(turn.end_time * fps)))
        if end > start:
            out[start:end] = [True] * (end - start)
    return out


__all__ = [
    "AudioAnalyzer",
    "active_speaker_lookup",
    "breathing_offset",
    "camera_tight_lookup",
    "emotion_lookup",
    "load_mono_waveform",
    "speaking_speaker_lookup",
]
