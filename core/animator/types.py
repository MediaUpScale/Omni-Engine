# -*- coding: utf-8 -*-
"""Shared, provider-agnostic dataclasses for the animation engine.

Nothing in this module imports Pillow, numpy, moviepy, or anything from any
channel package — it is the pure-data contract every other module in
``core/animator/`` speaks, so the engine stays usable by any channel (or
outside the factory entirely).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

Vec2 = tuple[int, int]

#: The nine canonical Rhubarb Lip Sync mouth shapes.
#:
#: A closed (rest/bilabial), B consonant, C open, D wide, E rounded,
#: F puckered, G bite (f/v), H tongue (l), X rest/silence.
VISEMES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "X")

#: Fallback viseme when a cue references a shape a skin does not ship.
REST_VISEME = "X"


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """One turn in a universal dialogue ledger.

    ``speaker`` is a free-form identifier the caller controls — for the
    shot-reverse-shot director it must match one of the ``character_id``
    values wired into the compositor, but the engine never assumes what
    those strings mean.

    ``audio_path`` optionally points at this turn's *isolated* audio track
    (before it was spliced into the session mixdown). Phonetic lip-sync
    analysers such as Rhubarb work per-utterance, so keeping the per-turn
    source around lets the engine reach a far better mouth track than
    slicing the mixdown back apart would.
    """

    speaker: str
    start_time: float
    end_time: float
    text: str = ""
    audio_path: str | None = None
    emotion: str = "neutral"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass(frozen=True, slots=True)
class SpeakerStyle:
    """Per-speaker presentation contract for the shot-reverse-shot director.

    This is how a channel tells the *generic* engine what to put on screen
    without the engine ever learning what a "Gemini" or an "orchestrator"
    is: an id to match turns against, a HUD label, an accent colour, and
    which way the hero looks when it holds the camera.
    """

    character_id: str
    label: str
    accent_hex: str = "#00F0FF"
    facing: str = "right"  # "right" | "left" — direction the hero is angled toward

    @property
    def accent_rgb(self) -> tuple[int, int, int]:
        value = self.accent_hex.lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


@dataclass(frozen=True, slots=True)
class WordTiming:
    """One spoken word with its own on-screen window.

    Produced by :mod:`core.animator.subtitles` and consumed by the ASS
    karaoke writer, which emits exactly one highlighted word per event.
    """

    word: str
    start_time: float
    end_time: float
    speaker: str


@dataclass(frozen=True, slots=True)
class PuppetAnchors:
    """Pixel anchors on a puppet's canvas, in the skin's own coordinate space."""

    mouth: Vec2
    eyes: Vec2
    head_pivot: Vec2
    neck_pivot: Vec2


@dataclass(frozen=True, slots=True)
class PuppetTheme:
    """Cosmetic accents read straight from ``puppet.json``."""

    glow_color: str = "#00F0FF"
    glow_radius: int = 25


@dataclass(slots=True)
class AnalyzedAudio:
    """Per-frame animation curves produced by :class:`AudioAnalyzer`.

    All arrays are the same length (``n_frames``) and index-aligned with the
    video timeline at ``fps``. Values are plain Python floats/ints/strings
    packed into lists (not numpy) so this module has zero heavy
    dependencies; numpy conversion happens inside the compositor.

    ``viseme`` carries the phonetic mouth track (one of :data:`VISEMES` per
    frame per speaker). ``mouth_state`` is the coarse 3-level RMS
    quantisation kept for skins that ship no viseme sprites.
    """

    fps: int
    duration_s: float
    n_frames: int
    rms: Sequence[float] = field(default_factory=list)
    active_speaker: Sequence[str | None] = field(default_factory=list)
    mouth_state: dict[str, Sequence[int]] = field(default_factory=dict)
    eye_state: dict[str, Sequence[int]] = field(default_factory=dict)
    viseme: dict[str, Sequence[str]] = field(default_factory=dict)
    emotion: Sequence[str] = field(default_factory=list)

    def frame_time(self, index: int) -> float:
        return index / float(self.fps)


__all__ = [
    "AnalyzedAudio",
    "DialogueTurn",
    "PuppetAnchors",
    "PuppetTheme",
    "REST_VISEME",
    "SpeakerStyle",
    "VISEMES",
    "Vec2",
    "WordTiming",
]
