# -*- coding: utf-8 -*-
"""Word-by-word karaoke subtitles as Advanced SubStation Alpha (``.ass``).

Never a static text block: the phrase on screen shows exactly ONE word lit
in the active speaker's accent colour while every other word in that phrase
stays clean white. That is achieved by emitting one Dialogue event per word
— each event re-renders the whole phrase with a colour override around only
the word being spoken at that instant — rather than ASS ``\\k`` karaoke
tags, which progressively latch every already-sung word into the highlight
colour and would leave the entire line lit by the end of the phrase.

The file is burned into the video by FFmpeg's libass ``subtitles`` filter;
see :mod:`core.animator.renderer`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

from .types import DialogueTurn, SpeakerStyle, WordTiming

_LOG = logging.getLogger("animator.subtitles")

#: Hard limits for one kinetic caption burst.
MIN_WORDS_PER_PHRASE = 2
MAX_WORDS_PER_PHRASE = 4
MAX_CHARS_PER_LINE = 24

#: Social-safe karaoke anchor across the upper portion of the lower chest.
SUBTITLE_CENTRE_Y = 1440

_FONT_NAME = "Arial Black"
_FONT_SIZE = 64
_WHITE = "&H00FFFFFF"

_WORD_SPLIT_RE = re.compile(r"\s+")


def _ass_colour(hex_rgb: str) -> str:
    """``#00F0FF`` -> ``&H00FFF000`` (ASS is &HAABBGGRR, alpha 00 = opaque)."""
    value = hex_rgb.lstrip("#")
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H00{b}{g}{r}".upper()


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"


def _parse_ass_time(value: str) -> float:
    hours, minutes, secs = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(secs)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def word_timings(turns: Iterable[DialogueTurn]) -> list[WordTiming]:
    """Distribute each turn's words across its own spoken window.

    Without a forced aligner, word length is the best available proxy for
    duration, so each word's slice of the turn is proportional to its
    character count (plus a constant, so one-letter words still get a
    readable beat). Word *order* and the turn's total span are exact, which
    is what keeps the highlight locked to the voice.
    """
    timings: list[WordTiming] = []
    for turn in turns:
        words = [w for w in _WORD_SPLIT_RE.split(turn.text.strip()) if w]
        if not words or turn.duration <= 0:
            continue
        weights = [len(w) + 2.0 for w in words]
        total = sum(weights)
        cursor = turn.speech_start
        for word, weight in zip(words, weights):
            span = turn.spoken_duration * (weight / total)
            timings.append(
                WordTiming(word=word, start_time=cursor, end_time=cursor + span, speaker=turn.speaker)
            )
            cursor += span
    return timings


def _phrase_chunks(words: Sequence[WordTiming]) -> list[list[WordTiming]]:
    """Build punctuation-aware 2–4-word bursts without orphaned words."""
    chunks: list[list[WordTiming]] = []
    current: list[WordTiming] = []
    for word in words:
        current.append(word)
        punctuation_break = word.word.rstrip().endswith((".", "!", "?", ";", ":"))
        long_enough = len(" ".join(item.word for item in current)) >= 22
        if len(current) >= MAX_WORDS_PER_PHRASE or (
            len(current) >= MIN_WORDS_PER_PHRASE and (punctuation_break or long_enough)
        ):
            chunks.append(current)
            current = []
    if current:
        if len(current) == 1 and chunks and len(chunks[-1]) < MAX_WORDS_PER_PHRASE:
            chunks[-1].extend(current)
        elif len(current) == 1 and chunks:
            current.insert(0, chunks[-1].pop())
            chunks.append(current)
        else:
            chunks.append(current)
    return chunks


def _balanced_lines(words: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split one short burst into at most two centered, <=24-char lines."""
    if len(" ".join(words)) <= MAX_CHARS_PER_LINE:
        return list(words), []
    candidates = range(1, len(words))
    split = min(
        candidates,
        key=lambda index: abs(
            len(" ".join(words[:index])) - len(" ".join(words[index:]))
        ),
    )
    return list(words[:split]), list(words[split:])


def build_ass(
    turns: Sequence[DialogueTurn],
    styles: dict[str, SpeakerStyle],
    *,
    destination: Path,
    width: int = 1080,
    height: int = 1920,
    words_per_phrase: int | None = None,
    fade_out_s: float = 0.0,
) -> Path:
    """Write a word-level karaoke ``.ass`` file and return its path."""
    del words_per_phrase  # Compatibility argument; bursts are now dynamically sized.
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{_FONT_NAME},{_FONT_SIZE},{_WHITE},{_WHITE},&H00000000,&H00000000,-1,0,0,0,100,100,1,0,1,4.5,2,5,90,90,320,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for turn in turns:
        style = styles.get(turn.speaker)
        accent = _ass_colour(style.accent_hex if style else "#00F0FF")
        turn_words = [t for t in word_timings([turn])]
        for phrase in _phrase_chunks(turn_words):
            for position, active in enumerate(phrase):
                # Whole phrase, with exactly one word wrapped in the accent
                # colour — the "currently spoken" word.
                rendered_words = [
                    (
                        f"{{\\c{accent}\\b1\\fscx112\\fscy112"
                        f"\\t(0,80,\\fscx100\\fscy100)}}{_escape(word.word)}"
                        f"{{\\c{_WHITE}\\b1\\fscx100\\fscy100}}"
                        if index == position
                        else _escape(word.word)
                    )
                    for index, word in enumerate(phrase)
                ]
                first_plain, second_plain = _balanced_lines([word.word for word in phrase])
                split = len(first_plain)
                rendered = " ".join(rendered_words[:split])
                if second_plain:
                    rendered += r"\N" + " ".join(rendered_words[split:])
                text = f"{{\\an5\\q2\\pos({width // 2},{SUBTITLE_CENTRE_Y})}}{rendered}"
                events.append(
                    "Dialogue: 0,"
                    f"{_ass_time(active.start_time)},{_ass_time(active.end_time)},"
                    f"Karaoke,,0,0,0,,{text}"
                )

    if fade_out_s > 0 and events:
        fade_ms = max(1, int(round(fade_out_s * 1000)))
        bits = events[-1].split(",", 9)
        bits[2] = _ass_time(_parse_ass_time(bits[2]) + fade_out_s)
        text = bits[9]
        bits[9] = (
            "{\\fad(0," + str(fade_ms) + ")" + text[1:]
            if text.startswith("{")
            else "{\\fad(0," + str(fade_ms) + ")}" + text
        )
        events[-1] = ",".join(bits)

    destination.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    _LOG.info("wrote %d karaoke subtitle events -> %s", len(events), destination)
    return destination


__all__ = [
    "MAX_CHARS_PER_LINE",
    "MAX_WORDS_PER_PHRASE",
    "MIN_WORDS_PER_PHRASE",
    "SUBTITLE_CENTRE_Y",
    "build_ass",
    "word_timings",
]
