# -*- coding: utf-8 -*-
"""Phonetic lip-sync via Rhubarb Lip Sync.

`Rhubarb <https://github.com/DanielSWolf/rhubarb-lip-sync>`_ analyses a
speech recording and emits time-stamped mouth cues using nine canonical
shapes (A-H plus X for rest). This module:

1. Locates ``rhubarb`` — an explicit ``RHUBARB_PATH``/``RHUBARB_EXE`` env
   var, a previous download under ``core/animator/bin/``, or ``PATH``.
2. Auto-downloads and unpacks the official Windows/Linux/macOS release into
   ``core/animator/bin/`` when it is missing.
3. Runs it over a turn's audio and parses the JSON cue list.

Every step degrades gracefully: no network, a blocked download, a corrupt
binary or an unreadable WAV all fall back to
:func:`visemes_from_envelope`, an RMS-driven approximation that keeps the
mouth moving in time with the voice even when no phonetic analyser is
available. The renderer therefore never hard-fails on a missing optional
dependency.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .types import REST_VISEME, VISEMES

_LOG = logging.getLogger("animator.rhubarb")

BIN_DIR: Path = Path(__file__).resolve().parent / "bin"

_RHUBARB_VERSION = "1.13.0"
_RELEASE_URL = (
    "https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/"
    f"v{_RHUBARB_VERSION}/Rhubarb-Lip-Sync-{_RHUBARB_VERSION}-Windows.zip"
)
_DOWNLOAD_TIMEOUT_S = 90

# Set once a download attempt has failed so a 45-turn batch does not retry
# (and re-stall on) the same unreachable URL dozens of times.
_DOWNLOAD_BLOCKED = False


def _exe_name() -> str:
    return "rhubarb.exe" if os.name == "nt" else "rhubarb"


def find_rhubarb() -> Path | None:
    """Locate an existing Rhubarb binary without attempting a download."""
    for env_var in ("RHUBARB_PATH", "RHUBARB_EXE"):
        raw = (os.getenv(env_var) or "").strip().strip('"')
        if raw and Path(raw).is_file():
            return Path(raw)

    local = BIN_DIR / _exe_name()
    if local.is_file():
        return local
    # Release zips unpack into a versioned subfolder; accept that layout too.
    if BIN_DIR.is_dir():
        for candidate in BIN_DIR.rglob(_exe_name()):
            if candidate.is_file():
                return candidate

    found = shutil.which("rhubarb")
    return Path(found) if found else None


def ensure_rhubarb(*, allow_download: bool = True) -> Path | None:
    """Return a usable Rhubarb binary, downloading it once if necessary.

    Returns ``None`` (never raises) when Rhubarb cannot be obtained, which
    is the caller's signal to fall back to the envelope approximation.
    """
    global _DOWNLOAD_BLOCKED

    existing = find_rhubarb()
    if existing is not None:
        return existing
    if not allow_download or _DOWNLOAD_BLOCKED:
        return None
    if os.name != "nt":
        _LOG.info("no Rhubarb binary found; auto-download is Windows-only — using envelope fallback")
        _DOWNLOAD_BLOCKED = True
        return None

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    archive = BIN_DIR / f"rhubarb-{_RHUBARB_VERSION}.zip"
    try:
        _LOG.info("downloading Rhubarb %s -> %s", _RHUBARB_VERSION, BIN_DIR)
        import urllib.request  # noqa: PLC0415

        with urllib.request.urlopen(_RELEASE_URL, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(BIN_DIR)
        archive.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 — offline/blocked/proxied environments are expected
        _LOG.warning("Rhubarb auto-download failed (%s); using RMS envelope fallback", exc)
        archive.unlink(missing_ok=True)
        _DOWNLOAD_BLOCKED = True
        return None

    resolved = find_rhubarb()
    if resolved is None:
        _LOG.warning("Rhubarb archive unpacked but no %s found; using envelope fallback", _exe_name())
        _DOWNLOAD_BLOCKED = True
    return resolved


def analyze_visemes(
    audio_path: Path,
    *,
    dialog_text: str = "",
    allow_download: bool = True,
) -> list[tuple[float, str]] | None:
    """Run Rhubarb over ``audio_path``.

    Returns ``[(start_time_s, viseme), ...]`` sorted by time, or ``None`` if
    Rhubarb is unavailable or failed — callers should then use
    :func:`visemes_from_envelope`.
    """
    binary = ensure_rhubarb(allow_download=allow_download)
    if binary is None:
        return None
    source_path = Path(audio_path)
    analysis_path = source_path
    transcoded_path: Path | None = None
    if source_path.suffix.lower() not in {".wav", ".ogg"}:
        transcoded_path = source_path.with_suffix(".rhubarb.wav")
        try:
            try:
                import imageio_ffmpeg  # noqa: PLC0415

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError):
                ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise FileNotFoundError("ffmpeg executable not found")
            converted = subprocess.run(
                [
                    str(ffmpeg),
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(transcoded_path),
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            if converted.returncode != 0 or not transcoded_path.is_file():
                raise RuntimeError((converted.stderr or "ffmpeg failed")[-400:])
            analysis_path = transcoded_path
        except Exception as exc:  # noqa: BLE001 - envelope fallback is intentional
            _LOG.warning(
                "could not transcode %s for Rhubarb (%s); using envelope fallback",
                source_path.name,
                exc,
            )
            transcoded_path.unlink(missing_ok=True)
            return None

    # Note: `-q` together with `--machineReadable` silences stdout entirely
    # (the cue list included), so neither is used. `--consoleLevel Error`
    # gets the same quiet console without swallowing the result, and
    # `--extendedShapes GHX` guarantees all nine canonical shapes are in
    # play rather than just the basic A-F set.
    cmd = [
        str(binary),
        "-f", "json",
        "-r", "phonetic",
        "--extendedShapes", "GHX",
        "--consoleLevel", "Error",
        str(analysis_path),
    ]
    dialog_file: Path | None = None
    if dialog_text.strip():
        # Rhubarb's phonetic recognizer is markedly more accurate when it
        # knows what was actually said.
        dialog_file = audio_path.with_suffix(".dialog.txt")
        try:
            dialog_file.write_text(dialog_text.strip(), encoding="utf-8")
            cmd += ["-d", str(dialog_file)]
        except OSError:
            dialog_file = None

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("Rhubarb execution failed on %s (%s); using envelope fallback", audio_path.name, exc)
        return None
    finally:
        if dialog_file is not None:
            dialog_file.unlink(missing_ok=True)
        if transcoded_path is not None:
            transcoded_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        _LOG.warning("Rhubarb exited %d on %s: %s", proc.returncode, audio_path.name, (proc.stderr or "")[-400:])
        return None

    try:
        payload = json.loads(proc.stdout)
        cues = [
            (float(cue["start"]), str(cue["value"]).upper()[:1] or REST_VISEME)
            for cue in payload.get("mouthCues", [])
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        _LOG.warning("could not parse Rhubarb output for %s (%s)", audio_path.name, exc)
        return None

    cues = [(t, v if v in VISEMES else REST_VISEME) for t, v in cues]
    cues.sort(key=lambda item: item[0])
    if not cues:
        return None

    # Degenerate-recognition guard. When the recognizer cannot find real
    # phonemes — non-speech audio, heavy processing, an unsupported
    # language — it still returns a valid cue list, but one that collapses
    # to rest plus a single generic consonant. Animating that gives a
    # near-frozen mouth, so it is treated as a failed analysis and handed
    # to the envelope fallback, which at least tracks the voice's
    # amplitude.
    distinct = {shape for _, shape in cues if shape != REST_VISEME}
    if len(distinct) < 3:
        _LOG.info(
            "Rhubarb returned a degenerate cue set for %s (shapes=%s); using envelope fallback",
            audio_path.name, sorted(distinct) or ["none"],
        )
        return None

    _LOG.info(
        "Rhubarb phonetic schedule for %s (%d cues, shapes=%s): %s",
        audio_path.name,
        len(cues),
        "".join(sorted({shape for _, shape in cues})),
        ", ".join(f"{timestamp:.2f}:{shape}" for timestamp, shape in cues),
    )
    return cues


# --------------------------------------------------------------------------- #
# Fallback: RMS-envelope-driven viseme approximation
# --------------------------------------------------------------------------- #
# Loudness bands mapped onto plausible mouth openings. Silence rests on X;
# quiet frames use the closed/consonant shapes; louder frames open through
# C/E and peak on the wide D. Cycling within a band (rather than pinning one
# sprite) prevents the "frozen open mouth" look on sustained vowels.
_ENVELOPE_LADDER: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0.06, (REST_VISEME,)),
    (0.18, ("A", "B")),
    (0.38, ("B", "C", "G")),
    (0.62, ("C", "E", "H")),
    (1.01, ("D", "C", "E")),
)


def visemes_from_envelope(levels: Sequence[float], fps: int) -> list[tuple[float, str]]:
    """Approximate a viseme cue list from a normalised 0..1 RMS envelope.

    Not phonetically correct — but it is frame-accurate to the voice's
    amplitude, which keeps the mouth alive and in sync when the phonetic
    analyser is unavailable.
    """
    cues: list[tuple[float, str]] = []
    previous: str | None = None
    for index, level in enumerate(levels):
        value = float(np.clip(level, 0.0, 1.0))
        for ceiling, shapes in _ENVELOPE_LADDER:
            if value < ceiling:
                shape = shapes[index % len(shapes)]
                break
        else:
            shape = "D"
        if shape != previous:
            cues.append((index / float(fps), shape))
            previous = shape
    if not cues:
        cues.append((0.0, REST_VISEME))
    return cues


def cues_to_frames(
    cues: Sequence[tuple[float, str]],
    *,
    fps: int,
    n_frames: int,
    offset_s: float = 0.0,
) -> list[str]:
    """Flatten a cue list into one viseme per video frame.

    ``offset_s`` shifts a per-turn cue list onto the session timeline, so a
    turn analysed in isolation lands on exactly the frames where that turn
    is actually heard in the mixdown.
    """
    frames = [REST_VISEME] * n_frames
    if not cues:
        return frames
    ordered = sorted(cues, key=lambda item: item[0])
    cursor = 0
    current = REST_VISEME
    for index in range(n_frames):
        t = index / float(fps) - offset_s
        while cursor < len(ordered) and ordered[cursor][0] <= t:
            current = ordered[cursor][1]
            cursor += 1
        frames[index] = current if t >= 0 else REST_VISEME
    return frames


__all__ = [
    "BIN_DIR",
    "analyze_visemes",
    "cues_to_frames",
    "ensure_rhubarb",
    "find_rhubarb",
    "visemes_from_envelope",
]
