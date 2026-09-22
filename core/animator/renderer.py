# -*- coding: utf-8 -*-
"""Zero-disk high-speed FFmpeg pipe.

Streams raw RGB24 frames straight from RAM into an ``ffmpeg`` subprocess via
stdin — no per-frame PNGs ever touch the disk. Uses a bounded-bitrate
``libx264 -preset ultrafast`` broadcast contract. Muxes the dialogue
audio track as normalized-rate AAC
(``-c:a aac -b:a 192k -ar 44100 -map 0:v -map 1:a -shortest``).
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_LOG = logging.getLogger("animator.renderer")

_NVENC_PROBE_CACHE: dict[str, bool] = {}


def _escape_filter_path(path: Path) -> str:
    """Make a Windows path safe inside an FFmpeg filtergraph argument.

    The filtergraph parser treats ``\\`` as an escape and ``:`` as an option
    separator, so ``G:\\My Drive\\subs.ass`` has to become
    ``G\\:/My Drive/subs.ass`` before it is wrapped in single quotes — the
    single most common reason a ``subtitles=`` filter silently fails on
    Windows.
    """
    text = str(path).replace("\\", "/")
    return text.replace(":", r"\:")


def _resolve_ffmpeg() -> str:
    """Reuse the parent engine's ffmpeg resolver when available, else fall
    back to ``imageio_ffmpeg`` / PATH so this package works standalone too."""
    try:
        from core.utils.fingerprint_engine import resolve_ffmpeg  # noqa: PLC0415

        return resolve_ffmpeg()
    except Exception:  # noqa: BLE001 — standalone extraction / no parent engine on path
        pass

    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg  # noqa: PLC0415

    return imageio_ffmpeg.get_ffmpeg_exe()


def _nvenc_available(ffmpeg_path: str) -> bool:
    """True only when ``h264_nvenc`` is both *listed* and actually usable.

    A CUDA-capable ffmpeg build can list ``h264_nvenc`` in ``-encoders`` on a
    machine with no NVIDIA driver at all (or a driver missing ``nvcuda.dll``),
    so the listing alone is not trustworthy — it must be able to open the
    encoder for a single throwaway frame, or every real render would pay for
    a failed GPU attempt before falling back to libx264.
    """
    if ffmpeg_path in _NVENC_PROBE_CACHE:
        return _NVENC_PROBE_CACHE[ffmpeg_path]
    available = False
    try:
        listed = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if "h264_nvenc" in (listed.stdout or ""):
            probe = subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:d=0.1",
                    "-frames:v",
                    "1",
                    "-c:v",
                    "h264_nvenc",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                timeout=15,
            )
            available = probe.returncode == 0
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("nvenc probe failed: %s", exc)
    _NVENC_PROBE_CACHE[ffmpeg_path] = available
    return available


@dataclass(slots=True)
class RenderStats:
    """Telemetry returned alongside the rendered file."""

    output_path: Path
    frames_written: int
    fps: int
    width: int
    height: int
    render_seconds: float
    video_seconds: float
    encoder: str
    file_size_bytes: int

    @property
    def effective_fps(self) -> float:
        return self.frames_written / self.render_seconds if self.render_seconds > 0 else 0.0

    @property
    def speedup_factor(self) -> float:
        """How many seconds of finished video came out per second of wall time."""
        return self.video_seconds / self.render_seconds if self.render_seconds > 0 else 0.0


class AnimationRenderer:
    """Pipes RGB24 frames into ffmpeg and muxes the dialogue audio track."""

    def __init__(self, *, width: int = 1080, height: int = 1920, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_path = _resolve_ffmpeg()

    def _video_codec_args(self, *, use_gpu: bool) -> list[str]:
        del use_gpu  # The delivery contract is intentionally encoder-stable.
        return [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-b:v",
            "2600k",
            "-maxrate",
            "3200k",
            "-bufsize",
            "6000k",
        ]

    def render(
        self,
        *,
        frame_iter: Iterable[np.ndarray],
        audio_path: str | Path | None,
        output_path: Path,
        duration_s: float,
        subtitles_path: str | Path | None = None,
    ) -> RenderStats:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        use_gpu = False
        frames_written, encoder_used, elapsed = self._run_ffmpeg(
            frame_iter,
            audio_path=audio_path,
            output_path=output_path,
            use_gpu=use_gpu,
            subtitles_path=Path(subtitles_path) if subtitles_path else None,
        )

        size = output_path.stat().st_size if output_path.is_file() else 0
        stats = RenderStats(
            output_path=output_path,
            frames_written=frames_written,
            fps=self.fps,
            width=self.width,
            height=self.height,
            render_seconds=elapsed,
            video_seconds=duration_s,
            encoder=encoder_used,
            file_size_bytes=size,
        )
        _LOG.info(
            "rendered %s: %d frames in %.2fs (%.1fx realtime, %s) -> %d bytes",
            output_path,
            frames_written,
            elapsed,
            stats.speedup_factor,
            encoder_used,
            size,
        )
        return stats

    def _build_cmd(
        self,
        *,
        audio_path: Path | None,
        output_path: Path,
        use_gpu: bool,
        audio_codec: str,
        subtitles_path: Path | None = None,
    ) -> list[str]:
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
        ]
        if audio_path is not None:
            # ``-i merged_audio.wav`` — the per-turn TTS/dialogue track,
            # already normalized + resampled to a consistent rate by
            # whichever bridge module built it (see build_session_audio()).
            cmd += ["-i", str(audio_path)]
        filters = ["setsar=1:1"]
        if subtitles_path is not None:
            # Burn karaoke after declaring square pixels, preventing players
            # from applying an anamorphic display transform to the final MP4.
            filters.append(f"subtitles='{_escape_filter_path(subtitles_path)}'")
        cmd += ["-vf", ",".join(filters)]
        cmd += self._video_codec_args(use_gpu=use_gpu)
        cmd += ["-pix_fmt", "yuv420p"]
        if audio_path is not None:
            cmd += ["-c:a", audio_codec]
            if audio_codec == "aac":
                cmd += ["-b:a", "192k", "-ar", "44100"]
            # Explicit stream mapping: video from input 0, audio from input 1.
            # ``-shortest`` trims to whichever stream is shorter so a
            # duration_override that truncates the video frame count never
            # produces a video-audio length mismatch in the final container.
            cmd += ["-map", "0:v", "-map", "1:a", "-shortest"]
        cmd.append(str(output_path))
        return cmd

    def _run_ffmpeg(
        self,
        frame_iter: Iterable[np.ndarray],
        *,
        audio_path: str | Path | None,
        output_path: Path,
        use_gpu: bool,
        subtitles_path: Path | None = None,
    ) -> tuple[int, str, float]:
        audio_p = Path(audio_path) if audio_path else None
        # Always encode the final stream to a known-good AAC/44.1 kHz
        # contract. Stream-copying arbitrary source audio was a source of
        # inaudible or incompatible MP4 tracks.
        codec_attempts = ["aac"]

        last_error: Exception | None = None
        for encoder_is_gpu in ([use_gpu, False] if use_gpu else [False]):
            for audio_codec in codec_attempts:
                cmd = self._build_cmd(
                    audio_path=audio_p,
                    output_path=output_path,
                    use_gpu=encoder_is_gpu,
                    audio_codec=audio_codec,
                    subtitles_path=subtitles_path,
                )
                encoder_name = "h264_nvenc" if encoder_is_gpu else "libx264"
                try:
                    frames_written, elapsed = self._pipe_frames(cmd, frame_iter, output_path)
                    return frames_written, f"{encoder_name}+{audio_codec}", elapsed
                except _FfmpegFailure as exc:
                    last_error = exc
                    _LOG.warning(
                        "ffmpeg attempt failed (encoder=%s audio_codec=%s): %s — retrying",
                        encoder_name,
                        audio_codec,
                        exc,
                    )
                    # Frame generators are exhausted after one attempt; the
                    # caller passed a fresh iterator per top-level render()
                    # call, so a retry needs the *same* frames again. We
                    # therefore materialize frames once, up front, on first
                    # failure so subsequent retries can replay them.
                    frame_iter = exc.consumed_frames + list(frame_iter)
        raise RuntimeError(f"ffmpeg failed for every encoder/audio-codec combination: {last_error}")

    def _pipe_frames(
        self, cmd: list[str], frame_iter: Iterable[np.ndarray], output_path: Path
    ) -> tuple[int, float]:
        start = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        frames_written = 0
        consumed: list[np.ndarray] = []
        try:
            for frame in frame_iter:
                consumed.append(frame)
                proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
                frames_written += 1
        except (BrokenPipeError, OSError) as exc:
            proc.stdin.close()
            _, stderr = proc.communicate(timeout=30)
            raise _FfmpegFailure(str(exc) + " :: " + stderr.decode("utf-8", "ignore")[-2000:], consumed)
        proc.stdin.close()
        _, stderr = proc.communicate(timeout=120)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0 or not output_path.is_file():
            raise _FfmpegFailure(stderr.decode("utf-8", "ignore")[-2000:], consumed)
        return frames_written, elapsed


class _FfmpegFailure(RuntimeError):
    def __init__(self, message: str, consumed_frames: list[np.ndarray]) -> None:
        super().__init__(message)
        self.consumed_frames = consumed_frames


def resolve_ffmpeg() -> str:
    """Public accessor for the ffmpeg executable this package will pipe into."""
    return _resolve_ffmpeg()


__all__ = ["AnimationRenderer", "RenderStats", "resolve_ffmpeg"]
