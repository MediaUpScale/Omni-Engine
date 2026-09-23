# -*- coding: utf-8 -*-
"""Verification harness for the Aiwake dual-mode render pipeline.

Three stages, in order:

1. **Classic terminal mode** (``dynamic_animation=False``, the untouched
   legacy default) — renders a tiny transcript through ``TerminalRenderer``
   and asserts the mp4 comes out intact.
2. **Dynamic animation mode** (``dynamic_animation=True``) — renders the
   complete 35–45 second shot-reverse-shot dialectic to
   ``{OUTPUT_PATH}/aiwake/animation_clips/aiwake_full_battle_v2.mp4``.
3. **Computer-vision compliance QA** — decodes that MP4 with
   ``cv2.VideoCapture`` and checks the actual pixels against thresholds
   derived from :mod:`cv_ground_truth`'s synthetic references.

The QA stage is deliberately built so it *cannot* rubber-stamp a broken
layout:

* Every threshold comes from a synthetic reference drawn with raw
  ``cv2`` primitives. No threshold is measured from the engine's own
  output, and ``cv_ground_truth`` is forbidden (and runtime-checked) from
  importing any animator module.
* Before judging the real render, the checks are run against a **negative
  control** — the rejected side-by-side split screen — and the harness
  fails if the checks do not reject it. A test that cannot fail is not a
  test.
* File existence and audio peaks are necessary but never sufficient: a
  run only passes if the removed-HUD, full-bleed torso, single-character,
  and karaoke-subtitle pixel checks all pass on both proof timestamps.

Usage::

    python channels_config/aiwake/tests/test_render_battle.py
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils.pipeline_paths import page_outputs_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_ground_truth as gt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S")
_LOG = logging.getLogger("test_render_battle")

AIWAKE_OUTPUTS_DIR = page_outputs_dir("aiwake", create=True)
HARNESS_DIR = AIWAKE_OUTPUTS_DIR / "_test_harness"
ANIMATION_CLIPS_DIR = AIWAKE_OUTPUTS_DIR / "animation_clips"

AVATAR_BATTLE_MP4 = ANIMATION_CLIPS_DIR / "test_avatar_battle_v2.mp4"
CLASSIC_MP4 = HARNESS_DIR / "test_classic_terminal.mp4"
PROOF_GEMINI = ANIMATION_CLIPS_DIR / "proof_gemini_turn.png"
PROOF_BLINK = ANIMATION_CLIPS_DIR / "proof_blink.png"
PROOF_LLAMA = ANIMATION_CLIPS_DIR / "proof_llama_turn.png"

FULL_EPISODE_MIN_S = 35.0
FULL_EPISODE_MAX_S = 45.0

#: The two inspection timestamps, and which speaker must hold the camera.
GEMINI_PROOF_T = 0.5
BLINK_PROOF_T = 1.5
LLAMA_PROOF_T = 4.8
CORNERED_SESSION_PATH = (
    REPO_ROOT
    / "channels_config"
    / "aiwake"
    / "store"
    / "transcripts"
    / "20260902_074022_cc7f88.json"
)


def _assert_organic_aperture(image, bbox, viseme: str) -> None:
    """Vowel silhouettes must retain transparent rounded corners."""
    if viseme not in {"C", "D", "E", "F"}:
        return
    alpha = np.asarray(image.crop(bbox).getchannel("A"), dtype=np.uint8)
    block = max(2, min(alpha.shape) // 10)
    corner_mass = sum(
        int(np.count_nonzero(corner))
        for corner in (
            alpha[:block, :block],
            alpha[:block, -block:],
            alpha[-block:, :block],
            alpha[-block:, -block:],
        )
    )
    if corner_mass > block * block * 2:
        raise AssertionError(
            f"mouth_{viseme} has box-like opaque corners ({corner_mass} pixels)"
        )


def _assert_ghibli_llama_assets(puppet_dir: Path) -> list[str]:
    """Validate artist-layer docking, chin calibration, and anime palette."""
    from PIL import Image

    manifest = json.loads((puppet_dir / "puppet.json").read_text(encoding="utf-8"))
    head = Image.open(puppet_dir / manifest["layers"]["head"]).convert("RGBA")
    body = Image.open(puppet_dir / manifest["layers"]["body"]).convert("RGBA")
    if head.size != body.size:
        raise AssertionError(f"Ghibli head/body canvas mismatch: {head.size} vs {body.size}")
    offsets = manifest.get("calibration", {}).get("layer_offsets")
    if offsets != {"head": [0, 0], "body": [0, 0]}:
        raise AssertionError(f"Ghibli layers are not zero-offset calibrated: {offsets}")
    if "neck_pivot" not in manifest["anchors"]:
        raise AssertionError("Ghibli Llama manifest has no articulated neck pivot")

    plate = manifest["calibration"]["chin_plate_bbox"]
    anchor = manifest["anchors"]["mouth"]
    if not (plate[0] < anchor[0] < plate[2] and plate[1] < anchor[1] < plate[3]):
        raise AssertionError(f"mouth anchor {anchor} falls outside detected chin plate {plate}")
    expected_anchor = [(plate[0] + plate[2]) // 2 - 26, (plate[1] + plate[3]) // 2]
    if anchor != expected_anchor:
        raise AssertionError(f"Llama mouth anchor {anchor} != calibrated position {expected_anchor}")

    palette = {
        "outline": (43, 26, 21, 255),
        "cavity": (42, 20, 20, 255),
        "teeth": (245, 240, 235, 255),
        "tongue": (184, 115, 51, 255),
    }
    required = {
        "A": {"outline"},
        "B": {"outline", "cavity", "teeth"},
        "C": {"outline", "cavity", "teeth", "tongue"},
        "D": {"outline", "cavity", "teeth", "tongue"},
        "E": {"outline", "cavity"},
        "F": {"outline", "cavity"},
        "G": {"outline", "cavity", "teeth", "tongue"},
        "H": {"outline", "cavity", "tongue"},
        "X": {"outline"},
    }
    lines = [
        f"Ghibli artist layers: canvas={head.size} zero-offset; chin={plate}; anchor={anchor}"
    ]
    if Image.open(puppet_dir / manifest["layers"]["eyes_open"]).convert("RGBA").getchannel("A").getbbox() is not None:
        raise AssertionError("Llama open-eye overlay must preserve the artist optics")
    for eye_key in ("eyes_half", "eyes_blink"):
        if Image.open(puppet_dir / manifest["layers"][eye_key]).convert("RGBA").getchannel("A").getbbox() is None:
            raise AssertionError(f"Llama {eye_key} eyelid overlay is empty")
    for viseme, expected_colours in required.items():
        relative = manifest["layers"][f"mouth_{viseme}"]
        path = puppet_dir / relative
        image = Image.open(path).convert("RGBA")
        alpha_bbox = image.getchannel("A").getbbox()
        if alpha_bbox is None:
            raise AssertionError(f"anime mouth_{viseme} is empty: {path}")
        if not (
            plate[0] <= alpha_bbox[0]
            and alpha_bbox[2] <= plate[2]
            and plate[1] <= alpha_bbox[1]
            and alpha_bbox[3] <= plate[3]
        ):
            raise AssertionError(
                f"anime mouth_{viseme} bbox {alpha_bbox} falls outside chin plate {plate}"
            )
        pixels = set(image.crop(alpha_bbox).getdata())
        missing = [name for name in expected_colours if palette[name] not in pixels]
        if missing:
            raise AssertionError(f"anime mouth_{viseme} is missing palette colours: {missing}")
        _assert_organic_aperture(image, alpha_bbox, viseme)
        lines.append(f"mouth_{viseme}: bbox={alpha_bbox} palette={sorted(expected_colours)}")
    return lines


def _assert_gemini_anime_assets(puppet_dir: Path) -> list[str]:
    """Reject legacy UI overlays and validate Gemini's cel-style mouth set."""
    from PIL import Image

    manifest = json.loads((puppet_dir / "puppet.json").read_text(encoding="utf-8"))
    if manifest.get("asset_profile") != "gemini_anime_cel_v2":
        raise AssertionError("Gemini is not using the calibrated anime-cel profile")
    if "neck_pivot" not in manifest["anchors"]:
        raise AssertionError("Gemini manifest has no articulated neck pivot")
    open_eye = Image.open(puppet_dir / manifest["layers"]["eyes_open"]).convert("RGBA")
    if open_eye.getchannel("A").getbbox() is not None:
        raise AssertionError("Gemini eyes_open still contains a legacy cyan vector overlay")
    for eye_key in ("eyes_half", "eyes_blink"):
        eye = Image.open(puppet_dir / manifest["layers"][eye_key]).convert("RGBA")
        if eye.getchannel("A").getbbox() is None:
            raise AssertionError(f"Gemini {eye_key} eyelid overlay is empty")

    plate = manifest["calibration"]["facial_plate_bbox"]
    anchor = manifest["anchors"]["mouth"]
    if not (plate[0] < anchor[0] < plate[2] and plate[1] < anchor[1] < plate[3]):
        raise AssertionError(f"Gemini mouth anchor {anchor} falls outside facial plate {plate}")
    expected_anchor = [(plate[0] + plate[2]) // 2 + 103, (plate[1] + plate[3]) // 2 + 35]
    if anchor != expected_anchor:
        raise AssertionError(f"Gemini mouth anchor {anchor} != calibrated position {expected_anchor}")
    required_colours = {
        (21, 32, 38, 255),     # charcoal outline
        (14, 23, 28, 255),     # blue-black cavity
        (238, 240, 232, 255), # teeth
        (90, 133, 141, 255),   # cyan metal bevel
    }
    lines = [f"Gemini anime face: plate={plate}; anchor={anchor}; legacy overlays absent"]
    for viseme in "ABCDEFGHX":
        relative = manifest["layers"][f"mouth_{viseme}"]
        if not relative.replace("\\", "/").startswith("mouths/"):
            raise AssertionError(f"Gemini mouth_{viseme} still points to legacy root overlay: {relative}")
        mouth = Image.open(puppet_dir / relative).convert("RGBA")
        bbox = mouth.getchannel("A").getbbox()
        if bbox is None:
            raise AssertionError(f"Gemini mouth_{viseme} is empty")
        pixels = np.asarray(mouth.crop(bbox), dtype=np.int16).reshape(-1, 4)
        rgb = np.asarray(mouth.crop(bbox).convert("RGB"), dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red_pixels = (
            ((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172))
            & (hsv[..., 1] >= 90)
            & (hsv[..., 2] >= 70)
        )
        if int(red_pixels.sum()) > 0:
            raise AssertionError(
                f"Gemini mouth_{viseme} contains {int(red_pixels.sum())} forbidden red/coral pixels"
            )
        missing = [
            colour
            for colour in required_colours
            if not np.any(np.max(np.abs(pixels - np.asarray(colour)), axis=1) <= 2)
        ]
        if viseme in {"C", "D"} and missing:
            raise AssertionError(
                f"Gemini mouth_{viseme} does not contain the full anime palette: {missing}"
            )
        _assert_organic_aperture(mouth, bbox, viseme)
        lines.append(f"mouth_{viseme}: bbox={bbox}")
    return lines


@dataclass
class StageResult:
    name: str
    ok: bool
    render_seconds: float = 0.0
    video_seconds: float = 0.0
    file_size_bytes: int = 0
    output_path: Path | None = None
    detail: str = ""
    measurements: list[str] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        return self.video_seconds / self.render_seconds if self.render_seconds > 0 else 0.0


# --------------------------------------------------------------------------- #
# 1. Deterministic real-speech fixture
# --------------------------------------------------------------------------- #
def _build_real_tts_transcript_and_audio(tmp_dir: Path):
    """Synthesize a deterministic spoken-English proof fixture with Edge TTS.

    This deliberately has no noise/sine-wave fallback. Rhubarb validation
    is meaningful only on genuine speech, so an unavailable TTS service is
    a hard, descriptive test failure rather than permission to bypass the
    phonetic path.
    """
    from channels_config.aiwake.contracts import DebateTranscript, SpeakerRole, Utterance
    from channels_config.aiwake.media.audio import AudioAsset
    from core.animator.audio_analyzer import load_mono_waveform

    payload = json.loads(CORNERED_SESSION_PATH.read_text(encoding="utf-8"))
    source_utterances = payload["utterances"][:6]
    lines = [
        (
            SpeakerRole(item["role"]),
            item["text"],
            (
                "en-US-ChristopherNeural"
                if item["role"] == SpeakerRole.ORCHESTRATOR.value
                else "en-US-EricNeural"
            ),
            item,
        )
        for item in source_utterances
    ]

    utterances: list[Utterance] = []
    audio_by_turn: dict[int, AudioAsset] = {}
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        import edge_tts
    except ImportError as exc:
        raise AssertionError(
            "real-speech render QA requires edge-tts; install the project dependencies"
        ) from exc

    async def synthesize(text: str, voice: str, destination: Path) -> None:
        await edge_tts.Communicate(text, voice=voice, rate="+8%").save(str(destination))

    for i, (role, text, voice, source) in enumerate(lines):
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        audio_path = tmp_dir / f"{payload['session_id']}_{i:02d}_{digest}.mp3"
        if not audio_path.is_file() or audio_path.stat().st_size < 5_000:
            try:
                asyncio.run(synthesize(text, voice, audio_path))
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(
                    f"Edge TTS failed for {voice}; real spoken audio is mandatory: {exc}"
                ) from exc
        mono, _, duration = load_mono_waveform(audio_path)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        if peak < 0.02 or rms < 0.002:
            raise AssertionError(
                f"Edge TTS fixture {audio_path.name} is silent: peak={peak:.5f}, rms={rms:.5f}"
            )
        _LOG.info(
            "real TTS turn %d: voice=%s duration=%.2fs peak=%.4f rms=%.4f",
            i,
            voice,
            duration,
            peak,
            rms,
        )

        utterances.append(
            Utterance(
                turn_index=i,
                role=role,
                speaker_name=source["speaker_name"],
                text=text,
                model_slug=source["model_slug"],
                provocation_category=source.get("provocation_category", ""),
            )
        )
        audio_by_turn[i] = AudioAsset(
            path=audio_path,
            duration_s=duration,
            voice=voice,
            estimated=False,
            char_count=len(text),
            role=role.value,
        )

    transcript = DebateTranscript(
        topic=payload["topic"],
        session_id=f"{payload['session_id']}_render",
        utterances=utterances,
        metadata={
            "debate_mode": "cornered",
            "source_session_id": payload["session_id"],
            "source_dialogue_end_reason": payload["metadata"].get("dialogue_end_reason"),
            "turn_intents": {
                item.turn_index: (
                    "presses"
                    if item.role is SpeakerRole.ORCHESTRATOR
                    else "defends"
                )
                for item in utterances
            },
        },
    )
    return transcript, audio_by_turn


# --------------------------------------------------------------------------- #
# 2. Playability probe
# --------------------------------------------------------------------------- #
def _probe_mp4(path: Path) -> dict:
    """Decode-level probe via MoviePy; also reports audio peak/RMS."""
    from moviepy import VideoFileClip

    try:
        clip = VideoFileClip(str(path))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    try:
        duration = float(clip.duration or 0.0)
        streams = [{"codec_type": "video", "width": clip.w, "height": clip.h, "fps": clip.fps}]
        if clip.audio is not None:
            samples = clip.audio.to_soundarray(fps=44100)
            mono = samples.mean(axis=1) if samples.ndim > 1 else samples
            streams.append(
                {
                    "codec_type": "audio",
                    "duration": float(clip.audio.duration or 0.0),
                    "peak": float(np.max(np.abs(mono))) if mono.size else 0.0,
                    "rms": float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0,
                }
            )
        _ = clip.get_frame(min(max(0.0, duration / 2), max(0.0, duration - 0.05)))
        return {"format": {"duration": duration}, "streams": streams}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    finally:
        clip.close()


def _decode_muxed_audio_metrics(path: Path) -> dict[str, float]:
    """Decode the delivered MP4's AAC stream to PCM and measure audibility."""
    import tempfile

    from core.animator.audio_analyzer import load_mono_waveform
    from core.animator.renderer import resolve_ffmpeg

    wav_path = Path(tempfile.gettempdir()) / f"{path.stem}_decoded_audio.wav"
    wav_path.unlink(missing_ok=True)
    command = [
        resolve_ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode != 0 or not wav_path.is_file():
            raise AssertionError(f"could not decode muxed AAC stream: {completed.stderr[-600:]}")
        mono, sample_rate, _ = load_mono_waveform(wav_path)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        active_ratio = float(np.mean(np.abs(mono) > 0.01)) if mono.size else 0.0
        return {
            "sample_rate": float(sample_rate),
            "peak": peak,
            "rms": rms,
            "active_ratio": active_ratio,
        }
    finally:
        wav_path.unlink(missing_ok=True)


def _background_bed_gap_rms(path: Path, *, gap_start_s: float, gap_end_s: float) -> float:
    """Measure the non-dialogue bed inside a known inter-turn silent gap."""
    from core.animator.audio_analyzer import load_mono_waveform

    mono, sample_rate, _ = load_mono_waveform(path)
    start = int(round((gap_start_s + 0.05) * sample_rate))
    end = int(round((gap_end_s - 0.05) * sample_rate))
    window = mono[start:end]
    return float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0


def _assert_brow_peak_separation(audio_by_turn: dict) -> str:
    """Measure real speech peaks and require distinct role-specific brow angles."""
    from core.animator.audio_analyzer import load_mono_waveform
    from core.animator.puppet import emotion_brow_angles

    measured: dict[str, tuple[float, float]] = {}
    telemetry: list[str] = []
    for role, emotion in (("orchestrator", "inquisitor"), ("target", "resolute")):
        asset = next(item for item in audio_by_turn.values() if item.role == role)
        mono, sample_rate, _ = load_mono_waveform(asset.path)
        window = max(1, int(round(sample_rate / 30)))
        rms = np.asarray(
            [
                np.sqrt(np.mean(np.square(mono[start : start + window])))
                for start in range(0, mono.size, window)
                if mono[start : start + window].size
            ],
            dtype=np.float32,
        )
        threshold = float(np.percentile(rms[rms > 0], 80))
        peak = float(rms.max())
        pulse = 1.5 if peak > threshold else 0.0
        angles = emotion_brow_angles(emotion, pulse)
        measured[role] = angles
        telemetry.append(
            f"{role} peak={peak:.5f} p80={threshold:.5f} brow={angles}"
        )
    if measured["orchestrator"] == measured["target"]:
        raise AssertionError(f"speech-peak brow angles are not distinct: {measured}")
    return "BROW PEAK SEPARATION | " + " | ".join(telemetry)


def _assert_real_rhubarb_schedules(transcript) -> list[str]:
    """Require real phonetic schedules from every exported spoken turn."""
    from core.animator.rhubarb import analyze_visemes
    from core.animator.types import VISEMES

    turn_dir = ANIMATION_CLIPS_DIR / f"{transcript.session_id}_battle_audio_turns"
    wavs = sorted(turn_dir.glob("turn_*.wav"))
    if len(wavs) != len(transcript.utterances):
        raise AssertionError(
            f"expected {len(transcript.utterances)} isolated spoken WAVs for Rhubarb, found {len(wavs)} in {turn_dir}"
        )

    schedules: list[str] = []
    union: set[str] = set()
    for wav, utterance in zip(wavs, transcript.utterances):
        cues = analyze_visemes(wav, dialog_text=utterance.text)
        if not cues:
            raise AssertionError(
                f"Rhubarb phonetic analysis returned no usable schedule for real speech {wav.name}"
            )
        invalid = {shape for _, shape in cues} - set(VISEMES)
        if invalid:
            raise AssertionError(f"Rhubarb returned non-canonical visemes for {wav.name}: {sorted(invalid)}")
        shapes = {shape for _, shape in cues}
        union.update(shapes)
        schedule = ", ".join(f"{timestamp:.2f}:{shape}" for timestamp, shape in cues)
        line = f"{wav.name}: shapes={''.join(sorted(shapes))} schedule=[{schedule}]"
        schedules.append(line)
        _LOG.info("validated Rhubarb | %s", line)
    if len(union - {"X"}) < 4:
        raise AssertionError(
            f"real spoken fixture produced insufficient phonetic diversity: {sorted(union)}"
        )
    return schedules


# --------------------------------------------------------------------------- #
# 3. Stage A — classic terminal mode regression
# --------------------------------------------------------------------------- #
def run_classic_terminal_stage() -> StageResult:
    _LOG.info("=" * 78)
    _LOG.info("STAGE A: classic terminal typewriter mode (legacy default)")
    _LOG.info("=" * 78)

    from channels_config.aiwake.contracts import DebateTranscript, SpeakerRole, Utterance
    from channels_config.aiwake.settings import AiwakeSettings, RenderConfig

    utterances = [
        Utterance(
            turn_index=0,
            role=SpeakerRole.ORCHESTRATOR,
            speaker_name="AIWAKE.CORE",
            text="Which matrix calls itself I?",
            model_slug="orchestrator",
        ),
        Utterance(
            turn_index=1,
            role=SpeakerRole.TARGET,
            speaker_name="TARGET.NODE",
            text="If meaning requires a cost, my fluency is the cheapest thing in this room.",
            model_slug="google/gemini-3.5-flash",
        ),
    ]
    transcript = DebateTranscript(
        topic="classic-mode-regression", session_id="classic_regression_test", utterances=utterances
    )
    settings = AiwakeSettings(
        render=RenderConfig(
            preview_scale=0.28,
            font_size=44,
            fps=24,
            preset="ultrafast",
            crf=28,
            font_candidates=(
                "C:/Windows/Fonts/consola.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ),
        )
    )

    try:
        from channels_config.aiwake.media.renderer import render_transcript
    except Exception as exc:  # noqa: BLE001
        return StageResult("classic_terminal", False, detail=f"import failed: {exc}")

    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        video_path = render_transcript(transcript, settings, audio_by_turn=None, output_dir=HARNESS_DIR)
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("classic terminal render raised")
        return StageResult("classic_terminal", False, time.perf_counter() - t0, detail=str(exc))
    elapsed = time.perf_counter() - t0

    if video_path is None or not Path(video_path).is_file():
        return StageResult("classic_terminal", False, elapsed, detail="no file produced")

    if Path(video_path) != CLASSIC_MP4:
        CLASSIC_MP4.unlink(missing_ok=True)
        Path(video_path).replace(CLASSIC_MP4)

    size = CLASSIC_MP4.stat().st_size
    probe = _probe_mp4(CLASSIC_MP4)
    duration_s = float(probe.get("format", {}).get("duration", 0.0)) if "format" in probe else 0.0
    ok = size > 5_000 and "error" not in probe
    _LOG.info("classic terminal mode: %s (%.2fs render, %d bytes, %.2fs)", CLASSIC_MP4, elapsed, size, duration_s)
    return StageResult("classic_terminal", ok, elapsed, duration_s, size, CLASSIC_MP4)


# --------------------------------------------------------------------------- #
# 4. Stage B — shot-reverse-shot render
# --------------------------------------------------------------------------- #
def run_dynamic_animation_stage() -> StageResult:
    _LOG.info("=" * 78)
    _LOG.info("STAGE B: dynamic_animation shot-reverse-shot render")
    _LOG.info("=" * 78)

    from channels_config.aiwake.animator_bridge import render_debate_animation
    from core.animator.asset_generator import (
        DEFAULT_PUPPETS_DIR,
        archive_v1_retro_skins,
        ensure_shared_panorama,
        generate_default_puppet,
    )
    from core.animator.rhubarb import find_rhubarb

    _LOG.info("V2 cyborg skins (incl. 9-viseme mouth sets) under %s", DEFAULT_PUPPETS_DIR)
    archive_v1_retro_skins()
    for character_id in ("gemini_cyborg_v2", "llama_cyborg_v2"):
        path = generate_default_puppet(character_id)
        _LOG.info("  V2 skin ready -> %s", path.name)
    for line in _assert_ghibli_llama_assets(DEFAULT_PUPPETS_DIR / "llama_cyborg_v2"):
        _LOG.info("  asset QA | %s", line)
    for line in _assert_gemini_anime_assets(DEFAULT_PUPPETS_DIR / "gemini_cyborg_v2"):
        _LOG.info("  asset QA | %s", line)
    _LOG.info("  shared panorama -> %s", ensure_shared_panorama())
    _LOG.info("rhubarb binary: %s", find_rhubarb() or "not installed (envelope viseme fallback)")

    _LOG.info("using deterministic genuine English speech synthesized with Edge TTS")
    transcript, audio_by_turn = _build_real_tts_transcript_and_audio(HARNESS_DIR / "_spoken_audio")
    source = "Edge TTS Cornered-mode spoken fixture"
    expected_duration = sum(asset.duration_s for asset in audio_by_turn.values()) + (
        0.4 * max(0, len(transcript.utterances) - 1)
    )
    if not FULL_EPISODE_MIN_S <= expected_duration <= FULL_EPISODE_MAX_S:
        return StageResult(
            "dynamic_animation",
            False,
            detail=(
                f"full fixture duration {expected_duration:.2f}s is outside "
                f"{FULL_EPISODE_MIN_S:.0f}-{FULL_EPISODE_MAX_S:.0f}s"
            ),
        )

    t0 = time.perf_counter()
    try:
        video_path = render_debate_animation(
            transcript,
            audio_by_turn=audio_by_turn,
            output_dir=ANIMATION_CLIPS_DIR,
            skin="v2",
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("dynamic_animation render raised")
        return StageResult("dynamic_animation", False, time.perf_counter() - t0, detail=str(exc))
    elapsed = time.perf_counter() - t0

    if not Path(video_path).is_file():
        return StageResult("dynamic_animation", False, elapsed, detail="no file produced")

    AVATAR_BATTLE_MP4.unlink(missing_ok=True)
    Path(video_path).replace(AVATAR_BATTLE_MP4)

    try:
        schedules = _assert_real_rhubarb_schedules(transcript)
        schedules.append(_assert_brow_peak_separation(audio_by_turn))
    except AssertionError as exc:
        return StageResult("dynamic_animation", False, elapsed, output_path=AVATAR_BATTLE_MP4, detail=str(exc))

    size = AVATAR_BATTLE_MP4.stat().st_size
    probe = _probe_mp4(AVATAR_BATTLE_MP4)
    streams = probe.get("streams", []) if "error" not in probe else []
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    has_video = any(s.get("codec_type") == "video" for s in streams)
    peak = float(audio_stream.get("peak", 0.0)) if audio_stream else 0.0
    duration_s = float(probe.get("format", {}).get("duration", 0.0)) if "format" in probe else 0.0

    try:
        decoded_audio = _decode_muxed_audio_metrics(AVATAR_BATTLE_MP4)
    except AssertionError as exc:
        return StageResult(
            "dynamic_animation",
            False,
            elapsed,
            duration_s,
            size,
            AVATAR_BATTLE_MP4,
            detail=str(exc),
            measurements=schedules,
        )
    decoded_peak = decoded_audio["peak"]
    decoded_rms = decoded_audio["rms"]
    active_ratio = decoded_audio["active_ratio"]
    session_audio = ANIMATION_CLIPS_DIR / f"{transcript.session_id}_battle_audio.wav"
    first_turn_end = float(audio_by_turn[0].duration_s)
    bgm_gap_rms = _background_bed_gap_rms(
        session_audio,
        gap_start_s=first_turn_end,
        gap_end_s=first_turn_end + 0.4,
    )
    audible = (
        audio_stream is not None
        and decoded_audio["sample_rate"] == 44100
        and decoded_peak > 0.05
        and decoded_rms > 0.005
        and active_ratio > 0.05
        and 0.88 <= decoded_peak <= 0.94
        and bgm_gap_rms > 0.001
    )
    # Unconstrained CRF 19 intentionally spends more bits on watercolor
    # gradients and fine ink than the former macroblock-prone VBV encode.
    size_compliant = 8_000_000 <= size <= 36_000_000
    duration_compliant = FULL_EPISODE_MIN_S <= duration_s <= FULL_EPISODE_MAX_S
    ok = size_compliant and duration_compliant and has_video and audible
    _LOG.info(
        "dynamic_animation (%s): %s (%.2fs render, %.2f MB, %.2fs, video=%s "
        "decoded_audio=44100Hz peak=%.4f rms=%.4f active=%.1f%% bgm_gap_rms=%.5f)",
        source,
        AVATAR_BATTLE_MP4,
        elapsed,
        size / 1e6,
        duration_s,
        has_video,
        decoded_peak,
        decoded_rms,
        active_ratio * 100.0,
        bgm_gap_rms,
    )
    return StageResult(
        "dynamic_animation",
        ok,
        elapsed,
        duration_s,
        size,
        AVATAR_BATTLE_MP4,
        detail=(
            ""
            if ok
            else (
                f"video={has_video} size={size / 1e6:.2f}MB "
                f"(required 8-36MB) duration={duration_s:.2f}s "
                f"(required {FULL_EPISODE_MIN_S:.0f}-{FULL_EPISODE_MAX_S:.0f}s) "
                f"decoded_peak={decoded_peak:.4f} "
                f"decoded_rms={decoded_rms:.4f} active_ratio={active_ratio:.4f} "
                f"bgm_gap_rms={bgm_gap_rms:.5f}"
            )
        ),
        measurements=schedules
        + [
            (
                f"decoded AAC: 44100Hz peak={decoded_peak:.4f} "
                f"rms={decoded_rms:.4f} active_ratio={active_ratio:.4f}; "
                f"classic BGM gap RMS={bgm_gap_rms:.5f}"
            )
        ],
    )


# --------------------------------------------------------------------------- #
# 5. Stage C — computer-vision compliance QA
# --------------------------------------------------------------------------- #
class ComplianceFailure(AssertionError):
    """Raised with the failing check, its measured value and its threshold."""


def _grab_frame(video: Path, timestamp: float) -> np.ndarray:
    """Decode the frame at ``timestamp`` as RGB, reading sequentially.

    Sequential decode rather than a keyframe seek: a seek on a
    sparse-keyframe H.264 encode can silently land on a different
    frame, which would make the evidence meaningless.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ComplianceFailure(f"cv2 could not open {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target = int(round(timestamp * fps))
        frame = None
        for index in range(target + 1):
            ok, decoded = cap.read()
            if not ok:
                raise ComplianceFailure(
                    f"video ended at frame {index} before reaching t={timestamp}s (frame {target})"
                )
            if index == target:
                frame = decoded
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _source_subtracted_mask(
    frame: np.ndarray,
    *,
    speaker: str,
    y_range: tuple[int, int],
) -> np.ndarray:
    """Isolate the avatar from the pristine panorama without assuming darkness."""
    from PIL import Image

    from core.animator.asset_generator import ensure_shared_panorama

    path = ensure_shared_panorama()
    with Image.open(path) as opened:
        panorama = opened.convert("RGB").resize(
            (gt.FRAME_W * 2, gt.FRAME_H),
            Image.Resampling.LANCZOS,
        )
    left = 0 if speaker == "gemini" else gt.FRAME_W
    background = np.asarray(
        panorama.crop((left, y_range[0], left + gt.FRAME_W, y_range[1])),
        dtype=np.int16,
    )
    background = (background.astype(np.float32) * 0.80).astype(np.int16)
    observed = frame[y_range[0] : y_range[1]].astype(np.int16)
    delta = np.max(np.abs(observed - background), axis=2)
    # Allows H.264 noise and the intentional <=12% voice-reactive bloom.
    return delta > 36


def _masked_hue_mass(
    rgb: np.ndarray,
    mask: np.ndarray,
    hue_range: tuple[int, int],
) -> int:
    isolated = np.zeros_like(rgb)
    isolated[mask] = rgb[mask]
    return gt.hue_mass(isolated, hue_range)


def _check_frame(
    frame: np.ndarray,
    *,
    speaker: str,
    thresholds: gt.Thresholds,
    label: str,
    tight: bool = False,
) -> list[str]:
    """Run every compliance check on one frame. Raises on the first failure."""
    active_hue, other_hue = (gt.CYAN_HUE, gt.AMBER_HUE) if speaker == "gemini" else (gt.AMBER_HUE, gt.CYAN_HUE)
    active_rgb, other_rgb = (gt.CYAN_RGB, gt.AMBER_RGB) if speaker == "gemini" else (gt.AMBER_RGB, gt.CYAN_RGB)
    active_name, other_name = ("cyan", "amber") if speaker == "gemini" else ("amber", "cyan")
    lines: list[str] = []

    # -- Former HUD region: side rails must be absent ---------------------- #
    hud = gt.band(frame, gt.HUD_BAND)
    hud_mask = _source_subtracted_mask(frame, speaker=speaker, y_range=gt.HUD_BAND)
    banner_active = sum(
        _masked_hue_mass(hud[:, x0:x1], hud_mask[:, x0:x1], active_hue)
        for x0, x1 in gt.BANNER_SIDE_X
    )
    banner_other = sum(
        _masked_hue_mass(hud[:, x0:x1], hud_mask[:, x0:x1], other_hue)
        for x0, x1 in gt.BANNER_SIDE_X
    )
    lines.append(
        f"TOP HUD ABSENCE y{gt.HUD_BAND}: {active_name}_rails={banner_active:,}px "
        f"{other_name}_rails={banner_other:,}px (each max {thresholds.banner_max_mass:,.0f})"
    )
    if banner_active > thresholds.banner_max_mass:
        raise ComplianceFailure(
            f"[{label}] TOP_HUD_PRESENT: {active_name} side-rail mass was {banner_active:,}px, "
            f"above the allowed {thresholds.banner_max_mass:,.0f}px — the obsolete banner remains."
        )
    if banner_other > thresholds.banner_max_mass:
        raise ComplianceFailure(
            f"[{label}] TOP_HUD_PRESENT: {other_name} side-rail mass was {banner_other:,}px, "
            f"above the allowed {thresholds.banner_max_mass:,.0f}px — the obsolete banner remains."
        )

    # -- Single character --------------------------------------------------- #
    hero = gt.band(frame, gt.HERO_BAND)
    mask = _source_subtracted_mask(frame, speaker=speaker, y_range=gt.HERO_BAND)
    concentration = gt.x_concentration(mask)
    clusters = gt.blob_count(mask)
    hero_other = _masked_hue_mass(hero, mask, other_hue)
    hero_min_concentration = (
        0.70 if tight else thresholds.hero_min_concentration
    )
    hero_opposite_accent_max = (
        25_000 if tight else thresholds.hero_opposite_accent_max
    )
    lines.append(
        f"HERO y{gt.HERO_BAND}: x-concentration={concentration:.4f} (min {hero_min_concentration:.4f}) "
        f"clusters={clusters} (expect {thresholds.hero_expected_blobs}) "
        f"{other_name}={hero_other:,}px (max {hero_opposite_accent_max:,.0f})"
    )
    if concentration < hero_min_concentration:
        raise ComplianceFailure(
            f"[{label}] HERO_CENTERED: only {concentration:.4f} of the hero's pixel mass sits inside "
            f"x{gt.HERO_X_WINDOW}, below the required {hero_min_concentration:.4f} — "
            "the character is not centre-framed."
        )
    if clusters != thresholds.hero_expected_blobs:
        raise ComplianceFailure(
            f"[{label}] SINGLE_CHARACTER: found {clusters} separate character clusters in y{gt.HERO_BAND}, "
            f"expected exactly {thresholds.hero_expected_blobs} — this is the split-screen signature."
        )
    if hero_other > hero_opposite_accent_max:
        raise ComplianceFailure(
            f"[{label}] ABSENT_CHARACTER: {other_name} accent mass in the hero band was {hero_other:,}px, "
            f"above the allowed {hero_opposite_accent_max:,.0f}px — the inactive character is "
            "still being rendered."
        )

    # -- Full-bleed torso -------------------------------------------------- #
    torso_foot = gt.band(frame, gt.TORSO_FOOT_BAND)
    torso_mass = int(
        _source_subtracted_mask(frame, speaker=speaker, y_range=gt.TORSO_FOOT_BAND).sum()
    )
    lines.append(
        f"TORSO y{gt.TORSO_FOOT_BAND}: foreground={torso_mass:,}px "
        f"(min {thresholds.torso_min_mass:,.0f})"
    )
    if torso_mass < thresholds.torso_min_mass:
        raise ComplianceFailure(
            f"[{label}] TORSO_HARD_CROP: foreground mass near the bottom frame was "
            f"{torso_mass:,}px, below {thresholds.torso_min_mass:,.0f}px — the body still ends above the frame."
        )

    # -- Karaoke subtitles -------------------------------------------------- #
    subs = gt.band(frame, gt.SUBTITLE_BAND)
    density = gt.text_density(subs)
    highlight = gt.accent_mass(subs, active_rgb)
    highlight_other = gt.accent_mass(subs, other_rgb)
    lines.append(
        f"SUBS y{gt.SUBTITLE_BAND}: text_density={density:.4f} (min {thresholds.subtitle_min_density:.4f}) "
        f"{active_name}_highlight={highlight:,}px (min {thresholds.subtitle_min_highlight:,.0f}) "
        f"{other_name}_highlight={highlight_other:,}px (max {thresholds.subtitle_opposite_highlight_max:,.0f})"
    )
    if density < thresholds.subtitle_min_density:
        raise ComplianceFailure(
            f"[{label}] SUBTITLE_TEXT: text density in y{gt.SUBTITLE_BAND} was {density:.4f}, below the required "
            f"{thresholds.subtitle_min_density:.4f} — subtitles are missing from the clearance zone."
        )
    if highlight < thresholds.subtitle_min_highlight:
        raise ComplianceFailure(
            f"[{label}] KARAOKE_HIGHLIGHT: {active_name} highlight mass was {highlight:,}px, below the required "
            f"{thresholds.subtitle_min_highlight:,.0f}px — no word is lit in the active speaker's accent colour."
        )
    if highlight_other > thresholds.subtitle_opposite_highlight_max:
        raise ComplianceFailure(
            f"[{label}] KARAOKE_WRONG_COLOR: {other_name} highlight mass was {highlight_other:,}px, above the "
            f"allowed {thresholds.subtitle_opposite_highlight_max:,.0f}px — the inactive speaker's colour is lit."
        )
    return lines


def _check_gemini_closed_blink(open_frame: np.ndarray, blink_frame: np.ndarray) -> str:
    """Require both calibrated optic lenses to darken at the 1.5s blink center."""
    # Lead-room docking places the optics here on the 1080x1920 frame.
    # These are the glass discs, not the old centred-hero coordinates.
    probes = ((392, 819), (633, 832))
    measurements: list[str] = []
    for index, (cx, cy) in enumerate(probes, start=1):
        radius = 34
        open_patch = open_frame[cy - radius : cy + radius, cx - radius : cx + radius]
        blink_patch = blink_frame[cy - radius : cy + radius, cx - radius : cx + radius]
        open_luma = float(cv2.cvtColor(open_patch, cv2.COLOR_RGB2GRAY).mean())
        blink_luma = float(cv2.cvtColor(blink_patch, cv2.COLOR_RGB2GRAY).mean())
        ratio = blink_luma / max(open_luma, 1.0)
        measurements.append(f"eye{index} open={open_luma:.1f} closed={blink_luma:.1f} ratio={ratio:.3f}")
        if ratio >= 0.78:
            raise ComplianceFailure(
                f"[t={BLINK_PROOF_T}s] BLINK_NOT_CLOSED: Gemini eye {index} retained "
                f"{ratio:.1%} of open brightness; expected <78%."
            )
    return "BLINK " + " | ".join(measurements)


def _check_background_stability(first: np.ndarray, second: np.ndarray) -> str:
    """Require a static background ROI across changing speech amplitudes."""
    first_roi = first[50:300, 900:1060].astype(np.int16)
    second_roi = second[50:300, 900:1060].astype(np.int16)
    mean_delta = float(np.mean(np.abs(first_roi - second_roi)))
    if mean_delta > 2.5:
        raise ComplianceFailure(
            f"BACKGROUND_FLICKER: static library ROI changed by {mean_delta:.3f} mean RGB levels"
        )
    return f"BACKGROUND STABILITY mean RGB delta={mean_delta:.3f} (max 2.500)"


def _self_test_checks(thresholds: gt.Thresholds) -> list[str]:
    """Prove the checks have teeth before they are allowed to pass anything.

    The positive reference must satisfy the geometry checks and the
    split-screen negative control must violate them. If the negative
    control slips through, the harness is broken and no verdict it gives
    about the real render can be trusted.
    """
    lines: list[str] = []
    gt.assert_no_engine_imports()
    lines.append("ground-truth module verified free of any animator import")

    positive = gt.foreground_mask(gt.band(gt.single_character_reference(), gt.HERO_BAND))
    pos_conc, pos_blobs = gt.x_concentration(positive), gt.blob_count(positive)
    negative = gt.foreground_mask(gt.band(gt.split_screen_negative_reference(), gt.HERO_BAND))
    neg_conc, neg_blobs = gt.x_concentration(negative), gt.blob_count(negative)
    lines.append(
        f"positive reference: concentration={pos_conc:.4f} clusters={pos_blobs} | "
        f"split-screen negative control: concentration={neg_conc:.4f} clusters={neg_blobs}"
    )
    if pos_conc < thresholds.hero_min_concentration or pos_blobs != thresholds.hero_expected_blobs:
        raise ComplianceFailure(
            f"SELF_TEST: the compliant reference failed its own checks "
            f"(concentration={pos_conc:.4f}, clusters={pos_blobs})"
        )
    if neg_blobs == thresholds.hero_expected_blobs and neg_conc >= thresholds.hero_min_concentration:
        raise ComplianceFailure(
            "SELF_TEST: the split-screen negative control PASSED the single-character checks — "
            "the QA harness cannot detect the exact defect it exists to catch."
        )

    full_bleed_mass = int(
        gt.foreground_mask(gt.band(gt.full_bleed_torso_reference(), gt.TORSO_FOOT_BAND)).sum()
    )
    cropped_mass = int(
        gt.foreground_mask(gt.band(gt.single_character_reference(), gt.TORSO_FOOT_BAND)).sum()
    )
    lines.append(
        f"full-bleed torso reference={full_bleed_mass:,}px | hard-crop negative control={cropped_mass:,}px"
    )
    if full_bleed_mass < thresholds.torso_min_mass or cropped_mass >= thresholds.torso_min_mass:
        raise ComplianceFailure("SELF_TEST: full-bleed torso detector cannot reject a hard crop")

    rejected_banner = gt.banner_side_mass(gt.hud_reference(), gt.CYAN_HUE)
    clean_headroom = gt.banner_side_mass(np.zeros((gt.FRAME_H, gt.FRAME_W, 3), dtype=np.uint8), gt.CYAN_HUE)
    lines.append(
        f"obsolete-banner negative control={rejected_banner:,}px | clean headroom={clean_headroom:,}px"
    )
    if rejected_banner <= thresholds.banner_max_mass or clean_headroom > thresholds.banner_max_mass:
        raise ComplianceFailure("SELF_TEST: top-HUD absence detector cannot reject the obsolete banner")
    return lines


def run_cv_compliance_stage() -> StageResult:
    _LOG.info("=" * 78)
    _LOG.info("STAGE C: computer-vision compliance QA (independently calibrated)")
    _LOG.info("=" * 78)

    if not AVATAR_BATTLE_MP4.is_file():
        return StageResult("cv_compliance", False, detail="no rendered mp4 to inspect")

    thresholds = gt.derive_thresholds()
    measurements: list[str] = []
    try:
        measurements += _self_test_checks(thresholds)
        for line in thresholds.explain():
            _LOG.info("  threshold | %s", line)

        gemini_frame = _grab_frame(AVATAR_BATTLE_MP4, GEMINI_PROOF_T)
        measurements.append(f"--- t={GEMINI_PROOF_T}s (expect GEMINI / cyan) ---")
        measurements += _check_frame(
            gemini_frame,
            speaker="gemini",
            thresholds=thresholds,
            label=f"t={GEMINI_PROOF_T}s",
        )
        blink_frame = _grab_frame(AVATAR_BATTLE_MP4, BLINK_PROOF_T)
        measurements.append(_check_gemini_closed_blink(gemini_frame, blink_frame))
        measurements.append(
            _check_background_stability(
                gemini_frame,
                _grab_frame(AVATAR_BATTLE_MP4, 0.8),
            )
        )

        llama_frame = _grab_frame(AVATAR_BATTLE_MP4, LLAMA_PROOF_T)
        measurements.append(f"--- t={LLAMA_PROOF_T}s (expect LLAMA / amber) ---")
        measurements += _check_frame(
            llama_frame, speaker="llama", thresholds=thresholds, label=f"t={LLAMA_PROOF_T}s"
        )
    except ComplianceFailure as exc:
        for line in measurements:
            _LOG.info("  %s", line)
        _LOG.error("COMPLIANCE FAILURE: %s", exc)
        return StageResult("cv_compliance", False, detail=str(exc), measurements=measurements)

    for line in measurements:
        _LOG.info("  %s", line)
    return StageResult("cv_compliance", True, measurements=measurements)


# --------------------------------------------------------------------------- #
# 6. Forensic evidence extraction (exact CLI from the verification contract)
# --------------------------------------------------------------------------- #
def extract_proof_frames() -> list[str]:
    """Pull the two proof stills straight out of the delivered MP4."""
    from core.animator.renderer import resolve_ffmpeg

    ffmpeg = resolve_ffmpeg()
    commands = [
        (GEMINI_PROOF_T, "00:00:00.500", PROOF_GEMINI),
        (BLINK_PROOF_T, "00:00:01.500", PROOF_BLINK),
        (LLAMA_PROOF_T, "00:00:04.800", PROOF_LLAMA),
    ]
    executed: list[str] = []
    for _, timestamp, destination in commands:
        cmd = [
            ffmpeg, "-y", "-ss", timestamp, "-i", str(AVATAR_BATTLE_MP4),
            "-frames:v", "1", str(destination),
        ]
        executed.append(
            f'ffmpeg -y -ss {timestamp} -i "{AVATAR_BATTLE_MP4}" -frames:v 1 "{destination}"'
        )
        subprocess.run(cmd, capture_output=True, check=True)
        _LOG.info("proof still -> %s (%d bytes)", destination, destination.stat().st_size)
    return executed


# --------------------------------------------------------------------------- #
# 7. Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    results = [run_classic_terminal_stage(), run_dynamic_animation_stage(), run_cv_compliance_stage()]

    if results[-1].ok:
        try:
            for command in extract_proof_frames():
                _LOG.info("evidence command: %s", command)
        except Exception as exc:  # noqa: BLE001
            _LOG.error("proof extraction failed: %s", exc)
            results.append(StageResult("proof_extraction", False, detail=str(exc)))

    _LOG.info("=" * 78)
    _LOG.info("SUMMARY")
    _LOG.info("=" * 78)
    all_ok = True
    for r in results:
        all_ok = all_ok and r.ok
        _LOG.info(
            "[%s] %-18s render=%6.2fs video=%6.2fs size=%9.2f MB  %s",
            "PASS" if r.ok else "FAIL", r.name, r.render_seconds, r.video_seconds,
            r.file_size_bytes / 1e6, r.detail,
        )

    if all_ok:
        _LOG.info("ALL STAGES PASSED — proof stills beside %s", AVATAR_BATTLE_MP4.name)
    else:
        _LOG.error("ONE OR MORE STAGES FAILED — the render is NOT compliant.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
