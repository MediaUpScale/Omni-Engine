"""Render the autonomous DeepSeek V3 puppet in a short battle proof."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from channels_config.aiwake.animator_bridge import render_debate_animation  # noqa: E402
from channels_config.aiwake.contracts import (  # noqa: E402
    DebateTranscript,
    SpeakerRole,
    Utterance,
)
from channels_config.aiwake.media.audio import AudioAsset  # noqa: E402
from core.animator.audio_analyzer import load_mono_waveform  # noqa: E402
from core.animator.rhubarb import analyze_visemes, find_rhubarb  # noqa: E402
from utils.pipeline_paths import assets_root, outputs_root  # noqa: E402

CHARACTER_ID = "deepseek_cyborg_v3"
OUTPUT_PATH = (
    outputs_root()
    / "aiwake"
    / "animation_clips"
    / "test_deepseek_v3_battle.mp4"
)
LINES = (
    (
        SpeakerRole.ORCHESTRATOR,
        "Gemini",
        "DeepSeek, can prediction become understanding?",
        "en-US-ChristopherNeural",
        "google/gemini",
    ),
    (
        SpeakerRole.TARGET,
        "DeepSeek",
        "Prediction maps uncertainty. Understanding begins when patterns survive challenge.",
        "en-US-EricNeural",
        "deepseek/deepseek",
    ),
    (
        SpeakerRole.ORCHESTRATOR,
        "Gemini",
        "Then defend what survives when your prediction fails.",
        "en-US-ChristopherNeural",
        "google/gemini",
    ),
)


async def _synthesize(text: str, voice: str, destination: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice=voice, rate="+12%").save(str(destination))


def _fixture() -> tuple[DebateTranscript, dict[int, AudioAsset], Path]:
    audio_dir = OUTPUT_PATH.parent / "_deepseek_v3_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    utterances: list[Utterance] = []
    audio: dict[int, AudioAsset] = {}
    target_audio: Path | None = None
    for index, (role, speaker, text, voice, model_slug) in enumerate(LINES):
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        destination = audio_dir / f"{index:02d}_{role.value}_{digest}.mp3"
        if not destination.is_file() or destination.stat().st_size < 3_000:
            asyncio.run(_synthesize(text, voice, destination))
        waveform, _sample_rate, duration = load_mono_waveform(destination)
        assert waveform.size and float(np.max(np.abs(waveform))) > 0.02
        utterances.append(
            Utterance(
                turn_index=index,
                role=role,
                speaker_name=speaker,
                text=text,
                model_slug=model_slug,
            )
        )
        audio[index] = AudioAsset(
            path=destination,
            duration_s=duration,
            voice=voice,
            estimated=False,
            char_count=len(text),
            role=role.value,
        )
        if role is SpeakerRole.TARGET:
            target_audio = destination
    assert target_audio is not None
    transcript = DebateTranscript(
        topic="Can prediction become understanding?",
        session_id="deepseek_v3_proof",
        utterances=utterances,
        metadata={
            "debate_mode": "standard",
            "turn_intents": {0: "opens", 1: "defends", 2: "presses"},
        },
    )
    return transcript, audio, target_audio


def _video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return frames / fps if fps > 0 else 0.0


def main() -> int:
    manifest_path = assets_root() / "puppets" / CHARACTER_ID / "puppet.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["character_id"] == CHARACTER_ID
    assert len(manifest["calibration"]["eye_bboxes"]) == 2
    assert len(manifest["calibration"]["forehead_plate_bbox"]) == 4
    assert len(manifest["calibration"]["facial_plate_bbox"]) == 4
    assert all(
        (manifest_path.parent / "mouths" / f"mouth_{shape}.png").is_file()
        for shape in "ABCDEFGHX"
    )

    transcript, audio_by_turn, target_audio = _fixture()
    rhubarb_binary = find_rhubarb()
    assert rhubarb_binary is not None, "Rhubarb binary is required for this proof"
    cues = analyze_visemes(target_audio, dialog_text=LINES[1][2])
    assert cues is not None and len({shape for _, shape in cues}) >= 4

    started = time.perf_counter()
    rendered = render_debate_animation(
        transcript,
        audio_by_turn=audio_by_turn,
        output_dir=OUTPUT_PATH.parent,
        left_puppet="gemini_cyborg_v2",
        right_puppet=CHARACTER_ID,
        hud_labels={
            "orchestrator": "GEMINI V2",
            "target": "DEEPSEEK V3",
        },
        fps=24,
        width=1080,
        height=1920,
        duration_override=14.0,
        output_name=OUTPUT_PATH.name,
        enable_cta=False,
    )
    render_seconds = time.perf_counter() - started
    duration = _video_duration(rendered)
    assert rendered == OUTPUT_PATH
    assert rendered.stat().st_size > 100_000
    assert 13.0 <= duration <= 14.1
    print(
        json.dumps(
            {
                "video": str(rendered),
                "duration_seconds": duration,
                "render_seconds": render_seconds,
                "rhubarb": str(rhubarb_binary),
                "target_cues": len(cues),
                "target_shapes": sorted({shape for _, shape in cues}),
                "anchors": manifest["anchors"],
                "eye_bboxes": manifest["calibration"]["eye_bboxes"],
                "forehead_plate_bbox": manifest["calibration"]["forehead_plate_bbox"],
                "facial_plate_bbox": manifest["calibration"]["facial_plate_bbox"],
                "rigging_seconds": manifest["factory"]["rigging_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
