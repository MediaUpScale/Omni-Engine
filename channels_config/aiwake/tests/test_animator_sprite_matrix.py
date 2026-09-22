from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from channels_config.aiwake.animator_bridge import resolve_character_map, resolve_dialectic_emotion
from core.animator.audio_analyzer import AudioAnalyzer, emotion_lookup
from core.animator.compositor import (
    ShotReverseShotCompositor,
    _HeroCamera,
    emphasis_head_target,
)
from core.animator.puppet import (
    BROW_STATES,
    PuppetRig,
    PuppetSkin,
    emotion_brow_angles,
    emotion_brow_state,
)
from core.animator.renderer import AnimationRenderer
from core.animator.subtitles import build_ass
from core.animator.types import VISEMES, DialogueTurn, SpeakerStyle


def _layer(path: Path, size: tuple[int, int], box: tuple[int, int, int, int]) -> None:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(box, fill=(70, 80, 95, 255))
    image.save(path)


def test_manifestless_high_resolution_external_sprite_matrix_is_preserved(tmp_path: Path) -> None:
    puppet_dir = tmp_path / "external_cyborg"
    puppet_dir.mkdir()
    size = (960, 1520)
    _layer(puppet_dir / "body.png", size, (160, 650, 800, 1480))
    _layer(puppet_dir / "head.png", size, (280, 180, 680, 760))
    _layer(puppet_dir / "eyes_open.png", size, (360, 420, 600, 470))
    original_body = (puppet_dir / "body.png").read_bytes()

    skin = PuppetSkin.load_or_create(puppet_dir)
    manifest_path = puppet_dir / "puppet.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("calibration", {})["eye_bboxes"] = [
        [350, 410, 470, 520],
        [500, 410, 620, 520],
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    skin = PuppetSkin.load(puppet_dir)
    rig = PuppetRig(skin)
    camera = _HeroCamera(rig, None)
    native_camera = _HeroCamera(rig, None, target_width=size[0], target_height=size[1])

    assert rig.canvas_size == size
    assert camera.pixel_aspect_error < 0.001
    assert native_camera.crop == (0, 0, size[0], size[1])
    assert (native_camera.out_w, native_camera.out_h) == size
    assert camera._update_head_angle(  # noqa: SLF001 - verifies stateful easing contract
        t=0.1,
        rms=0.5,
        emphasis_threshold=0.75,
        is_speaking=True,
    ) == 0.0
    spike_angle = camera._update_head_angle(  # noqa: SLF001
        t=0.5,
        rms=1.0,
        emphasis_threshold=0.75,
        is_speaking=True,
    )
    assert 0.0 < abs(spike_angle) <= 1.2
    eased_angle = camera._update_head_angle(  # noqa: SLF001
        t=0.6,
        rms=0.4,
        emphasis_threshold=0.75,
        is_speaking=False,
    )
    assert 0.0 < abs(eased_angle) < abs(spike_angle)
    assert camera._update_emotion("inquisitor") == ("neutral", 1 / 3)  # noqa: SLF001
    assert camera._update_emotion("inquisitor") == ("neutral", 2 / 3)  # noqa: SLF001
    assert camera._update_emotion("inquisitor") == ("neutral", 1.0)  # noqa: SLF001
    assert (puppet_dir / "eyes_half.png").is_file()
    assert (puppet_dir / "eyes_blink.png").is_file()
    assert (puppet_dir / "bg.png").is_file()
    assert all((puppet_dir / f"mouth_{viseme}.png").is_file() for viseme in VISEMES)
    assert (puppet_dir / "body.png").read_bytes() == original_body

    frame = rig.compose(viseme="D", eye_state=0, y_offset=3.0)
    assert frame.shape == (size[1], size[0], 4)

    neutral = rig.compose(viseme="D", eye_state=0, head_angle=0.0)
    inquisitor = rig.compose(viseme="D", eye_state=0, emotion="inquisitor")
    tilted = rig.compose(viseme="D", eye_state=0, head_angle=1.8)
    assert not np.array_equal(neutral[350:540], inquisitor[350:540])
    assert not np.array_equal(neutral[100:850], tilted[100:850])
    assert np.array_equal(neutral[900:1400], tilted[900:1400])

    brow_crop, brow_bbox = rig._brow_overlay("neutral")  # noqa: SLF001
    brow_alpha = np.zeros((size[1], size[0]), dtype=np.uint8)
    bx0, by0, bx1, by1 = brow_bbox
    brow_alpha[by0:by1, bx0:bx1] = brow_crop[..., 3]
    assert np.count_nonzero(brow_alpha[410:440]) > 0
    outer_thickness = np.count_nonzero(brow_alpha[:, 364:370], axis=0).max()
    inner_thickness = np.count_nonzero(brow_alpha[:, 450:456], axis=0).max()
    assert inner_thickness > outer_thickness
    brow_sprites = [rig._brow_overlay(state)[0].tobytes() for state in BROW_STATES]  # noqa: SLF001
    assert len(set(brow_sprites)) == 5


def test_speaking_head_kinetics_are_emphasis_gated_and_neck_safe() -> None:
    threshold = 0.75
    assert emphasis_head_target(0.74, threshold, direction=1.0) == 0.0
    assert emphasis_head_target(0.75, threshold, direction=-1.0) == 0.0
    assert 0.0 < emphasis_head_target(0.80, threshold, direction=1.0) <= 1.2
    assert -1.2 <= emphasis_head_target(1.0, threshold, direction=-1.0) < 0.0


def test_shared_panorama_has_stable_eighty_percent_exposure() -> None:
    compositor = object.__new__(ShotReverseShotCompositor)
    compositor.width = 4
    compositor.height = 2
    panorama = Image.new("RGB", (8, 2), (120, 80, 60))
    style = SpeakerStyle("speaker", "Speaker", facing="right")

    prepared = compositor._prepare_camera_background(None, style, panorama)  # type: ignore[arg-type]  # noqa: SLF001

    expected = (np.asarray(panorama)[:, :4].astype(np.float32) * 0.80).astype(np.uint8)
    assert np.array_equal(prepared, expected)


def test_blink_schedule_is_deterministic_five_frame_cel_cycle() -> None:
    fps = 30
    states = AudioAnalyzer(fps=fps)._blink_schedule(fps * 20, seed=17)
    closed_frames = [index for index, state in enumerate(states) if state == 2]
    assert closed_frames == [45, 135, 225, 315, 405, 495, 585]
    for center in closed_frames:
        assert states[center - 2 : center + 3] == [0, 1, 2, 1, 0]


def test_dialectic_emotions_are_deterministic_and_frame_aligned() -> None:
    assert resolve_dialectic_emotion("opens") == "neutral"
    assert resolve_dialectic_emotion("answers") == "neutral"
    assert resolve_dialectic_emotion("presses: premise") == "inquisitor"
    assert resolve_dialectic_emotion("probes") == "skeptical"
    assert resolve_dialectic_emotion("holds") == "resolute"
    assert resolve_dialectic_emotion("defends") == "resolute"
    assert resolve_dialectic_emotion("concedes") == "conceded"
    assert resolve_dialectic_emotion("") == "neutral"
    assert tuple(emotion_brow_state(state) for state in BROW_STATES) == BROW_STATES
    assert emotion_brow_angles("neutral") == (1.0, -1.0)
    assert emotion_brow_angles("inquisitor") == (6.0, -6.0)
    assert emotion_brow_angles("resolute") == (3.5, -3.5)
    assert emotion_brow_angles("inquisitor", 1.5) == (7.5, -7.5)
    assert emotion_brow_angles("resolute", 1.5) == (5.0, -5.0)
    assert emotion_brow_angles("inquisitor", 1.5) != emotion_brow_angles("resolute", 1.5)

    track = emotion_lookup(
        [
            DialogueTurn("gemini", 0.0, 0.1, emotion="inquisitor"),
            DialogueTurn("llama", 0.1, 0.2, emotion="resolute"),
        ],
        fps=30,
        n_frames=6,
    )
    assert track == ["inquisitor"] * 3 + ["resolute"] * 3


def test_ffmpeg_contract_is_square_pixel_bounded_bitrate(tmp_path: Path) -> None:
    renderer = AnimationRenderer(width=1080, height=1920, fps=30)
    command = renderer._build_cmd(  # noqa: SLF001 - command contract regression
        audio_path=tmp_path / "audio.wav",
        output_path=tmp_path / "out.mp4",
        use_gpu=False,
        audio_codec="aac",
        subtitles_path=tmp_path / "captions.ass",
    )
    vf = command[command.index("-vf") + 1]
    assert vf.startswith("setsar=1:1,subtitles=")
    for flag, value in (
        ("-c:v", "libx264"),
        ("-preset", "ultrafast"),
        ("-b:v", "2600k"),
        ("-maxrate", "3200k"),
        ("-bufsize", "6000k"),
        ("-c:a", "aac"),
        ("-b:a", "192k"),
        ("-pix_fmt", "yuv420p"),
    ):
        index = max(i for i, item in enumerate(command) if item == flag)
        assert command[index + 1] == value


def test_karaoke_subtitles_use_outline_without_opaque_box(tmp_path: Path) -> None:
    destination = build_ass(
        [
            DialogueTurn(
                "speaker",
                0.0,
                1.0,
                "indistinguishability demands careful evidence now",
            )
        ],
        {"speaker": SpeakerStyle("speaker", "Speaker", "#00F0FF")},
        destination=tmp_path / "captions.ass",
    )
    content = destination.read_text(encoding="utf-8")
    style = next(line for line in content.splitlines() if line.startswith("Style: Karaoke"))

    assert "WrapStyle: 2" in content
    assert ",1,4.5,2,5," in style
    assert ",3,10,0,5," not in style
    assert ",90,90,320,1" in style
    assert "&H00000000" in style
    assert "&H00FFF000" in content
    assert "\\pos(540,1440)" in content
    assert "\\fscx112\\fscy112" in content

    dialogues = [line.rsplit(",,", 1)[1] for line in content.splitlines() if line.startswith("Dialogue:")]
    for dialogue in dialogues:
        plain = re.sub(r"\{[^}]*\}", "", dialogue)
        assert 2 <= len(plain.replace(r"\N", " ").split()) <= 4
        assert len(plain.split(r"\N")) <= 2
        assert all(len(line) <= 24 for line in plain.split(r"\N"))


def test_versioned_skin_registry_defaults_to_v2_and_supports_overrides() -> None:
    assert resolve_character_map() == {
        "orchestrator": "gemini_cyborg_v2",
        "target": "llama_cyborg_v2",
    }
    assert resolve_character_map(skin="v1") == {
        "orchestrator": "gemini_robot_v1",
        "target": "llama_robot_v1",
    }
    assert resolve_character_map(skin="v2", left_puppet="custom_left") == {
        "orchestrator": "custom_left",
        "target": "llama_cyborg_v2",
    }
