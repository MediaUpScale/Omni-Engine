"""V3 autonomous puppet factory regression and acceptance harness."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.animator.asset_generator import DEFAULT_PUPPETS_DIR  # noqa: E402
from core.animator.factory import auto_rig_character  # noqa: E402
from core.animator.factory.create_environment import create_environment  # noqa: E402
from core.animator.factory.create_puppet import create_puppet  # noqa: E402
from core.animator.factory.layer_slicer import slice_character_layers  # noqa: E402
from core.animator.factory.puppet_matrix import PUPPET_MATRIX  # noqa: E402
from core.animator.factory.rigger import (  # noqa: E402
    chroma_key_cutout,
    solidify_character_alpha,
)
from core.animator.compositor import _HeroCamera  # noqa: E402
from core.animator.puppet import PuppetSkin  # noqa: E402
from core.animator.puppet import PuppetRig  # noqa: E402
from core.animator.types import SpeakerStyle  # noqa: E402
from core.animator.types import VISEMES  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_artist_override() -> list[str]:
    reports = []
    for character_id in ("gemini_cyborg_v2", "llama_cyborg_v2"):
        skin_dir = DEFAULT_PUPPETS_DIR / character_id
        manifest_path = skin_dir / "puppet.json"
        assert manifest_path.is_file(), f"missing V2 artist manifest: {manifest_path}"
        before_bytes = manifest_path.read_bytes()
        before_digest = _digest(manifest_path)
        before_mtime = manifest_path.stat().st_mtime_ns

        loaded = PuppetSkin.load_or_create(skin_dir)

        assert loaded.character_id == character_id
        assert manifest_path.read_bytes() == before_bytes
        assert _digest(manifest_path) == before_digest
        assert manifest_path.stat().st_mtime_ns == before_mtime
        reports.append(f"{character_id}: artist manifest unchanged ({before_digest[:12]})")
    return reports


def _raw_robot(path: Path) -> None:
    image = Image.new("RGBA", (640, 800), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    casing = (82, 118, 132, 255)
    ink = (12, 18, 24, 255)
    cavity = (3, 7, 10, 255)
    teeth = (238, 244, 240, 255)
    draw.ellipse((105, 55, 535, 590), fill=casing, outline=ink, width=18)
    draw.rounded_rectangle((235, 550, 405, 790), radius=35, fill=casing, outline=ink, width=16)
    for center_x in (235, 405):
        draw.ellipse(
            (center_x - 48, 245 - 48, center_x + 48, 245 + 48),
            fill=(20, 238, 255, 255),
            outline=ink,
            width=12,
        )
        draw.ellipse(
            (center_x - 17, 245 - 17, center_x + 17, 245 + 17),
            fill=teeth,
        )
    draw.rounded_rectangle((220, 405, 420, 475), radius=28, fill=cavity, outline=ink, width=10)
    draw.rectangle((252, 412, 388, 430), fill=teeth)
    image.save(path)


def _run_auto_rig(root: Path) -> tuple[float, dict]:
    skin_dir = root / "factory_test_robot"
    skin_dir.mkdir()
    _raw_robot(skin_dir / "character.png")
    started = time.perf_counter()
    manifest_path = auto_rig_character(skin_dir, "factory_test_robot")
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, f"auto-rigger exceeded 3.0 seconds: {elapsed:.3f}s"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["skin_version"] == "v3-auto-rig"
    assert manifest["character_id"] == "factory_test_robot"
    assert manifest["canvas_size"] == [640, 800]
    assert set(("left_eye", "right_eye", "mouth", "neck_pivot")) <= set(manifest["anchors"])
    assert all(len(manifest["anchors"][key]) == 2 for key in ("left_eye", "right_eye", "mouth", "neck_pivot"))
    assert manifest["anchors"]["left_eye"][0] < manifest["anchors"]["right_eye"][0]

    calibration = manifest["calibration"]
    assert calibration["detector"] in {"mediapipe_face_mesh", "opencv_robotic_fallback"}
    assert calibration["confidence"] >= 0.5
    assert len(calibration["eye_bboxes"]) == 2
    assert calibration["landmarks_uv"]["jawline"]
    assert all(
        (skin_dir / "mouths" / f"mouth_{viseme}.png").is_file()
        for viseme in VISEMES
    )

    palette = manifest["palette"]
    assert set(("ink_outline", "casing_color", "cavity_interior", "teeth_color")) <= set(palette)
    assert palette["ink_outline"] != palette["casing_color"]
    assert palette["cavity_interior"] != palette["casing_color"]
    assert PuppetSkin.load(skin_dir).available_visemes() == list(VISEMES)

    # Artist priority also applies to a V3 directory after initial ingestion.
    before = manifest_path.read_bytes()
    assert auto_rig_character(skin_dir, "different_id") == manifest_path
    assert manifest_path.read_bytes() == before
    return elapsed, manifest


def test_v2_artist_skins_use_override_path_unchanged() -> None:
    _assert_artist_override()


def test_legacy_v2_skins_use_immutable_54b1b5d_camera() -> None:
    for character_id, facing in (
        ("gemini_cyborg_v2", "right"),
        ("llama_cyborg_v2", "left"),
    ):
        rig = PuppetRig(PuppetSkin.load(DEFAULT_PUPPETS_DIR / character_id))
        camera = _HeroCamera(
            rig,
            SpeakerStyle(
                character_id=character_id,
                label=character_id,
                facing=facing,
            ),
            target_width=1080,
            target_height=1920,
        )
        expected = min(
            1080 / rig.canvas_size[0],
            1920 / rig.canvas_size[1],
        )
        expected_w = int(round(rig.canvas_size[0] * expected))
        expected_h = int(round(rig.canvas_size[1] * expected))
        assert camera.scale == camera.scale_x
        assert (camera.out_w, camera.out_h) == (expected_w, expected_h)
        assert camera.scale_x == expected_w / rig.canvas_size[0]
        assert camera.scale_y == expected_h / rig.canvas_size[1]
        assert camera.pixel_aspect_error < 0.001
        assert camera.offset_x == (1080 - camera.out_w) // 2
        assert camera.offset_y == 1920 - camera.out_h


def test_auto_rig_character_generates_compliant_puppet(tmp_path: Path) -> None:
    _run_auto_rig(tmp_path)


def test_local_image_prompt_to_puppet_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    create_puppet_module = importlib.import_module(
        "core.animator.factory.create_puppet"
    )
    monkeypatch.setattr(
        create_puppet_module,
        "birefnet_cutout",
        lambda _image: (_ for _ in ()).throw(RuntimeError("offline fixture")),
    )
    transparent = tmp_path / "transparent.png"
    opaque = tmp_path / "opaque.png"
    _raw_robot(transparent)
    with Image.open(transparent) as robot:
        background = Image.new("RGBA", robot.size, (0, 0, 0, 255))
        background.alpha_composite(robot.convert("RGBA"))
        background.convert("RGB").save(opaque)

    result = create_puppet(
        character_id="third_character",
        prompt="local autonomous fixture",
        facing="left",
        image_path=opaque,
        puppets_dir=tmp_path / "puppets",
    )

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    character_path = Path(result["character_path"])
    with Image.open(character_path) as character:
        alpha = character.convert("RGBA").getchannel("A")
        assert alpha.getextrema() == (0, 255)
        assert alpha.getbbox() != (0, 0, *character.size)
    assert manifest["facing"] == "left"
    assert manifest["factory"]["acquisition"] == "local"
    assert (
        manifest["factory"]["background_removal"]
        == "opencv_continuous_chroma"
    )
    assert result["rigging_seconds"] < 3.0
    assert all(
        (manifest_path.parent / "mouths" / f"mouth_{viseme}.png").is_file()
        for viseme in VISEMES
    )
    with Image.open(manifest_path.parent / "eyes_half.png") as half_eye:
        half_alpha = np.asarray(half_eye.convert("RGBA"))[..., 3]
    with Image.open(manifest_path.parent / "eyes_blink.png") as closed_eye:
        closed_alpha = np.asarray(closed_eye.convert("RGBA"))[..., 3]
    assert np.count_nonzero(half_alpha > 250) > 500
    assert np.count_nonzero(closed_alpha > 250) > np.count_nonzero(
        half_alpha > 250
    )
    assert manifest["calibration"]["neck_overlap_px"] == 45
    assert manifest["layers"]["collar"] == "collar.png"
    assert manifest["brows"]["angles"]["inquisitor"] == -5.5
    assert manifest["brows"]["angles"]["defeated"] == 12.0
    assert manifest["brows"]["stroke_width_px"] == 5.5

    rig = PuppetRig(PuppetSkin.load(manifest_path.parent))
    for state in ("neutral", "inquisitor", "troubled"):
        brow_crop, _brow_bbox = rig.brow_overlay(state)
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            np.where(brow_crop[..., 3] > 32, 255, 0).astype(np.uint8),
            connectivity=8,
        )
        areas = sorted(
            (int(area) for area in stats[1:, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        assert component_count >= 3
        assert abs(areas[0] - areas[1]) <= max(4, int(areas[0] * 0.05))
    camera = _HeroCamera(
        rig,
        SpeakerStyle(
            character_id="third_character",
            label="THIRD",
            facing="left",
        ),
        target_width=1080,
        target_height=1920,
    )
    assert camera.scale_x == camera.scale_y == camera.scale
    assert camera.out_w == int(rig.canvas_size[0] * camera.scale)
    assert camera.out_h == int(rig.canvas_size[1] * camera.scale)
    assert camera.static_body.shape[:2] == (
        int(round(rig.canvas_size[1] * camera.body_scale)),
        int(round(rig.canvas_size[0] * camera.body_scale)),
    )
    opaque_rows = np.nonzero(rig.body_rgba[..., 3] > 8)[0]
    assert camera.body_offset_y + int(
        round((int(opaque_rows.max()) + 1) * camera.body_scale)
    ) == PUPPET_MATRIX["body_anchor_y"]
    assert camera.body_scale == camera.scale
    assert camera.body_offset_x == camera.offset_x
    assert camera.eye_line_y == PUPPET_MATRIX["target_eye_y"]
    assert camera.body_offset_y != camera.offset_y
    assert not hasattr(camera, "_grounded_body_plane")


def test_solid_alpha_and_neck_overlap_layers() -> None:
    rgba = Image.new("RGBA", (180, 240), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rgba, "RGBA")
    draw.ellipse((30, 15, 150, 150), fill=(60, 90, 110, 255))
    draw.rectangle((15, 125, 165, 239), fill=(40, 70, 95, 255))
    draw.ellipse((68, 155, 112, 205), fill=(0, 0, 0, 0))
    repaired, filled = solidify_character_alpha(rgba)
    assert filled > 1_000
    assert repaired.getchannel("A").getpixel((90, 180)) == 255
    assert repaired.getchannel("A").getpixel((28, 80)) == 0

    sliced = slice_character_layers(
        repaired,
        neck_pivot=(90, 145),
        overlap_px=35,
    )
    assert sliced.head.getchannel("A").getbbox()[3] <= 146
    assert sliced.body.getchannel("A").getbbox()[1] <= 110
    assert sliced.collar.getchannel("A").getbbox() is not None


def test_magic_wand_chroma_key_preserves_dark_ink() -> None:
    source = Image.new("RGBA", (80, 80), (0, 255, 0, 255))
    draw = ImageDraw.Draw(source, "RGBA")
    draw.ellipse(
        (15, 10, 65, 70),
        fill=(55, 83, 102, 255),
        outline=(16, 20, 28, 255),
        width=3,
    )
    draw.rectangle((5, 5, 10, 10), fill=(55, 83, 102, 255))
    cutout = np.asarray(chroma_key_cutout(source, tolerance=22.0))
    assert cutout[0, 0, 3] == 0
    assert cutout[40, 40, 3] == 255
    assert cutout[10, 40, 3] > 160  # anti-aliased dark authored outline
    edge_alpha = np.unique(cutout[..., 3])
    assert np.count_nonzero((edge_alpha > 0) & (edge_alpha < 255)) >= 4


def test_environment_formatter_outputs_shared_panorama(tmp_path: Path) -> None:
    source = tmp_path / "room.png"
    Image.new("RGB", (640, 360), (90, 90, 90)).save(source)
    result = create_environment(
        theme_id="naval_lab",
        prompt="",
        image_path=source,
        puppets_dir=tmp_path / "puppets",
    )
    output = Path(result["path"])
    with Image.open(output) as panorama:
        assert panorama.size == (3840, 2160)
        left = np.asarray(panorama)[1080, 100]
        right = np.asarray(panorama)[1080, -100]
    assert int(left[2]) > int(left[0])
    assert int(right[0]) > int(right[2])


def main() -> None:
    for report in _assert_artist_override():
        print(f"PASS {report}")
    with tempfile.TemporaryDirectory(prefix="omni_factory_") as temporary:
        elapsed, manifest = _run_auto_rig(Path(temporary))
    calibration = manifest["calibration"]
    print(f"PASS auto-rigger benchmark: {elapsed:.3f}s (<3.000s)")
    print(
        "Landmarks:",
        json.dumps(
            {
                "backend": calibration["detector"],
                "confidence": calibration["confidence"],
                **calibration["landmarks_uv"],
            },
            separators=(",", ":"),
        ),
    )
    print("Palette:", json.dumps(manifest["palette"], separators=(",", ":")))


if __name__ == "__main__":
    main()
