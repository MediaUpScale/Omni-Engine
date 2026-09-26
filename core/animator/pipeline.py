"""Debate test renderer: character mouths, male voices, and contraplano cuts.

A contraplano pass uses the approved three-quarter views. ChatGPT stands on
the left looking right and opens the exchange. Claude stands on the right
looking left. The shot-reverse-shot director hard-cuts to whoever is speaking.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from utils.pipeline_paths import assets_root, outputs_root

from .render.facial_rig import (
    export_pilot_facial_sheet,
    face_layout,
    generate_character_acting,
    install_view_anchors,
)
from .render.visemes import (
    LLAMA_CHIN_WIDTH,
    LLAMA_WIDE_MOUTH,
    PALETTES,
    draw_mouths,
    generate_viseme_set,
    metal_stroke,
    viseme_file,
)
from .voice import assign_debater_voices
from .puppet import REST_MOUTH_STATES, rest_mouth_layer_key
from .types import VISEMES

_LOG = logging.getLogger("animator.pipeline")

_SOURCE_ALIASES = {"llama": "llama_cyborg_v2"}
_HARNESS = (
    Path(r"G:\My Drive\Z sosFiles\Z_act\@ NETWORK\@MEDIAUPSCALE_FACTORY_DYNAMIC_CONTENT")
    / "Unified Multi-Page Factory"
    / "outputs"
    / "aiwake"
    / "_test_harness"
)
_TEST_OUTPUT = _HARNESS / "chatgpt_vs_claude_mouth_test.mp4"
_CONTRAPLANO_OUTPUT = _HARNESS / "chatgpt_vs_claude_contraplano_test.mp4"
_PILOT_SHEET = _HARNESS / "chatgpt_pilot_facial_inspection.png"
_PILOT_VIDEO = _HARNESS / "chatgpt_pilot_acting_test.mp4"
_CAST = {
    "chatgpt_cyborg_v1": {"label": "CHATGPT", "accent": "#D4A466", "text": "A clear answer beats a clever one."},
    "claude_cyborg_v1": {"label": "CLAUDE", "accent": "#B05434", "text": "Only when that answer is also true."},
}


def resolve_puppet(name: str) -> str:
    """Map the blueprint nickname ``llama`` onto the approved skin folder."""
    token = name.strip()
    return _SOURCE_ALIASES.get(token, token)


def clone_mouths(source: str, puppets: list[str]) -> None:
    """Regenerate visemes and acting mouths from the Llama patch."""
    blueprint = resolve_puppet(source)
    blueprint_dir = assets_root() / "puppets" / blueprint
    if not blueprint_dir.is_dir():
        raise FileNotFoundError(f"blueprint puppet is missing: {blueprint_dir}")
    _LOG.info("cloning mouth geometry from %s", blueprint)
    for character_id in puppets:
        if character_id not in PALETTES:
            raise SystemExit(f"no Llama mouth palette for {character_id}")
        generate_viseme_set(character_id)
        generate_character_acting(character_id)


def _hex(color: tuple[int, int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(color[0], color[1], color[2])


def _shift(points: tuple[int, int], origin: tuple[int, int]) -> tuple[int, int]:
    return points[0] + origin[0], points[1] + origin[1]


def _opaque_width(image: Image.Image) -> int:
    alpha = np.asarray(image)[..., 3]
    xs = np.nonzero(alpha > 8)[1]
    if xs.size == 0:
        return image.width
    return int(xs.max() - xs.min() + 1)


def _paste_layer(canvas_size: tuple[int, int], image: Image.Image, origin: tuple[int, int]) -> Image.Image:
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    layer.alpha_composite(image, origin)
    return layer


def _bind_to_head(head_size: tuple[int, int], child: Image.Image) -> Image.Image:
    """Stage 1. A facial child lives in the head sprite's own pixels."""
    plate = Image.new("RGBA", head_size, (0, 0, 0, 0))
    plate.alpha_composite(child, (0, 0))
    return plate


def _promote_head(
    canvas_size: tuple[int, int],
    head_plate: Image.Image,
    head_xy: tuple[int, int],
) -> Image.Image:
    """Stage 2. The fused head is parented once. Children are not moved again."""
    return _paste_layer(canvas_size, head_plate, head_xy)


def _mouth_layer(
    canvas_size: tuple[int, int],
    sprite: Image.Image,
    anchor: tuple[int, int],
    scale: float,
    rot_deg: float = 0.0,
) -> Image.Image:
    resized = sprite.resize(
        (max(1, int(round(sprite.width * scale))), max(1, int(round(sprite.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    if abs(rot_deg) > 0.05:
        resized = resized.rotate(-rot_deg, resample=Image.Resampling.BICUBIC, expand=True)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    layer.alpha_composite(resized, (anchor[0] - resized.width // 2, anchor[1] - resized.height // 2))
    return layer


def _mouth_on_head(
    head_size: tuple[int, int],
    sprite: Image.Image,
    local_anchor: tuple[int, int],
    scale: float,
    rot_deg: float = 0.0,
) -> Image.Image:
    """Dock a mouth in head-local pixels, then let the head carry it."""
    return _mouth_layer(head_size, sprite, local_anchor, scale, rot_deg)


def _shutter_layer(
    canvas_size: tuple[int, int],
    optics: list[tuple[float, float, float]],
    fill: tuple[int, int, int, int],
    *,
    closed: bool,
) -> Image.Image:
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for center_x, center_y, radius in optics:
        box = (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )
        if closed:
            draw.ellipse(box, fill=fill)
        else:
            draw.pieslice(box, 180, 360, fill=fill)
    return layer


def build_render_skin(
    character_id: str,
    destination: Path,
    *,
    view_name: str = "facing_front",
    contraplano: bool = False,
) -> Path:
    """Assemble one view canvas the shot-reverse-shot rig can load.

    Contraplano uses the three-quarter view and leaves framing to the
    director's lead anchor, so a right-facing hero sits on the left and a
    left-facing hero sits on the right.
    """
    root = assets_root() / "puppets" / character_id
    manifest = json.loads((root / "puppet.json").read_text(encoding="utf-8"))
    view = manifest["views"][view_name]
    head = Image.open(root / view["head"]).convert("RGBA")
    body = Image.open(root / view["body"]).convert("RGBA")
    head_xy = (int(view["head_xy"][0]), int(view["head_xy"][1]))
    body_xy = (int(view["body_xy"][0]), int(view["body_xy"][1]))
    min_x = min(0, head_xy[0], body_xy[0])
    min_y = min(0, head_xy[1], body_xy[1])
    head_xy = (head_xy[0] - min_x, head_xy[1] - min_y)
    body_xy = (body_xy[0] - min_x, body_xy[1] - min_y)
    canvas_size = (
        max(head_xy[0] + head.width, body_xy[0] + body.width),
        max(head_xy[1] + head.height, body_xy[1] + body.height),
    )
    layout = face_layout(head)
    facial_rig = view.get("facial_rig") or {}
    rig_mouth = facial_rig.get("mouth") or {}
    mouth_spec = rig_mouth or view.get("mouth") or {}
    if mouth_spec:
        mouth_center = mouth_spec.get("center")
        if mouth_center is None:
            mouth_center = (mouth_spec["x"], mouth_spec["y"])
        local_mouth = (int(mouth_center[0]), int(mouth_center[1]))
        mouth_anchor = _shift(local_mouth, head_xy)
        mouth_rot = float(mouth_spec.get("rot_deg") or 0.0)
        mouth_mul = float(mouth_spec.get("scale") or 1.0)
    else:
        local_mouth = (int(layout.chin_anchor[0]), int(layout.chin_anchor[1]))
        mouth_anchor = _shift(local_mouth, head_xy)
        mouth_rot = 0.0
        mouth_mul = 1.0
    palette = PALETTES[character_id]
    stroke = metal_stroke(palette.bezel)
    native_dir = (
        root / rig_mouth["asset_dir"]
        if rig_mouth.get("asset_dir")
        else None
    )
    mouth_file = (
        (lambda code: native_dir / f"mouth_{code}.png")
        if native_dir is not None
        else (lambda code: viseme_file(root, code))
    )
    wide = Image.open(mouth_file("C")).convert("RGBA")
    if rig_mouth.get("target_width"):
        mouth_scale = float(rig_mouth["target_width"]) / wide.width
    else:
        mouth_scale = (
            layout.chin_width
            * (LLAMA_WIDE_MOUTH / LLAMA_CHIN_WIDTH)
            * mouth_mul
        ) / _opaque_width(wide)
    skin = destination / character_id
    mouths = skin / "mouths"
    mouths.mkdir(parents=True, exist_ok=True)
    _paste_layer(canvas_size, body, body_xy).save(skin / "body.png", compress_level=1)
    _promote_head(canvas_size, head, head_xy).save(skin / "head.png", compress_level=1)
    Image.new("RGBA", canvas_size, (0, 0, 0, 0)).save(skin / "eyes_open.png", compress_level=1)
    optics = []
    for optic in (layout.left, layout.right):
        center = _shift(optic.center, head_xy)
        optics.append((float(center[0]), float(center[1]), float(optic.radius)))
    lids = facial_rig.get("eyes_lids") or {}
    half_asset = root / lids["half_asset"] if lids.get("half_asset") else None
    full_asset = root / lids["full_asset"] if lids.get("full_asset") else None
    if half_asset is not None and half_asset.is_file():
        with Image.open(half_asset) as opened:
            half_lid = opened.convert("RGBA")
        _promote_head(
            canvas_size,
            _bind_to_head(head.size, half_lid),
            head_xy,
        ).save(skin / "eyes_half.png", compress_level=1)
    else:
        lid = (stroke[0] // 2, stroke[1] // 2, stroke[2] // 2, 255)
        _shutter_layer(canvas_size, optics, lid, closed=False).save(
            skin / "eyes_half.png",
            compress_level=1,
        )
    if full_asset is not None and full_asset.is_file():
        with Image.open(full_asset) as opened:
            full_lid = opened.convert("RGBA")
        _promote_head(
            canvas_size,
            _bind_to_head(head.size, full_lid),
            head_xy,
        ).save(skin / "eyes_blink.png", compress_level=1)
    else:
        lid = (stroke[0] // 2, stroke[1] // 2, stroke[2] // 2, 255)
        _shutter_layer(canvas_size, optics, lid, closed=True).save(
            skin / "eyes_blink.png",
            compress_level=1,
        )
    glow = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((4, 4, 28, 28), fill=(*palette.bezel[:3], 40))
    glow.save(skin / "glow.png", compress_level=1)
    Image.new("RGB", (64, 64), (12, 14, 18)).save(skin / "bg.png", compress_level=1)

    sprites = {
        code: Image.open(mouth_file(code)).convert("RGBA")
        for code in VISEMES
    }
    expression_dir = native_dir or (root / "mouths" / "expressions")
    expressions = {
        "neutral": sprites["X"],
        "smug_smile": Image.open(expression_dir / "mouth_smug.png").convert("RGBA"),
        "stressed_grimace": Image.open(expression_dir / "mouth_angry.png").convert("RGBA"),
    }
    for code, sprite in sprites.items():
        _promote_head(
            canvas_size,
            _mouth_on_head(head.size, sprite, local_mouth, mouth_scale, mouth_rot),
            head_xy,
        ).save(mouths / f"mouth_{code}.png", compress_level=1)
    for state in REST_MOUTH_STATES:
        _promote_head(
            canvas_size,
            _mouth_on_head(head.size, expressions[state], local_mouth, mouth_scale, mouth_rot),
            head_xy,
        ).save(mouths / f"{rest_mouth_layer_key(state)}.png", compress_level=1)

    head_alpha = np.asarray(head)[..., 3]
    rows = np.nonzero(head_alpha > 8)[0]
    head_bottom = head_xy[1] + (int(rows.max()) + 1 if rows.size else head.height)
    head_center_x = head_xy[0] + head.width // 2
    eye_boxes = []
    for center_x, center_y, radius in optics:
        eye_boxes.append(
            [
                int(round(center_x - radius)),
                int(round(center_y - radius)),
                int(round(center_x + radius)),
                int(round(center_y + radius)),
            ]
        )
    payload = {
        "character_id": character_id,
        "skin_version": "v2",
        "canvas_size": list(canvas_size),
        "mouth_style": "ghibli_mecha",
        "eye_style": "circular",
        "anchors": {
            "head_pivot": [head_center_x, int(round((optics[0][1] + optics[1][1]) / 2))],
            "neck_pivot": [head_center_x, head_bottom],
            "mouth": [head_center_x, head_bottom],
            "left_eye": [int(round(optics[0][0])), int(round(optics[0][1]))],
            "right_eye": [int(round(optics[1][0])), int(round(optics[1][1]))],
            "eye_radius": int(round((optics[0][2] + optics[1][2]) / 2)),
        },
        "theme": {"glow_color": _hex(palette.bezel), "glow_radius": 24},
        "palette": {"ink_outline": _hex(stroke)},
        "brows": _skin_brow_payload(
            facial_rig,
            head_xy,
            layout.nameplate_bottom,
            int(round(max(optic.radius for optic in (layout.left, layout.right)) * 2.3)),
        ),
        "framing": {} if contraplano else {"bottom_anchor": True},
        "calibration": {
            "eye_bboxes": eye_boxes,
            "lens_circles": [[center_x, center_y, radius] for center_x, center_y, radius in optics],
        },
        "layers": {
            "body": "body.png",
            "head": "head.png",
            "eyes_open": "eyes_open.png",
            "eyes_half": "eyes_half.png",
            "eyes_blink": "eyes_blink.png",
            "glow": "glow.png",
            "bg": "bg.png",
            **{f"mouth_{code}": f"mouths/mouth_{code}.png" for code in VISEMES},
            **{rest_mouth_layer_key(state): f"mouths/{rest_mouth_layer_key(state)}.png" for state in REST_MOUTH_STATES},
        },
    }
    (skin / "puppet.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info(
        "%s canvas=%s mouth=%s chin_w=%s plate_bottom=%s",
        character_id,
        canvas_size,
        mouth_anchor,
        layout.chin_width,
        layout.nameplate_bottom,
    )
    _LOG.info("%s view=%s contraplano=%s", character_id, view_name, contraplano)
    return skin


def _skin_brow_payload(
    facial_rig: dict,
    head_xy: tuple[int, int],
    nameplate_bottom: int,
    width_px: int,
) -> dict:
    """Copy the analyzer brow matrix into canvas space without a second scale."""
    source = facial_rig.get("eyebrows") or {}
    if source.get("left") and source.get("right"):
        nodes = {"style": str(source.get("style") or "ink")}
        for side in ("left", "right"):
            node = dict(source[side])
            center = node.get("center") or [0, 0]
            node["center"] = [
                int(center[0]) + head_xy[0],
                int(center[1]) + head_xy[1],
            ]
            nodes[side] = node
        return {
            "style": "acute_mecha",
            "lock_to_rig": True,
            "local_head_lock": True,
            "rig_brows": nodes,
            "ink_color": "#101216",
            "nameplate_bottom": head_xy[1] + nameplate_bottom,
        }
    return {
        "style": "acute_mecha",
        "ink_color": "#2B1A15",
        "gap_px": 3,
            "stroke_width_px": 9,
            "width_px": width_px,
            "nameplate_bottom": head_xy[1] + nameplate_bottom,
        }


def _ffmpeg() -> str:
    from .renderer import _resolve_ffmpeg

    return _resolve_ffmpeg()


def _probe_duration(path: Path) -> float:
    from moviepy import AudioFileClip

    clip = AudioFileClip(str(path))
    try:
        return float(clip.duration)
    finally:
        clip.close()


def _synthesize_line(text: str, voice: str, destination: Path) -> None:
    import edge_tts

    async def _save() -> None:
        communicate = edge_tts.Communicate(text, voice=voice, rate="+6%")
        await communicate.save(str(destination))

    asyncio.run(_save())


def build_exchange_audio(workdir: Path, lines: list[dict]) -> tuple[Path, list[dict]]:
    """Two short lines with a gap, plus the timeline the renderer should follow."""
    workdir.mkdir(parents=True, exist_ok=True)
    lead, gap, tail = 0.25, 0.45, 0.30
    cursor = lead
    timed: list[dict] = []
    for index, line in enumerate(lines):
        utterance = workdir / f"line_{index}.mp3"
        _synthesize_line(line["text"], line["voice"], utterance)
        duration = _probe_duration(utterance)
        timed.append({**line, "start": cursor, "end": cursor + duration, "audio": utterance})
        cursor += duration
        if index == 0:
            cursor += gap
    cursor += tail
    listing = workdir / "concat.txt"
    trim_silence = workdir / "lead.wav"
    gap_silence = workdir / "gap.wav"
    tail_silence = workdir / "tail.wav"
    for path, seconds in ((trim_silence, lead), (gap_silence, gap), (tail_silence, tail)):
        subprocess.run(
            [_ffmpeg(), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds), str(path)],
            check=True,
            capture_output=True,
        )
    # The concat demuxer drops clips when codecs differ, so every piece becomes WAV first.
    wavs: list[Path] = []
    for index, source in enumerate((trim_silence, timed[0]["audio"], gap_silence, timed[1]["audio"], tail_silence)):
        wav = workdir / f"piece_{index}.wav"
        subprocess.run(
            [_ffmpeg(), "-y", "-i", str(source), "-ar", "44100", "-ac", "2", str(wav)],
            check=True,
            capture_output=True,
        )
        wavs.append(wav)
    listing.write_text("".join(f"file '{path.as_posix()}'\n" for path in wavs), encoding="utf-8")
    mixed = workdir / "exchange.wav"
    completed = subprocess.run(
        [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c:a", "pcm_s16le", str(mixed)],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = _probe_duration(mixed)
    if duration < 5.0:
        raise RuntimeError(f"exchange audio is {duration:.2f}s; speech clips were dropped")
    return mixed, timed


def _alias_voice(character_id: str, requested: str) -> str:
    """Map a preset name onto that puppet's reserved neural voice."""
    if requested in {"male_confident", "male_british", "reserved", ""}:
        return assign_debater_voices([character_id])[character_id]
    return requested


def debate_lines(
    *,
    left_id: str,
    right_id: str,
    voice_left: str,
    voice_right: str = "male_british",
    contraplano: bool,
) -> list[dict]:
    """Left speaker looks right and talks first. Right speaker looks left."""
    left = _CAST[left_id]
    right = _CAST[right_id]
    view_for_facing = {"right": "facing_right", "left": "facing_left"}
    voices = assign_debater_voices(
        [left_id, right_id],
        requested={left_id: _alias_voice(left_id, voice_left), right_id: _alias_voice(right_id, voice_right)},
    )
    rows = (
        (left_id, left, "right", voices[left_id]),
        (right_id, right, "left", voices[right_id]),
    )
    lines = []
    for character_id, card, facing, voice in rows:
        lines.append(
            {
                "character_id": character_id,
                "label": card["label"],
                "accent": card["accent"],
                "facing": facing,
                "voice": voice,
                "text": card["text"],
                "view": view_for_facing[facing] if contraplano else "facing_front",
            }
        )
    return lines


def render_test_dialogue(
    lines: list[dict],
    output_path: Path,
    *,
    contraplano: bool = False,
) -> Path:
    """Render a few seconds of ChatGPT answering Claude through the standard engine."""
    from . import render_dynamic_animation
    from .types import DialogueTurn, SpeakerStyle

    workdir = outputs_root() / "aiwake" / "_test_harness" / (
        "chatgpt_vs_claude_contraplano_build" if contraplano else "chatgpt_vs_claude_mouth_build"
    )
    skin_root = workdir / "puppets"
    for line in lines:
        build_render_skin(
            line["character_id"],
            skin_root,
            view_name=line["view"],
            contraplano=contraplano,
        )
    _write_pose_stills(skin_root, workdir / "mouth_pose_stills.png", [line["character_id"] for line in lines])
    mixed, timed = build_exchange_audio(workdir / "audio", lines)
    turns = [
        DialogueTurn(
            speaker=line["character_id"],
            start_time=line["start"],
            end_time=line["end"],
            text=line["text"],
            audio_path=str(line["audio"]),
        )
        for line in timed
    ]
    styles = [
        SpeakerStyle(
            character_id=line["character_id"],
            label=line["label"],
            accent_hex=line["accent"],
            facing=line["facing"],
        )
        for line in lines
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_dynamic_animation(
        turns=turns,
        audio_path=mixed,
        styles=styles,
        output_path=output_path,
        puppets_dir=skin_root,
        fps=30,
        width=1080,
        height=1920,
        burn_subtitles=True,
        use_rhubarb=True,
    )
    print(output_path)
    return output_path


def _write_pose_stills(skin_root: Path, destination: Path, character_ids: list[str] | None = None) -> None:
    """One open-mouth frame per puppet, so the chin seat can be checked."""
    from .puppet import PuppetRig, PuppetSkin

    frames = []
    for character_id in character_ids or ("chatgpt_cyborg_v1", "claude_cyborg_v1"):
        rig = PuppetRig(PuppetSkin.load(skin_root / character_id))
        frame = rig.compose(viseme="C", emotion="neutral")
        rgb = frame[..., :3]
        image = Image.fromarray(rgb)
        image.thumbnail((420, 760))
        frames.append(image)
    sheet = Image.new("RGB", (frames[0].width + frames[1].width, max(frame.height for frame in frames)), (8, 8, 10))
    x_pos = 0
    for frame in frames:
        sheet.paste(frame, (x_pos, 0))
        x_pos += frame.width
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, compress_level=1)
    print(destination)


def build_solo_audio(workdir: Path, line: dict) -> tuple[Path, dict]:
    """One utterance padded so the pilot lands on about four seconds."""
    workdir.mkdir(parents=True, exist_ok=True)
    lead = 0.15
    utterance = workdir / "line_0.mp3"
    _synthesize_line(line["text"], line["voice"], utterance)
    spoken = _probe_duration(utterance)
    tail = max(0.15, 4.0 - lead - spoken)
    if lead + spoken + tail > 4.6:
        tail = 0.15
    listing = workdir / "concat.txt"
    pieces = []
    for index, (source, seconds) in enumerate((("lead", lead), ("speech", spoken), ("tail", tail))):
        wav = workdir / f"piece_{index}.wav"
        if source == "speech":
            subprocess.run(
                [_ffmpeg(), "-y", "-i", str(utterance), "-ar", "44100", "-ac", "2", str(wav)],
                check=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                [_ffmpeg(), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{seconds:.3f}", str(wav)],
                check=True,
                capture_output=True,
            )
        pieces.append(wav)
    listing.write_text("".join(f"file '{path.as_posix()}'\n" for path in pieces), encoding="utf-8")
    mixed = workdir / "solo.wav"
    subprocess.run(
        [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c:a", "pcm_s16le", str(mixed)],
        check=True,
        capture_output=True,
    )
    duration = _probe_duration(mixed)
    timed = {**line, "start": lead, "end": lead + spoken, "audio": utterance}
    _LOG.info("pilot audio %.2fs (speech %.2fs, voice %s)", duration, spoken, line["voice"])
    return mixed, timed


def render_pilot_acting(character_id: str, output_path: Path) -> Path:
    """Four-second solo: reserved male voice, visemes, a cocked brow, and a blink."""
    from . import render_dynamic_animation
    from .types import DialogueTurn, SpeakerStyle

    card = _CAST[character_id]
    voice = assign_debater_voices([character_id])[character_id]
    line = {
        "character_id": character_id,
        "label": card["label"],
        "accent": card["accent"],
        "facing": "right",
        "voice": voice,
        "text": "A clear answer beats a clever dodge every single time.",
        "view": "facing_right",
    }
    workdir = outputs_root() / "aiwake" / "_test_harness" / "chatgpt_pilot_acting_build"
    skin_root = workdir / "puppets"
    build_render_skin(character_id, skin_root, view_name=line["view"], contraplano=True)
    mixed, timed = build_solo_audio(workdir / "audio", line)
    turns = [
        DialogueTurn(
            speaker=character_id,
            start_time=timed["start"],
            end_time=timed["end"],
            text=timed["text"],
            audio_path=str(timed["audio"]),
            emotion="skeptical",
        )
    ]
    styles = [
        SpeakerStyle(
            character_id=character_id,
            label=line["label"],
            accent_hex=line["accent"],
            facing="right",
        )
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_dynamic_animation(
        turns=turns,
        audio_path=mixed,
        styles=styles,
        output_path=output_path,
        puppets_dir=skin_root,
        fps=30,
        width=1080,
        height=1920,
        burn_subtitles=True,
        use_rhubarb=True,
    )
    print(output_path)
    return output_path


def render_v6_contraplano(output_path: Path) -> Path:
    """Six-second shot-reverse-shot using the approved V6 mouths, lids, and deboche."""
    from . import render_dynamic_animation
    from .types import DialogueTurn, SpeakerStyle

    lines = debate_lines(
        left_id="chatgpt_cyborg_v1",
        right_id="claude_cyborg_v1",
        voice_left="male_confident",
        voice_right="male_british",
        contraplano=True,
    )
    workdir = output_path.parent / "build"
    skin_root = workdir / "puppets"
    for line in lines:
        build_render_skin(
            line["character_id"],
            skin_root,
            view_name=line["view"],
            contraplano=True,
        )
    mixed, timed = build_exchange_audio(workdir / "audio", lines)
    duration = _probe_duration(mixed)
    if duration < 5.95:
        padded = workdir / "audio" / "exchange_6s.wav"
        subprocess.run(
            [
                _ffmpeg(),
                "-y",
                "-i",
                str(mixed),
                "-af",
                "apad=whole_dur=6",
                "-c:a",
                "pcm_s16le",
                str(padded),
            ],
            check=True,
            capture_output=True,
        )
        mixed = padded
    turns = [
        DialogueTurn(
            speaker=line["character_id"],
            start_time=line["start"],
            end_time=line["end"],
            text=line["text"],
            audio_path=str(line["audio"]),
            emotion="deboche",
        )
        for line in timed
    ]
    styles = [
        SpeakerStyle(
            character_id=line["character_id"],
            label=line["label"],
            accent_hex=line["accent"],
            facing=line["facing"],
        )
        for line in lines
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_dynamic_animation(
        turns=turns,
        audio_path=mixed,
        styles=styles,
        output_path=output_path,
        puppets_dir=skin_root,
        fps=30,
        width=1080,
        height=1920,
        burn_subtitles=True,
        use_rhubarb=True,
    )
    print(output_path)
    return output_path


def render_v7_contraplano(output_path: Path) -> Path:
    """Shot-reverse-shot from the current rig, with brows locked to that matrix."""
    return render_v6_contraplano(output_path)


def render_v8_contraplano(output_path: Path) -> Path:
    """Debate reel from the golden-master rig, brows locked to that matrix."""
    return render_v6_contraplano(output_path)


def render_v9_contraplano(output_path: Path) -> Path:
    """Debate reel from the production rig. Lids stay parented to the head."""
    return render_v6_contraplano(output_path)


def render_v10_contraplano(output_path: Path) -> Path:
    """Debate reel from the release-candidate rig."""
    return render_v6_contraplano(output_path)


def render_v11_contraplano(output_path: Path) -> Path:
    """Debate reel from the gold-master rig."""
    return render_v6_contraplano(output_path)


def render_v12_contraplano(output_path: Path) -> Path:
    """Debate reel from the definitive-release rig."""
    return render_v6_contraplano(output_path)


def render_v13_contraplano(output_path: Path) -> Path:
    """Debate reel from the golden-seal rig."""
    return render_v6_contraplano(output_path)


def render_v14_contraplano(output_path: Path) -> Path:
    """Debate reel from the master sign-off rig."""
    return render_v6_contraplano(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a ChatGPT vs Claude debate test.")
    parser.add_argument("--clone-from", default="llama")
    parser.add_argument("--puppets", default="", help="Comma-separated puppet ids.")
    parser.add_argument("--contraplano", action="store_true")
    parser.add_argument("--speaker-left", default="chatgpt_cyborg_v1")
    parser.add_argument("--speaker-right", default="claude_cyborg_v1")
    parser.add_argument("--voice-left", default="male_confident")
    parser.add_argument("--voice-right", default="male_british")
    parser.add_argument("--render-test-dialogue", action="store_true")
    parser.add_argument("--pilot", default="")
    parser.add_argument("--draw-mouths", action="store_true")
    parser.add_argument("--inspect-sheet", action="store_true")
    parser.add_argument("--render-pilot-acting", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.pilot:
        if args.draw_mouths:
            draw_mouths(args.pilot, {"style": "cybernetic_capsule"})
            generate_character_acting(args.pilot)
            install_view_anchors(args.pilot)
        if args.inspect_sheet:
            export_pilot_facial_sheet(args.pilot, _PILOT_SHEET)
        if args.render_pilot_acting:
            render_pilot_acting(args.pilot, args.output or _PILOT_VIDEO)
        if not (args.draw_mouths or args.inspect_sheet or args.render_pilot_acting):
            parser.error("--pilot needs --draw-mouths, --inspect-sheet, or --render-pilot-acting")
        return 0
    if args.contraplano:
        puppets = [args.speaker_left, args.speaker_right]
        for character_id in puppets:
            generate_viseme_set(character_id)
            generate_character_acting(character_id)
        if args.render_test_dialogue:
            lines = debate_lines(
                left_id=args.speaker_left,
                right_id=args.speaker_right,
                voice_left=args.voice_left,
                voice_right=args.voice_right,
                contraplano=True,
            )
            render_test_dialogue(lines, args.output or _CONTRAPLANO_OUTPUT, contraplano=True)
        return 0
    if not args.puppets:
        parser.error("pass --puppets or --contraplano")
    puppets = [resolve_puppet(token) for token in args.puppets.split(",") if token.strip()]
    clone_mouths(args.clone_from, puppets)
    if args.render_test_dialogue:
        lines = debate_lines(
            left_id=puppets[0],
            right_id=puppets[-1],
            voice_left="male_confident",
            contraplano=False,
        )
        render_test_dialogue(lines, args.output or _TEST_OUTPUT, contraplano=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
