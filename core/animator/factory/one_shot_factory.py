"""One-shot puppet factory. BiRefNet alpha is saved exactly as the model returns it.

Usage:
    python -m core.animator.factory.one_shot_factory --character-id deepseek_cyborg_v3 --facing left
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.pipeline_paths import assets_root, outputs_root

from .create_puppet import DEEPSEEK_MASTER_PROMPT, generate_character_image
from .layer_slicer import slice_jaw_contour
from .rigger import auto_rig_character, birefnet_cutout


DEEPSEEK_PORTRAIT_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha character design, clean flat watercolor shading. "
    "Close medium bust portrait of a charming, dignified retro-futuristic tin-plate robot, angled 25 degrees facing toward the left. "
    "HEAD: Smooth rounded naval-blue dome helmet with brass rivets and circular ear dials. "
    "Forehead brass nameplate engraved 'DEEPSEEK'. Calm, expressive glowing emerald green concentric ring lenses (no human features). "
    "Smooth curved bronze chin shield plate. "
    "BODY: Sturdy rounded naval-blue chest with brass collar rim, broad mechanical shoulders with upper arms descending "
    "fully down and bleeding off the bottom of the frame. Continuous solid torso anchored at the bottom edge. "
    "Crisp, natural dark anime ink contours. Rendered against a solid flat muted light slate-blue background (#9FBACD). --ar 9:16"
)


class OneShotPuppetFactory:
    """Generate, matte, slice, and rig one character from a single 9:16 bust."""

    def create_character(
        self,
        character_id: str,
        prompt_template: str,
        facing: str = "left",
        bg_contrast: str = "black",
    ) -> Path:
        """Return the production skin directory."""
        skin_dir = assets_root() / "puppets" / character_id
        skin_dir.mkdir(parents=True, exist_ok=True)
        source = skin_dir / "source_oneshot.png"
        prompt = _contrast_prompt(prompt_template or DEEPSEEK_MASTER_PROMPT, bg_contrast)
        generate_character_image(prompt, source, aspect_ratio="9:16")
        with Image.open(source) as opened:
            master = birefnet_cutout(opened.convert("RGB"))
        transparent = skin_dir / "deepseek_master_transparent.png"
        master.save(transparent, format="PNG", compress_level=1)
        master.save(skin_dir / "character.png", format="PNG", compress_level=1)
        head, body = slice_jaw_contour(master, overlap_px=25)
        head.save(skin_dir / "head.png", format="PNG", compress_level=1)
        body.save(skin_dir / "body.png", format="PNG", compress_level=1)
        manifest = skin_dir / "puppet.json"
        if manifest.is_file():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("skin_version") != "v3-auto-rig":
                raise RuntimeError(f"refusing to replace artist manifest: {manifest}")
            manifest.unlink()
        auto_rig_character(
            skin_dir,
            character_id,
            facing=facing,
            component_mode=True,
        )
        self._write_preview(character_id, head, body, bg_contrast=bg_contrast)
        return skin_dir

    def _write_preview(
        self,
        character_id: str,
        head: Image.Image,
        body: Image.Image,
        bg_contrast: str = "black",
    ) -> Path:
        from core.animator.asset_generator import (  # noqa: PLC0415
            DEFAULT_PUPPETS_DIR,
            ensure_shared_panorama,
        )

        head_alpha = np.asarray(head.convert("RGBA"))[..., 3]
        body_alpha = np.asarray(body.convert("RGBA"))[..., 3]
        ys, xs = np.nonzero((head_alpha > 8) | (body_alpha > 8))
        if not ys.size:
            raise RuntimeError("one-shot bust has no visible pixels")
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        cropped_body = body.crop(box)
        cropped_head = head.crop(box)
        # The hem is pushed past the frame so an arched cutoff cannot sit inside it.
        scale = (1920 * 1.18) / max(1, cropped_body.height)
        placed_body = cropped_body.resize(
            (
                max(1, int(round(cropped_body.width * scale))),
                max(1, int(round(cropped_body.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        placed_head = cropped_head.resize(
            (
                max(1, int(round(cropped_head.width * scale))),
                max(1, int(round(cropped_head.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        frame = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        origin_x = (1080 - placed_body.width) // 2
        origin_y = 1920 - placed_body.height + int(1920 * 0.20)
        frame.alpha_composite(placed_body, (origin_x, origin_y))
        frame.alpha_composite(placed_head, (origin_x, origin_y))
        filename = (
            "deepseek_single_9x16_approved.png"
            if bg_contrast == "slate_blue"
            else f"{character_id.split('_cyborg')[0]}_v3_oneshot_master.png"
        )
        destination = outputs_root() / "aiwake" / "_test_harness" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.convert("RGB").save(destination, format="PNG", compress_level=1)
        print(f"oneshot: {destination}")
        return destination


def _contrast_prompt(prompt: str, bg_contrast: str) -> str:
    """White stage keeps dark ink separable from the backdrop."""
    proportions = (
        "Charming, prominent, slightly oversized mecha head with vintage anime proportions, "
        "slender compact mecha torso with lean mechanical shoulders. "
        "Bold jet-black anime contour outlines 4 to 8 pixels thick (#12151C). "
    )
    if bg_contrast == "slate_blue":
        return DEEPSEEK_PORTRAIT_PROMPT
    if bg_contrast != "white":
        return f"{prompt} {proportions}"
    stage = prompt.replace("solid pure black background", "solid pure white #FFFFFF background")
    stage = stage.replace("Solid pure black background", "Solid pure white #FFFFFF background")
    stage = stage.replace("pure solid black background", "solid pure white #FFFFFF background")
    return f"{stage} {proportions}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, matte, slice, and rig one puppet.")
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--facing", default="left", choices=("left", "right"))
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--bg-contrast",
        default="black",
        choices=("black", "white", "slate_blue"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = args.prompt or (
        DEEPSEEK_MASTER_PROMPT if args.character_id == "deepseek_cyborg_v3" else ""
    )
    if not prompt:
        raise SystemExit("--prompt is required for this character")
    skin = OneShotPuppetFactory().create_character(
        args.character_id,
        prompt,
        args.facing,
        bg_contrast=args.bg_contrast,
    )
    print(skin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
