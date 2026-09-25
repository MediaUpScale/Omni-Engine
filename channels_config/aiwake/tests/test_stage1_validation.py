"""Single 9:16 reverse-shot frame of DeepSeek. Gemini stays off this canvas."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.animator.asset_generator import (  # noqa: E402
    DEFAULT_PUPPETS_DIR,
    ensure_shared_panorama,
)
from core.animator.compositor import _HeroCamera, _alpha_blend_paste  # noqa: E402
from core.animator.factory.create_puppet import build_deepseek_stage1  # noqa: E402
from core.animator.puppet import PuppetRig, PuppetSkin  # noqa: E402
from core.animator.types import SpeakerStyle  # noqa: E402
from utils.pipeline_paths import outputs_root  # noqa: E402


def _deepseek_frame(background: Image.Image) -> np.ndarray:
    rig = PuppetRig(PuppetSkin.load(DEFAULT_PUPPETS_DIR / "deepseek_cyborg_v3"))
    camera = _HeroCamera(
        rig,
        SpeakerStyle("deepseek_cyborg_v3", "DeepSeek", facing="left"),
        target_width=1080,
        target_height=1920,
    )
    frame = np.asarray(background.resize((1080, 1920), Image.Resampling.LANCZOS), dtype=np.uint8).copy()
    _alpha_blend_paste(frame, camera.static_body, camera.body_offset_x, camera.body_offset_y)
    camera.render_dirty(
        frame,
        t=0.0,
        viseme="X",
        eye_state=0,
        rms=0.0,
        emphasis_threshold=0.8,
        brow_emphasis_threshold=0.9,
        is_speaking=False,
        emotion="neutral",
        body_offset_y=0,
    )
    print(
        f"deepseek_cyborg_v3 eye_line={camera.eye_line_y} "
        f"scale={camera.scale:.4f} facing=left"
    )
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek single-canvas reverse shot.")
    parser.add_argument("--single-character", default="deepseek_cyborg_v3")
    args = parser.parse_args(argv)
    if args.single_character != "deepseek_cyborg_v3":
        raise SystemExit(f"unsupported single character: {args.single_character}")
    stage = build_deepseek_stage1(regenerate=True)
    print("stage1", stage)
    panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
    width = panorama.width // 2
    library = panorama.crop((width, 0, panorama.width, panorama.height))
    sheet = Image.fromarray(_deepseek_frame(library))
    if sheet.size != (1080, 1920):
        raise SystemExit(f"expected 1080x1920, got {sheet.size}")
    destination = (
        outputs_root()
        / "aiwake"
        / "_test_harness"
        / "deepseek_single_test.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", compress_level=1)
    print(f"validation: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
