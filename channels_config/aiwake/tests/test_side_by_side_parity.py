"""Side-by-side proof: DeepSeek matched to Gemini's head presence. No video."""
from __future__ import annotations

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
from core.animator.factory.create_puppet import generate_character_image  # noqa: E402
from core.animator.factory.rigger import birefnet_cutout  # noqa: E402
from utils.pipeline_paths import outputs_root  # noqa: E402

HEAD_PROMPT = (
    "Studio Ghibli vintage anime mecha head, tall dignified oval dome helmet in "
    "dark matte naval blue, pale ice-blue faceplate with glowing green concentric "
    "ring optic eyes, clean rectangular brass plate on forehead 'DEEPSEEK', clean "
    "rounded chin plate. Cut cleanly at the jawline, NO neck attached. Angled 25 "
    "degrees facing left. Solid black background. --ar 1:1"
)
BODY_PROMPT = (
    "Studio Ghibli 90s vintage anime mecha, slender athletic humanoid robot torso, "
    "dark matte navy blue plating with delicate thin anime ink lines, rounded collar "
    "with an exposed mechanical neck column (copper conduits, cables, and hydraulic "
    "pistons rising from collar socket). Clean, elegant, non-bulky chest plate. "
    "Angled 25 degrees facing left. Solid black background. --ar 9:16"
)
HEAD_HEIGHT = 560


def _tight(image: Image.Image) -> Image.Image:
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    if not xs.size:
        raise RuntimeError("empty cutout")
    return image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def _scale(image: Image.Image, scale: float) -> Image.Image:
    return image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )


def _bbox(path: Path) -> tuple[int, int, int, int]:
    alpha = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _library_panel(background: Image.Image) -> Image.Image:
    return background.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")


def _gemini_panel(background: Image.Image, head: Image.Image, body: Image.Image, head_bbox_h: int) -> Image.Image:
    frame = _library_panel(background)
    scale = HEAD_HEIGHT / max(1, head_bbox_h)
    placed_head = _scale(head, scale)
    placed_body = _scale(body, scale)
    origin_x = (1080 - placed_body.width) // 2
    origin_y = 1920 - placed_body.height
    frame.alpha_composite(placed_body, (origin_x, origin_y))
    frame.alpha_composite(placed_head, (origin_x, origin_y))
    return frame


def _deepseek_panel(background: Image.Image, head: Image.Image, body: Image.Image, shoulder_px: int) -> Image.Image:
    frame = _library_panel(background)
    placed_head = _scale(head, HEAD_HEIGHT / max(1, head.height))
    placed_body = _scale(body, shoulder_px / max(1, body.width))
    body_x = (1080 - placed_body.width) // 2
    body_y = 1920 - placed_body.height
    frame.alpha_composite(placed_body, (body_x, body_y))
    rim = int(np.nonzero(np.asarray(placed_body)[..., 3] > 16)[0].min())
    head_y = body_y + rim - placed_head.height + 36
    frame.alpha_composite(placed_head, ((1080 - placed_head.width) // 2, head_y))
    return frame


def main() -> int:
    skin = DEFAULT_PUPPETS_DIR / "deepseek_cyborg_v3"
    skin.mkdir(parents=True, exist_ok=True)
    head_src = skin / "source_parity_head.png"
    body_src = skin / "source_parity_body.png"
    generate_character_image(HEAD_PROMPT, head_src, aspect_ratio="1:1")
    generate_character_image(BODY_PROMPT, body_src, aspect_ratio="9:16")
    with Image.open(head_src) as opened:
        head = _tight(birefnet_cutout(opened.convert("RGB")))
    with Image.open(body_src) as opened:
        body = _tight(birefnet_cutout(opened.convert("RGB")))
    head.save(skin / "head.png", format="PNG", compress_level=1)
    body.save(skin / "body.png", format="PNG", compress_level=1)

    gemini = DEFAULT_PUPPETS_DIR / "gemini_cyborg_v2"
    _gx0, _gy0, _gx1, gy1 = _bbox(gemini / "head.png")
    bx0, _by0, bx1, _by1 = _bbox(gemini / "body.png")
    gemini_head = Image.open(gemini / "head.png").convert("RGBA")
    gemini_body = Image.open(gemini / "body.png").convert("RGBA")
    shoulder_px = int(round((bx1 - bx0) * (HEAD_HEIGHT / max(1, gy1 - _gy0))))

    panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
    half = panorama.width // 2
    left_bg = panorama.crop((0, 0, half, panorama.height))
    right_bg = panorama.crop((half, 0, panorama.width, panorama.height))
    sheet = Image.new("RGB", (2160, 1920))
    sheet.paste(_gemini_panel(left_bg, gemini_head, gemini_body, gy1 - _gy0).convert("RGB"), (0, 0))
    sheet.paste(_deepseek_panel(right_bg, head, body, shoulder_px).convert("RGB"), (1080, 0))
    destination = outputs_root() / "aiwake" / "_test_harness" / "gemini_deepseek_side_by_side.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", compress_level=1)
    gemini_scale = HEAD_HEIGHT / max(1, gy1 - _gy0)
    print(f"gemini_head_px={HEAD_HEIGHT} gemini_shoulder_px={int(round((_bbox(gemini / 'body.png')[2] - _bbox(gemini / 'body.png')[0]) * gemini_scale))}")
    print(f"deepseek_head_px={HEAD_HEIGHT} deepseek_shoulder_px={shoulder_px}")
    print(f"parity: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
