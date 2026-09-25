"""Zero-credit visual inspection harness for the V3 BiRefNet cutout."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.animator.factory.rigger import birefnet_cutout  # noqa: E402
from utils.pipeline_paths import assets_root, outputs_root  # noqa: E402


def main() -> int:
    source = (
        assets_root()
        / "puppets"
        / "deepseek_cyborg_v3"
        / "source_head_original.png"
    )
    if not source.is_file():
        raise FileNotFoundError(f"raw DeepSeek head not found: {source}")

    destination = (
        outputs_root()
        / "aiwake"
        / "_test_harness"
        / "head_cutout_inspection.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as raw:
        cutout = birefnet_cutout(raw)
    alpha = np.asarray(cutout.getchannel("A"), dtype=np.uint8)
    intermediate = np.unique(alpha[(alpha > 0) & (alpha < 255)])
    if intermediate.size < 16:
        raise RuntimeError(
            "BiRefNet matte is not continuous enough: "
            f"{intermediate.size} intermediate alpha values"
        )

    panel_size = cutout.size
    white = Image.new("RGBA", panel_size, (255, 255, 255, 255))
    dark = Image.new("RGBA", panel_size, (38, 40, 46, 255))
    white.alpha_composite(cutout)
    dark.alpha_composite(cutout)
    inspection = Image.new(
        "RGB",
        (panel_size[0] * 2, panel_size[1]),
        (0, 0, 0),
    )
    inspection.paste(white.convert("RGB"), (0, 0))
    inspection.paste(dark.convert("RGB"), (panel_size[0], 0))
    inspection.save(destination, format="PNG", compress_level=1)

    print("BiRefNet initialized: birefnet-general")
    print(f"continuous alpha levels: {np.unique(alpha).size}")
    print(f"intermediate alpha levels: {intermediate.size}")
    print(f"inspection: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
