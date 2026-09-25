"""Generate shared 16:9 reverse-angle environments for the animator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from utils.pipeline_paths import assets_root

from .create_puppet import generate_character_image

PANORAMA_SIZE = (3840, 2160)


def _safe_theme_id(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if not normalized or any(
        not (character.isalnum() or character == "_")
        for character in normalized
    ):
        raise argparse.ArgumentTypeError(
            "theme id must contain only letters, numbers, underscores, or hyphens"
        )
    return normalized


def apply_reverse_angle_ambience(image: Image.Image) -> Image.Image:
    """Format a panorama with cool-left and warm-right atmospheric grading."""
    panorama = ImageOps.fit(
        image.convert("RGB"),
        PANORAMA_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    pixels = np.asarray(panorama, dtype=np.float32)
    width = pixels.shape[1]
    position = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    cool = np.clip(-position, 0.0, 1.0).reshape(1, width, 1)
    warm = np.clip(position, 0.0, 1.0).reshape(1, width, 1)
    cool_tint = np.asarray((0.94, 1.00, 1.08), dtype=np.float32).reshape(1, 1, 3)
    warm_tint = np.asarray((1.08, 1.01, 0.93), dtype=np.float32).reshape(1, 1, 3)
    graded = pixels * (1.0 + cool * (cool_tint - 1.0))
    graded *= 1.0 + warm * (warm_tint - 1.0)
    return Image.fromarray(np.clip(graded, 0, 255).astype(np.uint8))


def create_environment(
    *,
    theme_id: str,
    prompt: str,
    image_path: Path | None = None,
    puppets_dir: Path | None = None,
) -> dict:
    root = Path(puppets_dir) if puppets_dir else assets_root() / "puppets"
    destination_dir = root / "shared_backgrounds"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{theme_id}.png"
    source = destination_dir / f"{theme_id}_source.png"
    model = "local"
    if image_path is not None:
        with Image.open(Path(image_path).expanduser()) as supplied:
            supplied.convert("RGB").save(source, format="PNG", compress_level=1)
    else:
        if not prompt.strip():
            raise ValueError("--prompt is required when --image-path is omitted")
        environment_prompt = (
            "Wide cinematic 16:9 panoramic environment plate for a "
            "shot-reverse-shot dialogue. No people, no characters, no text. "
            "One continuous room with a consistent horizon, generous central "
            "depth, and matching architecture across both halves. "
            f"{prompt}"
        )
        model = generate_character_image(
            environment_prompt,
            source,
            aspect_ratio="16:9",
        )
    with Image.open(source) as generated:
        formatted = apply_reverse_angle_ambience(generated)
    formatted.save(destination, format="PNG", compress_level=1)
    return {
        "theme_id": theme_id,
        "path": str(destination),
        "source_path": str(source),
        "size": list(formatted.size),
        "acquisition": model,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a shared cool/warm 16:9 animator environment."
    )
    parser.add_argument("--theme", required=True, type=_safe_theme_id)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = create_environment(
        theme_id=args.theme,
        prompt=args.prompt,
        image_path=args.image_path,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PANORAMA_SIZE",
    "apply_reverse_angle_ambience",
    "create_environment",
]
