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

from .create_puppet import (
    DEEPSEEK_MASTER_PROMPT,
    generate_character_image,
    generate_imagen3_image,
)
from .layer_slicer import slice_jaw_contour
from .puppet_matrix import ANATOMICAL_TOLERANCE_ZONE
from .rigger import auto_rig_character, birefnet_cutout


HEAD_ISOLATION_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic, clean flat watercolor shading. "
    "Square close-up portrait of the mecha head of robot 'DEEPSEEK', angled 25 degrees facing to the left. "
    "Smooth rounded naval-blue dome helmet with antique brass circular ear dials and rivets. "
    "Rectangular brass nameplate engraved 'DEEPSEEK' across forehead. "
    "Calm, expressive glowing emerald-green concentric optical sensor lenses. "
    "Smooth blank curved bronze chin shield plate. "
    "CRITICAL CONSTRAINTS: Floating head only, ending cleanly at the lower bronze chin contour. "
    "The image must contain absolutely NO neck, NO body, NO torso, NO shoulders, NO human mouth, "
    "NO human nose, and zero magenta or pink bleeding onto the character. "
    "Rendered against a solid flat pure magenta background (#FF00FF). --ar 1:1"
)

BODY_CHASSIS_DEFINITIVE_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, "
    "clean flat watercolor cel shading. "
    "Vertical 9:16 portrait of a charming headless retro-futuristic tin-plate robot chassis, "
    "DYNAMIC ASYMMETRICAL THREE-QUARTER VIEW ANGLED 25 DEGREES TO THE LEFT. "
    "NECK & COLLAR: A SINGLE wide circular antique brass collar rim welded solidly directly to the top of the chest. "
    "Inside this single collar rim, a SHORT stubby dark corrugated mechanical neck column extends upwards just 40 pixels, "
    "ending in a flat clean flush metallic socket ready to receive a head. "
    "ABSOLUTELY NO detached floating rings, NO secondary halos, NO long giraffe neck. "
    "TORSO: Smooth curved cylindrical naval-blue tin-can barrel chassis (Laputa vintage aesthetic). "
    "At center chest, a vintage circular boiler pressure gauge with pointer needle and brass bezel. "
    "Slender tubular mechanical arms resting flush against the sides with round brass shoulder ball-joints. "
    "GROUNDED BASE: The naval-blue torso armor plates extend solidly all the way down, completely filling the lower canvas "
    "and bleeding off the bottom frame boundary (y=1920). NO floating capsule bottom cut. "
    "CRITICAL CONSTRAINTS: Headless body only. Angled 25 degrees facing left. "
    "Strictly NO head, NO face, NO floating rings, NO secondary neck collars, NO front-facing symmetry, "
    "NO muscular pecs, NO battle damage, and zero magenta or pink bleeding onto the chassis. "
    "Pristine factory condition. Warm nostalgic anime lighting, solid flat pure magenta background (#FF00FF). --ar 9:16"
)

BODY_CHASSIS_3QUARTERS_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, "
    "clean flat watercolor cel shading. "
    "Vertical 9:16 portrait of a charming headless retro-futuristic tin-plate robot chassis, "
    "DYNAMIC ASYMMETRICAL THREE-QUARTER VIEW ANGLED 25 DEGREES TO THE LEFT. "
    "PERSPECTIVE: The left shoulder is in the foreground closer to the camera, while the right shoulder is visibly recessed in perspective. "
    "COLLAR: Wide circular antique brass collar rim tilted in 3/4 oval perspective at top center, "
    "with dark corrugated mechanical neck column ready to receive a head. "
    "TORSO: Smooth curved cylindrical naval-blue tin-can barrel chassis (Laputa vintage aesthetic). "
    "Solid continuous smooth curved chest plate with antique brass trim, circular boiler dial with rivets. "
    "ARMS: Slender tubular mechanical arms resting naturally flush and straight against the sides of the torso. "
    "Torso solidly fills the lower canvas down to the bottom border (y=1920). "
    "CRITICAL CONSTRAINTS: Headless body only. Angled 25 degrees facing left. "
    "Strictly NO head, NO face, NO front-facing symmetry, NO muscular superhero pecs, NO six-pack abs, NO battle damage, "
    "and zero magenta or pink bleeding onto the chassis. "
    "Pristine factory condition. Warm nostalgic anime lighting, solid flat pure magenta background (#FF00FF). --ar 9:16"
)

BODY_CHASSIS_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, clean flat watercolor shading. "
    "Vertical portrait framing (9:16 canvas) of a sturdy vintage mecha chassis in naval-blue with antique brass trim. "
    "At top center, a wide circular antique brass collar rim opening with exposed dark corrugated mechanical neck column. "
    "Massive rounded naval-blue chest plates, broad articulated mechanical shoulders spanning across the frame width. "
    "Torso solidly fills the lower canvas and bleeds off the left, right, and bottom borders. "
    "CRITICAL CONSTRAINTS: Headless chassis only. The image must contain absolutely NO head, NO helmet, NO face, "
    "NO eyes, NO human features, NO floating torso, and zero magenta or pink bleeding onto the chassis. "
    "Pristine factory condition. Nostalgic warm anime lighting, solid flat pure magenta background (#FF00FF). --ar 9:16"
)

DEEPSEEK_CHROMA_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha character design, "
    "clean flat watercolor cel shading. "
    "TIGHT CLOSE-UP MECHA BUST PORTRAIT (9:16 vertical canvas) of an antique naval cybernetic robot named 'DEEPSEEK', "
    "facing 25 degrees to the left. "
    "HEAD: Massive, oversized rounded dome helmet in dark matte naval-blue plating with circular antique brass ear dials, "
    "occupying the upper 40% of the canvas. Clean brass nameplate engraved 'DEEPSEEK'. "
    "Expressive glowing emerald-green concentric optical sensor lenses. "
    "LOWER FACE: Smooth, solid, blank curved bronze chin shield plate (absolutely NO mouth, NO nose). "
    "BODY: Heavy industrial naval-blue chest armor plate with wide antique brass collar rim seated directly below the chin. "
    "Broad mechanical shoulders extending solidly to the left and right canvas edges. "
    "Chest plate solidly fills the entire lower canvas, bleeding off the bottom frame boundary (y=1920). "
    "NO waist, NO floating limbs, NO distant full-body shot. "
    "Pristine factory condition. Nostalgic warm anime lighting, rendered against a solid flat pure magenta background (#FF00FF). --ar 9:16"
)

DEEPSEEK_NEGATIVE_PROMPT = (
    "magenta, pink, purple, fuchsia, violet, colored ambient light on robot, "
    "color bleed, human nose, human mouth, human smile, floating body, waist, hips, "
    "distant full body shot, blurry outlines, rust, battle damage, scratches"
)

ONE_CLICK_MECHA_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha character design, "
    "clean flat watercolor cel shading. "
    "Waist-up vertical portrait framing (9:16 aspect ratio) of a charming retro-futuristic tin-plate robot named 'DEEPSEEK', "
    "facing 25 degrees to the left. "
    "HEAD: Prominent, oversized rounded naval-blue dome helmet with circular antique brass ear dials and rivets, "
    "occupying the upper 35% of the frame with strong visual presence. Sturdy brass nameplate engraved 'DEEPSEEK'. "
    "Calm glowing emerald-green concentric circular optic lenses. "
    "LOWER FACE: Clean, smooth curved blank bronze chin shield plate (blank surface, absolutely NO human nose, NO mouth). "
    "BODY: Charming vintage cylindrical tin-man mecha torso in matte naval-blue with wide brass collar ring. "
    "Both left and right tubular mechanical arms with brass elbow joints descending along frame edges. "
    "Torso solidly fills the lower half down to the bottom border. "
    "Pristine factory condition, zero rust. Nostalgic warm anime lighting, solid flat deep midnight blue background (#0A1128). --ar 9:16"
)

DEEPSEEK_FINAL_PRODUCTION_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, "
    "2d cel animation, clean flat watercolor shading. "
    "Waist-up medium portrait of a dignified cybernetic mecha robot named 'DEEPSEEK', "
    "facing 25 degrees to the left. "
    "HEAD: Smooth rounded naval-blue dome helmet with circular antique brass ear dials and rivets. "
    "Forehead clean rectangular brass nameplate neatly engraved with 'DEEPSEEK'. "
    "Soft glowing emerald green concentric circular optic sensor lenses. "
    "LOWER FACE: Smooth, blank curved bronze chin shield plate with absolutely NO human mouth, NO nose. "
    "BODY: Continuous two-tier naval-blue torso armor: rounded upper chest plate with wide brass collar rim, "
    "and solid lower abdominal midriff plating that extends continuously downward, completely filling the space "
    "between both arms down to the bottom border with zero gaps. "
    "BOTH left and right mechanical arms visible with articulated brass shoulder ball-joints descending along the edges. "
    "Nostalgic warm anime lighting, crisp dark anime contours, solid flat neutral grey background (#7B8C9E). --ar 9:16"
)

MASTER_GHIBLI_MECHA_TEMPLATE = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, "
    "2d cel animation, clean flat watercolor shading. "
    "Close medium bust portrait of a dignified cybernetic mecha robot named '{name}', "
    "facing 25 degrees to the {facing}. "
    "HEAD: Smooth rounded {casing_color} dome helmet with circular antique brass ear dials and rivets. "
    "On its forehead, a clean rectangular brass nameplate neatly engraved with '{name}'. "
    "Soft glowing emerald green concentric circular optic sensor lenses. "
    "LOWER FACE: Smooth, blank curved bronze chin shield plate with absolutely NO human mouth, NO nose, NO smile. "
    "BODY: Rounded {casing_color} mecha torso plate with wide brass collar rim, slender tubular mechanical arms "
    "framing the canvas edges. Upper torso solidly filling the lower half of the 9:16 vertical canvas down to the bottom edge. "
    "Nostalgic warm anime lighting, crisp dark anime contours, solid flat neutral grey background (#7B8C9E). --ar 9:16"
)

DEEPSEEK_TIN_MAN_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha character design, clean flat watercolor shading. "
    "Vertical portrait framing (9:16 aspect ratio) of a charming retro-futuristic tin-plate robot, facing 25 degrees to the left. "
    "HEAD: Prominent large rounded naval-blue dome helmet (occupying the top third of the frame), sturdy rectangular brass plate 'DEEPSEEK', "
    "calm glowing emerald-green concentric optical sensor lenses, smooth curved bronze chin shield plate. "
    "BODY: Charming vintage tin-man cylindrical torso in matte naval blue (inspired by vintage 1950s tin toys and Laputa robots), "
    "wide rounded brass collar rim, central circular bronze gauge portal with rivets, slender tubular mechanical arms with exposed brass elbow gears. "
    "Pristine factory condition, zero rust. Nostalgic warm anime lighting, solid flat pure chroma blue background (#0033CC). --ar 9:16"
)

DEEPSEEK_PRISTINE_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha character design, clean flat watercolor cel shading. "
    "Vertical portrait framing (9:16 aspect ratio) of an antique naval cybernetic humanoid robot, facing 25 degrees to the left. "
    "HEAD: Smooth, pristine rounded dome helmet in dark matte naval-blue plating with polished brass rivets and circular ear dials. "
    "On its forehead, a clean rectangular brass nameplate engraved 'DEEPSEEK'. "
    "Expressive glowing emerald-green concentric ring optical sensor lenses. "
    "LOWER FACE: Smooth, immaculate curved bronze chin shield plate (blank metallic surface, absolutely NO human nose, NO human mouth). "
    "BODY: Polished naval-blue chest armor with broad rounded brass collar rim, sturdy mechanical shoulders and upper arms. "
    "Torso solidly extends downwards, completely filling the lower half of the frame down to the bottom edge. "
    "FINISH: Museum-grade pristine factory condition, smooth flawless automotive paint, zero rust, zero scratches, zero battle damage. "
    "Nostalgic warm anime lighting, solid flat pure chroma blue background (#0033CC). --ar 9:16"
)

PORTRAIT_MASTER_PROMPT = (
    "Studio Ghibli vintage 1990s anime cel animation, clean flat 2d shading, crisp anime line art, Hayao Miyazaki mecha. "
    "Vertical portrait framing (9:16 aspect ratio) of an antique naval cybernetic robot, facing 25 degrees to the left. "
    "HEAD: Prominent rounded naval-blue dome helmet with brass rivets and side dial ears, occupying the upper third of the canvas. "
    "Rectangular brass nameplate engraved 'DEEPSEEK'. Calm, expressive glowing emerald green concentric ring lenses. "
    "Smooth blank curved bronze chin shield plate (absolutely NO human mouth, NO human smile, NO human nose). "
    "BODY: Heavy naval-blue armor plates with wide brass collar rim, broad mechanical shoulders with upper arms. "
    "Chest and torso solidly extend downwards, completely filling the lower half of the canvas and cropped naturally "
    "by the bottom frame boundary (y=1920). "
    "Nostalgic warm anime lighting, rendered against a solid flat pure chroma blue background (#0033CC). --ar 9:16"
)

DEEPSEEK_CANONICAL_PROMPT = (
    "Studio Ghibli character design, vintage 90s anime style, 2d cel animation, flat shading, muted pastel colors. "
    "Medium bust shot of a sleek, charming retro-futuristic robot, facing 25 degrees to the left. "
    "Smooth metallic chassis in muted naval blue, slate grey, and antique brass trim. "
    "On its forehead, a clean metal plate with the text 'DEEPSEEK'. "
    "Soft glowing emerald green eyes. Closed mouth. "
    "Nostalgic warm anime lighting, solid dark background."
)

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


class UniversalPuppetAssembler:
    """Autonomous docking engine for any 9:16 debater puppet."""

    CANVAS_W = 1080
    CANVAS_H = 1920
    TARGET_EYE_Y = 575
    TARGET_HEAD_HEIGHT = 640
    PUPPET_STAGE_X = 610

    @classmethod
    def auto_dock(
        cls,
        raw_head_rgba: np.ndarray,
        raw_body_rgba: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
        """Return scaled layers and paste points for the canonical horizon."""
        import cv2

        head_cropped = cls._crop_alpha(raw_head_rgba)
        body_cropped = cls._crop_alpha(raw_body_rgba)
        head_scale = cls.TARGET_HEAD_HEIGHT / max(1, head_cropped.shape[0])
        head = cv2.resize(
            head_cropped, None, fx=head_scale, fy=head_scale, interpolation=cv2.INTER_LANCZOS4
        )
        detected_eye_offset_y = int(head.shape[0] * 0.42)
        chin_y = head.shape[0]
        head_y = cls.TARGET_EYE_Y - detected_eye_offset_y
        head_x = cls.PUPPET_STAGE_X - (head.shape[1] // 2)
        target_body_w = int(head.shape[1] * 1.45)
        body_scale_x = target_body_w / max(1, body_cropped.shape[1])
        body = cv2.resize(
            body_cropped,
            None,
            fx=body_scale_x,
            fy=body_scale_x * 1.05,
            interpolation=cv2.INTER_LANCZOS4,
        )
        overlap_sink = int(head.shape[0] * 0.08)
        body_y = (head_y + chin_y) - overlap_sink
        body_x = cls.PUPPET_STAGE_X - (body.shape[1] // 2)
        print(
            f"assembler head={head.shape[1]}x{head.shape[0]} "
            f"body={body.shape[1]}x{body.shape[0]} "
            f"eye_y={cls.TARGET_EYE_Y} head_xy=({head_x},{head_y}) "
            f"body_xy=({body_x},{body_y}) sink={overlap_sink}"
        )
        return head, body, (head_x, head_y), (body_x, body_y)

    @staticmethod
    def _crop_alpha(img: np.ndarray) -> np.ndarray:
        import cv2

        alpha = img[:, :, 3]
        points = cv2.findNonZero((alpha > 0).astype(np.uint8))
        if points is None:
            raise RuntimeError("auto-dock crop found no opaque pixels")
        x, y, width, height = cv2.boundingRect(points)
        return img[y : y + height, x : x + width]


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
        if bg_contrast in {"portrait", "flow"}:
            generate_imagen3_image(prompt, source, aspect_ratio="9:16")
        else:
            generate_character_image(prompt, source, aspect_ratio="9:16")
        with Image.open(source) as opened:
            master = birefnet_cutout(opened.convert("RGB"))
        if bg_contrast == "bounds":
            master = _despill_bottom_chroma(master)
        transparent = skin_dir / "deepseek_master_transparent.png"
        master.save(transparent, format="PNG", compress_level=1)
        master.save(skin_dir / "character.png", format="PNG", compress_level=1)
        if bg_contrast in {"golden", "magenta"}:
            head, body = _slice_chin_only(master, overlap_px=15)
        else:
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

    def create_decoupled(self, character_id: str, facing: str = "left") -> Path:
        """Generate a separate head and chassis, then dock the tight crops."""
        skin_dir = assets_root() / "puppets" / character_id
        skin_dir.mkdir(parents=True, exist_ok=True)
        head_source = skin_dir / "source_head_stage.png"
        body_source = skin_dir / "source_body_stage.png"
        generate_character_image(HEAD_ISOLATION_PROMPT, head_source, aspect_ratio="1:1")
        generate_character_image(BODY_CHASSIS_PROMPT, body_source, aspect_ratio="9:16")
        with Image.open(head_source) as opened:
            head = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        with Image.open(body_source) as opened:
            body = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        head.save(skin_dir / "head.png", format="PNG", compress_level=1)
        body.save(skin_dir / "body.png", format="PNG", compress_level=1)
        manifest = skin_dir / "puppet.json"
        if manifest.is_file():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("skin_version") != "v3-auto-rig":
                raise RuntimeError(f"refusing to replace artist manifest: {manifest}")
            manifest.unlink()
        auto_rig_character(skin_dir, character_id, facing=facing, component_mode=True)
        self._write_preview(character_id, head, body, bg_contrast="decoupled")
        return skin_dir

    def create_body_only(self, character_id: str) -> Path:
        """Regenerate the chassis. The approved head.png stays untouched."""
        skin_dir = assets_root() / "puppets" / character_id
        head_path = skin_dir / "head.png"
        if not head_path.is_file():
            raise RuntimeError(f"approved head is missing: {head_path}")
        body_source = skin_dir / "source_body_stage.png"
        generate_character_image(BODY_CHASSIS_DEFINITIVE_PROMPT, body_source, aspect_ratio="9:16")
        with Image.open(body_source) as opened:
            body = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        body.save(skin_dir / "body.png", format="PNG", compress_level=1)
        with Image.open(head_path) as opened:
            head = opened.convert("RGBA")
        self._write_preview(character_id, head, body, bg_contrast="bodyonly")
        return skin_dir

    def run_assembler(
        self,
        character_id: str,
        *,
        head_scale_pct: float = 105,
        body_scale_pct: float = 97,
        body_drop_px: int = 18,
        head_target_h: int | None = None,
        body_target_w: int | None = None,
        head_shift_left: int = 10,
        head_drop_px: int = 15,
        restore_approved_head: bool = False,
        fix_rotation_padding: bool = False,
        body_shift_right: int = 0,
        head_rel_shift_left: int = 0,
        preview_name: str = "",
    ) -> Path:
        """Dock the saved head and body. Does not call an image API."""
        skin_dir = assets_root() / "puppets" / character_id
        bucket = skin_dir / "_head_isolated.png"
        if restore_approved_head and bucket.is_file():
            bucket.unlink()
        approved = skin_dir / "_head_approved_master.png"
        if restore_approved_head or not approved.is_file():
            stage = skin_dir / "source_head_stage.png"
            if restore_approved_head and stage.is_file():
                with Image.open(stage) as opened:
                    restored = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
                restored.save(approved, format="PNG", compress_level=1)
        head_path = approved if approved.is_file() else skin_dir / "head.png"
        body_master = skin_dir / "_body_tight_master.png"
        body_path = skin_dir / "body.png"
        if not body_master.is_file() and body_path.is_file():
            body_master.write_bytes(body_path.read_bytes())
        if body_master.is_file():
            body_path = body_master
        if not head_path.is_file() or not body_path.is_file():
            raise RuntimeError(f"approved layers missing in {skin_dir}")
        head_rgba = np.asarray(_tight_alpha_crop(Image.open(head_path).convert("RGBA")))
        body_rgba = np.asarray(_tight_alpha_crop(Image.open(body_path).convert("RGBA")))
        if fix_rotation_padding:
            from .puppet_assembler import diagnose_rotation_clip

            diagnose_rotation_clip(head_rgba, head_path.name)
        from .puppet_assembler import CanonicalDebaterContract

        head, body, head_xy, body_xy, pivot = CanonicalDebaterContract.assemble(
            head_rgba,
            body_rgba,
            head_height=head_target_h or CanonicalDebaterContract.TARGET_HEAD_HEIGHT,
            body_width=body_target_w or CanonicalDebaterContract.BODY_WIDTH,
            body_drop_px=body_drop_px,
            head_shift_x=-head_shift_left,
            head_drop_px=head_drop_px,
            body_shift_right=body_shift_right,
            head_rel_shift_left=head_rel_shift_left,
        )
        manifest_path = skin_dir / "puppet.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {"character_id": character_id, "skin_version": "v3-auto-rig"}
        )
        manifest["layer_order"] = ["body", "head", "mouths"]
        manifest["rest_rotation_deg"] = -3.5
        manifest["auto_dock"] = {
            "canvas": [1080, 1920],
            "contract": "CanonicalDebaterContract",
            "target_eye_y": 600,
            "target_head_height": int(head.shape[0]),
            "stage_x": 585,
            "head_xy": list(head_xy),
            "body_xy": list(body_xy),
            "chest_collar_ratio": 0.17,
            "chin_sink_px": 75,
            "body_scale_y_mult": 1.15,
            "body_target_w": int(body.shape[1]),
            "body_drop_px": body_drop_px,
            "body_shift_right": body_shift_right,
            "head_rel_shift_left": head_rel_shift_left,
            "head_scale_pct": head_scale_pct,
            "body_scale_pct": body_scale_pct,
            "rest_rotation_deg": -3.5,
            "head_pivot": [pivot[0], pivot[1]],
            "bones": {
                "eye": [585, 600],
                "chin": [body_xy[0] + int(body.shape[1] * 0.46), head_xy[1] + int(head.shape[0] * 0.95)],
                "collar": [body_xy[0] + int(body.shape[1] * 0.46), body_xy[1] + int(body.shape[0] * 0.17)],
                "head_pivot": [head_xy[0] + pivot[0], head_xy[1] + pivot[1]],
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        Image.fromarray(head).save(skin_dir / "head.png", format="PNG", compress_level=1)
        Image.fromarray(body).save(skin_dir / "body.png", format="PNG", compress_level=1)
        self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=((head_xy, body_xy)),
            preview_name=preview_name,
        )
        return skin_dir

    def _write_preview(
        self,
        character_id: str,
        head: Image.Image,
        body: Image.Image,
        bg_contrast: str = "black",
        placed: tuple[tuple[int, int], tuple[int, int]] | None = None,
        preview_name: str = "",
    ) -> Path:
        from core.animator.asset_generator import (  # noqa: PLC0415
            DEFAULT_PUPPETS_DIR,
            ensure_shared_panorama,
        )

        head_rgba = np.asarray(head.convert("RGBA"))
        body_rgba = np.asarray(body.convert("RGBA"))
        if bg_contrast == "bounds":
            zone = ANATOMICAL_TOLERANCE_ZONE
            overlap_px = 15
            target_head_height = 640
            head_alpha = head_rgba[..., 3]
            body_alpha = body_rgba[..., 3]
            hys, hxs = np.nonzero(head_alpha > 16)
            bys, bxs = np.nonzero(body_alpha > 16)
            raw_head_height = int(hys.max() - hys.min() + 1)
            center = int(np.median(hxs))
            half = max(12, int((hxs.max() - hxs.min()) * 0.18))
            chin_band = head_alpha[:, max(0, center - half) : center + half]
            collar_band = body_alpha[:, max(0, center - half) : center + half]
            chin_rows = np.nonzero(chin_band.any(axis=1))[0]
            collar_rows = np.nonzero(collar_band.any(axis=1))[0]
            chin_y = int(chin_rows.max()) if chin_rows.size else int(hys.max())
            collar_y = int(collar_rows.min()) if collar_rows.size else int(bys.min())
            scale = target_head_height / max(1, raw_head_height)
            placed_head = head.resize(
                (max(1, int(round(head.width * scale))), max(1, int(round(head.height * scale)))),
                Image.Resampling.LANCZOS,
            )
            placed_body = body.resize(
                (max(1, int(round(body.width * scale))), max(1, int(round(body.height * scale)))),
                Image.Resampling.LANCZOS,
            )
            placed_head_rgba = np.asarray(placed_head)
            green = (
                (placed_head_rgba[..., 1] > 120)
                & (placed_head_rgba[..., 1] > placed_head_rgba[..., 0] + 25)
                & (placed_head_rgba[..., 3] > 16)
            )
            eye_rows = np.nonzero(green.any(axis=1))[0]
            eye_cols = np.nonzero(green.any(axis=0))[0]
            eye_y = int(np.median(eye_rows)) if eye_rows.size else int(round((int(hys.min()) + raw_head_height * 0.42) * scale))
            eye_x = int(np.median(eye_cols)) if eye_cols.size else placed_head.width // 2
            head_x = int(zone["target_head_x"] - eye_x)
            head_y = int(zone["target_eye_y"] - eye_y)
            origin_x = head_x
            origin_y = int(round(head_y + (chin_y - collar_y) * scale - overlap_px))
            head_origin = (head_x, head_y)
            head_w = (int(hxs.max()) - int(hxs.min()) + 1) * scale
            body_bottom = int(round((int(bys.max()) + 1) * scale + origin_y))
            print(
                f"bounds head={head_w:.0f}x{target_head_height} "
                f"dock_overlap={overlap_px} eye_y={zone['target_eye_y']} "
                f"body_bottom={body_bottom}"
            )
        elif bg_contrast == "assembled" and placed is not None:
            placed_head = head
            placed_body = body
            head_origin, (origin_x, origin_y) = placed
        elif bg_contrast == "bodyonly":
            placed_head, placed_body, head_origin, (origin_x, origin_y) = fit_three_quarter_body(
                head, body
            )
        elif bg_contrast == "decoupled":
            from .puppet_assembler import assemble_puppet

            placed_head_np, placed_body_np, head_origin, (origin_x, origin_y) = assemble_puppet(
                np.asarray(head.convert("RGBA")),
                np.asarray(body.convert("RGBA")),
            )
            placed_head = Image.fromarray(placed_head_np)
            placed_body = Image.fromarray(placed_body_np)
        elif bg_contrast == "magenta":
            placed_head, placed_body, head_origin, (origin_x, origin_y) = fit_chroma_bounds(
                head, body
            )
        elif bg_contrast == "golden":
            placed_head, placed_body, head_origin, (origin_x, origin_y) = fit_to_golden_bounds(
                head, body
            )
        elif bg_contrast in {"flow", "approved"}:
            green = (
                (head_rgba[..., 1] > 120)
                & (head_rgba[..., 1] > head_rgba[..., 0] + 25)
                & (head_rgba[..., 3] > 16)
            )
            eye_rows = np.nonzero(green.any(axis=1))[0]
            eye_cols = np.nonzero(green.any(axis=0))[0]
            body_rows = np.nonzero((body_rgba[..., 3] > 16).any(axis=1))[0]
            eye_y = int(np.median(eye_rows)) if eye_rows.size else head.height // 3
            eye_x = int(np.median(eye_cols)) if eye_cols.size else head.width // 2
            bottom = int(body_rows.max()) + 1 if body_rows.size else body.height
            scale = (1920 - 620) / max(1, bottom - eye_y)
            placed_body = body.resize(
                (max(1, int(round(body.width * scale))), max(1, int(round(body.height * scale)))),
                Image.Resampling.LANCZOS,
            )
            placed_head = head.resize(
                (max(1, int(round(head.width * scale))), max(1, int(round(head.height * scale)))),
                Image.Resampling.LANCZOS,
            )
            origin_x = int(round(540 - eye_x * scale))
            origin_y = int(round(620 - eye_y * scale))
            print(f"flow eye_y=620 body_bottom=1920 scale={scale:.3f}")
        elif bg_contrast == "canonical":
            green = (
                (head_rgba[..., 1] > 120)
                & (head_rgba[..., 1] > head_rgba[..., 0] + 25)
                & (head_rgba[..., 3] > 16)
            )
            eye_rows = np.nonzero(green.any(axis=1))[0]
            body_rows = np.nonzero((body_rgba[..., 3] > 8).any(axis=1))[0]
            eye_y = int(np.median(eye_rows)) if eye_rows.size else head.height // 3
            bottom = int(body_rows.max()) + 1 if body_rows.size else body.height
            scale = (1920 - 640) / max(1, bottom - eye_y)
            placed_body = body.resize(
                (max(1, int(round(body.width * scale))), max(1, int(round(body.height * scale)))),
                Image.Resampling.LANCZOS,
            )
            placed_head = head.resize(
                (max(1, int(round(head.width * scale))), max(1, int(round(head.height * scale)))),
                Image.Resampling.LANCZOS,
            )
            origin_x = (1080 - placed_body.width) // 2
            origin_y = 640 - int(round(eye_y * scale))
        else:
            head_alpha = head_rgba[..., 3]
            body_alpha = body_rgba[..., 3]
            ys, xs = np.nonzero((head_alpha > 8) | (body_alpha > 8))
            if not ys.size:
                raise RuntimeError("one-shot bust has no visible pixels")
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            cropped_body = body.crop(box)
            cropped_head = head.crop(box)
            scale = (
                1920 / max(1, cropped_body.height)
                if bg_contrast == "portrait"
                else (1920 * 1.18) / max(1, cropped_body.height)
            )
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
            origin_x = (1080 - placed_body.width) // 2
            origin_y = 1920 - placed_body.height + (
                0 if bg_contrast == "portrait" else int(1920 * 0.20)
            )
        if bg_contrast not in {"bounds", "golden", "magenta", "decoupled", "bodyonly", "assembled"}:
            head_origin = (origin_x, origin_y)
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        frame = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        _composite_at(frame, placed_body, origin_x, origin_y)
        _composite_at(frame, placed_head, head_origin[0], head_origin[1])
        filename = (
            "deepseek_one_click_golden_static.png"
            if bg_contrast in {"golden", "magenta", "decoupled", "bodyonly", "assembled"}
            else "deepseek_master_approved.png"
            if bg_contrast == "approved"
            else "autonomous_factory_test.png"
            if bg_contrast == "flow"
            else "deepseek_perfect_bounding_box.png"
            if bg_contrast == "bounds"
            else "deepseek_portrait_test.png"
            if bg_contrast == "portrait"
            else "deepseek_canonical_v2_formula.png"
            if bg_contrast == "canonical"
            else "deepseek_single_9x16_approved.png"
            if bg_contrast == "slate_blue"
            else f"{character_id.split('_cyborg')[0]}_v3_oneshot_master.png"
        )
        destination = outputs_root() / "aiwake" / "_test_harness" / (preview_name or filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.convert("RGB").save(destination, format="PNG", compress_level=1)
        print(f"oneshot: {destination}")
        return destination


def fit_three_quarter_body(
    head_img: Image.Image,
    body_img: Image.Image,
) -> tuple[Image.Image, Image.Image, tuple[int, int], tuple[int, int]]:
    """Scale the frozen head to 640px and seat the chin inside the top collar rim."""
    import cv2

    head_rgba = np.asarray(head_img.convert("RGBA"))
    body_rgba = np.asarray(body_img.convert("RGBA"))
    head_scale = 640.0 / max(1, head_rgba.shape[0])
    body_scale = (1080 * 0.88) / max(1, body_rgba.shape[1])
    scaled_head = cv2.resize(head_rgba, None, fx=head_scale, fy=head_scale, interpolation=cv2.INTER_LANCZOS4)
    scaled_body = cv2.resize(body_rgba, None, fx=body_scale, fy=body_scale, interpolation=cv2.INTER_LANCZOS4)
    anchor_x = 580
    body_x = int(anchor_x - scaled_body.shape[1] / 2)
    body_y = 1920 - scaled_body.shape[0]
    red = scaled_body[..., 0].astype(np.int16)
    green = scaled_body[..., 1].astype(np.int16)
    blue = scaled_body[..., 2].astype(np.int16)
    alpha = scaled_body[..., 3]
    brass = (alpha > 16) & (red > 140) & (green > 110) & (red > blue + 30)
    band = brass[: int(scaled_body.shape[0] * 0.22), int(scaled_body.shape[1] * 0.35) : int(scaled_body.shape[1] * 0.65)]
    rows = np.nonzero(band.any(axis=1))[0]
    collar_y = int(rows.min()) if rows.size else 0
    chin_offset = scaled_head.shape[0] - 1
    head_y = int(body_y + collar_y - chin_offset + 24)
    head_x = int(anchor_x - scaled_head.shape[1] / 2)
    lenses = (
        (scaled_head[..., 1] > 120)
        & (scaled_head[..., 1] > scaled_head[..., 0] + 25)
        & (scaled_head[..., 3] > 16)
    )
    eye_rows = np.nonzero(lenses.any(axis=1))[0]
    eye_y = int(head_y + np.median(eye_rows)) if eye_rows.size else head_y
    print(
        f"bodyonly head={scaled_head.shape[1]}x640 body_w={scaled_body.shape[1]} "
        f"eye_y={eye_y} anchor_x=580 body_bottom=1920"
    )
    return (
        Image.fromarray(scaled_head),
        Image.fromarray(scaled_body),
        (head_x, head_y),
        (body_x, body_y),
    )


def fit_chroma_bounds(
    head_img: Image.Image,
    body_img: Image.Image,
) -> tuple[Image.Image, Image.Image, tuple[int, int], tuple[int, int]]:
    """Scale the bust to a 640px head, dock the chin, and seat the hem at y=1920."""
    import cv2

    head_rgba = np.asarray(head_img.convert("RGBA"))
    body_rgba = np.asarray(body_img.convert("RGBA"))
    head_alpha = head_rgba[..., 3]
    body_alpha = body_rgba[..., 3]
    hys, hxs = np.nonzero(head_alpha > 16)
    bys, bxs = np.nonzero(body_alpha > 16)
    if not hys.size or not bys.size:
        raise RuntimeError("chroma fit requires a visible head and body")
    raw_head_h = int(hys.max() - hys.min() + 1)
    scale = 640.0 / max(1, raw_head_h)
    scaled_head = cv2.resize(head_rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    scaled_body = cv2.resize(body_rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    center = int(np.median(hxs))
    half = max(8, int(raw_head_h * 0.08))
    chin_rows = np.nonzero(head_alpha[:, max(0, center - half) : center + half].any(axis=1))[0]
    collar_rows = np.nonzero(body_alpha[:, max(0, center - half) : center + half].any(axis=1))[0]
    chin_y = int(chin_rows.max()) if chin_rows.size else int(hys.max())
    under_chin = collar_rows[collar_rows >= chin_y - 4] if collar_rows.size else collar_rows
    collar_y = int(under_chin.min()) if under_chin.size else int(bys.min())
    body_cx = (int(bxs.min()) + int(bxs.max())) / 2
    origin_x = int(round(540 - body_cx * scale))
    origin_y = int(round(1920 - (int(bys.max()) + 1) * scale))
    head_y = int(round(origin_y + (collar_y - chin_y) * scale + 15))
    green = (
        (scaled_head[..., 1] > 120)
        & (scaled_head[..., 1] > scaled_head[..., 0] + 25)
        & (scaled_head[..., 3] > 16)
    )
    eye_rows = np.nonzero(green.any(axis=1))[0]
    eye_y = int(head_y + np.median(eye_rows)) if eye_rows.size else head_y
    head_w = (int(hxs.max()) - int(hxs.min()) + 1) * scale
    shoulder_w = (int(bxs.max()) - int(bxs.min()) + 1) * scale
    print(
        f"chroma head={head_w:.0f}x640 shoulders={shoulder_w:.0f} "
        f"eye_y={eye_y} body_bottom=1920 dock_overlap=15"
    )
    return (
        Image.fromarray(scaled_head),
        Image.fromarray(scaled_body),
        (origin_x, head_y),
        (origin_x, origin_y),
    )


def fit_to_golden_bounds(
    head_img: Image.Image,
    body_img: Image.Image,
) -> tuple[Image.Image, Image.Image, tuple[int, int], tuple[int, int]]:
    """Scale a docked puppet so the head is 640px and the eyes sit at y=580."""
    import cv2

    head_rgba = np.asarray(head_img.convert("RGBA"))
    body_rgba = np.asarray(body_img.convert("RGBA"))
    head_alpha = head_rgba[..., 3]
    body_alpha = body_rgba[..., 3]
    hys, hxs = np.nonzero(head_alpha > 16)
    bys, bxs = np.nonzero(body_alpha > 16)
    if not hys.size or not bys.size:
        raise RuntimeError("golden fit requires a visible head and body")
    raw_head_h = int(hys.max() - hys.min() + 1)
    scale = 640.0 / max(1, raw_head_h)
    scaled_head = cv2.resize(head_rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    scaled_body = cv2.resize(body_rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    green = (
        (scaled_head[..., 1] > 120)
        & (scaled_head[..., 1] > scaled_head[..., 0] + 25)
        & (scaled_head[..., 3] > 16)
    )
    eye_rows = np.nonzero(green.any(axis=1))[0]
    eye_cols = np.nonzero(green.any(axis=0))[0]
    eye_y = int(np.median(eye_rows)) if eye_rows.size else int(round((int(hys.min()) + raw_head_h * 0.42) * scale))
    eye_x = int(np.median(eye_cols)) if eye_cols.size else int(round(np.median(hxs) * scale))
    head_x = 540 - eye_x
    head_y = 580 - eye_y
    center = int(np.median(hxs))
    half = max(8, int(raw_head_h * 0.08))
    chin_rows = np.nonzero(head_alpha[:, max(0, center - half) : center + half].any(axis=1))[0]
    collar_rows = np.nonzero(body_alpha[:, max(0, center - half) : center + half].any(axis=1))[0]
    chin_y = int(chin_rows.max()) if chin_rows.size else int(hys.max())
    under_chin = collar_rows[collar_rows >= chin_y - 4] if collar_rows.size else collar_rows
    collar_y = int(under_chin.min()) if under_chin.size else int(bys.min())
    body_x = head_x
    body_y = int(round(head_y + (chin_y - collar_y) * scale - 15))
    shoulder_w = (int(bxs.max()) - int(bxs.min()) + 1) * scale
    head_w = (int(hxs.max()) - int(hxs.min()) + 1) * scale
    print(
        f"golden head={head_w:.0f}x640 shoulders={shoulder_w:.0f} "
        f"eye_y=580 dock_overlap=15"
    )
    return (
        Image.fromarray(scaled_head),
        Image.fromarray(scaled_body),
        (head_x, head_y),
        (body_x, body_y),
    )


def _slice_chin_only(
    character: Image.Image,
    *,
    overlap_px: int = 15,
) -> tuple[Image.Image, Image.Image]:
    """Keep the bronze chin on the head. The neck column stays on the body."""
    rgba = np.asarray(character.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    if not xs.size:
        raise RuntimeError("cannot slice an empty bust")
    red = rgba[..., 0].astype(np.int16)
    green = rgba[..., 1].astype(np.int16)
    blue = rgba[..., 2].astype(np.int16)
    span = max(1, int(ys.max()) - int(ys.min()))
    face_top = int(ys.min()) + int(span * 0.28)
    face_bottom = int(ys.min()) + int(span * 0.62)
    mid_left = int(xs.min()) + int((xs.max() - xs.min()) * 0.22)
    mid_right = int(xs.max()) - int((xs.max() - xs.min()) * 0.22)
    rows = np.arange(alpha.shape[0])[:, None]
    cols = np.arange(alpha.shape[1])[None, :]
    bronze = (
        (alpha > 16)
        & (red > 120)
        & (red > blue + 25)
        & (green > 70)
        & (green < red + 20)
        & (blue < 170)
        & (rows >= face_top)
        & (rows <= face_bottom)
        & (cols >= mid_left)
        & (cols <= mid_right)
    )
    bronze_y, bronze_x = np.nonzero(bronze)
    if not bronze_y.size:
        raise RuntimeError("chin-only slice found no bronze jaw plate")
    center = int(np.median(bronze_x))
    face_left = int(bronze_x.min())
    face_right = int(bronze_x.max())
    face_half = max(24, (face_right - face_left) // 2)
    curve = np.full(alpha.shape[1], -1, dtype=np.int32)
    for x in range(face_left, face_right + 1):
        column = np.nonzero(bronze[:, x])[0]
        if column.size:
            curve[x] = int(column.max())
    known = np.nonzero(curve >= 0)[0]
    curve[known[0] : known[-1] + 1] = np.interp(
        np.arange(known[0], known[-1] + 1),
        known,
        curve[known],
    ).astype(np.int32)
    pad = int(face_half * 0.35)
    side = int(curve[known[0]])
    curve[max(0, known[0] - pad) : known[0]] = side
    side = int(curve[known[-1]])
    curve[known[-1] + 1 : min(alpha.shape[1], known[-1] + 1 + pad)] = side
    visible = np.where(alpha > 16, 255, 0).astype(np.uint8)
    head_mask = np.zeros(alpha.shape, dtype=np.uint8)
    for x in np.nonzero(curve >= 0)[0]:
        column = np.nonzero(alpha[:, x] > 16)[0]
        kept = column[column <= int(curve[x])]
        if kept.size:
            head_mask[kept, x] = 255
    head_mask = np.bitwise_and(head_mask, visible)
    body_mask = np.bitwise_and(np.bitwise_not(head_mask), visible)
    chin_y = int(np.max(curve))
    band = slice(max(0, chin_y - overlap_px), chin_y + 1)
    body_mask[band, max(0, center - face_half) : center + face_half] = visible[
        band, max(0, center - face_half) : center + face_half
    ]
    return Image.fromarray(rgba * (head_mask[..., None] > 0)), Image.fromarray(rgba * (body_mask[..., None] > 0))


def _despill_bottom_chroma(image: Image.Image) -> Image.Image:
    """Turn leftover chroma-blue fringe on the lower hem into dark ink."""
    rgba = np.asarray(image.convert("RGBA")).copy()
    alpha = rgba[..., 3]
    rows = np.nonzero((alpha > 8).any(axis=1))[0]
    if not rows.size:
        return image
    bottom = int(rows.max())
    band = np.zeros(alpha.shape, dtype=bool)
    band[max(0, bottom - 23) : bottom + 1, :] = True
    red, green, blue = rgba[..., 0].astype(np.int16), rgba[..., 1].astype(np.int16), rgba[..., 2].astype(np.int16)
    fringe = band & (alpha > 0) & (alpha < 250) & (blue > 70) & (blue > red + 25) & (blue > green + 15)
    rgba[fringe, 0] = 0x12
    rgba[fringe, 1] = 0x15
    rgba[fringe, 2] = 0x1C
    rgba[fringe, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def _tight_alpha_crop(image: Image.Image) -> Image.Image:
    """Crop to the opaque alpha bounds before any scale or dock math."""
    import cv2

    rgba = np.asarray(image.convert("RGBA"))
    x, y, width, height = cv2.boundingRect((rgba[..., 3] > 0).astype(np.uint8))
    if width < 1 or height < 1:
        raise RuntimeError("tight crop found no opaque pixels")
    return Image.fromarray(rgba[y : y + height, x : x + width])


def _composite_at(frame: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    """Paste a layer that may extend past the 1080x1920 frame."""
    left = max(0, x)
    top = max(0, y)
    crop = layer.crop((
        max(0, -x),
        max(0, -y),
        min(layer.width, frame.width - x),
        min(layer.height, frame.height - y),
    ))
    if crop.width > 0 and crop.height > 0:
        frame.alpha_composite(crop, (left, top))


def _contrast_prompt(prompt: str, bg_contrast: str) -> str:
    """White stage keeps dark ink separable from the backdrop."""
    proportions = (
        "Charming, prominent, slightly oversized mecha head with vintage anime proportions, "
        "slender compact mecha torso with lean mechanical shoulders. "
        "Bold jet-black anime contour outlines 4 to 8 pixels thick (#12151C). "
    )
    if bg_contrast in {"canonical", "portrait", "bounds", "flow", "approved", "golden", "magenta"}:
        return prompt
    if bg_contrast == "slate_blue":
        return DEEPSEEK_PORTRAIT_PROMPT
    if bg_contrast != "white":
        return f"{prompt} {proportions}"
    stage = prompt.replace("solid pure black background", "solid pure white #FFFFFF background")
    stage = stage.replace("Solid pure black background", "Solid pure white #FFFFFF background")
    stage = stage.replace("pure solid black background", "solid pure white #FFFFFF background")
    return f"{stage} {proportions}"


def _discover_fast_image_model() -> str:
    """Return the Flash image model. Refuse every Gemini Pro image SKU."""
    import os

    from google import genai

    required = "models/gemini-2.5-flash-image"
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for model discovery")
    client = genai.Client(api_key=key)
    names = [getattr(model, "name", "") for model in client.models.list()]
    banned = [name for name in names if "gemini-3-pro-image" in name.lower()]
    match = next((name for name in names if name.endswith("gemini-2.5-flash-image")), "")
    if not match:
        raise RuntimeError(f"{required} is not available on this key")
    if "gemini-3-pro" in match.lower():
        raise RuntimeError(f"refusing prohibited model: {match}")
    print(f"discovered fast image model: {required}")
    if banned:
        print(f"prohibited models ignored: {', '.join(banned)}")
    return required


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, matte, slice, and rig one puppet.")
    parser.add_argument("--character-id", default="")
    parser.add_argument("--canonical-formula", default="")
    parser.add_argument("--portrait-formula", default="")
    parser.add_argument("--bounding-box-mode", default="")
    parser.add_argument("--fast-model-discovery", action="store_true")
    parser.add_argument("--one-click-golden", action="store_true")
    parser.add_argument("--magenta-isolation", action="store_true")
    parser.add_argument("--decoupled-stages", action="store_true")
    parser.add_argument("--generate-body-only", action="store_true")
    parser.add_argument("--run-assembler", action="store_true")
    parser.add_argument("--restore-approved-head", action="store_true")
    parser.add_argument("--freeze-golden-contract", action="store_true")
    parser.add_argument("--head-scale-pct", type=float, default=105)
    parser.add_argument("--body-scale-pct", type=float, default=97)
    parser.add_argument("--body-drop-px", type=int, default=18)
    parser.add_argument("--head-target-h", type=int, default=0)
    parser.add_argument("--head-shift-left", type=int, default=10)
    parser.add_argument("--body-shift-right", type=int, default=0)
    parser.add_argument("--head-rel-shift-left", type=int, default=0)
    parser.add_argument("--preview-name", default="")
    parser.add_argument("--fix-rotation-padding", action="store_true")
    parser.add_argument("--head-drop-px", type=int, default=15)
    parser.add_argument("--body-target-w", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--name", default="")
    parser.add_argument("--casing-color", default="naval-blue")
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
    if args.run_assembler or args.restore_approved_head:
        character_id = args.character_id or "deepseek_cyborg_v3"
        skin = OneShotPuppetFactory().run_assembler(
            character_id,
            head_scale_pct=args.head_scale_pct,
            body_scale_pct=args.body_scale_pct,
            body_drop_px=args.body_drop_px,
            head_target_h=args.head_target_h or None,
            body_target_w=args.body_target_w or None,
            head_shift_left=args.head_shift_left,
            head_drop_px=args.head_drop_px,
            restore_approved_head=args.restore_approved_head,
            fix_rotation_padding=args.fix_rotation_padding,
            body_shift_right=args.body_shift_right,
            head_rel_shift_left=args.head_rel_shift_left,
            preview_name=args.preview_name,
        )
        print(skin)
        return 0
    if args.generate_body_only:
        if not args.character_id:
            raise SystemExit("--character-id is required")
        _discover_fast_image_model()
        skin = OneShotPuppetFactory().create_body_only(args.character_id)
        print(skin)
        return 0
    if args.decoupled_stages:
        if not args.character_id:
            raise SystemExit("--character-id is required")
        _discover_fast_image_model()
        skin = OneShotPuppetFactory().create_decoupled(args.character_id, facing=args.facing)
        print(skin)
        return 0
    if args.magenta_isolation:
        if not args.character_id:
            raise SystemExit("--character-id is required")
        _discover_fast_image_model()
        prompt = f"{DEEPSEEK_CHROMA_PROMPT}\n\nNegative prompt: {DEEPSEEK_NEGATIVE_PROMPT}"
        skin = OneShotPuppetFactory().create_character(
            args.character_id,
            prompt,
            facing=args.facing,
            bg_contrast="magenta",
        )
        print(skin)
        return 0
    if args.one_click_golden:
        if not args.character_id:
            raise SystemExit("--character-id is required")
        _discover_fast_image_model()
        skin = OneShotPuppetFactory().create_character(
            args.character_id,
            ONE_CLICK_MECHA_PROMPT,
            facing=args.facing,
            bg_contrast="golden",
        )
        print(skin)
        return 0
    if args.fast_model_discovery:
        if not args.character_id:
            raise SystemExit("--character-id is required")
        _discover_fast_image_model()
        skin = OneShotPuppetFactory().create_character(
            args.character_id,
            DEEPSEEK_FINAL_PRODUCTION_PROMPT,
            facing="left",
            bg_contrast="approved",
        )
        print(skin)
        return 0
    if args.bounding_box_mode:
        skin = OneShotPuppetFactory().create_character(
            args.bounding_box_mode,
            DEEPSEEK_TIN_MAN_PROMPT,
            facing="left",
            bg_contrast="bounds",
        )
        print(skin)
        return 0
    if args.portrait_formula:
        skin = OneShotPuppetFactory().create_character(
            args.portrait_formula,
            PORTRAIT_MASTER_PROMPT,
            facing="left",
            bg_contrast="portrait",
        )
        print(skin)
        return 0
    if args.canonical_formula:
        skin = OneShotPuppetFactory().create_character(
            args.canonical_formula,
            DEEPSEEK_CANONICAL_PROMPT,
            facing="left",
            bg_contrast="canonical",
        )
        print(skin)
        return 0
    if args.name:
        if not args.character_id:
            raise SystemExit("--character-id is required")
        prompt = MASTER_GHIBLI_MECHA_TEMPLATE.format(
            name=args.name,
            facing=args.facing,
            casing_color=args.casing_color,
        )
        skin = OneShotPuppetFactory().create_character(
            args.character_id,
            prompt,
            args.facing,
            bg_contrast="flow",
        )
        print(skin)
        return 0
    if not args.character_id:
        raise SystemExit("--character-id is required")
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
