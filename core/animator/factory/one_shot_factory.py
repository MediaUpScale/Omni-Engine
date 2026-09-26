"""One-shot puppet factory. BiRefNet alpha is saved exactly as the model returns it.

Usage:
    python -m core.animator.factory.one_shot_factory --character-id deepseek_cyborg_v3 --facing left
"""
from __future__ import annotations

import argparse
import json
import shutil
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

CORRECT_TEXT_ON_FLIPPED_HEAD_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic. "
    "Square 1:1 close-up portrait of the mecha robot head from the reference image. "
    "CRITICAL: The character is ALREADY facing towards SCREEN-RIGHT (Viewer's Right) in the reference image. "
    "Maintain the EXACT SAME head pose, dark naval-blue dome helmet, bronze chin shield, concentric emerald eyes, "
    "and antique brass ear dials exactly as positioned in the reference image. "
    "TASK: Ensure the rectangular brass forehead nameplate is clearly engraved with the text 'DEEPSEEK' "
    "reading normally from LEFT TO RIGHT in English (NO mirrored letters, NO inverted spelling). "
    "Floating head only. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

HEAD_SCREEN_RIGHT_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic. "
    "Square 1:1 close-up portrait of the mecha head of robot 'DEEPSEEK'. "
    "CRITICAL ORIENTATION: ANGLED 25 DEGREES TOWARDS SCREEN-RIGHT (THE VIEWER'S RIGHT SIDE, LOOKING TOWARDS THE RIGHT EDGE OF THE FRAME). "
    "The robot's gaze, face, and chin point toward the right-hand side of the canvas. "
    "Maintain 100% design identity with the reference image: smooth dark naval-blue dome helmet, "
    "circular brass ear dials, glowing emerald-green concentric ring optical sensor lenses, and smooth curved bronze chin shield plate. "
    "On its forehead, a clean rectangular antique brass nameplate engraved 'DEEPSEEK' in clearly readable English text (from left to right, NO inverted letters). "
    "Floating head only, ending cleanly at the lower bronze chin contour. Strictly NO neck, NO body, NO human mouth, NO human nose. "
    "Rendered against a solid flat pure magenta background (#FF00FF). --ar 1:1"
)

HEAD_FRONTAL_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic. "
    "Square 1:1 close-up portrait of the mecha robot head from the reference image. "
    "CRITICAL ORIENTATION: PURE FRONTAL VIEW (0 DEGREES, FACING DIRECTLY TOWARD THE CAMERA/VIEWER). "
    "Perfect facial symmetry: smooth dark naval-blue dome helmet with antique brass circular ear dials visible on BOTH left and right sides. "
    "On its forehead, a centered rectangular brass nameplate engraved 'DEEPSEEK' in clean legible English text. "
    "Glowing concentric emerald-green optical sensor lenses staring straight ahead into the camera. "
    "Centered smooth blank curved bronze chin shield plate. "
    "Floating head only, ending at the lower chin contour. Strictly NO neck, NO body, NO human mouth, NO human nose. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

BODY_FRONTAL_ROUNDED_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, "
    "clean flat watercolor cel shading. "
    "Vertical 9:16 portrait of a charming headless retro-futuristic tin-plate robot chassis, "
    "PURE SYMMETRICAL FRONTAL VIEW (0 DEGREES FACING DIRECTLY FORWARD). "
    "TORSO: Charming rounded convex barrel-shaped chassis in naval-blue plating with gently bulging curved flanks "
    "(vintage Laputa tin-can aesthetic, matching the plump cylindrical volume of the profile views). "
    "At center chest, a vintage circular boiler pressure gauge with brass bezel and pointer needle. "
    "COLLAR: Wide circular antique brass collar rim centered at top, with dark corrugated neck socket column. "
    "ARMS: Slender tubular mechanical arms with round brass shoulder ball-joints, resting snugly FLUSH "
    "and straight against the curved sides of the torso, with NO empty air gaps between the arms and the body. "
    "Grounded base: solid torso armor completely filling lower canvas and bleeding off bottom boundary (y=1920). "
    "Headless body only. NO head, NO face, NO asymmetry, zero magenta bleed. "
    "Solid flat pure magenta background (#FF00FF). --ar 9:16"
)

BODY_FRONTAL_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, clean flat watercolor cel shading. "
    "Vertical 9:16 portrait of a charming headless retro-futuristic tin-plate robot chassis, "
    "PURE FRONTAL SYMMETRICAL VIEW (0 DEGREES FACING DIRECTLY FORWARD). "
    "COLLAR: Centered wide circular antique brass collar rim at top center with short dark corrugated neck column socket. "
    "TORSO: Smooth symmetrical cylindrical naval-blue barrel chassis. "
    "At center chest, a vintage circular boiler pressure gauge with brass bezel and pointer needle. "
    "Slender tubular mechanical arms resting flush and straight against both flanks, with brass shoulder ball-joints. "
    "Grounded base: solid torso armor completely filling lower canvas and bleeding off the bottom frame boundary (y=1920). "
    "Headless body only. NO head, NO face, NO asymmetry, zero magenta bleed. "
    "Solid flat pure magenta background (#FF00FF). --ar 9:16"
)

HEAD_RIGHT_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic. "
    "Square 1:1 close-up portrait of the mecha head of robot 'DEEPSEEK', "
    "CRITICAL ORIENTATION: ANGLED 25 DEGREES FACING TO THE RIGHT. "
    "Maintain 100% character design identity with the reference image: identical smooth dark naval-blue dome helmet, "
    "brass circular ear dials, glowing emerald-green concentric ring optical sensor lenses, and smooth curved bronze chin shield plate. "
    "On its forehead, a clean rectangular antique brass nameplate engraved 'DEEPSEEK' in clearly readable English text (from left to right, NO inverted text). "
    "Floating head only, ending at the lower chin contour. Strictly NO neck, NO body, NO human mouth, NO human nose. "
    "Rendered against a solid flat pure magenta background (#FF00FF). --ar 1:1"
)

BODY_RIGHT_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, clean flat watercolor cel shading. "
    "Vertical 9:16 portrait of a charming headless retro-futuristic tin-plate robot chassis, "
    "DYNAMIC ASYMMETRICAL THREE-QUARTER VIEW ANGLED 25 DEGREES TO THE RIGHT. "
    "PERSPECTIVE: The right shoulder is in the foreground closer to the camera, while the left shoulder is recessed in perspective. "
    "Wide circular antique brass collar rim tilted in 3/4 oval perspective at top center with dark corrugated neck column. "
    "Cylindrical naval-blue barrel chassis with circular boiler pressure gauge and rivets. "
    "Tubular mechanical arms flush against flanks. Torso solidly fills lower canvas down to y=1920. "
    "Headless body only. NO head, NO face, NO front symmetry, zero magenta bleed. "
    "Solid flat pure magenta background (#FF00FF). --ar 9:16"
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


CHATGPT_HEAD_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic, clean flat watercolor shading. "
    "Square 1:1 close-up portrait of the mecha head of a dignified cybernetic robot named 'CHATGPT', "
    "angled 25 degrees facing to the left. "
    "HEAD: Elegant polished ivory-cream ceramic dome helmet with muted celadon sage-green side panels and satin brass trim. "
    "Forehead clean rectangular brass nameplate neatly engraved 'CHATGPT'. "
    "Twin prominent circular camera-iris aperture lenses glowing with soft calm mint-cyan optical sensor light. "
    "LOWER FACE: Smooth, blank curved antique brushed-brass chin shield plate (clean untextured metallic surface for animation). "
    "CRITICAL: Floating head only, ending at lower chin contour. Strictly NO neck, NO body, NO human mouth, NO nose. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

CHATGPT_BODY_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, clean flat watercolor shading. "
    "Vertical 9:16 portrait of a charming headless retro-futuristic robot chassis, "
    "DYNAMIC ASYMMETRICAL THREE-QUARTER VIEW ANGLED 25 DEGREES TO THE LEFT. "
    "PERSPECTIVE: Left shoulder in foreground, right shoulder recessed in perspective. "
    "COLLAR: Symmetrical flat satin-brass collar rim seated flat against the shoulder armor with dark corrugated neck socket. "
    "TORSO: Sleek rounded tin-plate chassis in antique ivory-cream and muted celadon sage-green plating with brass pinstripe seams. "
    "CHEST ACCESSORY: At center chest, an illuminated horizontal glass radio frequency tuning dial window with brass bezel, "
    "showing delicate analog tuning scale numbers and a subtle amber pointer needle. "
    "NO circular gauge, NO vertical vacuum tube. Slender tubular mechanical arms flush against both flanks. "
    "Grounded continuous base: solid torso armor completely filling lower canvas and bleeding off bottom boundary (y=1920). "
    "Headless body only. NO head, NO face, NO superhero pecs, zero magenta bleed. "
    "Pristine museum factory condition. Nostalgic warm anime lighting, solid flat pure magenta background (#FF00FF). --ar 9:16"
)

TIN_MAN_A_HEAD_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki mecha aesthetic. "
    "Square 1:1 close-up of a charming vintage tin-man robot head named 'CHATGPT', angled 25 degrees left. "
    "Weathered tin-plate cranium with visible rows of tiny rivets, exposed rotating brass watch gear mechanism on temple dials. "
    "Rectangular engraved brass plate 'CHATGPT'. Twin warm glowing circular porthole eyes with brass bezels. "
    "High smooth blank curved bronze chin plate with NO mouth, NO nose. Floating head only. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
TIN_MAN_A_BODY_PROMPT = (
    "Studio Ghibli 1990s anime mecha cel animation. "
    "Vertical 9:16 portrait of a vintage tin-plate mecha chassis facing 25 degrees left. "
    "Riveted tin-man cylindrical chest plating with brass collar rim. "
    "In center chest, an ornate circular open glass window showing an exposed brass clockwork mechanical gear heart with copper pinions. "
    "Slender tubular tin arms with exposed gear elbow joints. Grounded base at y=1920. Headless body only. "
    "Solid flat pure magenta background (#FF00FF). --ar 9:16"
)
TIN_MAN_B_HEAD_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki mecha aesthetic. "
    "Square 1:1 close-up of an industrial retro-futuristic tin robot head named 'CHATGPT', angled 25 degrees left. "
    "Faceted tin-plate helmet with brass corner brackets and rivets. Forehead plate engraved 'CHATGPT'. "
    "Deep-set glowing turquoise circular ocular lenses. Smooth blank rectangular bronze jaw shield plate. "
    "Floating head only. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
TIN_MAN_B_BODY_PROMPT = (
    "Studio Ghibli 1990s anime mecha cel animation. "
    "Vertical 9:16 portrait of an industrial tin-man robot chassis facing 25 degrees left. "
    "Heavy riveted tin chest plates with horizontal art-deco brass ventilation louvers and exposed copper steam pipes. "
    "Round brass shoulder ball-joints with exposed teeth gears. Grounded solid base at y=1920. Headless body only. "
    "Solid flat pure magenta background (#FF00FF). --ar 9:16"
)
TIN_MAN_C_HEAD_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic. "
    "Square 1:1 close-up of a classic 1950s tin-toy robot head named 'CHATGPT', angled 25 degrees left. "
    "Rounded tin dome with a subtle vertical brass antenna fin on top, circular ear dials with exposed cogs. "
    "Brass plate 'CHATGPT'. Dual glowing amber-cyan circular glass bulb eyes. Smooth blank curved bronze chin shield. "
    "Floating head only. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
TIN_MAN_C_BODY_PROMPT = (
    "Studio Ghibli 1990s anime mecha cel animation. "
    "Vertical 9:16 portrait of a classic 1950s tin-toy robot chassis facing 25 degrees left. "
    "Stepped cylindrical tin-plate barrel chest with antique brass collar. "
    "Center chest features an exposed dual-gear kinetic escapement mechanism with rotating bronze cogs. "
    "Articulated tubular arms flush against sides. Grounded base at y=1920. Headless body only. "
    "Solid flat pure magenta background (#FF00FF). --ar 9:16"
)

CLEAN_A_HEAD_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, cel animation, clean flat watercolor shading. "
    "Square close-up of a dignified vintage mechanical robot named 'CHATGPT', facing 25 degrees left. "
    "Pristine smooth matte platinum slate-grey chassis with antique gold rivets and trim. "
    "On its FOREHEAD, a sturdy rectangular brass metal plate with text 'CHATGPT'. "
    "Temple ear dials with delicate exposed gold watch gears. Large glowing mint-cyan concentric ring eyes. "
    "Clean blank curved bronze chin shield with NO mouth, NO nose. Floating head only. "
    "Museum-grade factory condition, zero rust. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
CLEAN_A_BODY_PROMPT = (
    "Studio Ghibli 1990s anime mecha cel animation. "
    "Vertical 9:16 portrait of a dignified tin-plate robot chassis facing 25 degrees left. "
    "Pristine smooth matte platinum slate-grey torso plate matching the head, with simple antique gold collar rim and tiny rivets. "
    "Clean solid chest armor with NO gadgets, NO clocks, NO glass doors. "
    "Broad mechanical shoulders and slender tubular arms flush against flanks down to y=1920. "
    "Headless body only. Zero rust, zero magenta bleed. Solid flat pure magenta background (#FF00FF). --ar 9:16"
)
CLEAN_B_HEAD_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, cel animation, clean flat watercolor shading. "
    "Square close-up of a dignified vintage mechanical robot named 'CHATGPT', facing 25 degrees left. "
    "Smooth matte celadon sage-green chassis with antique satin-brass rivets and trim. "
    "On its FOREHEAD, a sturdy rectangular brass metal plate with text 'CHATGPT'. "
    "Circular ear dials with subtle gear accents. Glowing electric-mint concentric ring optical sensor eyes. "
    "Clean blank curved bronze chin shield with NO mouth, NO nose. Floating head only. "
    "Flawless factory condition, zero rust. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
CLEAN_B_BODY_PROMPT = (
    "Studio Ghibli 1990s anime mecha cel animation. "
    "Vertical 9:16 portrait of a dignified robot chassis facing 25 degrees left. "
    "Smooth matte celadon sage-green torso plate matching the head, with simple satin-brass collar rim and subtle pinstripes. "
    "Clean solid continuous chest armor with NO interior clocks, NO gadgets. "
    "Articulated tubular mechanical arms flush against flanks down to y=1920. "
    "Headless body only. Zero rust. Solid flat pure magenta background (#FF00FF). --ar 9:16"
)
CLEAN_C_HEAD_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, cel animation, clean flat watercolor shading. "
    "Square close-up of a dignified vintage mechanical robot named 'CHATGPT', facing 25 degrees left. "
    "Smooth matte dark gunmetal-grey chassis with warm antique bronze rivets and trim. "
    "On its FOREHEAD, a rectangular brass plate with text 'CHATGPT'. Circular brass ear dials. "
    "Soft glowing turquoise-green concentric optical sensor eyes. "
    "Clean blank curved bronze chin shield with NO mouth, NO nose. Floating head only. "
    "Pristine factory condition, zero rust. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
CLEAN_C_BODY_PROMPT = (
    "Studio Ghibli 1990s anime mecha cel animation. "
    "Vertical 9:16 portrait of a dignified robot chassis facing 25 degrees left. "
    "Smooth matte dark gunmetal-grey torso plate matching the head, with antique brass collar rim and small rivets. "
    "Clean solid chest armor with NO gadgets, NO open doors. "
    "Tubular mechanical arms flush against flanks bleeding off at y=1920. "
    "Headless body only. Zero rust. Solid flat pure magenta background (#FF00FF). --ar 9:16"
)

BLANK_A_HEAD_PROMPT = (
    "Studio Ghibli 1980s vintage anime mecha cel animation, Hayao Miyazaki retro-futurism aesthetic, clean flat watercolor shading. "
    "Square 1:1 close-up of a charming cybernetic communicator robot named 'CHATGPT', angled 25 degrees left. "
    "HEAD: Compact rounded tin-plate helmet in warm off-white, adorned with delicate etched copper circuit traces "
    "and circular ear dials housing exposed miniature rotating brass watch gears. "
    "Forehead sturdy rectangular brass plate neatly engraved 'CHATGPT'. "
    "Twin glowing emerald-cyan camera-aperture iris lenses with delicate aperture blades. "
    "LOWER FACE: Smooth, flat curved bronze jawplate with tiny exposed side hydraulic pivot bolts at the jaw corners. "
    "CRITICAL: ABSOLUTELY NO human nose, NO metal nose, NO mouth drawn. Lower face is 100% blank, flat and smooth. "
    "Floating head only, zero neck drawn. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
BLANK_A_BODY_PROMPT = (
    "Studio Ghibli 1980s vintage anime mecha cel animation. "
    "Vertical 9:16 portrait of a charming tin-plate robot chassis facing 25 degrees left. "
    "Matching warm off-white tin-can torso with flat antique brass collar rim and copper wire conduits. "
    "Clean solid continuous chest armor with NO gadgets, NO clocks, NO open doors. "
    "Slender tubular mechanical arms flush against flanks down to y=1920. "
    "Headless body only. Flawless factory condition, zero rust. Solid flat pure magenta background (#FF00FF). --ar 9:16"
)
BLANK_B_HEAD_PROMPT = (
    "Vintage 1950s retro-futuristic anime mecha, Studio Ghibli cel animation style, clean flat watercolor fills. "
    "Square 1:1 close-up of a charming laboratory automaton robot named 'CHATGPT', angled 25 degrees left. "
    "HEAD: Domed tin-plate helmet in pale sage-green with twin miniature brass gear-driven dials mounted on the ears "
    "and subtle brass circuit seams across the crown. Forehead plate engraved 'CHATGPT'. "
    "Twin circular glowing mint glass porthole lenses set in antique brass bezels. "
    "LOWER FACE: Angular flat brass faceplate covering the jaw, completely smooth, untextured and blank. "
    "CRITICAL: ABSOLUTELY ZERO nose, ZERO nostrils, ZERO mouth, ZERO lips. Completely clean metallic canvas. "
    "Floating head only, zero neck. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
BLANK_B_BODY_PROMPT = (
    "Vintage 1950s retro-futuristic anime mecha, Studio Ghibli cel animation style. "
    "Vertical 9:16 portrait of a charming robot chassis facing 25 degrees left. "
    "Pristine pale sage-green tin-plate chest plate matching the head, with flat brass collar rim "
    "and exposed copper joint rings. Clean solid chest armor with tiny brass rivets. "
    "Slender tubular arms flush along the borders bleeding off at y=1920. "
    "Headless body only. Zero rust. Solid flat pure magenta background (#FF00FF). --ar 9:16"
)
BLANK_C_HEAD_PROMPT = (
    "Studio Ghibli 1990s anime mecha cel animation, crisp dark ink contours, flat pastel shading. "
    "Square 1:1 close-up of a dignified calculating robot named 'CHATGPT', angled 25 degrees left. "
    "HEAD: Elegant two-tone helmet in antique ivory-cream and muted celadon sage-green plating with exposed miniature "
    "brass clockwork escapement gears visible on the side temples. Rectangular brass nameplate engraved 'CHATGPT'. "
    "Calm, brilliant glowing cyan optical aperture sensor eyes. "
    "LOWER FACE: Clean, smooth curved antique bronze chin shield plate, perfectly flat, blank and untextured. "
    "CRITICAL: STRICTLY NO nose, NO nostrils, NO human mouth, NO lips. "
    "Floating head only, zero neck drawn. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
BLANK_C_BODY_PROMPT = (
    "Studio Ghibli 1990s anime mecha cel animation, crisp line art, flat pastel shading. "
    "Vertical 9:16 portrait of a dignified robot chassis facing 25 degrees left. "
    "Symmetrical ivory and muted sage-green tin torso with flat satin-brass collar rim and articulated shoulder gear pivots. "
    "Clean solid chest armor plate down to bottom edge (y=1920). "
    "Headless body only. Flawless museum-grade finish, zero rust. Solid flat pure magenta background (#FF00FF). --ar 9:16"
)

PROVEN_A_HEAD_PROMPT = (
    "Studio Ghibli vintage anime mecha cel animation, 1980s retro-futuristic technology aesthetic, clean flat watercolor shading. "
    "Square 1:1 close-up of a charming cybernetic communicator robot named 'CHATGPT', facing 25 degrees to the left. "
    "HEAD: Compact rounded tin-plate helmet in warm off-white with delicate brass ventilation mesh and knobs on the temple dials. "
    "Forehead plate engraved 'CHATGPT'. Glowing emerald-cyan camera-iris aperture eyes. "
    "LOWER FACE: Smooth curved bronze jawplate with tiny exposed side hydraulic pivot joints at the jaw corners, "
    "center area 100% flat, clean and blank with NO mouth, NO nose. Floating head only. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
PROVEN_A_BODY_PROMPT = (
    "Studio Ghibli vintage anime mecha cel animation, 1980s retro-futuristic technology aesthetic. "
    "Vertical 9:16 portrait of a charming cybernetic robot chassis facing 25 degrees to the left. "
    "Matching off-white and bronze tin-can torso with flat collar, slender tubular arms descending to bottom frame edge. "
    "Grounded base at y=1920. Headless body only. Pristine condition, zero rust. "
    "Solid flat pure magenta background (#FF00FF). --ar 9:16"
)
PROVEN_B_HEAD_PROMPT = (
    "Studio Ghibli vintage anime mecha cel animation, 1980s retro-futuristic technology aesthetic, nostalgic warm lighting. "
    "Square 1:1 close-up of a charming cybernetic robot named 'CHATGPT', facing 25 degrees left. "
    "HEAD: Sleek ivory-cream tin-plate cranium with prominent brass acoustic ear dials featuring volume knobs. "
    "Rectangular brass forehead banner neatly inscribed 'CHATGPT'. "
    "Circular glowing mint-cyan camera-shutter optic lenses. "
    "LOWER FACE: Clean, smooth curved antique bronze chin shield plate, 100% flat and blank (NO mouth, NO nose). "
    "Floating head only. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
PROVEN_B_BODY_PROMPT = (
    "Studio Ghibli vintage anime mecha cel animation, 1980s retro-futurism. "
    "Vertical 9:16 portrait of an off-white and bronze mecha chassis facing 25 degrees left. "
    "Smooth ivory chest armor plate with flat antique brass collar rim, exposed hydraulic neck socket, "
    "slender tubular arms flush against sides down to bottom boundary (y=1920). "
    "Headless body only, zero rust. Solid flat pure magenta background (#FF00FF). --ar 9:16"
)
PROVEN_C_HEAD_PROMPT = (
    "Studio Ghibli vintage anime mecha cel animation, crisp dark ink contours, flat pastel watercolor fills. "
    "Square 1:1 close-up of an antique cybernetic communicator robot named 'CHATGPT', angled 25 degrees left. "
    "HEAD: Polished off-white ceramic-tin helmet with antique brass speaker mesh temple grills. "
    "Sturdy brass plate engraved 'CHATGPT' across forehead. "
    "Brilliant glowing emerald-cyan aperture porthole eyes. "
    "LOWER FACE: Generous smooth blank curved bronze jawplate shield, perfectly flat with NO mouth, NO nose. "
    "Floating head only, zero neck. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)
PROVEN_C_BODY_PROMPT = (
    "Studio Ghibli vintage anime mecha cel animation. "
    "Vertical 9:16 portrait of an off-white tin-plate mecha chassis facing 25 degrees left. "
    "Continuous solid off-white and bronze torso armor plates, flat brass collar, tubular arms bleeding off y=1920. "
    "Headless body only. Solid flat pure magenta background (#FF00FF). --ar 9:16"
)

CLAUDE_HEAD_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic, clean flat watercolor shading. "
    "Square 1:1 close-up portrait of the mecha head of a dignified cybernetic robot named 'CLAUDE', "
    "angled 25 degrees facing to the left. "
    "HEAD: Smooth rounded warm terracotta-orange dome helmet with antique brass circular ear dials and perimeter rivets. "
    "Forehead sturdy rectangular brass nameplate engraved 'CLAUDE'. "
    "Calm, highly intelligent glowing warm-amber concentric ring optical sensor lenses. "
    "Smooth blank curved bronze chin shield plate (clean metallic surface, NO human mouth, NO nose, NO smile). "
    "CRITICAL: Floating head only, ending cleanly at the lower bronze chin contour. ABSOLUTELY ZERO NECK DRAWN. "
    "Strictly NO human features, NO pink/magenta on character. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

CLAUDE_SCHOLAR_HEAD_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, clean flat watercolor cel shading. "
    "Square 1:1 close-up portrait of the mecha head of a dignified antique scholar robot named 'CLAUDE', "
    "angled 25 degrees facing to the left. "
    "HEAD: Refined antique terracotta-orange mecha cranium with polished brass trim. "
    "On the temples/sides, exposed miniature precision brass clockwork watch gears and intricate pinions. "
    "Engraved rectangular brass banner across forehead neatly inscribed 'CLAUDE'. "
    "Calm, thoughtful glowing warm-amber dual optical sensor lenses framed in ornate antique brass monocle bezels. "
    "LOWER FACE: Smooth, solid, blank curved bronze chin shield plate with a 100% CLEAN UNTEXTURED POLISHED METALLIC SURFACE. "
    "CRITICAL CONSTRAINTS: Absolutely NO vertical grilles, NO speaker slots, NO mouth slits, NO human mouth, NO nose. "
    "A clean, blank, generous bronze chin plate ready for animation. Floating head only, ending cleanly at the lower bronze chin contour. ABSOLUTELY ZERO NECK DRAWN. "
    "Strictly NO human features, NO pink/magenta bleed. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

CLAUDE_SMOOTH_CHIN_HEAD_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic, clean flat watercolor shading. "
    "Square 1:1 close-up portrait of the mecha head of robot 'CLAUDE', angled 25 degrees facing to the left. "
    "HEAD: Refined antique terracotta-orange mecha cranium with polished brass trim. "
    "On the temples/sides, exposed miniature precision brass clockwork watch gears and intricate pinions. "
    "Engraved rectangular brass banner across forehead neatly inscribed 'CLAUDE'. "
    "Calm, thoughtful glowing warm-amber dual optical sensor lenses framed in antique brass monocle bezels. "
    "LOWER FACE: Smooth, solid, blank curved bronze chin shield plate with a 100% CLEAN UNTEXTURED POLISHED METALLIC SURFACE. "
    "CRITICAL CONSTRAINTS: Absolutely NO vertical grilles, NO speaker slots, NO mouth slits, NO human mouth, NO nose. "
    "A clean, blank, generous bronze chin plate ready for animation. Floating head only, zero neck drawn. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

SUBTLE_HEAD_A_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, cel animation, clean flat watercolor shading. "
    "Square close-up of a dignified cybernetic robot facing 25 degrees to the left. "
    "HEAD: Chiseled terracotta mecha helmet with subtle beveled angular panels (less rounded, structured mecha volume), "
    "antique brass ear dials and 'CLAUDE' brass nameplate. "
    "Soft glowing warm amber lenses. Clean blank bronze lower faceplate with no mouth. "
    "Floating head only, ending at chin contour. Strictly NO neck, NO body, NO human mouth, NO nose. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

SUBTLE_HEAD_B_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, cel animation, clean flat watercolor shading. "
    "Square close-up of a dignified cybernetic robot facing 25 degrees to the left. "
    "HEAD: Terracotta mecha helmet with a subtle flat brow visor brim (structured retro anime silhouette, not a round ball), "
    "brass ear dials and 'CLAUDE' brass nameplate. "
    "Soft glowing warm amber lenses. Clean blank bronze lower faceplate with no mouth. "
    "Floating head only, ending at chin contour. Strictly NO neck, NO body, NO human mouth, NO nose. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

SUBTLE_HEAD_C_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, cel animation, clean flat watercolor shading. "
    "Square close-up of a dignified cybernetic robot facing 25 degrees to the left. "
    "HEAD: Gently tapered angular terracotta mecha helmet with clean geometric facet lines, "
    "brass ear dials and 'CLAUDE' brass nameplate. "
    "Soft glowing warm amber lenses. Clean blank bronze lower faceplate with no mouth. "
    "Floating head only, ending at chin contour. Strictly NO neck, NO body, NO human mouth, NO nose. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

HEAD_A_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, cel animation, clean flat watercolor shading. "
    "Square close-up of a dignified cybernetic robot facing 25 degrees to the left. "
    "HEAD: Tall vertically elongated oval dome helmet in warm terracotta plating (high forehead crown, antique diver automaton silhouette). "
    "Brass circular ear dials with delicate mechanical wires. Ornate vintage rectangular brass plate engraved 'CLAUDE'. "
    "Two prominent calm circular porthole optical sensor lenses glowing with soft warm amber light. "
    "LOWER FACE: Tall, smooth curved bronze lower chin shield plate with generous blank metallic surface and NO mouth. "
    "Floating head only, ending at lower chin contour. Strictly NO neck, NO body, NO human mouth, NO nose. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

HEAD_B_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, cel animation, clean flat watercolor shading. "
    "Square close-up of a dignified cybernetic robot facing 25 degrees to the left. "
    "HEAD: High cylindrical terracotta mecha helmet with vertical antique brass riveted reinforcement straps across the dome. "
    "Sturdy rectangular brass nameplate engraved 'CLAUDE'. "
    "Two large circular ocular sensor dials with concentric glowing amber glass lenses. "
    "LOWER FACE: High, smooth blank curved bronze jawplate shield (completely untextured flat metal for animation). "
    "Tall dignified head proportions, zero neck. Strictly NO human mouth, NO nose. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

HEAD_C_PROMPT = (
    "Studio Ghibli 1990s anime mecha portrait, Hayao Miyazaki retro-futurism aesthetic, cel animation, clean flat watercolor shading. "
    "Square close-up of a dignified cybernetic robot facing 25 degrees to the left. "
    "HEAD: Tall elongated rounded terracotta helmet dome with antique brass rim brow trim and circular side dials. "
    "Clean brass nameplate engraved 'CLAUDE'. "
    "Soft glowing warm amber concentric circular optic lenses set in deep dark visor recesses. "
    "LOWER FACE: Smooth, blank curved antique bronze chin shield plate with generous open canvas. "
    "Floating head only, tall vertical silhouette, zero neck drawn. "
    "Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

CLAUDE_SCHOLAR_BODY_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, clean flat watercolor cel shading. "
    "Vertical 9:16 portrait of a charming headless retro-futuristic tin-plate robot chassis, "
    "DYNAMIC ASYMMETRICAL THREE-QUARTER VIEW ANGLED 25 DEGREES TO THE LEFT. "
    "PERSPECTIVE: Left shoulder in foreground closer to camera, right shoulder recessed in perspective. "
    "COLLAR: Flat, low-profile flush antique brass rim seated flat against the shoulder armor with dark corrugated neck column socket. "
    "CRITICAL: ABSOLUTELY NO upturned flared funnel, NO trumpet cone, NO hovering Saturn ring. Low-profile flat collar only. "
    "CHEST: Dignified matte terracotta-orange chest plating with antique brass piping. "
    "At center chest, a protected vertical glass vacuum tube (nixie tube) glowing with warm amber filaments and visible tiny brass gears. "
    "NO circular pressure gauge. Slender articulated tubular mechanical arms resting flush against the sides. "
    "Grounded base: solid torso armor completely filling lower canvas and bleeding off the bottom border (y=1920). "
    "Headless body only. NO head, NO face, NO superhero pecs, NO battle damage, zero magenta bleed. "
    "Pristine museum factory condition. Warm nostalgic anime lighting, solid flat pure magenta background (#FF00FF). --ar 9:16"
)

CLAUDE_BODY_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki retro-futurism aesthetic, clean flat watercolor shading. "
    "Vertical 9:16 portrait of a charming headless retro-futuristic tin-plate robot chassis, "
    "DYNAMIC ASYMMETRICAL THREE-QUARTER VIEW ANGLED 25 DEGREES TO THE LEFT. "
    "PERSPECTIVE: Left shoulder in foreground closer to camera, right shoulder recessed in perspective. "
    "COLLAR: Wide circular antique brass collar rim tilted in 3/4 perspective at top center with dark corrugated neck socket. "
    "TORSO: Smooth curved cylindrical warm terracotta-orange tin-can barrel chassis (Laputa vintage aesthetic) with antique brass trim. "
    "At center chest, a vintage circular antique bronze boiler pressure gauge with pointer needle and tiny rivets. "
    "ARMS: Slender tubular mechanical arms with brass elbow ball-joints resting flush against the sides of the torso. "
    "Grounded base: solid torso armor completely filling lower canvas and bleeding off bottom boundary (y=1920). "
    "Headless body only. NO head, NO face, NO superhero pecs, NO battle damage, zero magenta bleed. "
    "Pristine factory condition. Warm nostalgic anime lighting, solid flat pure magenta background (#FF00FF). --ar 9:16"
)

CLAUDE_RIGHT_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic. "
    "Square 1:1 close-up portrait of the mecha robot head from the reference image. "
    "CRITICAL: The character is ALREADY facing towards SCREEN-RIGHT (Viewer's Right) in the reference image. "
    "Maintain the EXACT SAME head pose, terracotta dome with vertical brass riveted straps, "
    "glowing warm amber porthole lenses, and smooth curved bronze chin shield. "
    "TASK: Ensure the rectangular brass forehead nameplate is clearly engraved with the text 'CLAUDE' "
    "reading normally from LEFT TO RIGHT in English (NO mirrored letters). "
    "Floating head only. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

CLAUDE_FRONT_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic. "
    "Square 1:1 close-up portrait of the mecha robot head from the reference image. "
    "CRITICAL ORIENTATION: PURE FRONTAL VIEW (0 DEGREES, FACING DIRECTLY TOWARD THE CAMERA). "
    "Perfect symmetry: terracotta helmet with vertical antique brass riveted straps, "
    "centered forehead brass nameplate neatly engraved 'CLAUDE'. "
    "Two prominent glowing amber porthole optical lenses staring straight ahead. "
    "Centered smooth blank curved bronze chin shield plate with NO mouth, NO nose. "
    "Floating head only, zero neck. Solid flat pure magenta background (#FF00FF). --ar 1:1"
)

FRONT_BODY_NECK_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic, clean flat watercolor cel shading. "
    "Vertical 9:16 portrait of the retro-futuristic robot chassis from the reference image, "
    "PURE FRONTAL SYMMETRICAL VIEW (0 DEGREES FACING DIRECTLY FORWARD). "
    "NECK: A TALL, ELONGATED, dark charcoal accordion neck column with many deep horizontal bellows ridges, "
    "rising high above the shoulders exactly like the reference image. "
    "The corrugated neck is a real mechanical column, long enough to seat a robot head on top of it. "
    "It sits in the center of a round antique brass collar ring. "
    "ABSOLUTELY NOT a flush hole, NOT a missing neck, NOT a ghost stump. "
    "TORSO: Continuous solid terracotta mecha barrel matching the reference. "
    "CHEST: slender compact vertical nixie tube, small, matching the reference tube. "
    "Long slender arms bleed off the bottom of the frame. Headless body only. "
    "Solid flat pure magenta background (#FF00FF). --ar 9:16"
)

FRONT_BODY_GROUNDED_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic, clean flat watercolor cel shading. "
    "Vertical 9:16 portrait of the retro-futuristic robot chassis from the reference image, "
    "PURE FRONTAL SYMMETRICAL VIEW (0 DEGREES FACING DIRECTLY FORWARD). "
    "TORSO: Continuous solid terracotta mecha barrel chassis matching the exact styling of the reference image. "
    "CHEST ACCESSORY: At center chest, an ornate vertical glass vacuum tube (nixie tube) that is SLENDER, COMPACT, "
    "and proportionally SMALL (matching the exact height and width of the nixie tube in the reference image). "
    "COLLAR: Symmetrical flat brass collar rim centered at top. "
    "GROUNDED CONTINUOUS BASE: Torso plating and long slender tubular arms solidly extend all the way down "
    "and bleed off the bottom frame boundary (y=1920). ABSOLUTELY NO floating cuts, NO truncated midriff, NO hollow bottom. "
    "Headless body only. Solid flat pure magenta background (#FF00FF). --ar 9:16"
)

CLAUDE_FRONT_BODY_PROMPT = (
    "Studio Ghibli 1990s vintage anime cel animation, Hayao Miyazaki mecha aesthetic. "
    "Vertical 9:16 portrait of a symmetrical retro-futuristic tin-plate robot chassis, "
    "PURE FRONTAL SYMMETRICAL VIEW (0 DEGREES FACING FORWARD). "
    "TORSO: Rounded convex barrel chassis in terracotta plating. "
    "At center chest, an ornate vertical glass vacuum tube (nixie tube) glowing with warm amber filaments and gears. "
    "Centered flat brass collar rim at top with dark corrugated neck socket. "
    "Slender tubular arms flush against both flanks. Grounded solid base at y=1920. "
    "Headless body only, zero magenta bleed. Solid flat pure magenta background (#FF00FF). --ar 9:16"
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

    def generate_view(
        self,
        character_id: str,
        view: str,
        *,
        reference_head: bool = False,
        body_only: bool = False,
    ) -> Path:
        """Generate one opposite-facing view. The root left puppet stays in place."""
        if view == "facing_front":
            return self.generate_facing_front(character_id, body_only=body_only)
        if view != "facing_right":
            raise SystemExit(f"unsupported view: {view}")
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        for name in ("head.png", "body.png"):
            source = skin_dir / name
            destination = left_dir / name
            if source.is_file() and not destination.is_file():
                shutil.copy2(source, destination)
        if not (left_dir / "head.png").is_file():
            raise RuntimeError(f"approved head missing: {left_dir / 'head.png'}")
        reference = right_dir / "_head_reference_magenta.png"
        if reference_head:
            _flatten_on_magenta(left_dir / "head.png", reference)
        head_source = right_dir / "source_head.png"
        body_source = right_dir / "source_body.png"
        generate_character_image(
            HEAD_RIGHT_PROMPT,
            head_source,
            aspect_ratio="1:1",
            reference_path=reference if reference_head else None,
        )
        generate_character_image(BODY_RIGHT_PROMPT, body_source, aspect_ratio="9:16")
        with Image.open(head_source) as opened:
            head_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        with Image.open(body_source) as opened:
            body_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        head_master.save(right_dir / "_head_master.png", format="PNG", compress_level=1)
        body_master.save(right_dir / "_body_master.png", format="PNG", compress_level=1)
        from .puppet_assembler import CanonicalDebaterContract

        head, body, head_xy, body_xy, _pivot = CanonicalDebaterContract.assemble(
            np.asarray(head_master),
            np.asarray(body_master),
            head_height=721,
            body_width=876,
            body_drop_px=18,
            head_shift_x=0,
            head_drop_px=0,
            body_shift_right=0,
            head_rel_shift_left=0,
            rotation_deg=-3.5,
            stage_center_x=495,
        )
        body_bottom = body_xy[1] + int(body.shape[0])
        print(f"facing-right body_bottom_y={body_bottom}")
        Image.fromarray(head).save(right_dir / "head.png", format="PNG", compress_level=1)
        Image.fromarray(body).save(right_dir / "body.png", format="PNG", compress_level=1)
        manifest_path = skin_dir / "puppet.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {"character_id": character_id, "skin_version": "v3-auto-rig"}
        )
        manifest["name"] = manifest.get("name") or "DEEPSEEK"
        manifest["views"] = {
            "facing_left": {
                "head": "views/facing_left/head.png",
                "body": "views/facing_left/body.png",
                "stage_x": 585,
                "tilt_deg": -3.5,
            },
            "facing_right": {
                "head": "views/facing_right/head.png",
                "body": "views/facing_right/body.png",
                "stage_x": 495,
                "tilt_deg": 3.5,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        preview = self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=(head_xy, body_xy),
            preview_name="deepseek_facing_right_preview.png",
        )
        print(f"facing-right: {preview}")
        return skin_dir

    def generate_facing_front(self, character_id: str, *, body_only: bool = False) -> Path:
        """Generate the frontal head and body. Locked side views are not rewritten."""
        import cv2

        from .puppet_assembler import CanonicalDebaterContract, normalize_to_exact_ref

        skin_dir = assets_root() / "puppets" / character_id
        left_head = skin_dir / "views" / "facing_left" / "head.png"
        front_dir = skin_dir / "views" / "facing_front"
        front_dir.mkdir(parents=True, exist_ok=True)
        if not left_head.is_file():
            raise RuntimeError(f"locked reference missing: {left_head}")
        manifest_path = skin_dir / "puppet.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        views = dict(manifest.get("views") or {})
        for key in ("facing_left", "facing_right"):
            locked = dict(views.get(key) or {})
            locked["status"] = "locked_approved"
            views[key] = locked
        manifest["views"] = views
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        approved_front_head = front_dir / "head.png"
        if body_only:
            if not approved_front_head.is_file():
                raise RuntimeError(f"approved front head missing: {approved_front_head}")
            body_source = front_dir / "source_body_rounded.png"
            generate_character_image(BODY_FRONTAL_ROUNDED_PROMPT, body_source, aspect_ratio="9:16")
            with Image.open(body_source) as opened:
                body_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
            body_master.save(front_dir / "_body_master.png", format="PNG", compress_level=1)
            head = np.asarray(Image.open(approved_front_head).convert("RGBA"))
            exact_h = int(head.shape[0])
        else:
            reference = front_dir / "_head_reference_magenta.png"
            _flatten_on_magenta(left_head, reference)
            head_source = front_dir / "source_head.png"
            body_source = front_dir / "source_body.png"
            generate_character_image(
                HEAD_FRONTAL_PROMPT,
                head_source,
                aspect_ratio="1:1",
                reference_path=reference,
            )
            generate_character_image(BODY_FRONTAL_PROMPT, body_source, aspect_ratio="9:16")
            with Image.open(head_source) as opened:
                head_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
            with Image.open(body_source) as opened:
                body_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
            head_master.save(front_dir / "_head_master.png", format="PNG", compress_level=1)
            body_master.save(front_dir / "_body_master.png", format="PNG", compress_level=1)
            head, exact_h = normalize_to_exact_ref(left_head, np.asarray(head_master))
        raw_body = np.asarray(body_master)
        b_scale = CanonicalDebaterContract.BODY_WIDTH / max(1, raw_body.shape[1])
        body = cv2.resize(
            raw_body,
            None,
            fx=b_scale,
            fy=b_scale * CanonicalDebaterContract.BODY_SCALE_Y_MULT,
            interpolation=cv2.INTER_LANCZOS4,
        )
        stage_x = 540
        head_drop = 18
        pivot_x = head.shape[1] // 2
        head_y = CanonicalDebaterContract.TARGET_EYE_Y - int(head.shape[0] * 0.42)
        head_x = stage_x - pivot_x
        collar_center_x = int(body.shape[1] * 0.50)
        collar_rim_y = int(body.shape[0] * 0.17)
        chin_y = int(head.shape[0] * 0.95)
        body_x = stage_x - collar_center_x
        body_y = (head_y + chin_y) - collar_rim_y
        head_y += head_drop
        head_xy = (int(head_x), int(head_y))
        body_xy = (int(body_x), int(body_y))
        print(
            f"facing-front head={head.shape[1]}x{head.shape[0]} body={body.shape[1]}x{body.shape[0]} "
            f"stage_x={stage_x} head_xy={head_xy} body_xy={body_xy} "
            f"body_bottom_y={body_xy[1] + body.shape[0]} exact_h={exact_h} tilt=0"
        )
        if not body_only:
            Image.fromarray(head).save(front_dir / "head.png", format="PNG", compress_level=1)
        Image.fromarray(body).save(front_dir / "body.png", format="PNG", compress_level=1)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        views = dict(manifest.get("views") or {})
        views["facing_front"] = {
            "head": "views/facing_front/head.png",
            "body": "views/facing_front/body.png",
            "stage_x": stage_x,
            "tilt_deg": 0.0,
            "eye_y": 600,
            "head_height": int(head.shape[0]),
            "head_width": int(head.shape[1]),
            "body_width": int(body.shape[1]),
            "body_scale_y_mult": CanonicalDebaterContract.BODY_SCALE_Y_MULT,
            "head_drop_px": head_drop,
            "head_xy": list(head_xy),
            "body_xy": list(body_xy),
            "status": "head_frozen" if body_only else "generated",
        }
        manifest["views"] = views
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        preview = self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=(head_xy, body_xy),
            preview_name="deepseek_facing_front_preview.png",
        )
        print(f"facing-front: {preview}")
        return skin_dir

    def seal_deepseek_contract(self) -> Path:
        """Drop the front body 15px and lock the three approved views. No image API."""
        character_id = "deepseek_cyborg_v3"
        skin_dir = assets_root() / "puppets" / character_id
        front_dir = skin_dir / "views" / "facing_front"
        manifest_path = skin_dir / "puppet.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        views = dict(manifest.get("views") or {})
        for key in ("facing_left", "facing_right", "facing_front"):
            view = dict(views.get(key) or {})
            view["status"] = "locked_approved"
            views[key] = view
        front = views["facing_front"]
        body_xy = list(front["body_xy"])
        body_xy[1] = int(body_xy[1]) + 15
        front["body_xy"] = body_xy
        front["body_ground_drop_px"] = 15
        front["framing_contract"] = "ParametricPuppetContract"
        views["facing_front"] = front
        manifest["views"] = views
        manifest["framing_contract"] = "ParametricPuppetContract"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        head = Image.open(front_dir / "head.png").convert("RGBA")
        body = Image.open(front_dir / "body.png").convert("RGBA")
        head_xy = tuple(front["head_xy"])
        preview = self._write_preview(
            character_id,
            head,
            body,
            bg_contrast="assembled",
            placed=(head_xy, tuple(body_xy)),
            preview_name="deepseek_facing_front_preview.png",
        )
        print(
            f"deepseek-seal front body_xy={tuple(body_xy)} "
            f"body_bottom_y={body_xy[1] + body.height} preview={preview}"
        )
        return skin_dir

    def create_canonical_master(self, character_id: str, name: str, character_prompt: str) -> Path:
        """Generate the first 3/4 view and dock it with the parametric contract."""
        from .puppet_assembler import ParametricPuppetContract, safe_rotate_head

        key = (name.upper(), character_prompt)
        if key == ("CLAUDE", "scholar-terracotta"):
            head_prompt, body_prompt = CLAUDE_SCHOLAR_HEAD_PROMPT, CLAUDE_SCHOLAR_BODY_PROMPT
            target_head_h = None
            body_bulk = ParametricPuppetContract.DEFAULT_BODY_TO_HEAD_RATIO
            chin_sink = ParametricPuppetContract.CHIN_SINK_PX
            bottom_cut_y = None
            preview_name = "claude_one_click_master_preview.png"
        elif key == ("CLAUDE", "terracotta"):
            head_prompt, body_prompt = CLAUDE_HEAD_PROMPT, CLAUDE_BODY_PROMPT
            target_head_h = None
            body_bulk = ParametricPuppetContract.DEFAULT_BODY_TO_HEAD_RATIO
            chin_sink = ParametricPuppetContract.CHIN_SINK_PX
            bottom_cut_y = None
            preview_name = "claude_one_click_master_preview.png"
        elif key == ("CHATGPT", "ivory-celadon-radio"):
            head_prompt, body_prompt = CHATGPT_HEAD_PROMPT, CHATGPT_BODY_PROMPT
            target_head_h = 710
            body_bulk = 1.40
            chin_sink = 18
            bottom_cut_y = 1920
            preview_name = "chatgpt_one_click_master_preview.png"
        else:
            raise SystemExit(f"no prompt contract for {name} / {character_prompt}")
        skin_dir = assets_root() / "puppets" / character_id
        view_dir = skin_dir / "views" / "facing_left"
        view_dir.mkdir(parents=True, exist_ok=True)
        head_source = view_dir / "source_head.png"
        body_source = view_dir / "source_body.png"
        generate_character_image(head_prompt, head_source, aspect_ratio="1:1")
        generate_character_image(body_prompt, body_source, aspect_ratio="9:16")
        with Image.open(head_source) as opened:
            head_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        with Image.open(body_source) as opened:
            body_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        head_master.save(view_dir / "_head_master.png", format="PNG", compress_level=1)
        body_master.save(view_dir / "_body_master.png", format="PNG", compress_level=1)
        head, body, stage_x, tilt = ParametricPuppetContract.fit_puppet(
            np.asarray(head_master),
            np.asarray(body_master),
            view_name="facing_left",
            body_bulk_ratio=body_bulk,
            target_head_h=target_head_h,
        )
        if tilt:
            head = safe_rotate_head(head, angle_deg=-tilt, pivot_ratio=(0.5, 0.95))
        pivot_x = head.shape[1] // 2
        head_y = ParametricPuppetContract.CANONICAL_EYE_Y - int(head.shape[0] * 0.42)
        head_x = stage_x - pivot_x
        collar_center_x = int(body.shape[1] * 0.46)
        collar_rim_y = int(body.shape[0] * 0.17)
        chin_y = int(head.shape[0] * 0.95)
        body_x = (head_x + pivot_x) - collar_center_x
        body_y = (head_y + chin_y) - collar_rim_y
        head_y += chin_sink
        head_x += ParametricPuppetContract.jaw_offset_x("facing_left")
        head_xy = (int(head_x), int(head_y))
        body_xy = (int(body_x), int(body_y))
        if bottom_cut_y is not None:
            keep = int(bottom_cut_y) - body_xy[1]
            if 0 < keep < body.shape[0]:
                body = body[:keep]
        print(
            f"canonical-master head={head.shape[1]}x{head.shape[0]} body={body.shape[1]}x{body.shape[0]} "
            f"stage_x={stage_x} head_xy={head_xy} body_xy={body_xy} "
            f"body_bottom_y={body_xy[1] + body.shape[0]} tilt={tilt} chin_sink={chin_sink}"
        )
        Image.fromarray(head).save(view_dir / "head.png", format="PNG", compress_level=1)
        Image.fromarray(body).save(view_dir / "body.png", format="PNG", compress_level=1)
        manifest = {
            "character_id": character_id,
            "name": name,
            "skin_version": "v1-parametric",
            "framing_contract": "ParametricPuppetContract",
            "views": {
                "facing_left": {
                    "head": "views/facing_left/head.png",
                    "body": "views/facing_left/body.png",
                    "stage_x": stage_x,
                    "tilt_deg": tilt,
                    "eye_y": ParametricPuppetContract.CANONICAL_EYE_Y,
                    "head_height": int(head.shape[0]),
                    "head_width": int(head.shape[1]),
                    "body_width": int(body.shape[1]),
                    "body_to_head_ratio": body_bulk,
                    "head_drop_px": chin_sink,
                    "jaw_offset_x": ParametricPuppetContract.FACING_LEFT_JAW_OFFSET_X,
                    "head_xy": list(head_xy),
                    "body_xy": list(body_xy),
                    "status": "canonical_master",
                    "character_prompt": character_prompt,
                }
            },
        }
        (skin_dir / "puppet.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        preview = self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=(head_xy, body_xy),
            preview_name=preview_name,
        )
        print(f"canonical-master: {preview}")
        return skin_dir

    def regenerate_head_only(self, character_id: str, *, blank_viseme_plate: bool = False) -> Path:
        """Replace the facing-left head. The approved body file is not rewritten."""
        import cv2

        from .puppet_assembler import ParametricPuppetContract, safe_rotate_head

        if not blank_viseme_plate:
            raise SystemExit("--regenerate-head-only requires --blank-viseme-plate")
        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"head-only blank plate is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        view_dir = skin_dir / "views" / "facing_left"
        body_path = view_dir / "body.png"
        manifest_path = skin_dir / "puppet.json"
        if not body_path.is_file() or not manifest_path.is_file():
            raise SystemExit(f"frozen body missing: {body_path}")
        body_stamp = (body_path.stat().st_mtime_ns, body_path.stat().st_size)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        view = manifest["views"]["facing_left"]
        body_xy = (int(view["body_xy"][0]), int(view["body_xy"][1]))
        stage_x = int(view["stage_x"])
        tilt = float(view["tilt_deg"])
        body = np.asarray(Image.open(body_path).convert("RGBA"))
        head_source = view_dir / "source_head.png"
        generate_character_image(CLAUDE_SMOOTH_CHIN_HEAD_PROMPT, head_source, aspect_ratio="1:1")
        with Image.open(head_source) as opened:
            head_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        head_master.save(view_dir / "_head_master.png", format="PNG", compress_level=1)
        head_np = np.asarray(head_master)
        target_h = ParametricPuppetContract.target_head_height()
        scale = target_h / max(1, head_np.shape[0])
        head = cv2.resize(head_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        if tilt:
            head = safe_rotate_head(head, angle_deg=-tilt, pivot_ratio=(0.5, 0.95))
        pivot_x = head.shape[1] // 2
        collar_rim_y = int(body.shape[0] * 0.17)
        chin_y = int(head.shape[0] * 0.95)
        head_y = body_xy[1] + collar_rim_y - chin_y + 18
        head_x = (stage_x - pivot_x) + ParametricPuppetContract.jaw_offset_x("facing_left")
        head_xy = (int(head_x), int(head_y))
        Image.fromarray(head).save(view_dir / "head.png", format="PNG", compress_level=1)
        if (body_path.stat().st_mtime_ns, body_path.stat().st_size) != body_stamp:
            raise RuntimeError("frozen body.png was modified")
        view["head_height"] = int(head.shape[0])
        view["head_width"] = int(head.shape[1])
        view["head_xy"] = list(head_xy)
        view["head_drop_px"] = 18
        view["jaw_offset_x"] = ParametricPuppetContract.FACING_LEFT_JAW_OFFSET_X
        view["chin_plate"] = "blank_viseme"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        preview = self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=(head_xy, body_xy),
            preview_name="claude_one_click_master_preview.png",
        )
        print(
            f"claude-head-only head={head.shape[1]}x{head.shape[0]} "
            f"head_xy={head_xy} body_xy={body_xy} jaw_offset_x={view['jaw_offset_x']} "
            f"body_frozen={body_path.stat().st_size}"
        )
        print(f"claude-head-only: {preview}")
        return skin_dir

    def verify_health(self, character_id: str) -> Path:
        """Re-composite locked views. Does not move pixels or call the image API."""
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"health check is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        manifest_path = skin_dir / "puppet.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        locked = [manifest_path]
        for name in ("facing_left", "facing_right", "facing_front"):
            view = manifest["views"][name]
            locked.extend((skin_dir / view["head"], skin_dir / view["body"]))
        stamps = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in locked}
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        sheet = Image.new("RGB", (1080 * 3, 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        labels = ("LEFT", "FRONT", "RIGHT")
        for index, name in enumerate(("facing_left", "facing_front", "facing_right")):
            view = manifest["views"][name]
            head_xy = (int(view["head_xy"][0]), int(view["head_xy"][1]))
            body_xy = (int(view["body_xy"][0]), int(view["body_xy"][1]))
            panel = backdrop.copy()
            _composite_at(panel, Image.open(skin_dir / view["body"]).convert("RGBA"), body_xy[0], body_xy[1])
            _composite_at(panel, Image.open(skin_dir / view["head"]).convert("RGBA"), head_xy[0], head_xy[1])
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 280, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), labels[index], fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
            head = np.asarray(Image.open(skin_dir / view["head"]).convert("RGBA"))
            body = np.asarray(Image.open(skin_dir / view["body"]).convert("RGBA"))
            chin = int(np.nonzero(head[:, head.shape[1] // 2, 3] > 16)[0][-1]) + head_xy[1]
            body_rows = np.nonzero((body[..., 3] > 16).any(axis=1))[0]
            collar = int(body_rows[0]) + body_xy[1]
            print(f"health {name} chin_y={chin} collar_y={collar} overlap_px={chin - collar}")
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"locked file changed: {path}")
        destination = outputs_root() / "aiwake" / "_test_harness" / "claude_trinity_complete.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination, format="PNG", compress_level=1)
        print(f"health: {destination}")
        return destination

    def generate_tin_man_candidates(
        self,
        character_id: str,
        *,
        clean: bool = False,
        blank_face: bool = False,
        proven: bool = False,
    ) -> Path:
        """Sheet three ChatGPT concepts. The ivory production layers stay put."""
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama
        from .puppet_assembler import ParametricPuppetContract, safe_rotate_head

        if character_id != "chatgpt_cyborg_v1":
            raise SystemExit(f"tin-man variants are not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        view_dir = skin_dir / "views" / "facing_left"
        manifest_path = skin_dir / "puppet.json"
        head_path = view_dir / "head.png"
        body_path = view_dir / "body.png"
        if not head_path.is_file() or not body_path.is_file() or not manifest_path.is_file():
            raise SystemExit(f"ivory ChatGPT layers missing under {view_dir}")
        stamps = {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in (head_path, body_path, manifest_path)
        }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        view = manifest["views"]["facing_left"]
        head_xy = (int(view["head_xy"][0]), int(view["head_xy"][1]))
        body_xy = (int(view["body_xy"][0]), int(view["body_xy"][1]))
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        if not clean and not blank_face and not proven:
            archive_dir = skin_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archived = backdrop.copy()
            _composite_at(archived, Image.open(body_path).convert("RGBA"), body_xy[0], body_xy[1])
            _composite_at(archived, Image.open(head_path).convert("RGBA"), head_xy[0], head_xy[1])
            archive_path = archive_dir / "chatgpt_v1_ivory_radio.png"
            archived.convert("RGB").save(archive_path, format="PNG", compress_level=1)
            shutil.copy2(head_path, archive_dir / "chatgpt_v1_ivory_radio_head.png")
            shutil.copy2(body_path, archive_dir / "chatgpt_v1_ivory_radio_body.png")
            print(f"ivory-archive: {archive_path}")
        if proven:
            concepts = (
                ("A", "Proven Communicator", PROVEN_A_HEAD_PROMPT, PROVEN_A_BODY_PROMPT),
                ("B", "Dual Radio Dials", PROVEN_B_HEAD_PROMPT, PROVEN_B_BODY_PROMPT),
                ("C", "Hydraulic Joints", PROVEN_C_HEAD_PROMPT, PROVEN_C_BODY_PROMPT),
            )
            slot_prefix = "proven"
        elif blank_face:
            concepts = (
                ("A", "Off-White Communicator", BLANK_A_HEAD_PROMPT, BLANK_A_BODY_PROMPT),
                ("B", "Pale Sage Laboratory", BLANK_B_HEAD_PROMPT, BLANK_B_BODY_PROMPT),
                ("C", "Ivory Celadon Sentinel", BLANK_C_HEAD_PROMPT, BLANK_C_BODY_PROMPT),
            )
            slot_prefix = "option"
        elif clean:
            concepts = (
                ("A", "Platinum Slate", CLEAN_A_HEAD_PROMPT, CLEAN_A_BODY_PROMPT),
                ("B", "Celadon Sage", CLEAN_B_HEAD_PROMPT, CLEAN_B_BODY_PROMPT),
                ("C", "Gunmetal Bronze", CLEAN_C_HEAD_PROMPT, CLEAN_C_BODY_PROMPT),
            )
            slot_prefix = "clean"
        else:
            concepts = (
                ("A", "Clockwork Tin Man", TIN_MAN_A_HEAD_PROMPT, TIN_MAN_A_BODY_PROMPT),
                ("B", "Industrial Art-Deco", TIN_MAN_B_HEAD_PROMPT, TIN_MAN_B_BODY_PROMPT),
                ("C", "Tin-Toy Sentinel", TIN_MAN_C_HEAD_PROMPT, TIN_MAN_C_BODY_PROMPT),
            )
            slot_prefix = "tin"
        jaw_offset = -48
        panels: list[Image.Image] = []
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for key, _title, head_prompt, body_prompt in concepts:
            if proven:
                slot = skin_dir / "candidates" / f"proven_{key.lower()}"
            elif blank_face:
                slot = skin_dir / "candidates" / f"option_{key.lower()}"
            else:
                slot = view_dir / "candidates" / f"{slot_prefix}_{key.lower()}"
            slot.mkdir(parents=True, exist_ok=True)
            head_source = slot / "source_head.png"
            body_source = slot / "source_body.png"
            generate_character_image(head_prompt, head_source, aspect_ratio="1:1")
            generate_character_image(body_prompt, body_source, aspect_ratio="9:16")
            with Image.open(head_source) as opened:
                head_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
            with Image.open(body_source) as opened:
                body_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
            head_master.save(slot / "_head_master.png", format="PNG", compress_level=1)
            body_master.save(slot / "_body_master.png", format="PNG", compress_level=1)
            head, body, stage_x, tilt = ParametricPuppetContract.fit_puppet(
                np.asarray(head_master),
                np.asarray(body_master),
                view_name="facing_left",
                body_bulk_ratio=1.40,
                target_head_h=710,
            )
            if tilt:
                head = safe_rotate_head(head, angle_deg=-tilt, pivot_ratio=(0.5, 0.95))
            pivot_x = head.shape[1] // 2
            head_y = ParametricPuppetContract.CANONICAL_EYE_Y - int(head.shape[0] * 0.42)
            head_x = stage_x - pivot_x
            collar_center_x = int(body.shape[1] * 0.46)
            collar_rim_y = int(body.shape[0] * 0.17)
            chin_y = int(head.shape[0] * 0.95)
            dock_body_x = (head_x + pivot_x) - collar_center_x
            dock_body_y = (head_y + chin_y) - collar_rim_y
            head_y += ParametricPuppetContract.CHIN_SINK_PX
            head_x += jaw_offset
            placed_head = (int(head_x), int(head_y))
            placed_body = (int(dock_body_x), int(dock_body_y))
            keep = 1920 - placed_body[1]
            if 0 < keep < body.shape[0]:
                body = body[:keep]
            Image.fromarray(head).save(slot / "head.png", format="PNG", compress_level=1)
            Image.fromarray(body).save(slot / "body.png", format="PNG", compress_level=1)
            panel = backdrop.copy()
            _composite_at(panel, Image.fromarray(body), placed_body[0], placed_body[1])
            _composite_at(panel, Image.fromarray(head), placed_head[0], placed_head[1])
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 360, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), f"OPTION {key}", fill=(244, 214, 150, 255), font=font)
            panels.append(panel.convert("RGB"))
            print(
                f"tin-man {key} head={head.shape[1]}x{head.shape[0]} body={body.shape[1]}x{body.shape[0]} "
                f"head_xy={placed_head} body_xy={placed_body} jaw_offset_x={jaw_offset}"
            )
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"ivory production file changed: {path}")
        sheet = Image.new("RGB", (1080 * len(panels), 1920), (12, 10, 8))
        for index, panel in enumerate(panels):
            sheet.paste(panel, (index * 1080, 0))
        destination = outputs_root() / "aiwake" / "_test_harness" / "chatgpt_3_candidates_preview.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination, format="PNG", compress_level=1)
        print(f"tin-man: {destination}")
        return destination

    def style_picker(
        self,
        character_id: str,
        count: int = 3,
        *,
        subtle_shapes: bool = False,
        elongated_heads: bool = False,
    ) -> Path:
        """Generate facing-left head candidates over the frozen approved body."""
        import cv2
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama
        from .puppet_assembler import ParametricPuppetContract, safe_rotate_head

        if count != 3:
            raise SystemExit("--style-picker currently supports 3 candidates")
        if subtle_shapes == elongated_heads:
            raise SystemExit("--style-picker requires exactly one of --subtle-shapes or --elongated-heads")
        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"style picker is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        view_dir = skin_dir / "views" / "facing_left"
        body_path = view_dir / "body.png"
        manifest_path = skin_dir / "puppet.json"
        if not body_path.is_file() or not manifest_path.is_file():
            raise SystemExit(f"frozen body missing: {body_path}")
        body_stamp = (body_path.stat().st_mtime_ns, body_path.stat().st_size)
        head_stamp = (view_dir / "head.png").stat().st_mtime_ns
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        view = manifest["views"]["facing_left"]
        body_xy = (int(view["body_xy"][0]), int(view["body_xy"][1]))
        stage_x = int(view["stage_x"])
        tilt = float(view["tilt_deg"])
        body = np.asarray(Image.open(body_path).convert("RGBA"))
        candidate_dir = view_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if elongated_heads:
            prompts = (
                ("A", "High Oval Laputa", HEAD_A_PROMPT),
                ("B", "Vertical Strapped Sentinel", HEAD_B_PROMPT),
                ("C", "High-Dome Chrono", HEAD_C_PROMPT),
            )
        else:
            prompts = (
                ("A", "Beveled Paneled", SUBTLE_HEAD_A_PROMPT),
                ("B", "Visor Brim", SUBTLE_HEAD_B_PROMPT),
                ("C", "Tapered Octagonal", SUBTLE_HEAD_C_PROMPT),
            )
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        panels: list[Image.Image] = []
        for key, _title, prompt in prompts:
            source_path = candidate_dir / f"source_head_{key.lower()}.png"
            generate_character_image(prompt, source_path, aspect_ratio="1:1")
            with Image.open(source_path) as opened:
                head_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
            head_master.save(candidate_dir / f"_head_master_{key.lower()}.png", format="PNG", compress_level=1)
            head_np = np.asarray(head_master)
            target_h = ParametricPuppetContract.target_head_height()
            scale = target_h / max(1, head_np.shape[0])
            head = cv2.resize(head_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            if tilt:
                head = safe_rotate_head(head, angle_deg=-tilt, pivot_ratio=(0.5, 0.95))
            pivot_x = head.shape[1] // 2
            collar_rim_y = int(body.shape[0] * 0.17)
            chin_y = int(head.shape[0] * 0.95)
            jaw_offset = ParametricPuppetContract.jaw_offset_x("facing_left")
            head_y = body_xy[1] + collar_rim_y - chin_y + 18
            head_x = (stage_x - pivot_x) + jaw_offset
            Image.fromarray(head).save(
                candidate_dir / f"head_{key.lower()}.png",
                format="PNG",
                compress_level=1,
            )
            panel = backdrop.copy()
            _composite_at(panel, Image.fromarray(body), body_xy[0], body_xy[1])
            _composite_at(panel, Image.fromarray(head), int(head_x), int(head_y))
            draw = ImageDraw.Draw(panel)
            try:
                font = ImageFont.truetype("arialbd.ttf", 42)
            except OSError:
                font = ImageFont.load_default()
            label = f"OPTION {key}"
            draw.rectangle((36, 28, 36 + 280, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            panels.append(panel.convert("RGB"))
            print(
                f"style-picker {key} head={head.shape[1]}x{head.shape[0]} "
                f"head_xy=({int(head_x)},{int(head_y)}) body_xy={body_xy} jaw_offset_x={jaw_offset}"
            )
        if (body_path.stat().st_mtime_ns, body_path.stat().st_size) != body_stamp:
            raise RuntimeError("frozen body.png was modified")
        if (view_dir / "head.png").stat().st_mtime_ns != head_stamp:
            raise RuntimeError("production head.png was modified")
        sheet = Image.new("RGB", (1080 * len(panels), 1920), (12, 10, 8))
        for index, panel in enumerate(panels):
            sheet.paste(panel, (index * 1080, 0))
        destination = outputs_root() / "aiwake" / "_test_harness" / "claude_3_candidates_preview.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination, format="PNG", compress_level=1)
        print(f"style-picker: {destination}")
        return destination

    def lock_candidate(
        self,
        character_id: str,
        candidate: str,
        *,
        head_scale_pct: float = 110,
        head_compact_y: float = 0.94,
        head_compact_x: float = 1.02,
        head_sink_px: int = 38,
        body_drop_px: int = 15,
        jaw_offset_x: int = -45,
    ) -> Path:
        """Polish one generated head and dock it. No image API."""
        import cv2

        from .puppet_assembler import ParametricPuppetContract, safe_rotate_head

        key = candidate.strip().upper()
        if key != "B":
            raise SystemExit(f"only candidate B is approved for lock, got {candidate}")
        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"candidate lock is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        view_dir = skin_dir / "views" / "facing_left"
        master_path = view_dir / "candidates" / "_head_master_b.png"
        body_path = view_dir / "body.png"
        manifest_path = skin_dir / "puppet.json"
        if not master_path.is_file() or not body_path.is_file() or not manifest_path.is_file():
            raise SystemExit(f"option B master or frozen body missing under {view_dir}")
        body_stamp = (body_path.stat().st_mtime_ns, body_path.stat().st_size)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        view = manifest["views"]["facing_left"]
        stage_x = int(view["stage_x"])
        tilt = float(view["tilt_deg"])
        stored_base = view.get("body_xy_base")
        if isinstance(stored_base, list) and len(stored_base) == 2:
            origin = (int(stored_base[0]), int(stored_base[1]))
        else:
            prior_drop = int(view.get("body_ground_drop_px") or 0)
            origin = (int(view["body_xy"][0]), int(view["body_xy"][1]) - prior_drop)
        body_xy = (origin[0], origin[1] + int(body_drop_px))
        body = np.asarray(Image.open(body_path).convert("RGBA"))
        raw = np.asarray(Image.open(master_path).convert("RGBA"))
        target_h = ParametricPuppetContract.target_head_height()
        base = (target_h / max(1, raw.shape[0])) * (head_scale_pct / 100.0)
        head = cv2.resize(
            raw,
            None,
            fx=base * head_compact_x,
            fy=base * head_compact_y,
            interpolation=cv2.INTER_LANCZOS4,
        )
        if tilt:
            head = safe_rotate_head(head, angle_deg=-tilt, pivot_ratio=(0.5, 0.95))
        pivot_x = head.shape[1] // 2
        collar_rim_y = int(body.shape[0] * 0.17)
        chin_y = int(head.shape[0] * 0.95)
        head_y = body_xy[1] + collar_rim_y - chin_y + int(head_sink_px)
        head_x = (stage_x - pivot_x) + int(jaw_offset_x)
        head_xy = (int(head_x), int(head_y))
        Image.fromarray(head).save(view_dir / "head.png", format="PNG", compress_level=1)
        if (body_path.stat().st_mtime_ns, body_path.stat().st_size) != body_stamp:
            raise RuntimeError("frozen body.png was modified")
        view.update(
            {
                "head_height": int(head.shape[0]),
                "head_width": int(head.shape[1]),
                "head_xy": list(head_xy),
                "body_xy": list(body_xy),
                "body_xy_base": list(origin),
                "head_drop_px": int(head_sink_px),
                "body_ground_drop_px": int(body_drop_px),
                "jaw_offset_x": int(jaw_offset_x),
                "head_scale_pct": head_scale_pct,
                "head_compact_x": head_compact_x,
                "head_compact_y": head_compact_y,
                "locked_candidate": key,
                "status": "locked_approved",
                "chin_plate": "blank_viseme",
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        preview = self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=(head_xy, body_xy),
            preview_name="claude_one_click_master_preview.png",
        )
        print(
            f"claude-lock {key} head={head.shape[1]}x{head.shape[0]} "
            f"head_xy={head_xy} body_xy={body_xy} "
            f"body_bottom_y={body_xy[1] + body.shape[0]} "
            f"scale={head_scale_pct} compact=({head_compact_x},{head_compact_y}) "
            f"sink={head_sink_px} jaw_offset_x={jaw_offset_x} tilt={tilt}"
        )
        print(f"claude-lock: {preview}")
        return skin_dir

    def apply_approved_framing(self, character_id: str) -> Path:
        """Grow the locked Claude head and body 15% from the masters, then nudge the body."""
        import cv2

        from .puppet_assembler import ParametricPuppetContract, safe_rotate_head

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"approved framing is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        view_dir = skin_dir / "views" / "facing_left"
        manifest_path = skin_dir / "puppet.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        view = manifest["views"]["facing_left"]
        anchor_xy = view.get("framing_anchor_body_xy") or view["body_xy"]
        anchor_size = view.get("framing_anchor_body_size")
        body_path = view_dir / "body.png"
        if anchor_size is None:
            with Image.open(body_path) as current_body:
                anchor_size = [current_body.width, current_body.height]
        anchor_x, anchor_y = int(anchor_xy[0]), int(anchor_xy[1])
        anchor_w, anchor_h = int(anchor_size[0]), int(anchor_size[1])
        body_x = anchor_x + ParametricPuppetContract.APPROVED_BODY_SHIFT_X
        body_y = anchor_y + ParametricPuppetContract.APPROVED_BODY_SHIFT_Y
        head_master = np.asarray(
            Image.open(view_dir / "candidates" / "_head_master_b.png").convert("RGBA")
        )
        body_master = np.asarray(Image.open(view_dir / "_body_master.png").convert("RGBA"))
        boost = ParametricPuppetContract.APPROVED_UNIFORM_BOOST
        locked_target = int(1920 * 0.37)
        base = (locked_target / max(1, head_master.shape[0])) * (
            float(view.get("head_scale_pct") or 112.2) / 100.0
        )
        head = cv2.resize(
            head_master,
            None,
            fx=base * float(view.get("head_compact_x") or 1.02) * boost,
            fy=base * float(view.get("head_compact_y") or 0.8742) * boost,
            interpolation=cv2.INTER_LANCZOS4,
        )
        tilt = float(view["tilt_deg"])
        if tilt:
            head = safe_rotate_head(head, angle_deg=-tilt, pivot_ratio=(0.5, 0.95))
        body_scale_x = (anchor_w / max(1, body_master.shape[1])) * boost
        body_scale_y = (anchor_h / max(1, body_master.shape[0])) * boost
        body = cv2.resize(
            body_master,
            None,
            fx=body_scale_x,
            fy=body_scale_y,
            interpolation=cv2.INTER_LANCZOS4,
        )
        collar_x = body_x + int(body.shape[1] * 0.46)
        collar_y = body_y + int(body.shape[0] * 0.17)
        head_x = collar_x + ParametricPuppetContract.jaw_offset_x("facing_left") - (head.shape[1] // 2)
        head_y = collar_y + ParametricPuppetContract.CHIN_SINK_PX - int(head.shape[0] * 0.95)
        head_xy = (int(head_x), int(head_y))
        body_xy = (int(body_x), int(body_y))
        Image.fromarray(head).save(view_dir / "head.png", format="PNG", compress_level=1)
        Image.fromarray(body).save(body_path, format="PNG", compress_level=1)
        ratio = round(body.shape[1] / max(1, head.shape[1]), 3)
        view.update(
            {
                "head_height": int(head.shape[0]),
                "head_width": int(head.shape[1]),
                "head_xy": list(head_xy),
                "body_width": int(body.shape[1]),
                "body_height": int(body.shape[0]),
                "body_xy": list(body_xy),
                "body_to_head_ratio": ratio,
                "framing_anchor_body_xy": [anchor_x, anchor_y],
                "framing_anchor_body_size": [anchor_w, anchor_h],
                "approved_uniform_boost": boost,
                "body_shift_x": ParametricPuppetContract.APPROVED_BODY_SHIFT_X,
                "body_shift_y": ParametricPuppetContract.APPROVED_BODY_SHIFT_Y,
                "jaw_offset_x": ParametricPuppetContract.FACING_LEFT_JAW_OFFSET_X,
                "head_drop_px": ParametricPuppetContract.CHIN_SINK_PX,
                "status": "locked_approved",
            }
        )
        manifest["proportion_estimate"] = {
            "source": "claude_cyborg_v1 facing_left",
            "head_px": [int(head.shape[1]), int(head.shape[0])],
            "body_px": [int(body.shape[1]), int(body.shape[0])],
            "head_xy": list(head_xy),
            "body_xy": list(body_xy),
            "uniform_boost": boost,
            "body_shift_x": ParametricPuppetContract.APPROVED_BODY_SHIFT_X,
            "body_shift_y": ParametricPuppetContract.APPROVED_BODY_SHIFT_Y,
            "jaw_offset_x": ParametricPuppetContract.FACING_LEFT_JAW_OFFSET_X,
            "chin_sink_px": ParametricPuppetContract.CHIN_SINK_PX,
            "body_to_head_ratio": ratio,
            "note": "Rough starting size for the next drawn avatars.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        preview = self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=(head_xy, body_xy),
            preview_name="claude_final_approval_preview.png",
        )
        print(
            f"claude-approved head={head.shape[1]}x{head.shape[0]} head_xy={head_xy} "
            f"body={body.shape[1]}x{body.shape[0]} body_xy={body_xy} "
            f"bottom={body_xy[1] + body.shape[0]} ratio={ratio}"
        )
        print(f"claude-approved: {preview}")
        return preview

    def generate_trinity(self, character_id: str) -> Path:
        """Build facing-right and facing-front from the locked left master. Left pixels stay put."""
        import cv2
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama
        from .puppet_assembler import ParametricPuppetContract, mirror_approved_body, normalize_to_exact_ref

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"trinity generation is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        front_dir = skin_dir / "views" / "facing_front"
        right_dir.mkdir(parents=True, exist_ok=True)
        front_dir.mkdir(parents=True, exist_ok=True)
        left_head_path = left_dir / "head.png"
        left_body_path = left_dir / "body.png"
        manifest_path = skin_dir / "puppet.json"
        if not left_head_path.is_file() or not left_body_path.is_file() or not manifest_path.is_file():
            raise SystemExit("approved facing_left master is missing")
        left_stamp = (
            (left_head_path.stat().st_mtime_ns, left_head_path.stat().st_size),
            (left_body_path.stat().st_mtime_ns, left_body_path.stat().st_size),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        left_view = dict(manifest["views"]["facing_left"])
        approved_head = np.asarray(Image.open(left_head_path).convert("RGBA"))
        approved_body = np.asarray(Image.open(left_body_path).convert("RGBA"))
        body_right = mirror_approved_body(approved_body)
        Image.fromarray(body_right).save(right_dir / "body.png", format="PNG", compress_level=1)
        flipped_head = cv2.flip(approved_head, 1)
        reference = right_dir / "temp_pre_flipped_head.png"
        plate = Image.new("RGBA", (flipped_head.shape[1], flipped_head.shape[0]), (255, 0, 255, 255))
        plate.alpha_composite(Image.fromarray(flipped_head))
        plate.convert("RGB").save(reference, format="PNG", compress_level=1)
        right_source = right_dir / "source_head_preflipped.png"
        generate_character_image(
            CLAUDE_RIGHT_PROMPT,
            right_source,
            aspect_ratio="1:1",
            reference_path=reference,
        )
        with Image.open(right_source) as opened:
            right_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        right_master.save(right_dir / "_head_master.png", format="PNG", compress_level=1)
        head_right, exact_h = normalize_to_exact_ref(left_head_path, np.asarray(right_master))
        Image.fromarray(head_right).save(right_dir / "head.png", format="PNG", compress_level=1)
        right_head_xy, right_body_xy = _seat_view(
            head_right,
            body_right,
            stage_x=ParametricPuppetContract.STAGE_X_LEFT,
            collar_ratio=0.54,
            jaw_offset=-ParametricPuppetContract.FACING_LEFT_JAW_OFFSET_X,
        )
        front_reference = front_dir / "_head_reference_magenta.png"
        _flatten_on_magenta(left_head_path, front_reference)
        front_head_source = front_dir / "source_head.png"
        front_body_source = front_dir / "source_body.png"
        generate_character_image(
            CLAUDE_FRONT_PROMPT,
            front_head_source,
            aspect_ratio="1:1",
            reference_path=front_reference,
        )
        generate_character_image(CLAUDE_FRONT_BODY_PROMPT, front_body_source, aspect_ratio="9:16")
        with Image.open(front_head_source) as opened:
            front_head_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        with Image.open(front_body_source) as opened:
            front_body_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        front_head_master.save(front_dir / "_head_master.png", format="PNG", compress_level=1)
        front_body_master.save(front_dir / "_body_master.png", format="PNG", compress_level=1)
        head_front, _front_h = normalize_to_exact_ref(left_head_path, np.asarray(front_head_master))
        raw_front_body = np.asarray(front_body_master)
        canonical_w = int(approved_body.shape[1])
        front_scale = canonical_w / max(1, raw_front_body.shape[1])
        body_front = cv2.resize(
            raw_front_body,
            None,
            fx=front_scale,
            fy=front_scale,
            interpolation=cv2.INTER_LANCZOS4,
        )
        Image.fromarray(head_front).save(front_dir / "head.png", format="PNG", compress_level=1)
        Image.fromarray(body_front).save(front_dir / "body.png", format="PNG", compress_level=1)
        front_head_xy, front_body_xy = _seat_view(
            head_front,
            body_front,
            stage_x=ParametricPuppetContract.STAGE_X_CENTER,
            collar_ratio=0.50,
            jaw_offset=0,
        )
        if (
            (left_head_path.stat().st_mtime_ns, left_head_path.stat().st_size),
            (left_body_path.stat().st_mtime_ns, left_body_path.stat().st_size),
        ) != left_stamp:
            raise RuntimeError("approved facing_left files were modified")
        left_view["status"] = "locked_approved"
        views = dict(manifest.get("views") or {})
        views["facing_left"] = left_view
        views["facing_right"] = {
            "head": "views/facing_right/head.png",
            "body": "views/facing_right/body.png",
            "stage_x": ParametricPuppetContract.STAGE_X_LEFT,
            "tilt_deg": 3.5,
            "eye_y": ParametricPuppetContract.CANONICAL_EYE_Y,
            "head_height": int(head_right.shape[0]),
            "head_width": int(head_right.shape[1]),
            "body_width": int(body_right.shape[1]),
            "head_xy": list(right_head_xy),
            "body_xy": list(right_body_xy),
            "jaw_offset_x": -ParametricPuppetContract.FACING_LEFT_JAW_OFFSET_X,
            "head_drop_px": ParametricPuppetContract.CHIN_SINK_PX,
            "exact_height": exact_h,
            "status": "locked_approved",
        }
        views["facing_front"] = {
            "head": "views/facing_front/head.png",
            "body": "views/facing_front/body.png",
            "stage_x": ParametricPuppetContract.STAGE_X_CENTER,
            "tilt_deg": 0.0,
            "eye_y": ParametricPuppetContract.CANONICAL_EYE_Y,
            "head_height": int(head_front.shape[0]),
            "head_width": int(head_front.shape[1]),
            "body_width": int(body_front.shape[1]),
            "head_xy": list(front_head_xy),
            "body_xy": list(front_body_xy),
            "head_drop_px": ParametricPuppetContract.CHIN_SINK_PX,
            "exact_height": int(head_front.shape[0]),
            "status": "locked_approved",
        }
        manifest["views"] = views
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._write_preview(
            character_id,
            Image.fromarray(head_right),
            Image.fromarray(body_right),
            bg_contrast="assembled",
            placed=(right_head_xy, right_body_xy),
            preview_name="claude_facing_right_preview.png",
        )
        self._write_preview(
            character_id,
            Image.fromarray(head_front),
            Image.fromarray(body_front),
            bg_contrast="assembled",
            placed=(front_head_xy, front_body_xy),
            preview_name="claude_facing_front_preview.png",
        )
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        panels = (
            ("LEFT", approved_head, approved_body, tuple(left_view["head_xy"]), tuple(left_view["body_xy"])),
            ("FRONT", head_front, body_front, front_head_xy, front_body_xy),
            ("RIGHT", head_right, body_right, right_head_xy, right_body_xy),
        )
        sheet = Image.new("RGB", (1080 * len(panels), 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, head, body, head_xy, body_xy) in enumerate(panels):
            panel = backdrop.copy()
            _composite_at(panel, Image.fromarray(body), int(body_xy[0]), int(body_xy[1]))
            _composite_at(panel, Image.fromarray(head), int(head_xy[0]), int(head_xy[1]))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 320, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
        trinity = outputs_root() / "aiwake" / "_test_harness" / "claude_trinity_complete.png"
        trinity.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(trinity, format="PNG", compress_level=1)
        print(
            f"trinity right head={head_right.shape[1]}x{head_right.shape[0]} xy={right_head_xy} "
            f"body_xy={right_body_xy} front head={head_front.shape[1]}x{head_front.shape[0]} "
            f"xy={front_head_xy} body={body_front.shape[1]}x{body_front.shape[0]} "
            f"body_xy={front_body_xy} canonical_body_w={canonical_w}"
        )
        print(f"trinity: {trinity}")
        return trinity

    def fix_front_body(self, character_id: str) -> Path:
        """Regenerate the frontal body and raise the approved front head. Sides stay locked."""
        import cv2
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama
        from .puppet_assembler import CanonicalDebaterContract, dock_front_grounded

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"front-body fix is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        front_dir = skin_dir / "views" / "facing_front"
        manifest_path = skin_dir / "puppet.json"
        locked_paths = (
            left_dir / "head.png",
            left_dir / "body.png",
            right_dir / "head.png",
            right_dir / "body.png",
            front_dir / "head.png",
        )
        stamps = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in locked_paths}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        front = dict(manifest["views"]["facing_front"])
        head = np.asarray(Image.open(front_dir / "head.png").convert("RGBA"))
        head_x = 540 - (head.shape[1] // 2)
        head_xy = (int(head_x), int(front["head_xy"][1]) - 18)
        reference = front_dir / "_body_reference_magenta.png"
        _flatten_on_magenta(left_dir / "body.png", reference)
        source = front_dir / "source_body_grounded.png"
        generate_character_image(
            FRONT_BODY_GROUNDED_PROMPT,
            source,
            aspect_ratio="9:16",
            reference_path=reference,
        )
        with Image.open(source) as opened:
            body_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        body_master.save(front_dir / "_body_master_grounded.png", format="PNG", compress_level=1)
        raw = np.asarray(body_master)
        width_scale = CanonicalDebaterContract.BODY_WIDTH / max(1, raw.shape[1])
        body = cv2.resize(raw, None, fx=width_scale, fy=width_scale, interpolation=cv2.INTER_LANCZOS4)
        body, body_xy = dock_front_grounded(head, body, head_xy=head_xy)
        Image.fromarray(body).save(front_dir / "body.png", format="PNG", compress_level=1)
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"locked file changed: {path}")
        front.update(
            {
                "head_xy": list(head_xy),
                "body_xy": list(body_xy),
                "body_width": int(body.shape[1]),
                "body_height": int(body.shape[0]),
                "body_bottom_y": int(body_xy[1] + body.shape[0]),
                "stage_x": 540,
                "head_nudge_up_px": 18,
                "status": "locked_approved",
            }
        )
        manifest["views"]["facing_front"] = front
        manifest["views"]["facing_left"]["status"] = "locked_approved"
        manifest["views"]["facing_right"]["status"] = "locked_approved"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=(head_xy, body_xy),
            preview_name="claude_facing_front_preview.png",
        )
        left = manifest["views"]["facing_left"]
        right = manifest["views"]["facing_right"]
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        panels = (
            ("LEFT", left_dir / "head.png", left_dir / "body.png", left["head_xy"], left["body_xy"]),
            ("FRONT", front_dir / "head.png", front_dir / "body.png", head_xy, body_xy),
            ("RIGHT", right_dir / "head.png", right_dir / "body.png", right["head_xy"], right["body_xy"]),
        )
        sheet = Image.new("RGB", (1080 * 3, 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, head_path, body_path, panel_head_xy, panel_body_xy) in enumerate(panels):
            panel = backdrop.copy()
            panel_body = Image.open(body_path).convert("RGBA")
            panel_head = Image.open(head_path).convert("RGBA")
            _composite_at(panel, panel_body, int(panel_body_xy[0]), int(panel_body_xy[1]))
            _composite_at(panel, panel_head, int(panel_head_xy[0]), int(panel_head_xy[1]))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 320, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
        trinity = outputs_root() / "aiwake" / "_test_harness" / "claude_trinity_complete.png"
        sheet.save(trinity, format="PNG", compress_level=1)
        print(
            f"front-fix head_xy={head_xy} body={body.shape[1]}x{body.shape[0]} "
            f"body_xy={body_xy} bottom={body_xy[1] + body.shape[0]}"
        )
        print(f"front-fix: {trinity}")
        return trinity

    def settle_front_shoulders(self, character_id: str) -> Path:
        """Drop the front head a little, grow the approved body 5%, and lower it to the side shoulders."""
        import cv2
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"front settle is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        front_dir = skin_dir / "views" / "facing_front"
        manifest_path = skin_dir / "puppet.json"
        locked = (
            left_dir / "head.png",
            left_dir / "body.png",
            right_dir / "head.png",
            right_dir / "body.png",
            front_dir / "head.png",
        )
        stamps = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in locked}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        front = dict(manifest["views"]["facing_front"])
        head = np.asarray(Image.open(front_dir / "head.png").convert("RGBA"))
        body = np.asarray(Image.open(front_dir / "body.png").convert("RGBA"))
        body = cv2.resize(body, None, fx=1.05, fy=1.05, interpolation=cv2.INTER_LANCZOS4)
        head_xy = (540 - head.shape[1] // 2, int(front["head_xy"][1]) + 18)
        chin_local = int(np.nonzero((head[..., 3] > 16).any(axis=1))[0][-1])
        body_alpha = body[..., 3] > 16
        body_rows = np.nonzero(body_alpha.any(axis=1))[0]
        widths = body_alpha.sum(axis=1)
        top_band = body_rows[: max(1, int(len(body_rows) * 0.40))]
        shoulder_local = int(top_band[np.argmax(widths[top_band])])
        shoulder_y = (head_xy[1] + chin_local) + 208
        body_y = shoulder_y - shoulder_local
        body_x = 540 - body.shape[1] // 2
        body_xy = (int(body_x), int(body_y))
        Image.fromarray(body).save(front_dir / "body.png", format="PNG", compress_level=1)
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"locked file changed: {path}")
        front.update(
            {
                "head_xy": list(head_xy),
                "body_xy": list(body_xy),
                "body_width": int(body.shape[1]),
                "body_height": int(body.shape[0]),
                "body_bottom_y": int(body_xy[1] + body.shape[0]),
                "body_uniform_scale": 1.05,
                "head_nudge_down_px": 18,
                "shoulder_gap_px": 208,
                "status": "locked_approved",
            }
        )
        manifest["views"]["facing_front"] = front
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        left = manifest["views"]["facing_left"]
        right = manifest["views"]["facing_right"]
        panels = (
            ("LEFT", left_dir / "head.png", left_dir / "body.png", left["head_xy"], left["body_xy"]),
            ("FRONT", front_dir / "head.png", front_dir / "body.png", head_xy, body_xy),
            ("RIGHT", right_dir / "head.png", right_dir / "body.png", right["head_xy"], right["body_xy"]),
        )
        sheet = Image.new("RGB", (1080 * 3, 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, head_path, body_path, panel_head_xy, panel_body_xy) in enumerate(panels):
            panel = backdrop.copy()
            _composite_at(panel, Image.open(body_path).convert("RGBA"), int(panel_body_xy[0]), int(panel_body_xy[1]))
            _composite_at(panel, Image.open(head_path).convert("RGBA"), int(panel_head_xy[0]), int(panel_head_xy[1]))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 320, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
        trinity = outputs_root() / "aiwake" / "_test_harness" / "claude_trinity_complete.png"
        sheet.save(trinity, format="PNG", compress_level=1)
        print(
            f"front-settle head_xy={head_xy} body={body.shape[1]}x{body.shape[0]} "
            f"body_xy={body_xy} shoulder_y={shoulder_y} "
            f"bottom={body_xy[1] + body.shape[0]}"
        )
        print(f"front-settle: {trinity}")
        return trinity

    def restore_front_neck(self, character_id: str) -> Path:
        """Regenerate only the frontal body so it grows the side-view accordion neck."""
        import cv2
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"front neck restore is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        front_dir = skin_dir / "views" / "facing_front"
        manifest_path = skin_dir / "puppet.json"
        locked = (
            left_dir / "head.png",
            left_dir / "body.png",
            right_dir / "head.png",
            right_dir / "body.png",
            front_dir / "head.png",
        )
        stamps = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in locked}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        front = dict(manifest["views"]["facing_front"])
        head_xy = (int(front["head_xy"][0]), int(front["head_xy"][1]))
        shoulder_y = int(front.get("body_xy")[1])
        current_body = np.asarray(Image.open(front_dir / "body.png").convert("RGBA"))
        current_alpha = current_body[..., 3] > 16
        current_rows = np.nonzero(current_alpha.any(axis=1))[0]
        current_widths = current_alpha.sum(axis=1)
        current_top = current_rows[: max(1, int(len(current_rows) * 0.40))]
        shoulder_y = int(front["body_xy"][1]) + int(current_top[np.argmax(current_widths[current_top])])
        target_w = int(current_body.shape[1])
        reference = front_dir / "_body_reference_magenta.png"
        _flatten_on_magenta(left_dir / "body.png", reference)
        source = front_dir / "source_body_neck.png"
        generate_character_image(
            FRONT_BODY_NECK_PROMPT,
            source,
            aspect_ratio="9:16",
            reference_path=reference,
        )
        with Image.open(source) as opened:
            body_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        body_master.save(front_dir / "_body_master_neck.png", format="PNG", compress_level=1)
        raw = np.asarray(body_master)
        scale = target_w / max(1, raw.shape[1])
        body = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        alpha = body[..., 3] > 16
        widths = alpha.sum(axis=1)
        flare = np.nonzero(widths > int(widths.max() * 0.55))[0]
        shoulder_local = int(flare[0]) if flare.size else 0
        neck_px = shoulder_local
        body_x = 540 - body.shape[1] // 2
        body_y = shoulder_y - shoulder_local
        body_xy = (int(body_x), int(body_y))
        Image.fromarray(body).save(front_dir / "body.png", format="PNG", compress_level=1)
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"locked file changed: {path}")
        front.update(
            {
                "body_xy": list(body_xy),
                "body_width": int(body.shape[1]),
                "body_height": int(body.shape[0]),
                "body_bottom_y": int(body_xy[1] + body.shape[0]),
                "neck_px": int(neck_px),
                "shoulder_y": int(shoulder_y),
                "status": "locked_approved",
            }
        )
        manifest["views"]["facing_front"] = front
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        left = manifest["views"]["facing_left"]
        right = manifest["views"]["facing_right"]
        panels = (
            ("LEFT", left_dir / "head.png", left_dir / "body.png", left["head_xy"], left["body_xy"]),
            ("FRONT", front_dir / "head.png", front_dir / "body.png", head_xy, body_xy),
            ("RIGHT", right_dir / "head.png", right_dir / "body.png", right["head_xy"], right["body_xy"]),
        )
        sheet = Image.new("RGB", (1080 * 3, 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, head_path, body_path, panel_head_xy, panel_body_xy) in enumerate(panels):
            panel = backdrop.copy()
            _composite_at(panel, Image.open(body_path).convert("RGBA"), int(panel_body_xy[0]), int(panel_body_xy[1]))
            _composite_at(panel, Image.open(head_path).convert("RGBA"), int(panel_head_xy[0]), int(panel_head_xy[1]))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 320, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
        harness = outputs_root() / "aiwake" / "_test_harness"
        trinity = harness / "claude_trinity_complete.png"
        fresh = harness / "claude_trinity_neck.png"
        sheet.save(trinity, format="PNG", compress_level=1)
        sheet.save(fresh, format="PNG", compress_level=1)
        print(
            f"front-neck body={body.shape[1]}x{body.shape[0]} body_xy={body_xy} "
            f"neck_px={neck_px} shoulder_y={shoulder_y} bottom={body_xy[1] + body.shape[0]}"
        )
        print(f"front-neck: {fresh}")
        return fresh

    def shorten_front_neck(self, character_id: str) -> Path:
        """Put a short side-matched accordion on the approved frontal torso. No new generation."""
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"front neck shorten is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        front_dir = skin_dir / "views" / "facing_front"
        manifest_path = skin_dir / "puppet.json"
        locked = (
            left_dir / "head.png",
            left_dir / "body.png",
            right_dir / "head.png",
            right_dir / "body.png",
            front_dir / "head.png",
        )
        stamps = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in locked}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        front = dict(manifest["views"]["facing_front"])
        head_xy = (int(front["head_xy"][0]), int(front["head_xy"][1]))
        torso = Image.open(front_dir / "_body_master_grounded.png").convert("RGBA")
        torso = torso.resize((920, max(1, int(round(torso.height * 920 / torso.width)))), Image.Resampling.LANCZOS)
        neck_src = np.asarray(Image.open(front_dir / "_body_master_neck.png").convert("RGBA"))
        alpha = neck_src[..., 3] > 16
        widths = alpha.sum(axis=1)
        rows = np.nonzero(alpha.any(axis=1))[0]
        flare = np.nonzero(widths > int(widths.max() * 0.42))[0]
        neck_end = int(flare[0]) if flare.size else int(rows[0] + 80)
        neck_end = max(neck_end, int(rows[0]) + 40)
        column = Image.fromarray(neck_src[int(rows[0]) : neck_end]).crop(
            _opaque_box(neck_src[int(rows[0]) : neck_end])
        )
        target_w = max(48, int(round(torso.width * 0.22)))
        target_h = max(48, int(round(torso.height * (160 / 1440))))
        uniform = target_w / max(1, column.width)
        fitted = column.resize(
            (target_w, max(1, int(round(column.height * uniform)))),
            Image.Resampling.LANCZOS,
        )
        if fitted.height > target_h:
            fitted = fitted.crop((0, fitted.height - target_h, fitted.width, fitted.height))
        overlap = 36
        protrude = max(0, fitted.height - overlap)
        canvas = Image.new("RGBA", (torso.width, torso.height + protrude), (0, 0, 0, 0))
        canvas.paste(torso, (0, protrude))
        neck_x = (torso.width - fitted.width) // 2
        canvas.alpha_composite(fitted, (neck_x, 0))
        body = np.asarray(canvas)
        body_x = 540 - body.shape[1] // 2
        body_y = 1004 - protrude
        body_xy = (int(body_x), int(body_y))
        Image.fromarray(body).save(front_dir / "body.png", format="PNG", compress_level=1)
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"locked file changed: {path}")
        front.update(
            {
                "body_xy": list(body_xy),
                "body_width": int(body.shape[1]),
                "body_height": int(body.shape[0]),
                "body_bottom_y": int(body_xy[1] + body.shape[0]),
                "neck_px": int(fitted.height),
                "neck_protrude_px": int(protrude),
                "status": "locked_approved",
            }
        )
        manifest["views"]["facing_front"] = front
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        left = manifest["views"]["facing_left"]
        right = manifest["views"]["facing_right"]
        panels = (
            ("LEFT", left_dir / "head.png", left_dir / "body.png", left["head_xy"], left["body_xy"]),
            ("FRONT", front_dir / "head.png", front_dir / "body.png", head_xy, body_xy),
            ("RIGHT", right_dir / "head.png", right_dir / "body.png", right["head_xy"], right["body_xy"]),
        )
        sheet = Image.new("RGB", (1080 * 3, 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, head_path, body_path, panel_head_xy, panel_body_xy) in enumerate(panels):
            panel = backdrop.copy()
            _composite_at(panel, Image.open(body_path).convert("RGBA"), int(panel_body_xy[0]), int(panel_body_xy[1]))
            _composite_at(panel, Image.open(head_path).convert("RGBA"), int(panel_head_xy[0]), int(panel_head_xy[1]))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 320, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
        fresh = outputs_root() / "aiwake" / "_test_harness" / "claude_trinity_shortneck.png"
        sheet.save(fresh, format="PNG", compress_level=1)
        print(
            f"front-short-neck body={body.shape[1]}x{body.shape[0]} body_xy={body_xy} "
            f"neck={fitted.width}x{fitted.height} protrude={protrude} "
            f"bottom={body_xy[1] + body.shape[0]}"
        )
        print(f"front-short-neck: {fresh}")
        return fresh

    def seat_front_neck(self, character_id: str, neck_px: int = 12) -> Path:
        """Seat a 10–15px accordion between the front chin and the approved collar."""
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"front neck seat is not wired for {character_id}")
        neck_px = max(10, min(15, int(neck_px)))
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        front_dir = skin_dir / "views" / "facing_front"
        manifest_path = skin_dir / "puppet.json"
        locked = (
            left_dir / "head.png",
            left_dir / "body.png",
            right_dir / "head.png",
            right_dir / "body.png",
            front_dir / "head.png",
        )
        stamps = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in locked}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        front = dict(manifest["views"]["facing_front"])
        head_xy = (int(front["head_xy"][0]), int(front["head_xy"][1]))
        head = np.asarray(Image.open(front_dir / "head.png").convert("RGBA"))
        chin_local = int(np.nonzero(head[:, head.shape[1] // 2, 3] > 16)[0][-1])
        chin_y = head_xy[1] + chin_local
        torso = Image.open(front_dir / "_body_master_grounded.png").convert("RGBA")
        torso = torso.resize((920, max(1, int(round(torso.height * 920 / torso.width)))), Image.Resampling.LANCZOS)
        torso_px = np.asarray(torso)
        brass = (
            (torso_px[..., 3] > 16)
            & (torso_px[..., 0] > 140)
            & (torso_px[..., 1] > 110)
            & (torso_px[..., 0] > torso_px[..., 2] + 30)
        )
        ring_rows = np.nonzero(brass.sum(axis=1) > 40)[0]
        ring = int(ring_rows[0]) if ring_rows.size else 8
        # Bright rib from the locked side neck. The giraffe column averages to a black bar.
        side = np.asarray(Image.open(left_dir / "body.png").convert("RGBA"))
        band = side[80:100]
        band_img = Image.fromarray(band).crop(_opaque_box(band))
        neck_w = max(96, int(round(torso.width * 0.22)))
        fitted = band_img.resize((neck_w, neck_px), Image.Resampling.LANCZOS)
        # Ring starts at the bottom edge of the 12px neck, so the neck is the
        # first thing under the chin and the collar touches it.
        torso_y = max(0, neck_px - ring)
        canvas_h = torso_y + torso.height
        canvas = Image.new("RGBA", (torso.width, canvas_h), (0, 0, 0, 0))
        canvas.paste(torso, (0, torso_y))
        canvas.alpha_composite(fitted, ((torso.width - neck_w) // 2, 0))
        body = np.asarray(canvas)
        body_xy = (540 - body.shape[1] // 2, int(chin_y))
        Image.fromarray(body).save(front_dir / "body.png", format="PNG", compress_level=1)
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"locked file changed: {path}")
        front.update(
            {
                "head_xy": list(head_xy),
                "body_xy": list(body_xy),
                "body_width": int(body.shape[1]),
                "body_height": int(body.shape[0]),
                "body_bottom_y": int(body_xy[1] + body.shape[0]),
                "neck_px": neck_px,
                "status": "locked_approved",
            }
        )
        manifest["views"]["facing_front"] = front
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        left = manifest["views"]["facing_left"]
        right = manifest["views"]["facing_right"]
        panels = (
            ("LEFT", left_dir / "head.png", left_dir / "body.png", left["head_xy"], left["body_xy"]),
            ("FRONT", front_dir / "head.png", front_dir / "body.png", head_xy, body_xy),
            ("RIGHT", right_dir / "head.png", right_dir / "body.png", right["head_xy"], right["body_xy"]),
        )
        sheet = Image.new("RGB", (1080 * 3, 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, head_path, body_path, panel_head_xy, panel_body_xy) in enumerate(panels):
            panel = backdrop.copy()
            _composite_at(panel, Image.open(body_path).convert("RGBA"), int(panel_body_xy[0]), int(panel_body_xy[1]))
            _composite_at(panel, Image.open(head_path).convert("RGBA"), int(panel_head_xy[0]), int(panel_head_xy[1]))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 320, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
        fresh = outputs_root() / "aiwake" / "_test_harness" / "claude_trinity_neck12c.png"
        sheet.save(fresh, format="PNG", compress_level=1)
        print(
            f"front-neck12 head_xy={head_xy} chin_y={chin_y} body_xy={body_xy} "
            f"neck={neck_w if False else fitted.width}x{neck_px} bottom={body_xy[1] + body.shape[0]}"
        )
        print(f"front-neck12: {fresh}")
        return fresh

    def recover_giraffe_neck(self, character_id: str, *, scale: float = 1.10, head_drop_px: int = 28) -> Path:
        """Restore the accordion body from claude_trinity_neck.png, scale it, and seat the head on it."""
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"giraffe recover is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        front_dir = skin_dir / "views" / "facing_front"
        manifest_path = skin_dir / "puppet.json"
        locked = (
            left_dir / "head.png",
            left_dir / "body.png",
            right_dir / "head.png",
            right_dir / "body.png",
            front_dir / "head.png",
        )
        stamps = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in locked}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        front = dict(manifest["views"]["facing_front"])
        head = np.asarray(Image.open(front_dir / "head.png").convert("RGBA"))
        chin_local = int(np.nonzero(head[:, head.shape[1] // 2, 3] > 16)[0][-1])
        old_head_xy = (int(front["head_xy"][0]), int(front["head_xy"][1]))
        old_chin = old_head_xy[1] + chin_local
        master = Image.open(front_dir / "_body_master_neck.png").convert("RGBA")
        # Width the accordion had inside claude_trinity_neck.png, then +10%.
        placed_w = 920
        contact_local = int(round((old_chin - 406) * master.width / placed_w))
        final_w = int(round(placed_w * scale))
        final_h = max(1, int(round(master.height * final_w / master.width)))
        body_img = master.resize((final_w, final_h), Image.Resampling.LANCZOS)
        contact_final = int(round(contact_local * final_w / master.width))
        body_xy = (540 - final_w // 2, int(old_chin - contact_final))
        head_xy = (old_head_xy[0], old_head_xy[1] + int(head_drop_px))
        body_img.save(front_dir / "body.png", format="PNG", compress_level=1)
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"locked file changed: {path}")
        body = np.asarray(body_img)
        front.update(
            {
                "head_xy": list(head_xy),
                "body_xy": list(body_xy),
                "body_width": int(body.shape[1]),
                "body_height": int(body.shape[0]),
                "body_bottom_y": int(body_xy[1] + body.shape[0]),
                "neck_scale": float(scale),
                "head_drop_onto_neck_px": int(head_drop_px),
                "status": "locked_approved",
            }
        )
        manifest["views"]["facing_front"] = front
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        left = manifest["views"]["facing_left"]
        right = manifest["views"]["facing_right"]
        panels = (
            ("LEFT", left_dir / "head.png", left_dir / "body.png", left["head_xy"], left["body_xy"]),
            ("FRONT", front_dir / "head.png", front_dir / "body.png", head_xy, body_xy),
            ("RIGHT", right_dir / "head.png", right_dir / "body.png", right["head_xy"], right["body_xy"]),
        )
        sheet = Image.new("RGB", (1080 * 3, 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, head_path, body_path, panel_head_xy, panel_body_xy) in enumerate(panels):
            panel = backdrop.copy()
            _composite_at(panel, Image.open(body_path).convert("RGBA"), int(panel_body_xy[0]), int(panel_body_xy[1]))
            _composite_at(panel, Image.open(head_path).convert("RGBA"), int(panel_head_xy[0]), int(panel_head_xy[1]))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 320, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
        fresh = outputs_root() / "aiwake" / "_test_harness" / "claude_trinity_neck_plus10.png"
        sheet.save(fresh, format="PNG", compress_level=1)
        print(
            f"giraffe+10 head_xy={head_xy} body={body.shape[1]}x{body.shape[0]} "
            f"body_xy={body_xy} bottom={body_xy[1] + body.shape[0]}"
        )
        print(f"giraffe+10: {fresh}")
        return fresh

    def crop_neck_and_raise_body(
        self,
        character_id: str,
        *,
        crop_px: int = 15,
        raise_px: int = 40,
        preview_name: str = "claude_trinity_neck_up40.png",
    ) -> Path:
        """Cut the top of the front neck and move only the body upward."""
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"neck crop is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        front_dir = skin_dir / "views" / "facing_front"
        manifest_path = skin_dir / "puppet.json"
        locked = (
            left_dir / "head.png",
            left_dir / "body.png",
            right_dir / "head.png",
            right_dir / "body.png",
            front_dir / "head.png",
        )
        stamps = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in locked}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        front = dict(manifest["views"]["facing_front"])
        head_xy = (int(front["head_xy"][0]), int(front["head_xy"][1]))
        body_img = Image.open(front_dir / "body.png").convert("RGBA")
        alpha = np.asarray(body_img)[..., 3] > 16
        opaque_rows = np.nonzero(alpha.any(axis=1))[0]
        if opaque_rows.size == 0:
            raise RuntimeError("front body has no opaque pixels")
        cut = int(opaque_rows[0]) + int(crop_px)
        body_img = body_img.crop((0, cut, body_img.width, body_img.height))
        body_xy = (int(front["body_xy"][0]), int(front["body_xy"][1]) - int(raise_px))
        body_img.save(front_dir / "body.png", format="PNG", compress_level=1)
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"locked file changed: {path}")
        front.update(
            {
                "head_xy": list(head_xy),
                "body_xy": list(body_xy),
                "body_width": int(body_img.width),
                "body_height": int(body_img.height),
                "body_bottom_y": int(body_xy[1] + body_img.height),
                "neck_crop_px": int(crop_px),
                "body_raise_px": int(raise_px),
                "status": "locked_approved",
            }
        )
        manifest["views"]["facing_front"] = front
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        left = manifest["views"]["facing_left"]
        right = manifest["views"]["facing_right"]
        panels = (
            ("LEFT", left_dir / "head.png", left_dir / "body.png", left["head_xy"], left["body_xy"]),
            ("FRONT", front_dir / "head.png", front_dir / "body.png", head_xy, body_xy),
            ("RIGHT", right_dir / "head.png", right_dir / "body.png", right["head_xy"], right["body_xy"]),
        )
        sheet = Image.new("RGB", (1080 * 3, 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, head_path, body_path, panel_head_xy, panel_body_xy) in enumerate(panels):
            panel = backdrop.copy()
            _composite_at(panel, Image.open(body_path).convert("RGBA"), int(panel_body_xy[0]), int(panel_body_xy[1]))
            _composite_at(panel, Image.open(head_path).convert("RGBA"), int(panel_head_xy[0]), int(panel_head_xy[1]))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 320, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
        fresh = outputs_root() / "aiwake" / "_test_harness" / preview_name
        sheet.save(fresh, format="PNG", compress_level=1)
        print(
            f"neck-crop{crop_px} raise{raise_px} head_xy={head_xy} body_xy={body_xy} "
            f"body={body_img.size} bottom={body_xy[1] + body_img.height}"
        )
        print(f"neck-up40: {fresh}")
        return fresh

    def raise_and_scale_front_body(
        self, character_id: str, *, raise_px: int = 25, scale: float = 1.02
    ) -> Path:
        """Scale the front body and move it up. The head stays put."""
        from PIL import ImageDraw, ImageFont

        from core.animator.asset_generator import DEFAULT_PUPPETS_DIR, ensure_shared_panorama

        if character_id != "claude_cyborg_v1":
            raise SystemExit(f"front body nudge is not wired for {character_id}")
        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        front_dir = skin_dir / "views" / "facing_front"
        manifest_path = skin_dir / "puppet.json"
        locked = (
            left_dir / "head.png",
            left_dir / "body.png",
            right_dir / "head.png",
            right_dir / "body.png",
            front_dir / "head.png",
        )
        stamps = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in locked}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        front = dict(manifest["views"]["facing_front"])
        head_xy = (int(front["head_xy"][0]), int(front["head_xy"][1]))
        body_img = Image.open(front_dir / "body.png").convert("RGBA")
        final_w = max(1, int(round(body_img.width * scale)))
        final_h = max(1, int(round(body_img.height * scale)))
        body_img = body_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
        old_x, old_y = int(front["body_xy"][0]), int(front["body_xy"][1])
        old_w = int(front["body_width"])
        center_x = old_x + old_w // 2
        body_xy = (center_x - final_w // 2, old_y - int(raise_px))
        body_img.save(front_dir / "body.png", format="PNG", compress_level=1)
        for path, stamp in stamps.items():
            if (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise RuntimeError(f"locked file changed: {path}")
        front.update(
            {
                "head_xy": list(head_xy),
                "body_xy": list(body_xy),
                "body_width": int(body_img.width),
                "body_height": int(body_img.height),
                "body_bottom_y": int(body_xy[1] + body_img.height),
                "body_raise_px": int(front.get("body_raise_px", 0)) + int(raise_px),
                "body_scale_extra": float(scale),
                "status": "locked_approved",
            }
        )
        manifest["views"]["facing_front"] = front
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        panorama = Image.open(ensure_shared_panorama(puppets_dir=DEFAULT_PUPPETS_DIR)).convert("RGB")
        library = panorama.crop((panorama.width // 2, 0, panorama.width, panorama.height))
        backdrop = library.resize((1080, 1920), Image.Resampling.LANCZOS).convert("RGBA")
        left = manifest["views"]["facing_left"]
        right = manifest["views"]["facing_right"]
        panels = (
            ("LEFT", left_dir / "head.png", left_dir / "body.png", left["head_xy"], left["body_xy"]),
            ("FRONT", front_dir / "head.png", front_dir / "body.png", head_xy, body_xy),
            ("RIGHT", right_dir / "head.png", right_dir / "body.png", right["head_xy"], right["body_xy"]),
        )
        sheet = Image.new("RGB", (1080 * 3, 1920), (12, 10, 8))
        try:
            font = ImageFont.truetype("arialbd.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, head_path, body_path, panel_head_xy, panel_body_xy) in enumerate(panels):
            panel = backdrop.copy()
            _composite_at(panel, Image.open(body_path).convert("RGBA"), int(panel_body_xy[0]), int(panel_body_xy[1]))
            _composite_at(panel, Image.open(head_path).convert("RGBA"), int(panel_head_xy[0]), int(panel_head_xy[1]))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((36, 28, 320, 96), fill=(18, 12, 8, 210))
            draw.text((52, 40), label, fill=(244, 214, 150, 255), font=font)
            sheet.paste(panel.convert("RGB"), (index * 1080, 0))
        fresh = outputs_root() / "aiwake" / "_test_harness" / "claude_trinity_neck_up25.png"
        sheet.save(fresh, format="PNG", compress_level=1)
        print(
            f"body+2 raise{raise_px} head_xy={head_xy} body_xy={body_xy} "
            f"body={body_img.size} bottom={body_xy[1] + body_img.height}"
        )
        print(f"neck-up25: {fresh}")
        return fresh

    def generate_screen_right_head(self, character_id: str, *, preflip: bool = False) -> Path:
        """Generate only the screen-right head and mirror the approved body."""
        import cv2

        from .puppet_assembler import CanonicalDebaterContract, mirror_approved_body

        skin_dir = assets_root() / "puppets" / character_id
        left_dir = skin_dir / "views" / "facing_left"
        right_dir = skin_dir / "views" / "facing_right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        for name in ("head.png", "body.png"):
            source = skin_dir / name
            destination = left_dir / name
            if source.is_file() and not destination.is_file():
                shutil.copy2(source, destination)
        approved_body_path = left_dir / "body.png"
        approved_head_path = left_dir / "head.png"
        if not approved_body_path.is_file() or not approved_head_path.is_file():
            raise RuntimeError(f"approved facing_left layers missing in {left_dir}")
        approved_body = np.asarray(Image.open(approved_body_path).convert("RGBA"))
        body_right = mirror_approved_body(approved_body)
        Image.fromarray(body_right).save(right_dir / "body.png", format="PNG", compress_level=1)
        if preflip:
            approved_head = np.asarray(Image.open(approved_head_path).convert("RGBA"))
            flipped = cv2.flip(approved_head, 1)
            reference = right_dir / "temp_pre_flipped_head.png"
            plate = Image.new("RGBA", (flipped.shape[1], flipped.shape[0]), (255, 0, 255, 255))
            plate.alpha_composite(Image.fromarray(flipped))
            plate.convert("RGB").save(reference, format="PNG", compress_level=1)
            prompt = CORRECT_TEXT_ON_FLIPPED_HEAD_PROMPT
            head_source = right_dir / "source_head_preflipped.png"
        else:
            reference = right_dir / "_head_reference_magenta.png"
            _flatten_on_magenta(approved_head_path, reference)
            prompt = HEAD_SCREEN_RIGHT_PROMPT
            head_source = right_dir / "source_head_screen_right.png"
        generate_character_image(
            prompt,
            head_source,
            aspect_ratio="1:1",
            reference_path=reference,
        )
        with Image.open(head_source) as opened:
            head_master = _tight_alpha_crop(birefnet_cutout(opened.convert("RGB")))
        head_master.save(right_dir / "_head_master.png", format="PNG", compress_level=1)
        head, body, head_xy, body_xy, _pivot = CanonicalDebaterContract.dock_screen_right(
            np.asarray(head_master),
            body_right,
            head_height=721,
            stage_center_x=480,
        )
        Image.fromarray(head).save(right_dir / "head.png", format="PNG", compress_level=1)
        manifest_path = skin_dir / "puppet.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {"character_id": character_id, "skin_version": "v3-auto-rig"}
        )
        manifest["name"] = manifest.get("name") or "DEEPSEEK"
        views = dict(manifest.get("views") or {})
        views["facing_left"] = {
            "head": "views/facing_left/head.png",
            "body": "views/facing_left/body.png",
            "stage_x": 585,
            "tilt_deg": -3.5,
        }
        views["facing_right"] = {
            "head": "views/facing_right/head.png",
            "body": "views/facing_right/body.png",
            "stage_x": 480,
            "tilt_deg": 3.5,
        }
        manifest["views"] = views
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        preview = self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=(head_xy, body_xy),
            preview_name="deepseek_facing_right_preview.png",
        )
        print(f"screen-right: {preview}")
        return skin_dir

    def assemble_facing_right(
        self,
        character_id: str,
        *,
        body_shift_left: int = 55,
        head_shift_right: int = 45,
        head_drop_px: int = 18,
    ) -> Path:
        """Dock the saved screen-right head and mirrored body. Does not call an image API."""
        from .puppet_assembler import CanonicalDebaterContract

        skin_dir = assets_root() / "puppets" / character_id
        right_dir = skin_dir / "views" / "facing_right"
        head_master = right_dir / "_head_master.png"
        body_path = right_dir / "body.png"
        if not head_master.is_file() or not body_path.is_file():
            raise RuntimeError(f"facing_right layers missing in {right_dir}")
        head_rgba = np.asarray(Image.open(head_master).convert("RGBA"))
        body_rgba = np.asarray(Image.open(body_path).convert("RGBA"))
        head, body, head_xy, body_xy, _pivot = CanonicalDebaterContract.dock_screen_right(
            head_rgba,
            body_rgba,
            head_height=721,
            stage_center_x=480,
            body_shift_left=body_shift_left,
            head_shift_right=head_shift_right,
            head_drop_px=head_drop_px,
        )
        Image.fromarray(head).save(right_dir / "head.png", format="PNG", compress_level=1)
        Image.fromarray(body).save(right_dir / "body.png", format="PNG", compress_level=1)
        manifest_path = skin_dir / "puppet.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dock = manifest.get("auto_dock") or {}
        views = dict(manifest.get("views") or {})
        views["facing_left"] = {
            "head": "views/facing_left/head.png",
            "body": "views/facing_left/body.png",
            "stage_x": 585,
            "tilt_deg": -3.5,
            "eye_y": dock.get("target_eye_y", 600),
            "head_xy": dock.get("head_xy"),
            "body_xy": dock.get("body_xy"),
            "body_shift_right": dock.get("body_shift_right"),
            "head_rel_shift_left": dock.get("head_rel_shift_left"),
            "body_drop_px": dock.get("body_drop_px"),
            "body_target_w": dock.get("body_target_w", 876),
        }
        views["facing_right"] = {
            "head": "views/facing_right/head.png",
            "body": "views/facing_right/body.png",
            "stage_x": 480,
            "tilt_deg": 3.5,
            "eye_y": 600,
            "head_height": 721,
            "body_width": int(body.shape[1]),
            "body_scale_y_mult": 1.15,
            "body_shift_left": body_shift_left,
            "head_shift_right": head_shift_right,
            "head_drop_px": head_drop_px,
            "head_xy": list(head_xy),
            "body_xy": list(body_xy),
        }
        manifest["views"] = views
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        preview = self._write_preview(
            character_id,
            Image.fromarray(head),
            Image.fromarray(body),
            bg_contrast="assembled",
            placed=(head_xy, body_xy),
            preview_name="deepseek_facing_right_preview.png",
        )
        print(f"facing-right-dock: {preview}")
        return skin_dir

    def normalize_view(self, character_id: str, view: str, ref_view: str, *, exact_match: bool = False) -> Path:
        """Scale one saved head to the reference view's height. Does not call an image API."""
        from .puppet_assembler import CanonicalDebaterContract, normalize_to_exact_ref

        if view != "facing_right" or ref_view != "facing_left":
            raise SystemExit(f"unsupported normalize pair: {view} from {ref_view}")
        skin_dir = assets_root() / "puppets" / character_id
        ref_path = skin_dir / "views" / ref_view / "head.png"
        right_dir = skin_dir / "views" / view
        head_path = right_dir / "head.png"
        body_path = right_dir / "body.png"
        if not ref_path.is_file() or not head_path.is_file() or not body_path.is_file():
            raise RuntimeError("facing view layers missing")
        if not exact_match:
            raise SystemExit("--exact-match is required; boost multipliers are removed")
        target = np.asarray(Image.open(head_path).convert("RGBA"))
        head, exact_h = normalize_to_exact_ref(ref_path, target)
        manifest_path = skin_dir / "puppet.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        right = dict((manifest.get("views") or {}).get("facing_right") or {})
        stage_x = int(right.get("stage_x", 480))
        head_shift_right = int(right.get("head_shift_right", 45))
        head_drop = 18
        body_xy = tuple(right.get("body_xy") or (-28, 657))
        pivot_x = head.shape[1] // 2
        head_y = CanonicalDebaterContract.TARGET_EYE_Y - int(head.shape[0] * 0.42) + head_drop
        head_x = stage_x - pivot_x + head_shift_right
        head_xy = (int(head_x), int(head_y))
        print(
            f"parity-dock head={head.shape[1]}x{head.shape[0]} "
            f"head_xy={head_xy} body_xy={body_xy} eye_y=600 head_drop_px={head_drop}"
        )
        Image.fromarray(head).save(head_path, format="PNG", compress_level=1)
        right.pop("parity_boost", None)
        right.pop("chin_sink_extra_px", None)
        right.update(
            {
                "head": "views/facing_right/head.png",
                "head_height": int(head.shape[0]),
                "head_width": int(head.shape[1]),
                "exact_ref_height": int(exact_h),
                "parity_ref": "views/facing_left/head.png",
                "eye_y": 600,
                "head_drop_px": head_drop,
                "head_xy": list(head_xy),
                "tilt_deg": 3.5,
            }
        )
        views = dict(manifest.get("views") or {})
        views["facing_right"] = right
        manifest["views"] = views
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        body = Image.open(body_path).convert("RGBA")
        preview = self._write_preview(
            character_id,
            Image.fromarray(head),
            body,
            bg_contrast="assembled",
            placed=(head_xy, (int(body_xy[0]), int(body_xy[1]))),
            preview_name="deepseek_facing_right_preview.png",
        )
        print(f"parity-preview: {preview}")
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


def _opaque_box(rgba: np.ndarray) -> tuple[int, int, int, int]:
    alpha = rgba[..., 3] > 16
    ys, xs = np.nonzero(alpha)
    if not ys.size:
        return (0, 0, rgba.shape[1], rgba.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _seat_view(
    head: np.ndarray,
    body: np.ndarray,
    *,
    stage_x: int,
    collar_ratio: float,
    jaw_offset: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Keep the eyes on y=600 and sink the chin into the collar."""
    from .puppet_assembler import ParametricPuppetContract

    pivot_x = head.shape[1] // 2
    head_y = ParametricPuppetContract.CANONICAL_EYE_Y - int(head.shape[0] * 0.42)
    head_x = stage_x + jaw_offset - pivot_x
    collar_rim_y = int(body.shape[0] * 0.17)
    chin_y = int(head.shape[0] * 0.95)
    body_x = stage_x - int(body.shape[1] * collar_ratio)
    body_y = (head_y + chin_y) - collar_rim_y - ParametricPuppetContract.CHIN_SINK_PX
    return (int(head_x), int(head_y)), (int(body_x), int(body_y))


def _flatten_on_magenta(source: Path, destination: Path) -> None:
    """Lay a transparent head on flat magenta for the Flash reference input."""
    with Image.open(source) as opened:
        art = opened.convert("RGBA")
    plate = Image.new("RGBA", art.size, (255, 0, 255, 255))
    plate.alpha_composite(art)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plate.convert("RGB").save(destination, format="PNG", compress_level=1)


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
    """Swap a black stage for white. The artistic prompt is otherwise unchanged."""
    if bg_contrast == "slate_blue":
        return DEEPSEEK_PORTRAIT_PROMPT
    if bg_contrast != "white":
        return prompt
    stage = prompt.replace("solid pure black background", "solid pure white #FFFFFF background")
    stage = stage.replace("Solid pure black background", "Solid pure white #FFFFFF background")
    return stage.replace("pure solid black background", "solid pure white #FFFFFF background")


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
    parser.add_argument("--view", default="")
    parser.add_argument("--normalize-view", default="")
    parser.add_argument("--exact-match", action="store_true")
    parser.add_argument("--ref-view", default="facing_left")
    parser.add_argument("--body-shift-left", type=int, default=0)
    parser.add_argument("--head-shift-right", type=int, default=0)
    parser.add_argument("--generate-view", default="")
    parser.add_argument("--lock-candidate", default="")
    parser.add_argument("--head-compact-y", type=float, default=0.94)
    parser.add_argument("--head-compact-x", type=float, default=1.02)
    parser.add_argument("--head-sink-px", type=int, default=38)
    parser.add_argument("--style-picker", type=int, default=0)
    parser.add_argument("--subtle-shapes", action="store_true")
    parser.add_argument("--elongated-heads", action="store_true")
    parser.add_argument("--tin-man-variants", action="store_true")
    parser.add_argument("--clean-canonical-tinman", action="store_true")
    parser.add_argument("--blank-face-candidates", action="store_true")
    parser.add_argument("--proven-communicator", action="store_true")
    parser.add_argument("--verify-health", action="store_true")
    parser.add_argument("--regenerate-head-only", action="store_true")
    parser.add_argument("--blank-viseme-plate", action="store_true")
    parser.add_argument("--create", default="")
    parser.add_argument("--character-prompt", default="")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--auto-views", default="")
    parser.add_argument("--generate-trinity", action="store_true")
    parser.add_argument("--fix-front-body", action="store_true")
    parser.add_argument("--seal-deepseek-and-setup-parametric-contract", action="store_true")
    parser.add_argument("--body-only", action="store_true")
    parser.add_argument("--generate-screen-right-head", action="store_true")
    parser.add_argument("--generate-screen-right-head-preflipped", action="store_true")
    parser.add_argument("--reference-head", action="store_true")
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
    if args.verify_health:
        if not args.character_id:
            raise SystemExit("--verify-health requires --character-id")
        preview = OneShotPuppetFactory().verify_health(args.character_id)
        print(preview)
        return 0
    if args.seal_deepseek_and_setup_parametric_contract:
        skin = OneShotPuppetFactory().seal_deepseek_contract()
        print(skin)
        return 0
    if args.lock_candidate:
        if not args.character_id:
            raise SystemExit("--lock-candidate requires --character-id")
        skin = OneShotPuppetFactory().lock_candidate(
            args.character_id,
            args.lock_candidate,
            head_scale_pct=args.head_scale_pct,
            head_compact_y=args.head_compact_y,
            head_compact_x=args.head_compact_x,
            head_sink_px=args.head_sink_px,
            body_drop_px=args.body_drop_px,
        )
        print(skin)
        return 0
    if args.style_picker:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
        if not args.character_id:
            raise SystemExit("--style-picker requires --character-id")
        if (
            args.tin_man_variants
            or args.clean_canonical_tinman
            or args.blank_face_candidates
            or args.proven_communicator
        ):
            preview = OneShotPuppetFactory().generate_tin_man_candidates(
                args.character_id,
                clean=args.clean_canonical_tinman,
                blank_face=args.blank_face_candidates,
                proven=args.proven_communicator,
            )
            print(preview)
            return 0
        preview = OneShotPuppetFactory().style_picker(
            args.character_id,
            args.style_picker,
            subtle_shapes=args.subtle_shapes,
            elongated_heads=args.elongated_heads,
        )
        print(preview)
        return 0
    if args.regenerate_head_only:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
        if not args.character_id:
            raise SystemExit("--regenerate-head-only requires --character-id")
        skin = OneShotPuppetFactory().regenerate_head_only(
            args.character_id,
            blank_viseme_plate=args.blank_viseme_plate,
        )
        print(skin)
        return 0
    if args.create:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
        if args.character_prompt:
            if not args.name:
                raise SystemExit("--create requires --name")
            skin = OneShotPuppetFactory().create_canonical_master(
                args.create,
                args.name,
                args.character_prompt,
            )
        elif args.prompt and args.name:
            skin = OneShotPuppetFactory().create_character(args.create, args.prompt)
        else:
            raise SystemExit("--create requires --name and --character-prompt")
        print(skin)
        return 0
    if args.fix_front_body:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
        if not args.character_id:
            raise SystemExit("--fix-front-body requires --character-id")
        trinity = OneShotPuppetFactory().fix_front_body(args.character_id)
        print(trinity)
        return 0
    if args.auto_views and args.generate_trinity:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
        trinity = OneShotPuppetFactory().generate_trinity(args.auto_views)
        print(trinity)
        return 0
    if args.auto_views:
        from .puppet_assembler import ParametricPuppetContract

        manifest_path = assets_root() / "puppets" / args.auto_views / "puppet.json"
        locked = []
        if manifest_path.is_file():
            views = json.loads(manifest_path.read_text(encoding="utf-8")).get("views") or {}
            locked = [name for name, view in views.items() if view.get("status") == "locked_approved"]
        print(
            f"parametric-contract eye_y={ParametricPuppetContract.CANONICAL_EYE_Y} "
            f"head_h={ParametricPuppetContract.target_head_height()} "
            f"stage=({ParametricPuppetContract.STAGE_X_LEFT},"
            f"{ParametricPuppetContract.STAGE_X_CENTER},"
            f"{ParametricPuppetContract.STAGE_X_RIGHT}) "
            f"locked={','.join(locked) or 'none'}"
        )
        return 0
    if args.generate_screen_right_head or args.generate_screen_right_head_preflipped:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
        character_id = args.character_id or "deepseek_cyborg_v3"
        skin = OneShotPuppetFactory().generate_screen_right_head(
            character_id,
            preflip=args.generate_screen_right_head_preflipped,
        )
        print(skin)
        return 0
    if args.generate_view:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
        if not args.character_id:
            raise SystemExit("--character-id is required")
        skin = OneShotPuppetFactory().generate_view(
            args.character_id,
            args.generate_view,
            reference_head=args.reference_head,
            body_only=args.body_only,
        )
        print(skin)
        return 0
    if args.normalize_view:
        character_id = args.character_id or "deepseek_cyborg_v3"
        skin = OneShotPuppetFactory().normalize_view(
            character_id,
            args.normalize_view,
            args.ref_view,
            exact_match=args.exact_match,
        )
        print(skin)
        return 0
    if args.run_assembler and args.view == "facing_right":
        character_id = args.character_id or "deepseek_cyborg_v3"
        skin = OneShotPuppetFactory().assemble_facing_right(
            character_id,
            body_shift_left=args.body_shift_left,
            head_shift_right=args.head_shift_right,
            head_drop_px=args.head_drop_px,
        )
        print(skin)
        return 0
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
