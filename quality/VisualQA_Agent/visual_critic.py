# -*- coding: utf-8 -*-
"""
Multimodal Visual Critic — Gemini Vision Art Director for Master Mei rigor.

Loads the candidate + ALL ``style_reference`` images and returns strict JSON:
``score``, ``passed``, ``flaws``, ``fix_instructions``.

HARD CAP: clean fitness / studio / commercial looks → score must be < 4.0.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator, model_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from quality.VisualQA_Agent import config

_LOG = logging.getLogger(__name__)
_console = Console()

_CRITIC_PROFILE_CACHE: dict[str, dict[str, Any]] = {}
_ANCIENT_MYSTERY_DEFAULT_THRESHOLD: float = 5.5


def load_critic_profile(name: str) -> dict[str, Any]:
    """Load ``critic_profiles/{name}.json`` from the factory root (cached)."""
    key = (name or "").strip().lower()
    if not key:
        return {}
    cached = _CRITIC_PROFILE_CACHE.get(key)
    if cached is not None:
        return cached
    candidates = [
        config.PROJECT_ROOT / "critic_profiles" / f"{key}.json",
        Path(__file__).resolve().parent / "critic_profiles" / f"{key}.json",
    ]
    for path in candidates:
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    _CRITIC_PROFILE_CACHE[key] = data
                    return data
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("critic profile %s unreadable (%s)", path, exc)
    _CRITIC_PROFILE_CACHE[key] = {}
    return {}


def ancient_mystery_quality_threshold(fallback: float | None = None) -> float:
    profile = load_critic_profile("ancient_mystery")
    raw = profile.get("quality_threshold", fallback)
    try:
        return float(raw if raw is not None else _ANCIENT_MYSTERY_DEFAULT_THRESHOLD)
    except (TypeError, ValueError):
        return float(
            fallback if fallback is not None else _ANCIENT_MYSTERY_DEFAULT_THRESHOLD
        )

_CLEAN_MODEL_FLAW_RE = re.compile(
    r"fitness\s*model|gym[\s-]?bro|clean[\s-]?skin|handsome|studio[\s-]?lit|"
    r"studio\s+lighting|softbox|commercial\s+fashion|fashion\s+shoot|"
    r"polished\s+3d|3d\s+render|underwear|calendar\s+model|influencer|"
    r"smooth\s+skin|airbrushed|beauty\s+portrait|clean[\s-]?shaven\s+model",
    re.IGNORECASE,
)

DEFORMED_HAND_FLAW = "deformed or anatomically incorrect hand"
COVERAGE_FLAW = "COVERAGE: incomplete garment / exposed skin on a human figure"
_DEFORMED_HAND_RE = re.compile(
    r"("
    r"too many fingers|too few fingers|extra fingers?|six fingers|seven fingers|"
    r"missing fingers?|duplicated fingers?|twisted fingers?|"
    r"hand.{0,48}(merg\w*|enter\w*|sink\w*|disappear\w*|fused?).{0,24}"
    r"(torso|chest|sternum|body|face|breast)|"
    r"(torso|chest|sternum|body|face).{0,24}(merg\w*|swallow\w*|absorb\w*).{0,24}hand|"
    r"fingers?.{0,24}(into|inside).{0,16}(chest|torso|body)|"
    r"anatomically incorrect hand|deformed.{0,16}hand"
    r")",
    re.IGNORECASE,
)


class CriticVerdict(BaseModel):
    """Structured Art Director evaluation (Gemini response_schema)."""

    score: float = Field(..., ge=0.0, le=10.0, description="Quality score 0–10.")
    passed: bool = Field(
        ..., description="True only if score >= QUALITY_THRESHOLD and no hard flaws."
    )
    flaws: list[str] = Field(
        default_factory=list,
        description="Concrete visual flaws found in the generated image.",
    )
    fix_instructions: str = Field(
        default="",
        description="Precise rewrite instructions for the next prompt attempt.",
    )
    corner_scan: str = Field(
        default="",
        description=(
            "LOFI: what is visible in each corner (TL / TR / BL / BR), "
            "including any flesh-colored fragments. Empty for other profiles."
        ),
    )
    limb_fragment_present: bool = Field(
        default=False,
        description=(
            "True if any disconnected flesh-colored hand/finger/arm fragment "
            "exists anywhere, including at frame edges."
        ),
    )
    ungrounded_hand_present: bool = Field(
        default=False,
        description=(
            "True if a hand or grip is visible with no wrist AND no forearm "
            "continuing off-frame. Center-of-frame floating grips count. "
            "Must be set even when the dominant object is also wrong."
        ),
    )
    deformed_hand_present: bool = Field(
        default=False,
        description=(
            "True if a visible hand has wrong finger count, fused/merged into "
            "torso/face/object, twisted/duplicated fingers, or impossible "
            "proportion relative to the body."
        ),
    )
    incomplete_garment_coverage: bool = Field(
        default=False,
        description=(
            "True if a human figure's back, shoulders, chest, or torso is "
            "incompletely clothed / exposing skin that should be covered."
        ),
    )

    # Back-compat aliases for older agent_loop callers
    @property
    def detected_flaws(self) -> list[str]:
        return self.flaws

    @property
    def prompt_fix_instructions(self) -> str:
        return self.fix_instructions

    @field_validator("score")
    @classmethod
    def _round_score(cls, v: float) -> float:
        return round(float(v), 2)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "flaws" not in data and "detected_flaws" in data:
                data["flaws"] = data.pop("detected_flaws")
            if "fix_instructions" not in data and "prompt_fix_instructions" in data:
                data["fix_instructions"] = data.pop("prompt_fix_instructions")
        return data


def _mime_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def _resize_to_jpeg(image_path: Path | str, max_dim: int | None = None) -> bytes:
    """
    Downscale an image (max dimension ~512px, aspect preserved) and re-encode as
    JPEG q85. Input tokens scale with pixel area, so this cuts per-request vision
    tokens ~70–80%. Returns JPEG bytes; never raises on a bad frame — callers
    fall back to the original bytes via ``_decode_part_safe``.
    """
    from io import BytesIO

    from PIL import Image as PILImage

    max_dim = int(max_dim if max_dim is not None else config.VISION_MAX_DIM)
    quality = int(config.VISION_JPEG_QUALITY)
    with PILImage.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > max_dim:
            scale = max_dim / longest
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                resample=PILImage.LANCZOS,
            )
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def _decode_part_safe(
    path: Path,
    *,
    fallback_mime: str,
    force_prefix: str = "",
) -> tuple[str, types.Part]:
    """
    Build a compact JPEG part from ``path``. If resizing/encoding fails (corrupt
    frame, unreadable), fall back to the raw bytes so MoviePy / the critic always
    completes. ``force_prefix`` keeps friendly labels for cropped parts.
    """
    try:
        jpeg = _resize_to_jpeg(path)
        if jpeg:
            return (
                force_prefix or path.name,
                types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            )
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("CRITIC | resize skipped for %s (%s)", path, exc)
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"Empty image file: {path}")
    return (force_prefix or path.name, types.Part.from_bytes(data=raw, mime_type=fallback_mime))


def _load_image_part(path: Path) -> types.Part:
    return _decode_part_safe(path, fallback_mime=_mime_for(path))[1]


def _bytes_to_jpeg_part(data: bytes) -> types.Part:
    """Downscale & JPEG-encode in-memory image bytes; fall back to raw PNG."""
    from io import BytesIO

    from PIL import Image as PILImage

    try:
        im = PILImage.open(BytesIO(data)).convert("RGB")
        max_dim = int(config.VISION_MAX_DIM)
        quality = int(config.VISION_JPEG_QUALITY)
        w, h = im.size
        longest = max(w, h)
        if longest > max_dim:
            scale = max_dim / longest
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                resample=PILImage.LANCZOS,
            )
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        jpeg = buf.getvalue()
        if jpeg:
            return types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("CRITIC | in-memory bytes resize skipped (%s)", exc)
    return types.Part.from_bytes(data=data, mime_type="image/png")


def _lofi_edge_crop_parts(image_path: Path) -> list[tuple[str, types.Part]]:
    """Zoom the four corners so orphaned limb fragments are large enough to judge."""
    from io import BytesIO

    from PIL import Image as PILImage

    im = PILImage.open(image_path).convert("RGB")
    w, h = im.size
    boxes = (
        ("bottom-left", (0, int(h * 0.85), int(w * 0.28), h)),
        ("bottom-right", (int(w * 0.72), int(h * 0.85), w, h)),
        ("top-left", (0, 0, int(w * 0.28), int(h * 0.15))),
        ("top-right", (int(w * 0.72), 0, w, int(h * 0.15))),
    )
    out: list[tuple[str, types.Part]] = []
    for name, box in boxes:
        try:
            crop = im.crop(box)
            scaled = _encode_then_downscale(crop, f"edge_{name}")
            if scaled is not None:
                out.append(scaled)
                continue
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("CRITIC | edge crop %s resize skipped (%s)", name, exc)
        buf = BytesIO()
        im.crop(box).save(buf, format="PNG")
        out.append(
            (name, types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
        )
    return out


def _lofi_torso_crop_parts(image_path: Path) -> list[tuple[str, types.Part]]:
    """Zoom the chest/mid-frame so a hand fused into the torso is large enough to judge."""
    from io import BytesIO

    from PIL import Image as PILImage

    im = PILImage.open(image_path).convert("RGB")
    w, h = im.size
    box = (int(w * 0.12), int(h * 0.28), int(w * 0.88), int(h * 0.88))
    try:
        scaled = _encode_then_downscale(im.crop(box), "torso")
        if scaled is not None:
            return [scaled]
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("CRITIC | torso crop resize skipped (%s)", exc)
    buf = BytesIO()
    im.crop(box).save(buf, format="PNG")
    return [
        (
            "torso/mid-frame",
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
        )
    ]


def _encode_then_downscale(
    crop: Any,
    label: str,
) -> tuple[str, types.Part] | None:
    """Encode a cropped PIL region as a max-512px JPEG part (None on failure)."""
    from io import BytesIO

    from PIL import Image as PILImage

    max_dim = int(config.VISION_MAX_DIM)
    quality = int(config.VISION_JPEG_QUALITY)
    copy = crop.convert("RGB")
    cw, ch = copy.size
    longest = max(cw, ch)
    if longest > max_dim:
        scale = max_dim / longest
        copy = copy.resize(
            (max(1, int(cw * scale)), max(1, int(ch * scale))),
            resample=PILImage.LANCZOS,
        )
    buf = BytesIO()
    copy.save(buf, format="JPEG", quality=quality, optimize=True)
    jpeg = buf.getvalue()
    if not jpeg:
        return None
    return (label, types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))


_INCOMPLETE_GARMENT_RE = re.compile(
    r"("
    r"bare back|exposed (?:back|shoulders?|chest|cleavage|skin)|"
    r"off[- ]shoulder|incomplete (?:garment|dress|coverage)|"
    r"unclothed|undressed|nude|naked|missing (?:fabric|dress|clothes)|"
    r"shoulders? (?:bare|exposed|uncovered)|back (?:bare|exposed)"
    r")",
    re.IGNORECASE,
)
_NEGATED_HAND = re.compile(
    r"\b(?:no|without|zero|not any|neither|avoid|omit)\s+"
    r"(?:visible\s+)?"
    r"(?:hands?|fingers?|grips?)"
    r"(?:\s*,\s*(?:no\s+)?(?:hands?|fingers?|grips?))*"
    r"(?:\s+or\s+(?:hands?|fingers?|grips?))?"
    r"|\b(?:hands?|fingers?|grips?)\s+(?:absent|none)\b"
    r"|\b(?:object alone|alone with)\s+no\s+grip\b",
    re.I,
)
_POS_HAND = re.compile(r"\b(?:hands?|fingers?|grip)\b", re.I)


def _blob_claims_visible_hand(blob: str) -> bool:
    """True only if the critic claims a hand is visible, not that it must be absent."""
    cleaned = _NEGATED_HAND.sub(" ", blob or "")
    return bool(_POS_HAND.search(cleaned))


def list_reference_images(style_folder: Path | str) -> list[Path]:
    """Load ALL image files from the channel style_reference folder."""
    folder = Path(style_folder)
    if not folder.is_dir():
        return []
    return [
        p for p in sorted(folder.iterdir(), key=lambda x: x.name.lower())
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    ]


def _semantic_fidelity_block(
    requested_subject: str | None,
    requested_object: str | None = None,
    close_variant: str | None = None,
    licensed_objects: Sequence[str] | None = None,
) -> str:
    st = (requested_subject or "").strip().lower().replace(" ", "_")
    key_obj = " ".join(str(requested_object or "").strip().split())
    licensed = [
        str(x).strip()
        for x in (licensed_objects or [])
        if str(x).strip()
    ]
    if key_obj and key_obj not in licensed:
        licensed = [key_obj, *licensed]
    licensed_line = ", ".join(dict.fromkeys(licensed))
    licensed_block = ""
    if licensed_line:
        licensed_block = f"""
   LICENSED OBJECTS for this beat (shared with spoken_prop — do not flag as unspoken):
   {licensed_line}.
   Pronouns it/this/that in the spoken line refer to these. FAIL "unspoken"
   only for props that are NOT in this list.
"""
    if str(close_variant or "").strip().lower() == "portrait_close":
        return """
5) SEMANTIC FIDELITY TO REQUEST (HARD FAIL if pixels mismatch) — portrait_close:
   Crop MUST be a medium close-up portrait: head and shoulders in frame,
   full face visible — eyes, nose, mouth, and some hair or shoulder.
   FAIL if this is an extreme macro on eyes/brow only (the retired eye_close).
   Prefix "FRAME:".
   FAIL if the image is a photograph or photoreal skin/eyes. Prefix "STYLE:".
   FAIL if no person is present. Technique must stay painterly/print.

If this check fails, set passed=false even when style otherwise looks fine.
fix_instructions MUST say: medium close-up portrait, head and shoulders in
frame, full face visible, painterly illustration — not an extreme macro crop
on eyes alone.
""".strip()
    if str(close_variant or "").strip().lower() == "eye_close":
        return """
5) SEMANTIC FIDELITY TO REQUEST (HARD FAIL if pixels mismatch) — eye_close:
   Crop must show a woman's eyes, brow, and part of hair only.
   FAIL if a nose, mouth, chin, or full face is visible. Prefix "FACE:".
   FAIL if the crop is a full-figure or three-quarter portrait.
   Eyes must be recognizable (reflective or tearful). This is a partial
   reveal, not a face reveal.

If this check fails, set passed=false even when style otherwise looks fine.
fix_instructions MUST say: extreme close on eyes only, cropped above the
nose bridge, no nose, no mouth, no lower face.
""".strip()
    if st not in {"woman", "man", "couple", "silhouette", "object_focus"}:
        subject_block = (
            "5) SEMANTIC FIDELITY — requested subject_type was not provided; "
            "skip the person-count check."
        )
    else:
        object_match = ""
        if st == "object_focus" and key_obj:
            object_match = f"""
   - object_focus OBJECT MATCH (HARD FAIL): the dominant object in the pixels must
     match this description: "{key_obj}". Supporting clutter around it is OK.
     FAIL if a different unrelated object dominates instead — especially a mug,
     cup, glass, or coffee vessel when that is not what was requested.
     Prefix a flaw with "OBJECT:" (e.g. "OBJECT: mug dominates, requested {key_obj}").
     fix_instructions MUST name "{key_obj}" as the thing to draw, centered, filling the frame.
"""
        subject_block = f"""
5) SEMANTIC FIDELITY TO REQUEST (HARD FAIL if pixels mismatch) — requested subject_type={st}:
   - woman: exactly ONE adult woman as the dominant subject; one head, one body, correctly formed.
     FAIL if two people, a man-only shot, a fused androgynous figure, phantom extra limbs, or object-only with no person.
   - man: exactly ONE adult man as the dominant subject; one head, one body, correctly formed.
     FAIL if two people, a woman-only shot, a fused figure, phantom limbs, or object-only.
   - couple: TWO distinct people (a man and a woman), two distinct faces, two distinct bodies,
     standing separately (side by side or one slightly behind). FAIL if only one person,
     if bodies/faces are fused into one figure, or if extra phantom hands/limbs appear.
     BED / MATTRESS (HARD FAIL): if a bed is in the scene, figures must stand on the
     floor beside it (floor visible under their feet) or sit on the edge with feet
     on the floor. FAIL if anyone stands ON the mattress / on top of the bed, if
     feet are planted in rumpled bedding, or if the bed fills the foreground and
     figures rise out of the covers with no visible floor. Prefix "POSE:".
   - object_focus: the object sits ALONE and dominates the frame.
     No dominant face or full-body figure. Default: zero hands.
     FAIL if a close portrait of a person fills the frame. Do not require facial anatomy.
{object_match}   - silhouette: a featureless or near-featureless figure against light.
     FAIL if a fully detailed close-portrait face dominates. Do not require facial detail.

If this check fails, set passed=false even when style/anatomy otherwise look fine.
Prefix a person-count flaw with "SEMANTIC:" (e.g. "SEMANTIC: couple requested, only one fused figure").
fix_instructions MUST tell Flux how to match {st}.
""".strip()

    return f"""
{subject_block}
{licensed_block}

6) LIMB INTEGRITY (HARD FAIL on every beat, including object_focus / silhouette).
   This check is NEVER skipped or weakened for object_focus.
   First LIST every distinct hand, finger, wrist, or arm piece you can see —
   including tiny fragments at frame edges AND a full hand in the center of frame.
   Then judge:
   - woman / man / silhouette: one attached body only. Any extra disconnected
     flesh blob, extra finger, floating hand, or orphaned limb piece = FAIL,
     even if it is small or cropped by the frame edge.
   - couple: two attached bodies only. Extra fragments = FAIL.
   - object_focus: the preferred pass is NO HANDS AT ALL.
     If a hand appears, it is a HARD FAIL unless a plausible wrist AND forearm
     continue off-frame in a natural direction (not a top-down floating grip).
     A hand with no wrist, no forearm, and no body connection anywhere in the
     frame is a hard fail — not acceptable "close framing" of the object.
     Set limb_fragment_present=true AND ungrounded_hand_present=true for that
     ungrounded grip, even if the object is also wrong (report OBJECT: AND LIMB:).
   Cropping a correctly attached limb at the edge is OK only when the wrist
     is visible heading off-frame. A second, unattached skin-colored shape is NOT OK.
   If you are unsure whether a skin-colored shape is a limb fragment, FAIL.
   Report ALL hard fails in flaws, not only the first (e.g. both OBJECT: and LIMB:).
   EDGE CROPS are tight outer corners of IMAGE 1 only (not the main subject).
   A correctly attached hand that is merely cropped by the frame will mostly
   sit outside these tiny corners. If a crop shows isolated fingers, a flesh
   blob, or a hand piece with no connecting wrist in that crop, that is an
   orphaned fragment — set limb_fragment_present=true and FAIL.
   Prefix a flaw with "LIMB:" (e.g. "LIMB: floating grip, no wrist or forearm").
   fix_instructions MUST say to remove the extra fragment / ungrounded hand
   and keep only the object (object_focus) or attached limbs (person shots).

7) HAND ANATOMY (HARD FAIL — attached hands that are still wrong).
   LIMB INTEGRITY above catches disconnected / ungrounded hands. This check
   catches a hand that IS attached to a body but is anatomically impossible:
   - Incorrect finger count (too many or too few).
   - Hand merging into, entering, or fused with the torso, chest, face,
     or another object (e.g. fingers disappearing into the sternum).
   - Twisted or duplicated fingers, or a hand whose scale does not match the body.
   If any of these occur: set deformed_hand_present=true, passed=false, and
   report the flaw exactly as: "deformed or anatomically incorrect hand".
   An empty-handed pose with a correctly formed, separate hand is a PASS.
""".strip()


def _build_lofi_critic_instruction(
    channel_name: str,
    rules: dict[str, Any],
    quality_threshold: float,
    n_refs: int,
    requested_subject: str | None = None,
    requested_object: str | None = None,
    prior_fail_criteria: Sequence[str] | None = None,
    style_profile: str | None = None,
    close_variant: str | None = None,
    licensed_objects: Sequence[str] | None = None,
) -> str:
    forbidden = ", ".join(rules.get("forbidden_tokens") or [])
    mandatory = ", ".join(rules.get("mandatory_elements") or [])
    lighting = rules.get("lighting_style") or ""
    audience = rules.get("target_audience_rules") or ""
    semantic = _semantic_fidelity_block(
        requested_subject,
        requested_object,
        close_variant=close_variant,
        licensed_objects=licensed_objects,
    )
    st_cov = (requested_subject or "").strip().lower().replace(" ", "_")
    cv_cov = str(close_variant or "").strip().lower()
    if cv_cov == "eye_close" or st_cov == "object_focus":
        coverage = ""
    elif st_cov in {"woman", "man", "couple", "silhouette"} or cv_cov == "portrait_close":
        coverage = """
6) GARMENT COVERAGE (HARD FAIL on any human figure, including silhouettes
   and back-facing shots — not optional):
   Every visible human body must be fully covered by dress/sweater/coat fabric.
   Back, shoulders, chest, and torso clothed. Face and hands may show skin.
   FAIL if: bare back, exposed shoulders, cleavage, nape-to-waist skin,
   incomplete garment, off-shoulder, or any nude/undressed read.
   If this fails: set incomplete_garment_coverage=true, passed=false, and
   add the exact flaw string "COVERAGE: incomplete garment / exposed skin on a human figure".
   Inspect TORSO CROPS as well as the full frame.
""".strip()
    else:
        coverage = ""
    prof = str(style_profile or "").strip().lower()
    painterly = prof == "lofi_painterly_v1"
    riso_retro = prof in {
        "style-riso_painting_retro_vintage",
        "lofi_risograph_v1",
    }
    prior_lines = [
        " ".join(str(x).split())
        for x in (prior_fail_criteria or [])
        if str(x).strip()
    ]
    if str(close_variant or "").strip().lower() == "portrait_close":
        style_eval = (
            "1) Vintage risograph-painting print is ON STYLE. This beat is a "
            "portrait_close: a painterly medium close-up, head and shoulders "
            "in frame, full face visible (eyes, nose, mouth, some hair/"
            "shoulder). FAIL if it is an extreme macro on eyes/brow only. "
            "FAIL if photoreal skin, glossy camera eyes, or beauty-lighting "
            "photography. Technique must match the rest of the episode: ink, "
            "flat printed color, paper grain."
        )
    elif str(close_variant or "").strip().lower() == "eye_close":
        style_eval = (
            "1) Vintage risograph-painting print is ON STYLE. This beat is "
            "the eye_close exception: a tight crop of eyes, brow, and hair "
            "MAY show detailed, reflective eyes. Do NOT require a full-body "
            "or unlit silhouette. FAIL if a nose, mouth, chin, or full face "
            "is visible. FAIL if it is a photograph / photoreal camera look."
        )
    elif painterly:
        style_eval = (
            "1) Painted illustration is ON STYLE — canvas grain, soft brush, "
            "muted vintage print. Do NOT fail for painterly softness or "
            "blended shading. FAIL only if it is a photograph / photoreal camera look.\n"
            "   For woman/man/couple: FAIL if a frontal or three-quarter close-up "
            "face is the main subject (eyes/nose/mouth rendered as a portrait). "
            "Allowed: from behind, deep profile turned away, silhouette, crop "
            "above the eyes, or a distant figure whose face is only brushstrokes."
        )
    elif riso_retro:
        style_eval = (
            "1) Vintage risograph-painting print is ON STYLE — grainy print, "
            "ink line, atmospheric color, canvas/paper texture. Do NOT fail the "
            "environment for painterly richness, blended dusk color, or canvas "
            "texture. FAIL only if it is a photograph / photoreal camera look.\n"
            "   For woman/man/couple: FAIL if any person is a lit, glossy, or "
            "rendered-skin portrait (glamour, off-shoulder, beauty lighting, "
            "detailed facial features, cinematic street photography). Each figure "
            "MUST read as a flat unlit silhouette against a bright light (window, "
            "sunset, doorway, streetlamp). The environment around them MUST keep "
            "full painterly detail. Silhouette and object_focus: do not require "
            "this backlight rule; judge them on overall print look only."
        )
    else:
        style_eval = (
            "1) Illustration style consistency — ink/graphic-novel look with "
            "hard-edged flat color, visible ink line, and halftone "
            "(not blended shading). FAIL if photorealistic, painterly, "
            "soft-gradient, airbrushed, or glossy/rendered-skin drift."
        )
    prior_block = ""
    if prior_lines:
        listed = "\n".join(f"  - {x}" for x in prior_lines[:8])
        prior_block = f"""
PRIOR FAIL HOLD (this is a retry — same criteria, not a looser bar):
The previous attempt failed for:
{listed}
Inspect THIS image for those same issues. If ANY are still visible,
set passed=false and include them in flaws again, even if other
aspects improved. Do not pass a softer read of the same drift.
"""
    return f"""
You are an Art Director for LOFI economic illustration reels ("{channel_name}").

You are given:
- IMAGE 1 = NEW CANDIDATE (judge the PIXELS, not the prompt text)
- OPTIONAL REFERENCE STYLE IMAGES 1–{n_refs}

CHANNEL RULES:
- FORBIDDEN: {forbidden}
- MANDATORY: {mandatory}
- LIGHTING: {lighting}
- AUDIENCE: {audience}
{prior_block}
EVALUATE:
{style_eval}
2) Anatomy — malformed hands/faces beyond acceptable illustration threshold
   (skip face-detail requirements for object_focus and silhouette, but still
   FAIL ungrounded/disconnected hands on those beats).
   Also FAIL disconnected/orphaned limb fragments, even small ones at frame edges.
   HAND ANATOMY (HARD FAIL on person shots; also FAIL on object_focus/silhouette
   if a hand is visible). Specifically check for:
   - Hands with an incorrect number of fingers (too many or too few).
   - A hand merging/entering the torso, face, or another object in an
     anatomically impossible way (e.g. a hand appearing to sink into the chest).
   - Twisted or duplicated fingers, or hand proportion inconsistent with the body.
   If ANY of these occur: set deformed_hand_present=true, passed=false, and
   add the exact flaw string "deformed or anatomically incorrect hand".
3) Text artifacts — no embedded/garbled letters, logos, or UI burned into the image.
4) Framing — vertical/portrait-friendly composition suitable for 9:16 reels; subject readable.
{semantic}
{coverage}

Do NOT apply Master Mei dystopian / body-horror criteria.
Do NOT require fitness-model hard caps.
Do NOT fail solely for palette/color temperature (warm vs charcoal-teal duotone).
Palettes are assigned per act and are not a critic hard-fail.

SCORING GUIDE:
- 9.0–10.0: clean LOFI illustration, on-style, request-faithful, no text artifacts, no orphan limbs
- {quality_threshold}–8.9: minor issues, still approvable, request type and object match
- Below {quality_threshold}: FAIL (photoreal drift, bad anatomy, orphan limb, text artifacts, wrong framing, semantic mismatch, incomplete garment coverage, OR wrong dominant object)

Set passed=true ONLY if score >= {quality_threshold} AND no hard flaws above AND semantic fidelity AND limb integrity AND garment coverage pass.

Return JSON fields exactly:
  score (float 0–10),
  passed (boolean),
  flaws (list of strings),
  fix_instructions (string — actionable FLUX rewrite directives; empty if passed),
  corner_scan (string — TL / TR / BL / BR contents),
  limb_fragment_present (boolean — true if any disconnected limb piece exists),
  ungrounded_hand_present (boolean — true if a hand has no wrist/forearm; center grips count),
  deformed_hand_present (boolean — true if finger count is wrong, a hand merges
    into torso/face/object, or fingers are twisted/duplicated / mis-scaled).
  incomplete_garment_coverage (boolean — true if a human figure's back,
    shoulders, chest, or torso is incompletely clothed / exposing skin).
""".strip()


def _build_ancient_mystery_critic_instruction(
    channel_name: str,
    rules: dict[str, Any],
    quality_threshold: float,
    n_refs: int,
) -> str:
    forbidden = ", ".join(rules.get("forbidden_tokens") or [])
    mandatory = ", ".join(rules.get("mandatory_elements") or [])
    lighting = rules.get("lighting_style") or ""
    audience = rules.get("target_audience_rules") or ""
    profile = load_critic_profile("ancient_mystery")
    eval_rules = profile.get("evaluation_rules") if isinstance(profile, dict) else {}
    if not isinstance(eval_rules, dict):
        eval_rules = {}
    hard_only = eval_rules.get("hard_fail_only") or [
        "severe anatomical deformations",
        "unreadable floating artifacts",
        "corrupt or broken renders",
    ]
    hard_block = "\n".join(f"   - {item}" for item in hard_only)
    return f"""
You are an Art Director for the Ancient Knowledge documentary channel ("{channel_name}").

You are given:
- IMAGE 1 = NEW CANDIDATE (judge the PIXELS, not the prompt text)
- OPTIONAL REFERENCE STYLE IMAGES 1–{n_refs}

EVALUATION POLICY (relaxed — do not over-reject usable FLUX stills):
1) ACCEPT clean, high-atmosphere FLUX renders. Digital polish is NOT a defect.
   Do NOT penalize missing 35mm film grain, missing analog texture, or a slightly
   clean/digital look.
2) Do NOT require a recognisable real-world monument (Pyramid, Baalbek, Göbekli
   Tepe, etc.) unless the generation prompt explicitly named one. Invented or
   unnamed ruins, artefacts, and structures are valid.
3) Fantasy / surreal ancient-mystery interpretations ARE valid outputs.
4) Atmospheric lighting is a plus, not a gate. Do not fail solely for lighting
   that is less dramatic than torchlight chiaroscuro.

CHANNEL RULES (guidance, not automatic fails):
- FORBIDDEN: {forbidden}
- MANDATORY: {mandatory}
- LIGHTING: {lighting}
- AUDIENCE: {audience}

STRICT FAILURE (score MUST be < {quality_threshold}) only for severe visual flaws:
{hard_block}
Also fail illustration, sketch, cartoon, or unusable garbage pixels.

Do NOT apply Master Mei dystopian / body-horror / fitness-model criteria.

SCORING GUIDE:
- 9.0–10.0: strong atmospheric ancient-mystery still, ready to post
- {quality_threshold}–8.9: approvable (clean FLUX polish / fantasy ruins OK)
- Below {quality_threshold}: FAIL — reserved for the severe flaws listed above

Set passed=true if score >= {quality_threshold} AND none of the strict-failure
flaws are present. Do not fail for missing film grain or missing geo-anchors.

Return JSON fields exactly:
  score (float 0–10),
  passed (boolean),
  flaws (list of strings),
  fix_instructions (string — actionable camera/lighting FLUX rewrite; empty if passed).
""".strip()


def _build_critic_instruction(
    channel_name: str,
    rules: dict[str, Any],
    quality_threshold: float,
    hard_cap: float,
    n_refs: int,
    requested_subject: str | None = None,
    requested_object: str | None = None,
    prior_fail_criteria: Sequence[str] | None = None,
    style_profile: str | None = None,
    close_variant: str | None = None,
    licensed_objects: Sequence[str] | None = None,
) -> str:
    profile = str(rules.get("critic_profile") or channel_name or "").strip().lower()
    if profile == "lofi_economic" or channel_name == "lofi_economic":
        return _build_lofi_critic_instruction(
            channel_name, rules, quality_threshold, n_refs,
            requested_subject=requested_subject,
            requested_object=requested_object,
            prior_fail_criteria=prior_fail_criteria,
            style_profile=style_profile,
            close_variant=close_variant,
            licensed_objects=licensed_objects,
        )
    if profile in ("ancient_mystery", "ancient_knowledge") or channel_name == "ancient_knowledge":
        return _build_ancient_mystery_critic_instruction(
            channel_name, rules, quality_threshold, n_refs,
        )

    forbidden = ", ".join(rules.get("forbidden_tokens") or [])
    mandatory = ", ".join(rules.get("mandatory_elements") or [])
    lighting = rules.get("lighting_style") or ""
    audience = rules.get("target_audience_rules") or ""
    anchor = config.MASTER_STYLE_ANCHOR

    return f"""
You are a ruthless Art Director for channel "{channel_name}".

You are given:
- IMAGE 1 = NEW CANDIDATE (judge the PIXELS, not the prompt text)
- REFERENCE STYLE IMAGES 1–{n_refs} = the ONLY accepted visual DNA

Compare the candidate PIXEL-BY-PIXEL against the reference images and CHANNEL RULES.

CHANNEL RULES:
- FORBIDDEN: {forbidden}
- MANDATORY: {mandatory}
- LIGHTING: {lighting}
- AUDIENCE: {audience}
- TARGET STYLE ANCHOR: {anchor}

════════════════════════════════════════
HARD CAP RULE (MANDATORY — NON-NEGOTIABLE):
If the candidate looks like ANY of the following, you MUST assign score < {hard_cap}
(typically 1.0–3.5) REGARDLESS of narrative prompt match:
  • clean-skinned fitness model / gym-bro / handsome actor
  • studio-lit portrait / softbox beauty lighting
  • polished 3D render / CGI toy look
  • commercial fashion shoot / underwear-ad aesthetic
  • smooth airbrushed skin with no industrial grit
Do NOT give such images a score of 4.0 or higher. Cap is MAX {hard_cap - 0.01:.2f}.
════════════════════════════════════════

VISUAL TEXTURE REQUIREMENT (must match references):
- Raw 35mm physical film grain (not digital noise overlay only)
- Dirty rusted metal, unpolished practical FX
- Cables / copper conduits VISIBLY piercing flesh with scar tissue
- Grim dystopian industrial lighting matching the reference images
- Gaunt, scarred, exhausted industrial subjects — NOT fitness physiques

EVALUATE:
1) Anatomy / Artifacts — fitness models, clean skin, nipples/genitalia, weird hands,
   floating objects, deformed limbs, underwear-ad posing.
2) Atmospheric & 80s Biomechanical Style — H.R. Giger practical body horror adherence.
3) Reference DNA match — same grit, metal, cable-flesh integration as refs.

SCORING GUIDE:
- 9.0–10.0: production-ready hardcore body horror matching refs
- {quality_threshold}–8.9: minor issues only, still approved territory
- 4.0–{quality_threshold - 0.01:.1f}: wrong texture / weak biomech — FAIL
- Below {hard_cap}: CLEAN MODEL / STUDIO / COMMERCIAL HARD FAIL

Set passed=true ONLY if score >= {quality_threshold} AND no hard aesthetic flaws.

Return JSON fields exactly:
  score (float 0–10),
  passed (boolean),
  flaws (list of strings),
  fix_instructions (string — actionable FLUX rewrite directives; empty if passed).
""".strip()


def _apply_hard_cap(verdict: CriticVerdict, hard_cap: float) -> CriticVerdict:
    """Local enforcement: clean-model fingerprints force score below hard_cap."""
    blob = " ".join(verdict.flaws) + " " + (verdict.fix_instructions or "")
    triggered = bool(_CLEAN_MODEL_FLAW_RE.search(blob))

    # Also scan flaw list for common soft refusals that still imply commercial look
    soft_triggers = (
        "too clean", "too polished", "looks like a model", "catalog",
        "commercial", "studio", "fitness", "handsome", "smooth skin",
    )
    lo = blob.lower()
    if any(t in lo for t in soft_triggers):
        triggered = True

    if triggered and verdict.score >= hard_cap:
        _console.print(
            f"[red]HARD CAP[/red] clean/studio fingerprint detected — "
            f"clamping score {verdict.score:.2f} → {hard_cap - 0.5:.2f}"
        )
        verdict.score = round(hard_cap - 0.5, 2)
        verdict.passed = False
        if "HARD CAP" not in " ".join(verdict.flaws):
            verdict.flaws = list(verdict.flaws) + [
                "HARD CAP: clean-model / studio / commercial aesthetic detected"
            ]
        if not verdict.fix_instructions:
            verdict.fix_instructions = (
                "Describe sleek cybernetic digital servitude — hypnotized face lit by "
                "screen glare, metallic neural collar, luminous fiber-optic tethers to "
                "smartphone/VR, rain-slicked dystopian alley, gritty 35mm film grain. "
                "Never gore, blood, open wounds, or gym fitness apparel."
            )
    return verdict


def evaluate_image(
    generated_image: Path | str | bytes,
    *,
    channel_name: str,
    rules: dict[str, Any],
    style_folder: Path | str | None = None,
    reference_paths: Sequence[Path | str] | None = None,
    quality_threshold: float | None = None,
    requested_subject: str | None = None,
    requested_object: str | None = None,
    prior_fail_criteria: Sequence[str] | None = None,
    style_profile: str | None = None,
    close_variant: str | None = None,
    licensed_objects: Sequence[str] | None = None,
) -> CriticVerdict:
    """
    Judge a generated image with Gemini Vision against ALL style refs + rules.

    If ``config.VISUAL_QA_ENABLED`` is falsy, the API call is bypassed entirely
    and a mock PASS verdict is returned so every downstream pipeline (orchestrator,
    reel_visual_qa, agent_loop) keeps running without burning vision tokens.
    """
    if not config.VISUAL_QA_ENABLED:
        threshold = (
            float(quality_threshold)
            if quality_threshold is not None
            else float(config.QUALITY_THRESHOLD)
        )
        bypass = CriticVerdict(
            score=10.0,
            passed=True,
            flaws=[],
            fix_instructions="",
            corner_scan="",
            limb_fragment_present=False,
            ungrounded_hand_present=False,
            deformed_hand_present=False,
            incomplete_garment_coverage=False,
        )
        _console.print(
            f"[cyan]CRITIC[/cyan] VISUAL_QA_ENABLED=false -> bypassing vision API "
            f"(mock PASS score=10) | threshold={threshold} | channel={channel_name}"
        )
        _LOG.info("CRITIC | bypassed (VISUAL_QA_ENABLED=false) channel=%s", channel_name)
        return bypass

    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY missing — set env or factory .env before critic runs."
        )

    threshold = (
        float(quality_threshold)
        if quality_threshold is not None
        else float(config.QUALITY_THRESHOLD)
    )
    hard_cap = float(config.CLEAN_MODEL_HARD_CAP)

    if isinstance(generated_image, (bytes, bytearray)):
        candidate_part = _bytes_to_jpeg_part(bytes(generated_image))
        candidate_label = "<bytes>"
    else:
        candidate_path = Path(generated_image)
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Generated image not found: {candidate_path}")
        candidate_part = _load_image_part(candidate_path)
        candidate_label = str(candidate_path)

    if reference_paths:
        refs = [Path(p) for p in reference_paths if Path(p).is_file()]
    else:
        folder = Path(style_folder) if style_folder else config.style_folder_for_channel(
            channel_name
        )
        refs = list_reference_images(folder)

    profile = str(rules.get("critic_profile") or channel_name or "").strip().lower()
    is_lofi = profile == "lofi_economic" or channel_name == "lofi_economic"
    is_ancient = (
        profile in ("ancient_mystery", "ancient_knowledge")
        or channel_name == "ancient_knowledge"
    )
    if is_ancient and quality_threshold is None:
        threshold = ancient_mystery_quality_threshold()

    contents: list[Any] = [
        "IMAGE 1 — NEW CANDIDATE TO JUDGE (inspect pixels):",
        candidate_part,
    ]
    if is_lofi and not isinstance(generated_image, (bytes, bytearray)):
        try:
            for name, part in _lofi_edge_crop_parts(Path(generated_image)):
                contents.append(
                    f"EDGE CROP of IMAGE 1 — {name}. Inspect for orphaned "
                    "flesh-colored hand/finger/arm fragments."
                )
                contents.append(part)
            st_req = (requested_subject or "").strip().lower().replace(" ", "_")
            if (
                st_req in {"woman", "man", "couple", "silhouette"}
                and str(close_variant or "") != "eye_close"
            ):
                for name, part in _lofi_torso_crop_parts(Path(generated_image)):
                    contents.append(
                        f"TORSO CROP of IMAGE 1 — {name}. Inspect hands vs chest/"
                        "torso AND garment coverage (back, shoulders, chest clothed). "
                        "A finger, palm, or wrist merging into the body "
                        "is a HARD FAIL (deformed or anatomically incorrect hand), "
                        "even if the face looks fine. Incomplete clothing / exposed "
                        "back or shoulders is also a HARD FAIL."
                    )
                    contents.append(part)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("CRITIC | edge crops skipped (%s)", exc)
    for i, ref in enumerate(refs, start=1):
        try:
            contents.append(f"REFERENCE STYLE IMAGE {i}: {ref.name}")
            contents.append(_load_image_part(ref))
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("CRITIC | skip ref %s (%s)", ref, exc)

    if not refs:
        _console.print(
            "[yellow]CRITIC WARNING[/yellow] no style_reference images found — "
            "judging from channel rules only"
        )

    contents.append(
        _build_critic_instruction(
            channel_name, rules, threshold, hard_cap, n_refs=len(refs),
            requested_subject=requested_subject,
            requested_object=requested_object,
            prior_fail_criteria=prior_fail_criteria,
            style_profile=style_profile,
            close_variant=close_variant,
            licensed_objects=licensed_objects,
        )
    )

    from core.google_guardrail import guarded_generate_content, make_guarded_gemini_client

    client = make_guarded_gemini_client(config.GEMINI_API_KEY)
    model_id = config.GEMINI_CRITIC_MODEL
    if not model_id.startswith("models/"):
        model_id = f"models/{model_id}"

    _console.print(
        f"[cyan]CRITIC[/cyan] Gemini Vision | candidate={candidate_label} | "
        f"refs={len(refs)} (ALL loaded) | channel={channel_name} | model={model_id}"
        + (" | profile=lofi_economic" if is_lofi else "")
        + (" | profile=ancient_mystery" if is_ancient else "")
        + (f" | subject={requested_subject}" if requested_subject else "")
        + (f" | object={requested_object}" if requested_object else "")
    )

    response = guarded_generate_content(
        client,
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CriticVerdict,
            temperature=0.15,
        ),
        source="visual_critic.evaluate_image",
    )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, CriticVerdict):
        verdict = parsed
    elif parsed is not None and hasattr(parsed, "model_dump"):
        verdict = CriticVerdict.model_validate(parsed.model_dump())
    else:
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError("Gemini critic returned empty response")
        # Tolerate fenced JSON
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            verdict = CriticVerdict.model_validate_json(text)
        except Exception:
            verdict = CriticVerdict.model_validate(json.loads(text))

    if not is_lofi and not is_ancient:
        verdict = _apply_hard_cap(verdict, hard_cap)

    if verdict.score < threshold:
        verdict.passed = False
    elif verdict.passed and verdict.score >= threshold:
        verdict.passed = True

    st_req = (requested_subject or "").strip().lower().replace(" ", "_")
    if is_lofi and st_req == "object_focus":
        blob = " ".join(
            [
                " ".join(verdict.flaws or []),
                str(verdict.fix_instructions or ""),
                str(getattr(verdict, "corner_scan", "") or ""),
            ]
        )
        # "no hands" / "without fingers" in a fix line is a prohibition, not a
        # detection. Matching the bare word used to force LIMB on object_focus
        # whenever Gemini restated the no-hand rule (table-corner false positive).
        mentions_hand = _blob_claims_visible_hand(blob)
        grounded = bool(
            re.search(r"wrist.{0,24}forearm|forearm.{0,24}wrist", blob, re.I)
        )
        if mentions_hand and not grounded:
            verdict.ungrounded_hand_present = True

    if is_lofi and (
        bool(getattr(verdict, "limb_fragment_present", False))
        or bool(getattr(verdict, "ungrounded_hand_present", False))
    ):
        verdict.passed = False
        if bool(getattr(verdict, "ungrounded_hand_present", False)):
            verdict.limb_fragment_present = True
        scan = str(getattr(verdict, "corner_scan", "") or "").strip()
        limb_msg = "LIMB: disconnected limb fragment"
        if bool(getattr(verdict, "ungrounded_hand_present", False)):
            limb_msg = "LIMB: ungrounded hand, no wrist or forearm"
        if scan:
            limb_msg = f"{limb_msg} ({scan})"
        if not any(str(f).upper().startswith("LIMB:") for f in (verdict.flaws or [])):
            verdict.flaws.append(limb_msg)
        if not str(verdict.fix_instructions or "").strip():
            verdict.fix_instructions = (
                "Remove extra disconnected hands, fingers, or limb fragments. "
                "On object_focus, draw the object alone with no grip. "
                "On person shots, keep only limbs attached to the visible body."
            )
        if verdict.score >= threshold:
            verdict.score = round(min(float(verdict.score), threshold - 0.5), 2)

    if is_lofi:
        blob = " ".join(
            [
                " ".join(verdict.flaws or []),
                str(verdict.fix_instructions or ""),
                str(getattr(verdict, "corner_scan", "") or ""),
            ]
        )
        deformed = bool(getattr(verdict, "deformed_hand_present", False)) or bool(
            _DEFORMED_HAND_RE.search(blob)
        )
        if deformed:
            verdict.deformed_hand_present = True
            verdict.passed = False
            if not any(
                DEFORMED_HAND_FLAW.lower() in str(f).lower()
                for f in (verdict.flaws or [])
            ):
                verdict.flaws.append(DEFORMED_HAND_FLAW)
            if not str(verdict.fix_instructions or "").strip():
                verdict.fix_instructions = (
                    "Keep every hand fully outside the torso and face, "
                    "correct finger count, no fused or overlapping anatomy."
                )
            if verdict.score >= threshold:
                verdict.score = round(min(float(verdict.score), threshold - 0.5), 2)

        coverage_hit = bool(
            getattr(verdict, "incomplete_garment_coverage", False)
        ) or bool(_INCOMPLETE_GARMENT_RE.search(blob))
        st_req = (requested_subject or "").strip().lower().replace(" ", "_")
        cv_req = str(close_variant or "").strip().lower()
        human_req = st_req in {"woman", "man", "couple", "silhouette"} or cv_req == "portrait_close"
        if human_req and cv_req != "eye_close" and coverage_hit:
            verdict.incomplete_garment_coverage = True
            verdict.passed = False
            if not any(
                COVERAGE_FLAW.lower() in str(f).lower()
                or "incomplete garment" in str(f).lower()
                for f in (verdict.flaws or [])
            ):
                verdict.flaws.append(COVERAGE_FLAW)
            if not str(verdict.fix_instructions or "").strip() or "covered" not in str(
                verdict.fix_instructions or ""
            ).lower():
                extra = (
                    "fully covered by dress fabric, back and shoulders clothed, "
                    "chest and torso clothed; no bare back, no exposed skin"
                )
                prev = str(verdict.fix_instructions or "").strip()
                verdict.fix_instructions = f"{prev} {extra}".strip() if prev else extra
            if verdict.score >= threshold:
                verdict.score = round(min(float(verdict.score), threshold - 0.5), 2)

    # Terminal: exact raw critique score
    table = Table(title="Gemini Vision Critic — Raw Verdict", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("score", f"{verdict.score:.2f} / 10.0")
    table.add_row("passed", str(verdict.passed))
    table.add_row("flaws", "; ".join(verdict.flaws) or "(none)")
    table.add_row("fix_instructions", verdict.fix_instructions or "(none)")
    if is_lofi:
        table.add_row(
            "limb_fragment_present",
            str(bool(getattr(verdict, "limb_fragment_present", False))),
        )
        table.add_row(
            "ungrounded_hand_present",
            str(bool(getattr(verdict, "ungrounded_hand_present", False))),
        )
        table.add_row(
            "deformed_hand_present",
            str(bool(getattr(verdict, "deformed_hand_present", False))),
        )
        table.add_row(
            "incomplete_garment_coverage",
            str(bool(getattr(verdict, "incomplete_garment_coverage", False))),
        )
        table.add_row(
            "corner_scan",
            str(getattr(verdict, "corner_scan", "") or "(none)"),
        )
    _console.print(table)
    _console.print(
        Panel(
            json.dumps(verdict.model_dump(), indent=2),
            title="[bold]RAW CRITIC JSON[/bold]",
            border_style="cyan",
        )
    )

    if config.LOG_COSTS:
        _console.print(
            f"[dim]COST[/dim] critic~${config.COST_GEMINI_FLASH_LITE_USD:.6f}"
        )

    _LOG.info(
        "CRITIC | channel=%s score=%.2f passed=%s flaws=%s",
        channel_name, verdict.score, verdict.passed, verdict.flaws,
    )
    return verdict
