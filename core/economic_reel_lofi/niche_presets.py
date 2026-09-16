# -*- coding: utf-8 -*-
"""Modular niche presets for the single-pass LOFI writer + image wrapper.

Each niche owns its golden few-shot references and the Aesthetic Master
notes applied to writer-supplied visual concepts before Flux/Together. The
active style module owns the final rendering vocabulary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


from core.economic_reel_lofi.style_modules.riso_retro_flat_v4 import (
    STYLE as RISO_STYLE,
)


RISO_PREFIX = f"{RISO_STYLE.open} {RISO_STYLE.technique}".strip()
RELATIONSHIP_LOCATION_ANCHOR = (
    "a story-specific cinematic environment chosen for this narrative"
)
PARENTING_LOCATION_ANCHOR = (
    "a story-specific lived-in environment shared by parent and child"
)

CAMERA_STAGING_BY_SCENE: dict[int, str] = {
    1: "wide establishing landscape or exterior, human subject small and upright",
    2: "architectural interior framing, human subject properly seated or standing",
    3: "tactile environmental prop detail in realistic context",
    4: "close-up side profile in dramatic light, upright head and natural anatomy",
    5: "motion transit through a door, platform, or corridor, subject upright",
    6: "looking through rain or reflective glass, subject upright behind the glass",
    7: "dynamic walking silhouette with natural upright limbs",
    8: "expansive panoramic closure, upright subject grounded in the landscape",
}
_UPRIGHT_ANATOMY_GUARD = (
    "ANATOMY CONTRACT: every human is upright—standing, walking, seated properly "
    "on furniture, or tucked into bed with visible pillows; nobody lies on a "
    "floor, rug, street, or doorstep, and nobody crawls or appears prone"
)


@dataclass(frozen=True)
class NichePreset:
    key: str
    label: str
    aesthetic_prefix: str
    negative_prompt: str
    few_shot_examples: tuple[str, ...]
    voice_notes: str
    visual_notes: str

    def few_shot_block(self) -> str:
        numbered = []
        for i, example in enumerate(self.few_shot_examples, start=1):
            numbered.append(f"EXAMPLE {i}:\n\"{example}\"")
        return (
            "GOLDEN FEW-SHOT REFERENCES — learn cadence, emotional depth, and "
            "hook energy. Do not copy their wording or force their structure:\n\n"
            + "\n\n".join(numbered)
        )


RELATIONSHIP = NichePreset(
    key="relationship",
    label="reflective adult relationship",
    aesthetic_prefix=RISO_PREFIX,
    negative_prompt=RISO_STYLE.style_negative,
    few_shot_examples=(
        "Once Kafka said, what is love? After all, it is quite simple. Love is "
        "everything which enhances, widens and enriches our life. In its heights "
        "and in its depths, it makes every moment of life worth living.",
        "Let people lose you. Let them misunderstand you. Let them create their "
        "own stories. Don't rush to fix them. Let time answer what you never "
        "needed to explain. Every truth reveals itself when the right time comes.",
        "Never ask a liar why they lied. To explain it, they would have to lie "
        "again. One lie is never born alone. It always needs another to keep "
        "alive, then another to protect the last, until even they forget where "
        "the truth ended and the lie becomes the only story they remember.",
        "If someone values you, they make time for you. Genuine love is easy to "
        "recognize because it shows up in simple ways. It's a text to check in, "
        "a callback when they said they would, a plan that actually happens. "
        "Even on busy days, they still find a moment because you matter to them, "
        "because they want to stay connected.",
        "I kept my words to myself, but I saw everything and noticed everything. "
        "When I stay quiet, it doesn't mean I'm clueless. Not everything needs "
        "an immediate response, and sometimes the best approach is to let people "
        "reveal themselves over time.",
        "Once the rain is over, an umbrella becomes a burden to everyone. The "
        "day a blind man sees, the first thing he throws away is the stick that "
        "helped him all his life. When the sun returns, the first thing people "
        "extinguish is the candle that guided them through the dark. That's how "
        "loyalty ends. When benefits stop.",
    ),
    voice_notes=(
        "First-person, concrete, unperformed. No parenting advice. "
        "No therapist-speak slogans."
    ),
    visual_notes=(
        "Atmospheric painterly risograph, golden rim-lit silhouettes, gouache "
        "depth, paper tooth, and fine ink. Never photoreal or flat vector. "
        "Footwear or cropped feet. Mood-match the spoken beat."
    ),
)

PARENTING = NichePreset(
    key="parenting",
    label="parenting presence and time",
    aesthetic_prefix=RISO_PREFIX,
    negative_prompt=RISO_STYLE.style_negative,
    few_shot_examples=(
        "They won't remember how clean the house was. They will remember if "
        "you were present when they looked up.",
        "One day you will put your child down and never pick them up again. "
        "Honor the moments that slip away quietly.",
    ),
    voice_notes=(
        "Intimate, present-tense observation of childhood time. No lectures, "
        "no productivity tips, no shame. Write from inside the ache of "
        "presence and the moments that disappear."
    ),
    visual_notes=(
        "Warm domestic twilight, parent-and-child silhouettes, wooden toys, "
        "worn shoes, painterly risograph gouache, paper grain, golden rim light. "
        "Never photoreal or flat vector."
    ),
)

_REGISTRY: dict[str, NichePreset] = {
    RELATIONSHIP.key: RELATIONSHIP,
    PARENTING.key: PARENTING,
}

_LOCATION_TERMS = (
    "kitchen",
    "hallway",
    "street",
    "train",
    "station",
    "bedroom",
    "living room",
    "park",
    "diner",
    "booth",
    "table",
    "window",
    "apartment",
    "office",
    "platform",
    "sidewalk",
    "cafe",
    "beach",
    "forest",
)
_UNSAFE_HUMAN_TERMS = (
    "front-facing",
    "front facing",
    "direct eye contact",
    "looking at camera",
    "smiling mouth",
    "speaking mouth",
)
_CAMERA_ARC = (
    "wide establishing angle",
    "medium side angle",
    "over-the-shoulder angle",
    "macro object detail",
    "low angle",
    "soft-focus silhouette",
    "high angle",
    "quiet closing detail",
)

DEFAULT_CELESTIAL_DISC_SCENES = frozenset({8})
_NON_CELESTIAL_PALETTES = {
    "WARM": (
        "Nostalgic practical-light palette: dusty rose, terracotta, cream paper, "
        "and amber tungsten illumination"
    ),
    "COLD": (
        "Atmospheric practical-light palette: deep cobalt, pale slate grey, "
        "and restrained amber tungsten accents"
    ),
    "CONTRAST": (
        "High-contrast practical-light palette: warm amber lamps against deep "
        "indigo shadow planes"
    ),
}
_NON_CELESTIAL_LIGHTING = {
    1: "a wet exterior with one practical amber lantern",
    3: "rain on window glass with distant city lights",
    4: "a single warm desk lamp in a dark room",
    5: "shadows in an unlit hallway",
    6: "a wet street with one practical lantern",
    7: "a wet walking route under one practical amber streetlamp",
}
_CELESTIAL_NEGATIVE = (
    "sun disc, moon disc, glowing celestial circle, celestial orb, eclipse, "
    "giant sun, giant moon"
)


def normalize_niche_key(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"relationship_reflective", "relationship"}:
        return "relationship"
    if raw in {"parenting", "momma_circle"}:
        return "parenting"
    return "relationship"


def get_niche_preset(key: str | None = None) -> NichePreset:
    return _REGISTRY[normalize_niche_key(key)]


def wrap_visual_prompt(
    visual_concept: str,
    niche: str | NichePreset | None = None,
) -> str:
    """Compatibility wrapper using scene-one's active Risograph palette."""
    del niche
    return build_flux_prompt(visual_concept, 1)[0]


def build_flux_prompt(
    beat_visual_concept: str,
    scene_idx: int,
    *,
    celestial_scenes: set[int] | frozenset[int] | None = None,
) -> tuple[str, str]:
    """Build the complete Flux prompt from the active style and 3-act palette."""
    scene = max(1, int(scene_idx))
    allowed_celestial = (
        DEFAULT_CELESTIAL_DISC_SCENES
        if celestial_scenes is None
        else frozenset(int(value) for value in celestial_scenes)
    )
    palette_key = "WARM" if scene <= 3 else ("COLD" if scene <= 6 else "CONTRAST")
    palette_text = RISO_STYLE.palettes[palette_key]
    if scene not in allowed_celestial:
        palette_text = _NON_CELESTIAL_PALETTES[palette_key]
    concept = " ".join(str(beat_visual_concept or "").split()).rstrip(".,; ")
    concept_low = concept.lower()
    if scene == 1:
        spatial_guard = (
            "SPATIAL ISOLATION: exterior establishing environment, no beds, "
            "mattresses, dressers, nightstands, sofas, or indoor furniture"
        )
    elif scene == 2:
        spatial_guard = (
            "SPATIAL ISOLATION: strictly architectural interior framing; "
            "furniture only where naturally supported by this interior"
        )
    elif scene == 3:
        spatial_guard = (
            "SPATIAL ISOLATION: tactile prop in its realistic indoor context, "
            "supported by a table, shelf, rack, or other proper surface"
        )
    elif scene == 4:
        spatial_guard = (
            "SPATIAL ISOLATION: intimate interior close-up beside a practical "
            "desk lamp; subject properly seated or standing"
        )
    elif scene == 5:
        spatial_guard = (
            "SPATIAL ISOLATION: interior corridor transit, no outdoor furniture "
            "and no person on the floor"
        )
    elif scene == 6:
        spatial_guard = (
            "SPATIAL ISOLATION: subject indoors behind rain-streaked or reflective "
            "glass, wet street visible outside"
        )
    elif scene == 7:
        spatial_guard = (
            "SPATIAL ISOLATION: exterior walking route, no beds, mattresses, "
            "dressers, nightstands, sofas, chairs, or indoor furniture"
        )
    elif scene == 8:
        spatial_guard = (
            "SPATIAL ISOLATION: expansive exterior panorama, no beds, mattresses, "
            "dressers, nightstands, sofas, or indoor furniture"
        )
    elif any(term in concept_low for term in ("bedroom", "nursery")):
        spatial_guard = (
            "SPATIAL ISOLATION: strictly indoors, cozy interior, wooden "
            "floorboards, no outdoor placement"
        )
    elif any(
        term in concept_low
        for term in ("street", "alley", "platform", "sidewalk", "outdoors")
    ):
        spatial_guard = (
            "SPATIAL ISOLATION: exterior environment, no beds, mattresses, "
            "dressers, nightstands, sofas, or indoor furniture"
        )
    else:
        spatial_guard = "SPATIAL ISOLATION: realistic object placement"
    if scene in allowed_celestial:
        lighting_guard = (
            "A visible sunset sun or moon disc is permitted in this scene only "
            "when motivated by the story; it is not required"
        )
        negative = RISO_STYLE.style_negative
    else:
        treatment = _NON_CELESTIAL_LIGHTING.get(
            scene, "intimate tactile interior chiaroscuro"
        )
        enclosure = (
            "uniform overcast mist fills the sky with no focal shape"
            if scene in {1, 7}
            else "the composition is enclosed or tightly cropped so sky and horizon stay outside frame"
        )
        lighting_guard = (
            f"PRACTICAL LIGHTING ONLY: illumination comes exclusively from "
            f"{treatment}; {enclosure}"
        )
        negative = f"{RISO_STYLE.style_negative}, {_CELESTIAL_NEGATIVE}"
    positive = (
        f"{RISO_STYLE.open} {RISO_STYLE.technique} "
        f"{palette_text} {concept}. {spatial_guard}. "
        f"{lighting_guard}. {_UPRIGHT_ANATOMY_GUARD}. "
        f"{RISO_STYLE.mood} {RISO_STYLE.linework_guard}, "
        f"{RISO_STYLE.format}"
    )
    return " ".join(positive.split()), negative


def select_celestial_disc_scenes(script: dict[str, Any]) -> frozenset[int]:
    """Reserve one intentional disc at resolution, leaving leak headroom."""
    requested = script.get("celestial_disc_scenes")
    if isinstance(requested, (list, tuple, set, frozenset)):
        scenes = frozenset(int(value) for value in requested if int(value) in {1, 8})
        if 1 <= len(scenes) <= 2:
            return scenes
    return frozenset({8})


def _non_celestial_anchor(anchor: str) -> str:
    """Remove time-of-day cues that cause FLUX to invent a sky disc."""
    cleaned = str(anchor or "")
    replacements = (
        (r"\bgolden[\s-]+hour\b", "amber lamplight"),
        (r"\bsunrise\b|\bdawn\b", "misty overcast morning"),
        (r"\bsunset\b", "rainy blue hour"),
        (r"\bsunlit\b|\bsunny\b", "softly lamp-lit"),
        (r"\bmoonlit\b", "lantern-lit"),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _scene_location_anchor(
    anchor: str,
    scene: int,
    celestial_scenes: frozenset[int],
) -> str:
    """Keep non-disc scenes free of outdoor cues that provoke sky orbs."""
    if scene in celestial_scenes:
        return anchor
    base = _non_celestial_anchor(anchor)
    grounded = {
        1: base,
        2: "windowless architectural interior in the same story world",
        3: "tight tabletop detail inside a windowless room",
        4: "windowless intimate room with one practical desk lamp",
        5: "enclosed shadowed corridor with closed doors and no windows",
        6: "night interior behind rain-streaked glass with distant streetlights",
        7: "misty night walking route lit only by vintage streetlamps",
    }
    return grounded.get(scene, base)


def anchor_visual_concept(
    location_anchor: str,
    visual_concept: str,
    *,
    scene: int = 1,
) -> str:
    """Attach the writer's story world without erasing its scene staging."""
    anchor = " ".join(str(location_anchor or "").split()).rstrip(".,; ")
    concept = " ".join(str(visual_concept or "").split()).rstrip(".,; ")
    if not anchor:
        raise ValueError("single-pass writer returned no location_anchor")
    if not concept:
        raise ValueError("single-pass writer returned an empty visual_concept")
    concept_low = concept.lower()
    unsafe_human = any(term in concept_low for term in _UNSAFE_HUMAN_TERMS)
    if unsafe_human:
        angle = _CAMERA_ARC[(max(1, int(scene)) - 1) % len(_CAMERA_ARC)]
        concept = (
            f"{angle}, natural side profile or silhouette, changing light across "
            "tactile surfaces in the declared environment"
        )
    if concept.lower().startswith(anchor.lower()):
        return concept
    return f"{anchor}. {concept}"


def inject_prompt_fields(
    script: dict[str, Any],
    niche: str | NichePreset | None = None,
) -> dict[str, Any]:
    """Add full Flux fields without another model call or output-token cost."""
    preset = niche if isinstance(niche, NichePreset) else get_niche_preset(
        str(niche or script.get("niche") or script.get("module") or "relationship")
    )
    fallback_anchor = (
        PARENTING_LOCATION_ANCHOR
        if preset.key == "parenting"
        else RELATIONSHIP_LOCATION_ANCHOR
    )
    anchor = str(script.get("location_anchor") or fallback_anchor).strip()
    script["location_anchor"] = anchor
    celestial_scenes = select_celestial_disc_scenes(script)
    script["celestial_disc_scenes"] = sorted(celestial_scenes)
    script["celestial_disc_budget"] = len(celestial_scenes)
    rows = [row for row in (script.get("lines") or []) if isinstance(row, dict)]
    for i, row in enumerate(rows, start=1):
        scene = int(row.get("scene") or i)
        writer_concept = anchor_visual_concept(
            anchor,
            str(row.get("visual_concept") or ""),
            scene=scene,
        )
        camera_contract = CAMERA_STAGING_BY_SCENE.get(
            scene,
            CAMERA_STAGING_BY_SCENE[((scene - 1) % 8) + 1],
        )
        subject_contract = (
            "parent and child in a coherent shared story world"
            if preset.key == "parenting"
            else "the recurring adult subject in a coherent story world"
        )
        if scene == 3:
            subject_contract = "one story-relevant prop in its realistic context"
        scene_anchor = _scene_location_anchor(anchor, scene, celestial_scenes)
        anchored = (
            f"{scene_anchor}. REQUIRED CAMERA AND POSE—OVERRIDES WRITER FRAMING: "
            f"{camera_contract}. SUBJECT: {subject_contract}. Emotional action "
            "must match the spoken narration"
        )
        row["writer_visual_concept"] = writer_concept
        row["visual_concept"] = anchored
        row["camera_staging"] = camera_contract
        row["celestial_disc_allowed"] = scene in celestial_scenes
        if row["celestial_disc_allowed"]:
            row["lighting_prompt_guard"] = (
                "CELESTIAL BUDGET: this is one of the reel's designated giant "
                "sun or moon disc scenes; use at most one clear celestial disc"
            )
        else:
            row["lighting_allocation"] = _NON_CELESTIAL_LIGHTING.get(
                scene, "intimate tactile interior chiaroscuro"
            )
            retry_enclosure = (
                "render a uniform overcast mist field behind the subject"
                if scene in {1, 7}
                else "keep sky and horizon completely outside the crop"
            )
            row["lighting_prompt_guard"] = (
                "PRACTICAL-LIGHT-ONLY RETRY: all illumination comes from "
                f"{row['lighting_allocation']}; {retry_enclosure}"
            )
        positive, negative = build_flux_prompt(
            anchored,
            scene,
            celestial_scenes=celestial_scenes,
        )
        row["final_positive_prompt"] = positive
        row["negative_prompt"] = negative
    script["lines"] = rows
    script["niche"] = preset.key
    script["aesthetic_wrapper"] = preset.aesthetic_prefix
    return script


def writer_visual_clause(preset: NichePreset) -> str:
    participants = (
        "Keep the parent-child relationship legible through natural shared action."
        if preset.key == "parenting"
        else "Keep recurring people visually coherent without forcing one protagonist."
    )
    return (
        "VISUAL STAGING — Use these eight distinct perspectives in order: wide "
        "establishing exterior; architectural interior framing; tactile prop "
        "detail; dramatic close-up side profile; motion transit; rain or "
        "reflective glass; dynamic walking silhouette; expansive panoramic "
        "closure. Choose a fresh, story-specific bedroom, porch, train "
        "platform, library, or rainy street. Domestic bathrooms, public restrooms, "
        "washrooms, toilets, urinals, and commodes are strictly forbidden. Do not "
        "default to a doorway, giant sun, isolated cup, hallway, or sunset alley. "
        "A bedroom or nursery is always a cozy indoor interior with wooden "
        "floorboards. Exterior scenes contain no beds, mattresses, dressers, "
        "nightstands, sofas, or other indoor furniture. "
        "Every human remains upright: standing, walking, properly seated on "
        "furniture, or tucked into bed with visible pillows. Never stage anyone "
        "lying on a floor, rug, street, or doorstep, crawling, or prone. "
        "The location_anchor names the coherent story world, not a mandatory prop. "
        "Every prop must belong in its "
        "real context: cups on tables or counters, bags on racks or seats, books on "
        "desks or shelves. Never scatter symbolic objects on floors or thresholds. "
        f"{participants} Render as {preset.visual_notes} No front-facing portrait, "
        "direct eye contact, visible speaking mouth, photorealism, or flat vector. "
        "Do not include the style prefix; the pipeline adds it."
    )


def writer_system_extras(preset: NichePreset) -> str:
    return f"{preset.few_shot_block()}\n\n{writer_visual_clause(preset)}"


def preset_notes(preset: NichePreset) -> dict[str, Any]:
    return {
        "niche_id": preset.key,
        "niche_label": preset.label,
        "audience": preset.label,
        "voice_notes": preset.voice_notes,
        "aesthetic_prefix": preset.aesthetic_prefix,
        "negative_prompt": preset.negative_prompt,
    }
