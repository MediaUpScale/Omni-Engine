from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from agents.mcp.text_model import TextResult, _complete_gemini_flash
from agents.writer.freeform_writer import (
    ScriptDraft,
    _SYSTEM,
    _normalize_total_word_budget,
    _output_contract,
    _validate_golden_contract,
    system_prompt_for,
)
from agents.writer.script_brain import BrainResult, draft_to_script
from agents.writer.writer_brief import (
    COGNITIVE_BRIDGE_STYLES,
    PHILOSOPHICAL_ANCHORS,
    PSYCHOLOGICAL_REALITY_ANCHORS,
    WriterBrief,
    get_narrative_harness,
    select_single_conflict,
)
from core.economic_reel_lofi.niche_presets import (
    PARENTING,
    PARENTING_LOCATION_ANCHOR,
    RELATIONSHIP,
    RELATIONSHIP_LOCATION_ANCHOR,
    build_flux_prompt,
    get_niche_preset,
    inject_prompt_fields,
    select_celestial_disc_scenes,
    wrap_visual_prompt,
)
from core.economic_reel_lofi import config as lofi_cfg
from core.economic_reel_lofi import lofi_collections as rag
from core.economic_reel_lofi.image_gen import (
    enforce_full_bleed,
    generate_scene_image_gemini,
)
from core.economic_reel_lofi.output_hygiene import (
    enforce_clips_mp4_only,
    migrate_channel_clips,
    migrate_outputs_root,
)
from core.economic_reel_lofi.pipeline import (
    _assemble_stage3_prompts,
    _current_run_scene_audio_paths,
    _gate2_review_rows,
    _generate_validated_script,
    _load_locked_script,
    _print_script_report,
    _produce_one,
    _purge_stale_temp_voice_files,
    _resolve_page_dirs,
)
from core.economic_reel_lofi.style_modules.riso_retro_flat_v4 import STYLE as RISO_STYLE
from core.economic_reel_lofi.visual_concept import _fallback_concept
from core.economic_reel_lofi.visual_identity import assign_palette_arc
from core.economic_reel_lofi.assembler import (
    _visual_only,
    audit_watermark_native_size,
    is_verified_instrumental_bgm,
    list_library_bgm_tracks,
    render_logo_layer,
)
from core.economic_reel_lofi.voiceover import ensure_script_voiceover


def test_emotional_brief_prioritizes_depth_without_literal_props() -> None:
    brief = WriterBrief.from_emotional(theme="detachment", subtheme="quiet_boundary")
    block = brief.assignment_block()
    assert brief.mode == "emotional"
    assert "EVERYDAY CADENCE & ANAPHORA" in block
    assert "Selected anchor" in block
    assert "Every beat contains 4–6 simple spoken words" in block
    assert "at most 48 words" in block
    assert "SUBJECT GROUNDING" in block
    assert "ONE LOCKED CONFLICT" in block
    assert "SINGLE-CONFLICT UNITY" in block
    assert "SPATIAL ISOLATION" in block


def test_fast_prompt_embeds_all_six_golden_examples_and_candidate_schema() -> None:
    for number in range(1, 7):
        assert f"EXAMPLE {number}" in _SYSTEM
    assert "Once Kafka said" in _SYSTEM
    assert "Never ask a liar why they lied" in _SYSTEM
    assert "When benefits stop" in _SYSTEM
    contract = _output_contract(WriterBrief.from_emotional(theme="boundaries"))
    assert '"writer_mode"' not in contract
    assert '"niche"' not in contract
    assert '"location_anchor"' in contract
    assert "visual_concept" in contract
    assert '"beats": [' in contract
    assert "exactly eight" in _SYSTEM
    assert "4 RULES OF THE GOLDEN SCRIPT" in _SYSTEM
    assert "Every beat contains 4–6" in _SYSTEM
    assert "PROVEN MOMMA CIRCLE GOLDEN BLUEPRINT" in _SYSTEM
    assert "POV SUBJECT INTEGRITY" in _SYSTEM
    assert "ANAPHORA AND CADENCE" in _SYSTEM
    assert "before any" in _SYSTEM
    assert "bathrooms" in _SYSTEM
    assert "Do not default" in _SYSTEM
    assert "cups on tables" in _SYSTEM


def test_parenting_preset_swaps_few_shots_and_aesthetic_wrapper() -> None:
    brief = WriterBrief.from_emotional(theme="presence", module="parenting")
    prompt = system_prompt_for(brief)
    assert "They won't remember how clean the house was" in prompt
    assert "One day you will put your child down" in prompt
    assert "Never ask a liar why they lied" not in prompt
    assert "parent-child relationship" in prompt
    assert "train platform" in prompt
    assert "washrooms" in prompt
    assert get_niche_preset("parenting") is PARENTING
    wrapped = wrap_visual_prompt(
        "A cinematic, melancholic shot of a dimly lit hallway with two umbrellas leaning on separate walls.",
        "relationship",
    )
    assert wrapped.startswith(RELATIONSHIP.aesthetic_prefix)
    assert "Nostalgic practical-light palette" in wrapped
    assert "vertical 9:16 full-bleed composition" in wrapped.lower()
    assert "vintage graphic novel poster" in wrapped.lower()
    assert "35mm" not in wrapped.lower()
    parent_wrap = wrap_visual_prompt("small shoes by a sunlit doorway", "parenting")
    assert parent_wrap.startswith(PARENTING.aesthetic_prefix)
    assert "photorealistic" in PARENTING.negative_prompt


def test_draft_to_script_keeps_writer_visual_concepts() -> None:
    brief = WriterBrief.from_emotional(theme="distance", module="relationship")
    draft = ScriptDraft(
        lines=[
            "The loudest departure always begins long before anyone packs a bag.",
            "I stopped translating absence into excuses that protected everyone else.",
            "Because love without presence slowly becomes another form of waiting.",
            "The boundary felt cruel only while abandonment still felt familiar.",
            "Then distance showed me what closeness had kept carefully hidden.",
            "Some endings hurt because they return our neglected selves.",
            "Maturity begins when grief no longer negotiates against self-respect.",
        ],
        location_anchor="A dim diner booth beside a rain-slicked window",
        brief=brief,
        visual_concepts=[
            "A melancholic wide view across the rain-slicked diner window."
        ]
        + [f"quiet still {i}" for i in range(2, 8)],
        meta={"niche": "relationship"},
    )
    script = draft_to_script(draft)
    inject_prompt_fields(script)
    assert script["niche"] == "relationship"
    assert script["location_anchor"] == "A dim diner booth beside a rain-slicked window"
    assert "A melancholic wide view" in script["lines"][0]["writer_visual_concept"]
    assert "REQUIRED CAMERA AND POSE" in script["lines"][0]["visual_concept"]
    assert script["lines"][0]["visual_source"] == "writer_single_pass"


def test_prompt_fields_are_programmatic_and_single_location_anchored() -> None:
    anchor = "A kitchen table with cold coffee cups at dusk"
    script = {
        "niche": "relationship",
        "location_anchor": anchor,
        "celestial_disc_scenes": [1, 8],
        "lines": [
            {
                "scene": i,
                "text": "Silence settled where our honest answers once belonged.",
                "visual_concept": f"{angle} view of one untouched cup under fading light",
            }
            for i, angle in enumerate(
                (
                    "wide",
                    "medium",
                    "over-the-shoulder",
                    "macro",
                    "low-angle",
                    "side-profile",
                    "high-angle",
                    "close detail",
                ),
                start=1,
            )
        ],
    }
    inject_prompt_fields(script)
    assert script["location_anchor"] == anchor
    camera_staging = [beat["camera_staging"] for beat in script["lines"]]
    assert len(set(camera_staging)) == 8
    for i, beat in enumerate(script["lines"], start=1):
        if i in {1, 8}:
            assert beat["visual_concept"].startswith(anchor)
        else:
            assert not beat["visual_concept"].startswith(anchor)
        assert beat["final_positive_prompt"].startswith(RELATIONSHIP.aesthetic_prefix)
        assert "photorealistic" in beat["negative_prompt"]
        assert beat["celestial_disc_allowed"] is (i in {1, 8})
        if i in {1, 8}:
            assert "sun disc, moon disc" not in beat["negative_prompt"]
            assert beat["lighting_prompt_guard"].startswith("CELESTIAL BUDGET")
        else:
            assert "sun disc, moon disc" in beat["negative_prompt"]
            assert "PRACTICAL LIGHTING ONLY" in beat["final_positive_prompt"]
            assert beat["lighting_prompt_guard"].startswith(
                "PRACTICAL-LIGHT-ONLY RETRY"
            )
        expected_palette = "WARM" if i <= 3 else ("COLD" if i <= 6 else "CONTRAST")
        if i in {1, 8}:
            assert RISO_STYLE.palettes[expected_palette] in beat["final_positive_prompt"]
        else:
            assert "practical-light palette" in beat["final_positive_prompt"].lower()
            assert "golden hour sunset" not in beat["final_positive_prompt"]
            assert "blazing orange-amber sun" not in beat["final_positive_prompt"]
        assert "ANATOMY CONTRACT" in beat["final_positive_prompt"]
    assert "wide establishing landscape or exterior" in camera_staging[0]
    assert "architectural interior framing" in camera_staging[1]
    assert "tactile environmental prop detail" in camera_staging[2]
    assert "close-up side profile" in camera_staging[3]
    assert "motion transit" in camera_staging[4]
    assert "rain or reflective glass" in camera_staging[5]
    assert "dynamic walking silhouette" in camera_staging[6]
    assert "expansive panoramic closure" in camera_staging[7]
    assert "exterior establishing environment" in script["lines"][0]["final_positive_prompt"]
    assert "strictly architectural interior framing" in script["lines"][1]["final_positive_prompt"]
    assert "tactile prop in its realistic indoor context" in script["lines"][2]["final_positive_prompt"]
    assert "intimate interior close-up" in script["lines"][3]["final_positive_prompt"]
    assert "interior corridor transit" in script["lines"][4]["final_positive_prompt"]
    assert "indoors behind rain-streaked" in script["lines"][5]["final_positive_prompt"]
    assert "exterior walking route" in script["lines"][6]["final_positive_prompt"]
    assert "expansive exterior panorama" in script["lines"][7]["final_positive_prompt"]


def test_riso_flux_builder_uses_three_act_palette_and_style_negative() -> None:
    assert RISO_STYLE.model == "black-forest-labs/FLUX.1-dev"
    assert RISO_STYLE.steps == 20
    assert RISO_STYLE.guidance_scale == 5.5
    assert RISO_STYLE.profile_name == "style-riso_painting_retro_vintage"
    assert "full-bleed composition" in RISO_STYLE.format
    assert "white border" in RISO_STYLE.style_negative
    assert "photorealistic" in RISO_STYLE.style_negative
    assert "flat vector" in RISO_STYLE.style_negative
    assert "bathroom" in RISO_STYLE.style_negative
    assert "porcelain toilet" in RISO_STYLE.style_negative
    assert "bed outdoors" in RISO_STYLE.style_negative
    assert "furniture on street" in RISO_STYLE.style_negative
    assert "surreal placement" in RISO_STYLE.style_negative
    assert "lying on floor" in RISO_STYLE.style_negative
    assert "crawling" in RISO_STYLE.style_negative
    assert "prone body" in RISO_STYLE.style_negative
    assert "melted limbs" in RISO_STYLE.style_negative
    assert "anatomical glitches" in RISO_STYLE.style_negative
    positive, negative = build_flux_prompt("woman waiting beside tea", 7)
    assert positive.startswith(RISO_STYLE.open)
    assert RISO_STYLE.technique.strip() in positive
    assert "warm amber lamps" in positive
    assert "blazing orange-amber sun" not in positive
    assert RISO_STYLE.linework_guard in positive
    assert "35mm" not in positive.lower()
    assert "vintage graphic novel poster" in positive.lower()
    assert "full-bleed" in positive.lower()
    assert "paper tooth" in positive.lower()
    assert negative.startswith(RISO_STYLE.style_negative)
    assert "sun disc, moon disc" in negative


def test_full_bleed_crop_removes_generated_paper_margin(tmp_path: Path) -> None:
    image_path = tmp_path / "bordered.png"
    image = Image.new("RGB", (100, 100), "white")
    image.paste("red", (5, 5, 95, 95))
    image.save(image_path)

    enforce_full_bleed(image_path, trim_frac=0.05)

    with Image.open(image_path) as result:
        assert result.size == (100, 100)
        assert result.getpixel((0, 0))[0] > 240
        assert result.getpixel((0, 0))[1] < 20


def test_parenting_prompt_fields_lock_parent_child_arc() -> None:
    script = {
        "niche": "parenting",
        "location_anchor": "somewhere else",
        "lines": [
            {
                "scene": i,
                "text": "Time moves quietly through the rooms we share.",
                "visual_concept": "soft twilight, paper grain, quiet gesture",
            }
            for i in range(1, 9)
        ],
    }

    inject_prompt_fields(script)

    assert script["location_anchor"] == "somewhere else"
    for beat in script["lines"]:
        if beat["scene"] == 8:
            assert beat["visual_concept"].startswith("somewhere else")
        elif beat["scene"] == 3:
            assert "story-relevant prop" in beat["visual_concept"]
        else:
            assert "parent and child" in beat["visual_concept"]
        assert "soft twilight" in beat["writer_visual_concept"].lower()


def test_theme_selector_never_repeats_consecutively(tmp_path: Path) -> None:
    bank = tmp_path / "lofi_theme_bank_relationship.json"
    rotation = tmp_path / "lofi_theme_rotation_relationship.json"
    bank.write_text(
        json.dumps(
            [
                {
                    "theme": "trust",
                    "subtheme": "consistency",
                    "status": "used",
                    "last_used_date": "2026-09-01",
                },
                {
                    "theme": "closure",
                    "subtheme": "unfinished",
                    "status": "used",
                    "last_used_date": "2026-09-01",
                },
                {
                    "theme": "pride",
                    "subtheme": "cost",
                    "status": "used",
                    "last_used_date": "2026-09-01",
                },
            ]
        ),
        encoding="utf-8",
    )

    def fake_store(name: str) -> Path:
        if name == "lofi_theme_bank_relationship":
            return bank
        if name == "lofi_theme_rotation_relationship":
            return rotation
        return tmp_path / f"{name}.json"

    with (
        patch.object(rag, "ensure_seeded"),
        patch.object(rag, "_store_path", side_effect=fake_store),
    ):
        first = rag.select_theme("relationship")
        rag.mark_theme_used(
            "relationship", first["theme"], first.get("subtheme")
        )
        second = rag.select_theme("relationship")
        rag.mark_theme_used(
            "relationship", second["theme"], second.get("subtheme")
        )
        third = rag.select_theme("relationship")

    assert (first["theme"], first["subtheme"]) != (
        second["theme"],
        second["subtheme"],
    )
    assert (second["theme"], second["subtheme"]) != (
        third["theme"],
        third["subtheme"],
    )
    persisted = json.loads(bank.read_text(encoding="utf-8"))
    stamped = next(row for row in persisted if row["theme"] == first["theme"])
    assert "T" in stamped["last_used_at"]


def test_total_narration_budget_is_normalized_without_llm_repair() -> None:
    lines = ["One two three four five six seven eight nine ten."] * 8
    normalized = _normalize_total_word_budget(lines)
    assert sum(len(line.split()) for line in normalized) == 48
    assert all(4 <= len(line.split()) <= 6 for line in normalized)
    assert len(normalized[0].split()) <= 6


def test_golden_contract_rejects_banned_words_and_bathrooms() -> None:
    valid_lines = ["Let your partner show true care."] * 8
    visuals = ["side profile on a rainy porch"] * 8

    _validate_golden_contract(
        valid_lines,
        location_anchor="rainy front porch",
        visuals=visuals,
    )
    with pytest.raises(ValueError, match="word counts outside contract"):
        _validate_golden_contract(
            ["Seneca said our thoughts hurt us deeply.", *valid_lines[1:]],
            location_anchor="rainy front porch",
            visuals=visuals,
        )
    with pytest.raises(ValueError, match="forbidden vocabulary"):
        _validate_golden_contract(
            [*valid_lines[:-1], "Leave this torment behind right now."],
            location_anchor="rainy front porch",
            visuals=visuals,
        )
    with pytest.raises(ValueError, match="bathroom staging"):
        _validate_golden_contract(
            valid_lines,
            location_anchor="dim domestic bathroom",
            visuals=visuals,
        )
    with pytest.raises(ValueError, match="ambiguous pronoun"):
        _validate_golden_contract(
            ["They left without one last call.", *valid_lines[1:]],
            location_anchor="rainy front porch",
            visuals=visuals,
        )
    grounded = [
        "Seneca said we fear shadows.",
        "A child needs simple comfort.",
        "They fear being left alone.",
        *["Let your child feel safe here."] * 5,
    ]
    _validate_golden_contract(
        grounded,
        location_anchor="cozy bedroom interior",
        visuals=["wooden floorboards under warm light"] * 8,
    )
    with pytest.raises(ValueError, match="indoor furniture"):
        _validate_golden_contract(
            valid_lines,
            location_anchor="rainy city street",
            visuals=["an empty bed under streetlights", *visuals[1:]],
        )


def test_subject_integrity_rejects_role_switches_and_family_anecdotes() -> None:
    romantic = ["Let your partner show true care."] * 8
    parenting = ["Let your child feel safe here."] * 8
    visuals = ["side profile under a table lamp"] * 8
    with pytest.raises(ValueError, match="romantic and parenting"):
        _validate_golden_contract(
            [*romantic[:4], "Your father waits beside you.", *romantic[5:]],
            location_anchor="windowless room",
            visuals=visuals,
            niche="relationship",
        )
    with pytest.raises(ValueError, match="romantic and parenting"):
        _validate_golden_contract(
            [*parenting[:4], "Your partner waits beside you.", *parenting[5:]],
            location_anchor="windowless room",
            visuals=visuals,
            niche="parenting",
        )
    with pytest.raises(ValueError, match="family anecdote"):
        _validate_golden_contract(
            ["My grandfather taught this lesson.", *romantic[1:]],
            location_anchor="windowless room",
            visuals=visuals,
            niche="relationship",
        )


def test_parenting_contract_rejects_multiple_friction_families() -> None:
    lines = [
        "Mothers often carry heavy guilt.",
        "They scrub every dish twice.",
        "Then anger breaks their patience.",
        "Their child waits beside them.",
        "Quiet minutes keep moving past.",
        "Warm presence matters much more.",
        "Set that pressure down now.",
        "Hold your child with calm.",
    ]
    with pytest.raises(ValueError, match="multiple emotional frictions"):
        _validate_golden_contract(
            lines,
            location_anchor="windowless family room",
            visuals=["warm table lamp and upright mother"] * 8,
            niche="parenting",
        )
    assert select_single_conflict(
        "parenting", "self_compassion", "good_enough_mother"
    ) == "perfectionism guilt about keeping the home perfect"
    assert select_single_conflict(
        "parenting", "presence", "phones_down_eye_contact"
    ) == "screen distraction stealing present childhood time"


def test_narrative_harness_has_deterministic_approved_anchor_branches() -> None:
    philosophy = get_narrative_harness("relationship", rng=random.Random(1))
    psychology = get_narrative_harness("parenting", rng=random.Random(2))

    assert philosophy["use_philosophy"] is True
    assert philosophy["narrative_anchor"] in PHILOSOPHICAL_ANCHORS
    assert philosophy["narrative_mode"] == "famous_thinker_hook"
    assert psychology["use_philosophy"] is False
    assert psychology["narrative_anchor"] in PSYCHOLOGICAL_REALITY_ANCHORS
    assert psychology["narrative_mode"] == "original_freewriting"
    assert psychology["narrative_niche"] == "parenting"
    style_names = {name for name, _ in COGNITIVE_BRIDGE_STYLES}
    assert philosophy["cognitive_bridge_style"] in style_names
    assert psychology["cognitive_bridge_style"] in style_names
    assert philosophy["cognitive_bridge_direction"]
    thinker_rate = sum(
        bool(get_narrative_harness("relationship", rng=random.Random(seed))["use_philosophy"])
        for seed in range(1000)
    ) / 1000
    assert 0.60 <= thinker_rate <= 0.80


def test_celestial_budget_is_stable_and_limited_to_hook_or_resolution() -> None:
    script = {
        "theme": "grief",
        "subtheme": "learning to carry it",
        "location_anchor": "rainy coastal cottage",
    }
    first = select_celestial_disc_scenes(script)
    second = select_celestial_disc_scenes(script)
    assert first == second
    assert 1 <= len(first) <= 2
    assert first <= {1, 8}
    assert select_celestial_disc_scenes(
        {**script, "celestial_disc_scenes": [8]}
    ) == {8}


def test_non_budget_scenes_strip_celestial_location_cues() -> None:
    script = {
        "niche": "relationship",
        "location_anchor": "dawn porch beside moonlit tracks",
        "celestial_disc_scenes": [8],
        "lines": [
            {"scene": 1, "text": "Simple words fit this test.", "visual_concept": "wide view"},
            {"scene": 2, "text": "Simple words bridge this test.", "visual_concept": "inside view"},
            {"scene": 8, "text": "Simple words close this test.", "visual_concept": "wide close"},
        ],
    }
    inject_prompt_fields(script)
    first = script["lines"][0]["visual_concept"].lower()
    second = script["lines"][1]["visual_concept"].lower()
    last = script["lines"][2]["visual_concept"].lower()
    assert "dawn" not in first and "moonlit" not in first
    assert "misty overcast morning" in first and "lantern-lit" in first
    assert second.startswith("windowless architectural interior")
    assert "tracks" not in second
    assert last.startswith("dawn porch beside moonlit tracks")


def test_lofi_voice_speed_is_meditative() -> None:
    assert lofi_cfg.tts_speed() == 0.69
    assert lofi_cfg.VO_INTERLINE_SILENCE_S == 0.60


def test_gemini38_provider_uses_only_configured_primary_model() -> None:
    fake = TextResult("{}", "gemini", "models/gemini-3.8-flash", 1, 1, 0.0)
    with patch(
        "agents.mcp.text_model._complete_gemini",
        return_value=fake,
    ) as complete:
        result = _complete_gemini_flash("prompt", system="system")
    assert result is fake
    assert complete.call_count == 1
    assert complete.call_args.kwargs["model_chain"] == ["models/gemini-3.8-flash"]
    config = complete.call_args.kwargs["generation_config"]
    thinking = getattr(config, "thinking_config", None)
    assert thinking is None or int(getattr(thinking, "thinking_budget", 0) or 0) == 0
    assert float(config.temperature) == 0.35
    assert float(config.top_p) == 0.85
    assert int(config.max_output_tokens) == 768
    assert config.response_json_schema["properties"]["beats"]["minItems"] == 8


def test_fast_stage1_accepts_first_draft_without_judge_validator_or_repairs() -> None:
    lines = [
        "Silence taught me which promises were never meant to stay.",
        "I stopped translating absence into excuses that protected everyone else.",
        "Because love without presence slowly becomes another form of waiting.",
        "The boundary felt cruel only while abandonment still felt familiar.",
        "Then distance showed me what closeness had kept carefully hidden.",
        "Some endings hurt because they return our neglected selves.",
        "Maturity begins when grief no longer negotiates against self-respect.",
    ]
    lines.append("Leaving became the first honest kindness I offered myself.")
    draft = ScriptDraft(
        lines=lines,
        location_anchor="A kitchen table with cold coffee cups at dusk",
        human_situation="A person stops explaining absence.",
        visual_concepts=[f"relative camera angle {i} across the same cold cups" for i in range(8)],
        meta={"niche": "relationship"},
    )
    with (
        patch(
            "agents.writer.freeform_writer.write_draft",
            return_value=draft,
        ) as writer,
        patch(
            "core.economic_reel_lofi.pipeline.validate_script",
            side_effect=AssertionError("validator must be bypassed"),
        ),
        patch(
            "core.economic_reel_lofi.pipeline._repair_targeted_line",
            side_effect=AssertionError("repair must be bypassed"),
        ),
        patch(
            "agents.writer.script_brain.compose",
            side_effect=AssertionError("judge brain must be bypassed"),
        ),
        patch(
            "core.economic_reel_lofi.pipeline.rag.select_concrete_details",
            return_value=[],
        ),
        patch("core.economic_reel_lofi.pipeline.rag.note_batch_script"),
    ):
        script, errors, manual = _generate_validated_script(
            module="relationship",
            theme_row={"theme": "distance", "subtheme": "boundaries"},
            scene_count=9,
            duration_s=27,
            writer_mode="emotional",
            persist_on_pass=False,
            strict_judge=False,
        )
    assert writer.call_count == 1
    assert not errors and manual is False
    assert script is not None
    assert script["writer_diagnostics"]["attempt_count"] == 1
    assert script["writer_diagnostics"]["judge_calls"] == 0
    assert script["writer_diagnostics"]["repair_calls"] == 0


def test_strict_stage1_keeps_archived_judge_and_validator_path() -> None:
    brief = WriterBrief.from_emotional(theme="distance")
    draft = ScriptDraft(
        lines=[
            "Silence taught me which promises were never meant to stay.",
            "I stopped translating absence into excuses that protected everyone else.",
            "Because love without presence slowly becomes another form of waiting.",
            "The boundary felt cruel only while abandonment still felt familiar.",
            "Then distance showed me what closeness had kept carefully hidden.",
            "Some endings hurt because they return our neglected selves.",
            "Maturity begins when grief no longer negotiates against self-respect.",
        ],
        brief=brief,
    )
    brain = BrainResult(brief=brief, ok=True, draft=draft)
    accepted = {
        "theme": "distance",
        "writer_mode": "emotional",
        "arc_template": "thematic_arc",
        "lines": [{"scene": i + 1, "text": line} for i, line in enumerate(draft.lines)],
    }
    validation = SimpleNamespace(ok=True, reasons=[], script=accepted, feedback=lambda: "")
    with (
        patch("agents.writer.script_brain.compose", return_value=brain) as judge_brain,
        patch(
            "core.economic_reel_lofi.pipeline._repair_targeted_line",
            return_value=(None, 0),
        ),
        patch(
            "core.economic_reel_lofi.pipeline.validate_script",
            return_value=validation,
        ) as validator,
        patch(
            "core.economic_reel_lofi.pipeline.rag.select_concrete_details",
            return_value=[],
        ),
        patch("core.economic_reel_lofi.pipeline.rag.note_batch_script"),
    ):
        script, errors, manual = _generate_validated_script(
            module="relationship",
            theme_row={"theme": "distance"},
            scene_count=9,
            duration_s=27,
            writer_mode="emotional",
            strict_judge=True,
        )
    assert script is not None and not errors and manual is False
    assert judge_brain.call_count == 1
    assert validator.call_count >= 1


def test_default_beat_contract_caps_seven_words() -> None:
    assert lofi_cfg.beat_word_budget(3.0) == 6
    assert lofi_cfg.beat_word_ceiling(3.0) == 7
    assert lofi_cfg.thematic_caption_limits() == (7, 84)


def test_audio_slots_follow_voice_with_calm_floor_and_no_cut() -> None:
    assert lofi_cfg.slot_duration_for_vo(1.0, trailing_silence_s=0.3) == (2.5, True)
    assert lofi_cfg.slot_duration_for_vo(3.5, trailing_silence_s=0.3) == (3.8, True)
    assert lofi_cfg.slot_duration_for_vo(4.2, trailing_silence_s=0.3) == (4.5, True)


def test_atmospheric_fallback_does_not_require_caption_noun() -> None:
    row = {"text": "Some silences are boundaries finally learning to breathe."}
    concept = _fallback_concept(
        row,
        episode_state={"mood": "quiet detachment", "motif": ""},
        index=0,
    )
    assert concept["source"] == "atmospheric_fallback"
    assert concept["beat_mood"] == "quiet detachment"
    assert concept["scene_description"]
    assert concept["licensed_objects"]


def test_stage2_mood_lighting_survives_palette_arc() -> None:
    row = {
        "scene": 1,
        "visual_source": "llm",
        "lighting_condition": "rainy_grey",
        "beat_mood": "restrained grief",
    }
    assign_palette_arc([row])
    assert row["lighting_condition"] == "rainy_grey"


def test_stage3_wraps_writer_concepts_with_niche_aesthetic(tmp_path: Path) -> None:
    script = {
        "niche": "parenting",
        "module": "parenting",
        "lines": [
            {
                "scene": 1,
                "text": "They will remember if you were present when they looked up.",
                "visual_concept": "small shoes waiting by a sunlit doorway",
            }
        ],
    }
    _assemble_stage3_prompts(
        script,
        theme_row={},
        lock_visuals=False,
        clips_dir=tmp_path,
        stamp="test",
        scene_count=1,
    )
    prompt = script["lines"][0]["visual_prompt"]
    assert prompt.startswith(PARENTING.aesthetic_prefix)
    assert "small shoes waiting by a sunlit doorway" in prompt
    assert "Nostalgic practical-light palette" in prompt
    assert "golden hour sunset" not in prompt
    assert "vertical 9:16 full-bleed composition" in prompt.lower()


def test_gate2_payload_contains_complete_prompt_artifact() -> None:
    rows = _gate2_review_rows(
        [
            {
                "scene": 1,
                "text": "Silence kept the boundary I could not name.",
                "meaning": "quiet self-protection",
                "beat_mood": "melancholic restraint",
                "visual_prompt": "riso positive",
                "negative_prompt": "structural negative",
                "subject_type": "silhouette",
            }
        ]
    )
    assert rows[0]["beat_mood"] == "melancholic restraint"
    assert rows[0]["visual_prompt"] == "riso positive"
    assert rows[0]["negative_prompt"] == "structural negative"


def test_script_report_prints_parseable_candidate_json(capsys) -> None:
    script = {
        "writer_mode": "emotional",
        "theme": "distance",
        "niche": "relationship",
        "location_anchor": "A diner booth beside a rain-slicked window",
        "lines": [
            {
                "scene": 1,
                "text": "Distance began where honesty stopped feeling safe.",
                "visual_concept": "two umbrellas leaning on opposite walls",
                "final_positive_prompt": "cinematic prompt",
                "negative_prompt": "bad anatomy, clear front portrait",
            }
        ],
    }
    _print_script_report(script, index=1, qty=1)
    output = capsys.readouterr().out
    payload = output.split("SCRIPT_CANDIDATE_JSON\n", 1)[1].split(
        "\nEND_SCRIPT_CANDIDATE_JSON", 1
    )[0]
    parsed = json.loads(payload)
    assert parsed["writer_mode"] == "emotional"
    assert parsed["niche"] == "relationship"
    assert parsed["location_anchor"]
    assert parsed["beats"][0]["visual_concept"]
    assert parsed["beats"][0]["final_positive_prompt"] == "cinematic prompt"
    assert parsed["beats"][0]["negative_prompt"]


def test_script_only_hard_stops_before_visual_and_tts_apis(tmp_path: Path) -> None:
    (tmp_path / "clips").mkdir()
    (tmp_path / "assets").mkdir()
    lines = [
        {"scene": i + 1, "text": f"This is approved emotional beat number {i + 1}."}
        for i in range(9)
    ]
    script = {
        "theme": "distance",
        "writer_mode": "emotional",
        "arc_template": "thematic_arc",
        "hook_type": "bold_claim",
        "lines": lines,
    }
    validation = SimpleNamespace(ok=True, reasons=[], script=script)
    with (
        patch("core.economic_reel_lofi.pipeline.validate_script", return_value=validation),
        patch(
            "core.economic_reel_lofi.visual_concept.translate_episode_visuals",
            side_effect=AssertionError("visual API must not run"),
        ),
        patch(
            "core.economic_reel_lofi.voiceover.ensure_script_voiceover",
            side_effect=AssertionError("TTS API must not run"),
        ),
    ):
        result = _produce_one(
            page_id="wonder_feed",
            module="relationship",
            duration_s=27,
            clips_dir=tmp_path / "clips",
            assets_dir=tmp_path / "assets",
            index=1,
            script_only=True,
            locked_script=script,
        )
    assert result.ok is True
    assert result.pipeline_stage == "1_script"
    assert result.video_path is None


def test_tts_first_reuses_audio_without_mutating_approved_text_or_visuals(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fake_generate(text: str, output: Path, **_: object):
        calls.append(text)
        Path(output).write_bytes(b"fake mp3")
        return Path(output), [("word", 0.0, 0.2)]

    state: dict = {}
    script = {
        "theme": "distance",
        "lines": [{"scene": 1, "text": "Silence became the boundary I was afraid to name."}],
    }
    with (
        patch(
            "agents.media.audio_engine.generate_voiceover_with_timestamps",
            side_effect=fake_generate,
        ),
        patch(
            "core.economic_reel_lofi.assembler.measure_vo_speech_duration",
            return_value=2.6,
        ),
    ):
        ensure_script_voiceover(state, script, tmp_path)
        ensure_script_voiceover(state, script, tmp_path)
        script["lines"][0]["text"] = "Distance became the truth I could finally name."
        script["lines"][0]["visual_prompt"] = "stale"
        ensure_script_voiceover(state, script, tmp_path)

    assert len(calls) == 2
    assert calls == [
        "Silence became the boundary I was afraid to name.",
        "Distance became the truth I could finally name.",
    ]
    assert state["tts_stage_complete"] is True
    assert state["scene_durations"] == [2.8]
    assert state["audio_durations_s"] == [2.6]
    assert script["lines"][0]["visual_prompt"] == "stale"


def test_locked_atmospheric_hold_reuses_stills_after_legacy_prop_only_failure(
    tmp_path: Path,
) -> None:
    stills = [str(tmp_path / f"scene_{i:02d}.png") for i in range(1, 3)]
    source = tmp_path / "hold.json"
    source.write_text(
        json.dumps(
            {
                "script": {
                    "lines": [
                        {
                            "scene": i,
                            "text": f"Approved line {i}",
                            "visual_source": "writer_single_pass",
                        }
                        for i in range(1, 3)
                    ]
                },
                "scene_images": stills,
                "visual_qa_flags": [
                    "scene_1: SPOKEN-PROP: unspoken objects in pixels: window",
                    "scene_2: SPOKEN-PROP: unspoken objects in pixels: window",
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_locked_script(source)

    assert loaded["_locked_sidecar_assets"]["scene_images"] == stills
    assert loaded["_locked_sidecar_assets"]["manual_accept_scenes"] == [1, 2]


def test_stale_temp_voice_files_are_purged_before_assembly(tmp_path: Path) -> None:
    stale = tmp_path / "temp_voice_old.mp3"
    leftover = tmp_path / "cached_mix.mp3"
    extra_scene = tmp_path / "vo_scene_09.mp3"
    kept = tmp_path / "vo_scene_01.mp3"
    stale.write_bytes(b"stale")
    leftover.write_bytes(b"leftover")
    extra_scene.write_bytes(b"old-scene")
    kept.write_bytes(b"approved")

    removed = _purge_stale_temp_voice_files(
        tmp_path, keep_names={"vo_scene_01.mp3"}
    )

    assert stale in removed
    assert leftover in removed
    assert extra_scene in removed
    assert not stale.exists()
    assert not leftover.exists()
    assert not extra_scene.exists()
    assert kept.exists()


def test_current_run_scene_audio_paths_ignore_glob_leftovers(tmp_path: Path) -> None:
    (tmp_path / "temp_voice_old.mp3").write_bytes(b"stale")
    (tmp_path / "random.mp3").write_bytes(b"noise")
    (tmp_path / "vo_scene_01.mp3").write_bytes(b"one")
    (tmp_path / "vo_scene_02.mp3").write_bytes(b"two")
    (tmp_path / "vo_scene_09.mp3").write_bytes(b"extra")

    paths = _current_run_scene_audio_paths(
        tmp_path,
        2,
        [tmp_path / "random.mp3", Path("/other/run/vo_scene_02.mp3")],
    )

    assert [p.name if p else None for p in paths] == [
        "vo_scene_01.mp3",
        "vo_scene_02.mp3",
    ]
    assert all(p.parent == tmp_path for p in paths if p)


def test_bgm_pool_rejects_reference_reel_dumps(tmp_path: Path) -> None:
    assert is_verified_instrumental_bgm(tmp_path / "Fading_Embers_2026-08-16.mp3")
    assert not is_verified_instrumental_bgm(tmp_path / "0816.MP3")
    assert not is_verified_instrumental_bgm(tmp_path / "0816(1).MP3")
    assert not is_verified_instrumental_bgm(tmp_path / "lofi_bed_01.mp3")
    assert not is_verified_instrumental_bgm(
        tmp_path / "_quarantine_vocals" / "Fading_Embers_2026-08-16.mp3"
    )
    engine = tmp_path
    bgm_dir = engine / "channels_config" / "wonder_feed" / "audio" / "bgm"
    bgm_dir.mkdir(parents=True)
    (bgm_dir / "0816.MP3").write_bytes(b"x" * 2000)
    (bgm_dir / "Fading_Embers_ok.mp3").write_bytes(b"x" * 2000)
    tracks = list_library_bgm_tracks(engine)
    assert [p.name for p in tracks] == ["Fading_Embers_ok.mp3"]


def test_visual_only_strips_clip_audio() -> None:
    clip = SimpleNamespace(audio="leaked-voice")

    def without_audio():
        return SimpleNamespace(audio="still-there")

    clip.without_audio = without_audio
    silent = _visual_only(clip)
    assert silent.audio is None


def test_voiceover_writes_canonical_scene_files_only(tmp_path: Path) -> None:
    stale = tmp_path / "temp_voice_old.mp3"
    leftover = tmp_path / "echo_cache.mp3"
    stale.write_bytes(b"stale")
    leftover.write_bytes(b"echo")
    script = {
        "theme": "presence",
        "lines": [{"scene": 1, "text": "Stay with the moment you still have."}],
    }

    def fake_generate(text: str, output: Path, **kwargs: object):
        Path(output).write_bytes(b"fresh-vo")
        return Path(output), [("Stay", 0.0, 0.2)]

    with (
        patch(
            "agents.media.audio_engine.generate_voiceover_with_timestamps",
            side_effect=fake_generate,
        ),
        patch(
            "core.economic_reel_lofi.assembler.measure_vo_speech_duration",
            return_value=1.2,
        ),
    ):
        ensure_script_voiceover({}, script, tmp_path)

    assert not stale.exists()
    assert not leftover.exists()
    assert (tmp_path / "vo_scene_01.mp3").read_bytes() == b"fresh-vo"


def test_elastic_timeline_clamps_hook_and_adds_outro_tail(tmp_path: Path) -> None:
    generated: list[tuple[str, float]] = []
    lines = [
        {"scene": i + 1, "text": f"Exact approved line number {i + 1} stays fully unchanged."}
        for i in range(8)
    ]
    script = {"theme": "distance", "lines": lines}

    def fake_generate(text: str, output: Path, **kwargs: object):
        generated.append((text, float(kwargs["speed"])))
        Path(output).write_bytes(b"fake mp3")
        return Path(output), [("word", 0.0, 0.2)]

    measured = [2.9, 2.2, 2.3, 2.4, 2.5, 2.1, 2.6, 2.3]
    with (
        patch(
            "agents.media.audio_engine.generate_voiceover_with_timestamps",
            side_effect=fake_generate,
        ),
        patch(
            "core.economic_reel_lofi.assembler.measure_vo_speech_duration",
            side_effect=measured,
        ),
        patch(
            "agents.writer.spoken_budget.rewrite_single_line",
            side_effect=AssertionError("approved text must never be rewritten"),
        ),
    ):
        state: dict = {}
        ensure_script_voiceover(state, script, tmp_path)

    assert [text for text, _ in generated] == [row["text"] for row in lines]
    assert all(speed == 0.70 for _, speed in generated)
    assert all(row["tts_speed"] == 0.69 for row in script["lines"])
    assert all(row["tts_api_speed"] == 0.70 for row in script["lines"])
    assert state["audio_durations_s"] == measured
    assert state["scene_durations"] == [
        3.5,
        2.8,
        2.9,
        3.0,
        3.1,
        2.7,
        3.2,
        4.55,
    ]


def test_gemini_generator_normalizes_output_path(tmp_path: Path) -> None:
    class FakeAdapter:
        last_gemini_image_model_used = "models/gemini-2.5-flash-image"

        def __init__(self, **_: object) -> None:
            pass

        def generate(self, *_: object, output_directory: Path, **__: object) -> Path:
            path = Path(output_directory) / "generated.png"
            Image.new("RGB", (512, 1024), "navy").save(path)
            return path

    out = tmp_path / "scene_01.png"
    with patch(
        "agents.media.providers.image_provider.GeminiImageAdapter",
        FakeAdapter,
    ):
        path, meta = generate_scene_image_gemini("riso scene", out, width=720, height=1280)
    assert path == out
    assert Image.open(out).size == (720, 1280)
    assert meta["gen_model"] == "models/gemini-2.5-flash-image"


def test_momma_circle_uses_png_logo_not_text_handle() -> None:
    cfg = lofi_cfg.channel_assembly_cfg("momma_circle")
    assert cfg["use_text_watermark"] is False
    assert cfg["logo_size_multiplier"] == pytest.approx(0.337)
    assert cfg["logo_bottom_px"] == 252
    assert cfg["trim_logo_transparent_padding"] is True
    assert not lofi_cfg.channel_assembly_cfg("wonder_feed").get(
        "trim_logo_transparent_padding", False
    )
    logo = lofi_cfg.resolve_logo_path(
        "momma_circle", Path(__file__).resolve().parents[1]
    )
    assert logo is not None
    assert logo == (
        Path(__file__).resolve().parents[1]
        / "channels_config"
        / "momma_circle"
        / "logo"
        / "logo.png"
    )
    assert logo.suffix.lower() == ".png"
    assert logo.is_file()


def test_momma_circle_logo_trims_padding_and_applies_final_calibration() -> None:
    cfg = lofi_cfg.channel_assembly_cfg("momma_circle")
    logo = lofi_cfg.resolve_logo_path(
        "momma_circle", Path(__file__).resolve().parents[1]
    )
    assert logo is not None
    with Image.open(logo) as source:
        source_w, source_h = source.size
    layer = render_logo_layer(
        logo,
        opacity=float(cfg["logo_opacity"]),
        scale=float(cfg["logo_scale"]),
        bottom_px=int(cfg["logo_bottom_px"]),
        size_multiplier=float(cfg["logo_size_multiplier"]),
        trim_transparent_padding=bool(cfg["trim_logo_transparent_padding"]),
    )
    audit = audit_watermark_native_size(
        layer,
        logo_path=logo,
        use_text=False,
        expected_scale=float(cfg["logo_size_multiplier"]),
    )
    assert layer is not None
    assert audit["watermark_native_size"] == 1
    assert audit["painted_w"] <= round(source_w * 0.337) + 2
    assert audit["painted_h"] <= round(source_h * 0.337) + 2
    assert audit["painted_w"] >= 285
    ys, _ = (layer[..., 3] > 10).nonzero()
    visible_bottom_inset = layer.shape[0] - 1 - int(ys.max())
    assert visible_bottom_inset == pytest.approx(252, abs=2)


def test_resolve_page_dirs_separates_clips_and_metadata(tmp_path: Path) -> None:
    page, clips, assets, metadata = _resolve_page_dirs("wonder_feed", tmp_path)
    assert page == tmp_path
    assert clips == tmp_path / "clips"
    assert assets == tmp_path / "assets"
    assert metadata == tmp_path / "metadata"
    assert clips.is_dir() and metadata.is_dir()


def test_output_hygiene_moves_json_and_root_tests(tmp_path: Path) -> None:
    (tmp_path / "test_music_track.mp3").write_bytes(b"probe")
    (tmp_path / "_api_test_ping.json").write_text("{}", encoding="utf-8")
    clips = tmp_path / "wonder_feed" / "clips"
    clips.mkdir(parents=True)
    (clips / "keep.mp4").write_bytes(b"mp4")
    (clips / "lofi_pipeline_x.json").write_text("{}", encoding="utf-8")
    (clips / "reel_vo_concat.mp3").write_bytes(b"tmp")

    migrate_outputs_root(tmp_path)
    report = migrate_channel_clips(tmp_path / "wonder_feed")

    assert (tmp_path / "_tests" / "test_music_track.mp3").is_file()
    assert not (clips / "lofi_pipeline_x.json").exists()
    assert (tmp_path / "wonder_feed" / "metadata" / "lofi_pipeline_x.json").is_file()
    assert "reel_vo_concat.mp3" in report["deleted"]
    assert enforce_clips_mp4_only(clips) == []
