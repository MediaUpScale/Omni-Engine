from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from inspect import signature
from types import SimpleNamespace
from unittest.mock import patch

from agents.writer.claude_compose import (
    compose_instruction,
    compose_system,
    format_compose_user,
)
from agents.writer.script_agent import assess_story_quality
from agents.writer.script_brain import _anaphora_profile, _function_shape
from agents.writer.spoken_budget import trim_at_clause_boundary
from agents.writer.writer_brief import WriterBrief
from core.economic_reel_lofi.aphorism_bank import bank_path, entries, get_entry
from core.economic_reel_lofi import config as lofi_cfg
from core.economic_reel_lofi.pipeline import (
    _repair_targeted_line,
    _targeted_line_failure,
    run_economic_reel_lofi,
)
from core.economic_reel_lofi.validator_agent import validate_script
from core.economic_reel_lofi.visual_concept import stamp_stage1_timing
from core.economic_reel_lofi.visual_identity import (
    apply_anchor_callback_beats,
    apply_windowless_interior_frame,
    assemble_v2_prompt,
    hook_silhouette_dissolve_scene,
)


def test_theme_compose_prompt_is_duration_aware() -> None:
    instruction = compose_instruction(15, duration_s=45, beat_s=3)
    user = format_compose_user(
        theme="security",
        thesis="Security is practiced.",
        anchor={"name": "blanket"},
        rhetoric={},
        pool_block="blanket",
        scene_count=15,
        duration_s=45,
        beat_s=3,
    )
    assert "exactly 15 spoken lines" in instruction
    assert "Each line has roughly 3.0s" in instruction
    assert "exactly 15 spoken lines for 45s" in user
    assert "TARGETS 6 words" in user
    assert "HOOK LINE (beat 1)" in instruction
    assert "HOOK LINE (beat 1)" in user
    assert "target 6" in instruction and "hard maximum 7 words" in instruction
    assert "Only three writing priorities" in instruction
    assert "ANCHOR OBJECT" not in user
    assert "ASSIGNED PATTERN" not in user
    assert "load-bearing connectives" not in compose_system(15, duration_s=45)


def test_quote_brief_requires_four_part_parable_arc() -> None:
    brief = WriterBrief.from_quote(
        quote="Hedgehogs huddle, wound one another, retreat, then find a distance.",
        theme="connection",
        meta={"duration_s": 27},
    )
    block = brief.assignment_block()
    assert brief.mode == "quote"
    assert "PARABLE ARC" in block
    assert "workable equilibrium" in block
    assert "HOOK LINE (beat 1)" in block
    assert "target 4" in block and "6 words" in block


def test_paraphrase_bank_and_brief() -> None:
    row = get_entry("more_tears_than_smiles")
    brief = WriterBrief.from_paraphrase(
        aphorism=row["text"],
        source_id=row["id"],
        theme="leaving",
    )
    assert brief.mode == "paraphrase"
    assert brief.meta["aphorism_id"] == "more_tears_than_smiles"
    assert "20–30%" in brief.assignment_block()
    assert "HOOK LINE (beat 1)" in brief.assignment_block()
    assert "protected anaphora" in brief.assignment_block()
    assert "ANAPHORA" in brief.assignment_block()
    assert "do not merge or split those sentences" in brief.assignment_block()
    assert bank_path() == (
        lofi_cfg.STORE_DIR / "aphorism_bank.json"
    )


def test_paraphrase_picks_random_bank_entry_without_id() -> None:
    ids = [
        get_entry(module="relationship", exclude_ids=[])["id"]
        for _ in range(12)
    ]
    assert len(set(ids)) > 1
    assert set(ids) <= {row["id"] for row in entries()}


def test_emotional_is_pipeline_default() -> None:
    default = signature(run_economic_reel_lofi).parameters["writer_mode"].default
    assert default == "emotional"


def test_anaphora_profile_survives_internal_clause_reorder() -> None:
    source = [
        "When there are more tears than smiles, leave.",
        "When there are more fights than jokes, leave.",
        "When it hurts more than it feels good, leave.",
        "When you have to leave what you love just to live, leave.",
    ]
    varied = [
        "When smiles become fewer than tears, leave.",
        "When there are more battles than jokes, leave.",
        "When it wounds more than it feels right, leave.",
        "When you have to leave what you love just to survive, leave.",
    ]
    before = _anaphora_profile(source)
    after = _anaphora_profile(varied)
    assert before is not None and after is not None
    assert before["opening_count"] == after["opening_count"] == 4
    assert before["ending_count"] == after["ending_count"] == 4
    changed = [
        i
        for i, (left, right) in enumerate(zip(source, varied), start=1)
        if _function_shape(left) != _function_shape(right)
    ]
    assert changed == [1]


def test_short_forms_can_clear_scaled_story_spine() -> None:
    report = assess_story_quality(
        [
            "When tears outnumber smiles, leave.",
            "When fights replace laughter, leave.",
            "When love hurts more than heals, leave.",
            "When staying costs your life, leave.",
        ],
        theme="leaving",
    )
    assert report["need"] == 3
    assert not report["fails"]


def test_deterministic_trim_uses_clause_boundary_and_punctuation() -> None:
    line = "Stay when it is safe, but leave before you disappear completely!"
    trimmed = trim_at_clause_boundary(line, max_words=6, max_chars=32)
    assert trimmed == "Stay when it is safe!"
    assert len(trimmed.split()) <= 6
    assert len(trimmed) <= 32


def test_thematic_scene_cap_expands_with_duration() -> None:
    assert lofi_cfg.thematic_max_scenes() == 9
    assert lofi_cfg.thematic_max_scenes(45) == 15


def test_validator_groups_one_beats_cap_failures_for_targeted_repair() -> None:
    script = {
        "arc_template": "thematic_arc",
        "lines": [{"text": "This single line is much too long for one beat."}],
    }
    target = _targeted_line_failure(
        script,
        [
            "scene 1 spoken line has 10 words (max 9)",
            "scene 1 caption exceeds 56 chars (61)",
        ],
    )
    assert target is not None
    assert target[0] == 0


def test_story_close_and_spine_failures_are_scoped_to_one_beat() -> None:
    script = {
        "lines": [
            {"text": "He stayed when leaving was easier."},
            {"text": "I finally told him why I flinch."},
            {"text": "The room went quiet."},
        ],
        "story_quality": {"links": ["0->1:shared=['stayed']"]},
    }
    close = _targeted_line_failure(
        script,
        ["story_quality: story-close: last line has no usable takeaway"],
    )
    assert close is not None and close[0] == 2
    spine = _targeted_line_failure(
        script,
        ["story_quality: story-spine: 1/2 consecutive lines linked"],
    )
    assert spine is not None and spine[0] == 2
    assert "prior beat" in spine[1]


def test_visual_anchor_callbacks_do_not_rewrite_spoken_lines() -> None:
    original = [
        "He asked why I flinched at his hug.",
        "I did not have an answer ready.",
        "So I told him about my father.",
        "He listened without trying to fix me.",
        "That was the first time I felt safe.",
        "Nothing dramatic happened after that.",
        "He simply stayed for the hard part.",
        "I stopped waiting for him to leave.",
        "Being known no longer felt dangerous.",
    ]
    lines = [
        {"scene": i + 1, "text": text, "beat_text": text}
        for i, text in enumerate(original)
    ]
    beats = apply_anchor_callback_beats(
        lines,
        {
            "name": "shared knitted blanket",
            "setting": "sofa in a quiet living room",
            "initial_state": "folded",
            "final_state": "still warm",
        },
    )
    assert beats == [1, 7]
    assert [row["text"] for row in lines] == original
    assert lines[1]["key_object"] == "shared knitted blanket"
    assert lines[7]["anchor_beat"] == "callback"


def test_two_failed_scoped_repairs_use_clause_trim_without_third_call() -> None:
    original = (
        "Stay while the room is safe, but leave before you disappear "
        "completely tonight!"
    )
    script = {
        "arc_template": "thematic_arc",
        "lines": [{"text": original, "beat_text": original}],
    }
    output = StringIO()
    with (
        patch(
            "agents.writer.spoken_budget.repair_one_line",
            side_effect=ValueError("forced impossible constraint"),
        ) as repair,
        patch(
            "core.economic_reel_lofi.pipeline.validate_script",
            return_value=SimpleNamespace(ok=True, reasons=[], script=script),
        ),
        redirect_stdout(output),
    ):
        result, attempts = _repair_targeted_line(
            script,
            module="relationship",
            scene_count=9,
            reasons=["scene 1 spoken line has 13 words (max 9)"],
            persist_on_pass=False,
        )
    assert result.ok
    assert attempts == 2
    assert repair.call_count == 2
    assert script["lines"][0]["text"] == "Stay while the room is safe!"
    assert "deterministic fallback after 2 LLM attempts" in output.getvalue()


def test_story_quality_is_blocking_in_validator() -> None:
    lines = [
        {
            "scene": i + 1,
            "text": text,
            "beat_text": text,
            "caption_beats": [text],
            "duration_s": 3.0,
        }
        for i, text in enumerate(
            [
                "The room is quiet.",
                "A chair waits nearby.",
                "The window stays open.",
                "A cup rests there.",
            ]
        )
    ]
    result = validate_script(
        {
            "writer": "freeform_v1",
            "writer_mode": "quote",
            "hook_type": "bold_claim",
            "arc_template": "thematic_arc",
            "theme": "quiet",
            "duration_requested_s": 27,
            "scene_duration_s": 3,
            "lines": lines,
            "monologue": " ".join(row["text"] for row in lines),
        },
        module="relationship",
        scene_count=9,
        persist_on_pass=False,
    )
    assert not result.ok
    assert any(reason.startswith("story_quality:") for reason in result.reasons)


def test_stamp_stage1_timing_preserves_approved_beat_durations() -> None:
    script = {
        "duration_requested_s": 18,
        "lines": [
            {"scene": 1, "duration_s": 6.0, "text": "a"},
            {"scene": 2, "duration_s": 6.0, "text": "b"},
            {"scene": 3, "duration_s": 3.0, "text": "c"},
            {"scene": 4, "duration_s": 3.0, "text": "d"},
        ],
    }
    stamp_stage1_timing(script, beat_s=3.0, duration_s=27.0)
    assert [row["duration_s"] for row in script["lines"]] == [6.0, 6.0, 3.0, 3.0]
    assert script["duration_s"] == 18.0
    assert script["duration_requested_s"] == 18.0


def test_windowless_keeps_place_meaning_environment() -> None:
    row = {
        "scene": 2,
        "text": "They barge in uninvited, then sit where love died.",
        "meaning": "The desecration of a place where profound emotional loss occurred.",
        "episode_place": "A forgotten sunroom",
        "setting": "tight crop against a plain painted wall, no room beyond the wall plane",
        "subject_type": "couple",
    }
    apply_windowless_interior_frame(row)
    assert "sunroom" in str(row["setting"]).lower()
    assert "painted wall" not in str(row["setting"]).lower()
    assert "paper-grain" in str(row["setting"])


def test_hook_assemble_uses_dissolve_only() -> None:
    beat = {
        "hook_template": True,
        "shot_type": "hook",
        "subject_type": "silhouette",
        "dissolve_element": "a flock of small birds breaking from the silhouette at the head",
        "hook_memory_overlay": "a hazy indistinct room, soft and out of focus, like a fading memory",
        "setting": "sunlit kitchen",
        "key_object": "object",
        "arc_position": "act1",
    }
    prompt = assemble_v2_prompt(beat)
    expected = hook_silhouette_dissolve_scene(
        dissolve_element=beat["dissolve_element"],
        memory_overlay=beat["hook_memory_overlay"],
    )
    assert expected in prompt
    assert "Close portrait" not in prompt
    assert "object is the only object" not in prompt
    assert "sunlit kitchen" not in prompt
    assert "high-neck sweater" not in prompt.lower()


def test_closed_journal_prompt_is_physically_shut() -> None:
    beat = {
        "scene": 3,
        "subject_type": "object_focus",
        "key_object": "closed journal",
        "setting": "Cozy living room, gouache furniture",
        "arc_position": "act2",
    }
    prompt = assemble_v2_prompt(beat).lower()
    assert "hardcover book shut" in prompt
    assert "spine and cover only visible" in prompt
    assert "no pages" in prompt
    assert "no bookmark ribbon" in prompt


def test_couple_place_uses_small_figures_not_close_portrait() -> None:
    beat = {
        "scene": 2,
        "subject_type": "couple",
        "text": "They barge in uninvited, then sit where love died.",
        "meaning": "a place where love died",
        "episode_place": "A forgotten sunroom",
        "setting": "A forgotten sunroom, gouache furniture and paper-grain walls",
        "key_object": "figure",
        "arc_position": "act1",
        "time_of_day": "evening",
    }
    prompt = assemble_v2_prompt(beat)
    low = prompt.lower()
    assert "two figures small in frame" in low
    assert "environment dominates" in low
    assert "not a close couple portrait" in low
    assert "close up" not in low
    assert "standing close" not in low
    assert "close portrait, face and torso" not in low


def test_portrait_has_neck_and_sweater() -> None:
    beat = {
        "scene": 4,
        "subject_type": "woman",
        "close_variant": "portrait_close",
        "close_character": "woman",
        "setting": "tight crop against a plain painted wall",
        "key_object": "figure",
        "arc_position": "act3",
    }
    prompt = assemble_v2_prompt(beat).lower()
    assert "high-neck sweater" in prompt
    assert "natural head turn" in prompt
    assert "no extreme neck rotation" in prompt


def test_critic_model_remaps_retired_flash_lite() -> None:
    from quality.VisualQA_Agent.config import (
        CURRENT_CRITIC_MODEL,
        GEMINI_CRITIC_MODEL,
        resolve_critic_model,
    )

    assert CURRENT_CRITIC_MODEL == "models/gemini-3.5-flash-lite"
    assert "2.5-flash-lite" not in GEMINI_CRITIC_MODEL
    assert resolve_critic_model("models/gemini-2.5-flash-lite") == CURRENT_CRITIC_MODEL
    assert resolve_critic_model("gemini-2.5-flash-lite") == CURRENT_CRITIC_MODEL


def test_together_flux2_backend_and_cost() -> None:
    import os

    from agents.media.providers.together_image import (
        FLUX_2_DEV_MODEL,
        estimate_together_image_cost,
    )

    usd = estimate_together_image_cost(FLUX_2_DEV_MODEL)
    assert abs(usd - 0.0154) < 0.00001
    from agents.media.providers.together_image import _is_flux2_dev_model

    assert _is_flux2_dev_model("black-forest-labs/FLUX-2-dev")
    assert _is_flux2_dev_model("black-forest-labs/FLUX.2-dev")
    prev = os.environ.get("LOFI_FLUX_BACKEND")
    os.environ["LOFI_FLUX_BACKEND"] = "flux2-dev"
    try:
        assert lofi_cfg.uses_flux2_dev() is True
        assert lofi_cfg.uses_flux_dev() is True
        cost, meta = lofi_cfg.lofi_image_cost_per_call_usd()
        assert meta["backend"] == "flux2-dev"
        assert meta["provider"] == "together"
        assert meta["model"] == FLUX_2_DEV_MODEL
        assert meta["steps"] == 24
        assert meta["guidance_scale"] == 5.5
        assert abs(cost - 0.0154) < 0.00001
    finally:
        if prev is None:
            os.environ.pop("LOFI_FLUX_BACKEND", None)
        else:
            os.environ["LOFI_FLUX_BACKEND"] = prev


def test_hook_line_brevity_is_writer_target_not_still_hold() -> None:
    from agents.writer.freeform_writer import _output_contract
    from agents.writer.writer_brief import WriterBrief

    clause = lofi_cfg.hook_line_brevity_clause()
    assert "target 4" in clause and "6 words" in clause
    assert "roughly 3s" in clause
    assert "still duration follows the rendered VO" in clause
    theme = WriterBrief.from_theme(theme="healing").assignment_block()
    assert "HOOK LINE (beat 1)" in theme
    contract = _output_contract(
        WriterBrief.from_theme(theme="healing", meta={"duration_s": 27})
    )
    assert "Every scene contains exactly 4–6" in contract
    assert "at most 48 words" in contract


def test_slot_duration_follows_measured_vo_not_estimate() -> None:
    from core.economic_reel_lofi.config import slot_duration_for_vo

    slot, driven = slot_duration_for_vo(4.13, base_s=6.0, trailing_silence_s=0.30)
    assert driven is True
    assert abs(slot - 4.43) < 0.001
    slot_last, driven_last = slot_duration_for_vo(
        3.80, base_s=4.0, trailing_silence_s=0.12
    )
    assert driven_last is True
    assert abs(slot_last - 3.92) < 0.001
    fallback, driven_fb = slot_duration_for_vo(0.0, base_s=6.0, trailing_silence_s=0.30)
    assert driven_fb is False
    assert fallback == 6.0


def test_coverage_infra_error_does_not_fail_closed() -> None:
    from core.economic_reel_lofi.pipeline import _is_critic_infra_error

    err = RuntimeError(
        "404 NOT_FOUND. This model models/gemini-2.5-flash-lite "
        "is no longer available to new users."
    )
    assert _is_critic_infra_error(err)
    assert not _is_critic_infra_error(RuntimeError("incomplete garment"))

