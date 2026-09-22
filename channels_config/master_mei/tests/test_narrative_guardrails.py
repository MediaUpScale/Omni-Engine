# -*- coding: utf-8 -*-
"""Verify Master Mei focused narrative engine, humble voice, RAG directive."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from agents.media.mei_narrative import (
        MASTER_SCRIPTWRITER_DIRECTIVE,
        apply_mei_word_budget_from_duration,
        build_three_act_script_instructions,
        enforce_humble_mei_voice,
        episode_theme_meta,
        master_scriptwriter_directive,
        mei_voice_prompt_block,
        narration_word_budget,
        resolve_mei_duration_profile,
        sanitize_mei_narration_body,
    )
    from agents.media.visual_roles import PENULTIMATE_LIBERATION_PROMPT
    from channels_config.master_mei import page_config as mei_cfg

    fails: list[str] = []

    rag_path = ROOT / "channels_config" / "master_mei" / "rag_directive.txt"
    if not rag_path.is_file():
        fails.append("rag_directive.txt missing")
    directive = master_scriptwriter_directive()
    for needle in (
        "MASTER MEI NARRATIVE ENGINE",
        "ANTI-INSTRUCTIONAL",
        "PHILOSOPHICAL HOOK",
        "FOCUSED UNCONSCIOUS TRAP",
        "SPIRITUAL & FINANCIAL LIBERATION",
        "HUMBLE PRACTICAL DISCIPLINE",
        "fellow human seekers",
        "ONE specific focal idea",
    ):
        if needle.lower() not in directive.lower():
            fails.append(f"RAG directive missing {needle!r}")

    p105 = resolve_mei_duration_profile(105)
    if p105["frames"] not in (8, 9, 10) or p105["words_max"] < 130:
        fails.append(f"105s profile unexpected: {p105}")
    apply_mei_word_budget_from_duration(105)
    wmin, wtarget, wmax = narration_word_budget()
    if not (115 <= wmin <= wtarget <= wmax <= 180):
        fails.append(f"word budget out of expected 90–120s band: {narration_word_budget()}")
    if abs(float(mei_cfg.BGM_START_TIME) - 0.5) > 1e-6:
        fails.append(f"BGM_START_TIME not 0.5: {mei_cfg.BGM_START_TIME}")

    voice = mei_voice_prompt_block(duration_s=105)
    for needle in ("ANTI-INSTRUCTIONAL", "PHILOSOPHICAL HOOK", "fellow seekers", "ONE"):
        if needle.lower() not in voice.lower():
            fails.append(f"voice block missing {needle!r}")
    if "my disciples" in voice.lower() and "never" not in voice.lower():
        fails.append("voice block still promotes 'my disciples'")

    ep = episode_theme_meta("Digital stimulation drains mental capital")
    instr = build_three_act_script_instructions(9, ep)
    for needle in (
        "PHILOSOPHICAL HOOK",
        "FOCUSED UNCONSCIOUS TRAP",
        "SPIRITUAL & FINANCIAL LIBERATION",
        "HUMBLE PRACTICAL DISCIPLINE",
        "SINGLE FOCAL IDEA",
    ):
        if needle not in instr:
            fails.append(f"script instructions missing {needle}")
    if "I demand" in instr and "NEVER" not in instr and "PROHIBITED" not in instr:
        fails.append("script instructions still use aggressive I demand examples")

    scrubbed = sanitize_mei_narration_body(
        "My disciples, I demand you wake up. I studied Plato. Don't do that."
    )
    lo = scrubbed.lower()
    if "my disciples" in lo:
        fails.append("sanitize left 'my disciples'")
    if "i demand" in lo:
        fails.append("sanitize left 'I demand'")
    if "i studied" in lo:
        fails.append("sanitize left 'I studied'")
    humble = enforce_humble_mei_voice("Followers and students, do this now.")
    if "followers" in humble.lower() or "students" in humble.lower():
        fails.append("humble voice left followers/students")

    if "attention monopoly" not in PENULTIMATE_LIBERATION_PROMPT.lower():
        fails.append("penultimate missing attention-monopoly prefix")
    if "ninja" not in PENULTIMATE_LIBERATION_PROMPT.lower() and "warrior" not in PENULTIMATE_LIBERATION_PROMPT.lower():
        fails.append("penultimate missing ninja-warrior")
    if "gear" not in PENULTIMATE_LIBERATION_PROMPT.lower():
        fails.append("penultimate missing devastated gears")
    if "MASTER SCRIPTWRITER" not in MASTER_SCRIPTWRITER_DIRECTIVE and "NARRATIVE ENGINE" not in MASTER_SCRIPTWRITER_DIRECTIVE:
        fails.append("embedded directive header missing")

    pen_file = ROOT / "channels_config" / "master_mei" / "prompts" / "penultimate_frame.txt"
    if not pen_file.is_file():
        fails.append("penultimate_frame.txt missing")

    # CTA color helper + no black-bar dependency
    from core.reel_sequence_engine import _rgb_tuple_to_moviepy_color

    if _rgb_tuple_to_moviepy_color((255, 204, 0)) != "#ffcc00":
        fails.append("CTA color helper mismatch for subtitle fill")

    if fails:
        print("[FAIL]")
        for f in fails:
            print(" -", f)
        return 1

    print("[PASS] RAG directive (focused 5-beat + humble persona)")
    print(f"[PASS] Duration profile 105s -> frames={p105['frames']} words={wmin}-{wmax}")
    print("[PASS] Humble voice sanitize (no disciples / I demand / I studied)")
    print("[PASS] CTA subtitle-matching color helper")
    print("[PASS] Penultimate harsh wasteland gear-training prompt")
    print("\nALL LOCAL GUARDRAIL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
