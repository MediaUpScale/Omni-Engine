# -*- coding: utf-8 -*-
"""Smoke-test Master Mei Scene1/3/7/penultimate RAG overrides + audio knobs."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from agents.media.audio_engine import (
        MUSIC_PROMPT_SYSTEM_DIRECTIVE,
        _mei_section_styles,
        generate_dynamic_music_prompt,
    )
    from agents.media.mei_narrative import resolve_mei_duration_profile
    from agents.media.visual_roles import (
        GLOBAL_FIREARM_BAN,
        MEI_BASE_ANCHOR,
        MEI_PRIMARY_ENVIRONMENTS,
        MEI_SECONDARY_ENVIRONMENTS,
        SCENE_03_BIOMECHANICAL_PROMPT,
        SCENE_07_CYBORG_CLOSEUP_PROMPT,
        SCENE_FINAL_HOOK_PROMPT,
        SCENE_01_HOOK_PROMPT,
        PENULTIMATE_LIBERATION_PROMPT,
        _load_penultimate_prompt,
        build_role_prompt,
        compute_mei_act_durations,
        frame_lore_for_acts,
        pick_mei_meditation_environment,
        pick_mei_frame1_setup,
        pick_mei_frame_final_setup,
        compose_scene_01_prompt,
        compose_scene_final_prompt,
        select_style_references_for_mei_frame,
        MEI_FRAME1_ACTIONS,
        MEI_FRAME1_CAMERAS,
        MEI_FINAL_ACTIONS,
        MEI_FINAL_CAMERAS,
        MEI_OUTDOOR_TOKENS,
        validate_mei_visual_prompt,
    )
    from channels_config.master_mei import page_config as mei_cfg

    fails: list[str] = []

    p = resolve_mei_duration_profile(105)
    if p["frames"] != 9:
        fails.append(f"105s profile frames expected 9 got {p['frames']}")
    p90 = resolve_mei_duration_profile(95)
    if p90["frames"] != 8:
        fails.append(f"95s profile frames expected 8 got {p90['frames']}")
    p120 = resolve_mei_duration_profile(120)
    if p120["frames"] != 10:
        fails.append(f"120s frames expected 10 got {p120['frames']}")

    durs = compute_mei_act_durations(9, 105.0, hook_max_s=8.0)
    if durs[0] > 8.01:
        fails.append(f"hook duration >8s: {durs[0]}")
    if abs(sum(durs) - 105.0) > 0.05:
        fails.append(f"act durs sum drift: {sum(durs)}")
    body_avg = sum(durs[1:]) / (len(durs) - 1)
    if not (9.5 <= body_avg <= 13.5):
        fails.append(f"body avg out of 10-12 band: {body_avg}")

    lore = frame_lore_for_acts(9)
    if lore[0][1] != "intro" or lore[2][1] != "gears" or lore[-2][1] != "break_free":
        fails.append(f"lore locks wrong: {lore}")
    if lore[6][1] != "dopamine_trap":
        fails.append(f"Scene 7 lore expected dopamine_trap got {lore[6]}")

    if "topknot" not in MEI_BASE_ANCHOR.lower() or "snow-white" not in MEI_BASE_ANCHOR.lower():
        fails.append("MEI_BASE_ANCHOR missing snow-white topknot DNA")
    if "mala" not in MEI_BASE_ANCHOR.lower():
        fails.append("MEI_BASE_ANCHOR missing mala prayer beads")
    if "strand" not in MEI_BASE_ANCHOR.lower() and "locks" not in MEI_BASE_ANCHOR.lower():
        fails.append("MEI_BASE_ANCHOR missing two long white hair strands/locks")
    if "eyebrow" not in MEI_BASE_ANCHOR.lower() or "temple" not in MEI_BASE_ANCHOR.lower():
        fails.append("MEI_BASE_ANCHOR missing extra-long eyebrows past temples")
    if "topknot" not in SCENE_01_HOOK_PROMPT.lower() or "snow-white" not in SCENE_01_HOOK_PROMPT.lower():
        fails.append("Scene 1 missing Master Mei snow-white DNA")
    if not any(k in SCENE_01_HOOK_PROMPT.lower() for k in MEI_OUTDOOR_TOKENS):
        fails.append("Scene 1 missing outdoor environment")
    if re.search(r"(?i)\binside\b.*\btemple\b|\btemple\s+hall\b", SCENE_01_HOOK_PROMPT):
        fails.append("Scene 1 still defaults to indoor temple")
    if "topknot" not in SCENE_FINAL_HOOK_PROMPT.lower():
        fails.append("Final Hook missing Master Mei base anchor")
    if "attention monopoly" not in SCENE_03_BIOMECHANICAL_PROMPT.lower():
        fails.append("Scene 3 prompt missing attention-monopoly prefix/content")
    if "vr headset" not in SCENE_03_BIOMECHANICAL_PROMPT.lower() and "propaganda" not in SCENE_03_BIOMECHANICAL_PROMPT.lower():
        fails.append("Scene 3 missing post-apocalyptic slave / VR headset details")
    if "computing apparatus" not in SCENE_07_CYBORG_CLOSEUP_PROMPT.lower() and "hydraulic" not in SCENE_07_CYBORG_CLOSEUP_PROMPT.lower():
        fails.append("Scene 7 prompt missing computing-apparatus captive details")
    from agents.media.visual_roles import SCENE_08_UNPLUG_ESCAPE_PROMPT
    if "samurai" not in SCENE_08_UNPLUG_ESCAPE_PROMPT.lower() or "katana" not in SCENE_08_UNPLUG_ESCAPE_PROMPT.lower():
        fails.append("Scene 8 missing samurai matrix-slice details")
    if "ninja" not in PENULTIMATE_LIBERATION_PROMPT.lower() and "warrior" not in PENULTIMATE_LIBERATION_PROMPT.lower():
        fails.append("Penultimate prompt missing ninja-warrior")
    if "gear" not in PENULTIMATE_LIBERATION_PROMPT.lower():
        fails.append("Penultimate prompt missing devastated gears")
    if not re.search(r"(?i)push-?up|balance|horse stance|train", PENULTIMATE_LIBERATION_PROMPT):
        fails.append("Penultimate prompt missing a rigorous training drill")
    pen_drills = {_load_penultimate_prompt(seed=f"drill-{i}") for i in range(18)}
    if len(pen_drills) < 3:
        fails.append(f"Penultimate drills not rotating: {len(pen_drills)} unique prompts")
    if re.search(r"(?i)shirtless|bare[- ]chest|muscular warrior", PENULTIMATE_LIBERATION_PROMPT):
        fails.append("Penultimate still describes a shirtless strongman")
    if "firearm" not in GLOBAL_FIREARM_BAN.lower():
        fails.append("firearm ban missing")

    from agents.media.prompt_builder import (
        FLUX_MAX_CHARS,
        FLUX_MAX_WORDS,
        finalize_flux_prompt,
        flux_prompt_stats,
    )

    for label, prompt, need_prefix in (
        ("scene1", SCENE_01_HOOK_PROMPT, False),
        ("scene3", SCENE_03_BIOMECHANICAL_PROMPT, True),
        ("scene7", SCENE_07_CYBORG_CLOSEUP_PROMPT, True),
        ("scene8", SCENE_08_UNPLUG_ESCAPE_PROMPT, True),
        ("penultimate", PENULTIMATE_LIBERATION_PROMPT, True),
    ):
        stats = flux_prompt_stats(prompt)
        if stats["chars"] > FLUX_MAX_CHARS:
            fails.append(f"{label} exceeds {FLUX_MAX_CHARS} chars: {stats['chars']}")
        if stats["words"] > FLUX_MAX_WORDS:
            fails.append(f"{label} exceeds {FLUX_MAX_WORDS} words: {stats['words']}")
        if stats["has_tag_spam"]:
            fails.append(f"{label} still contains FLUX tag spam")
        if need_prefix and not stats["has_mandatory_prefix"]:
            fails.append(f"{label} missing mandatory cyberpunk prefix")
        if not need_prefix and stats["has_mandatory_prefix"]:
            fails.append(f"{label} unexpectedly has cyberpunk prefix")

    bloated = finalize_flux_prompt(("ultra detailed masterpiece 8k photorealistic " + ("word " * 200)))
    if len(bloated.split()) > FLUX_MAX_WORDS:
        fails.append("finalize_flux_prompt failed 180-word hard trim")
    if "8k" in bloated.lower() or "masterpiece" in bloated.lower() or "photorealistic" in bloated.lower():
        fails.append("finalize_flux_prompt failed to strip tag spam")
    if "attention monopoly era" not in bloated.lower():
        fails.append("finalize_flux_prompt failed to inject mandatory prefix")

    pos1, _ = build_role_prompt(
        role="master", beat="intro", act_index=0, subject="awakening dawn clarity",
        spoken_beat="dawn awakens sovereign clarity against the matrix drain",
    )
    if "topknot" not in pos1.lower() or "snow-white" not in pos1.lower():
        fails.append("Scene 1 build_role_prompt missing snow-white DNA")
    if "strand" not in pos1.lower() and "locks" not in pos1.lower():
        fails.append("Scene 1 missing two long white hair strands")
    if "eyebrow" not in pos1.lower():
        fails.append("Scene 1 missing iconic eyebrows DNA")
    if not any(k in pos1.lower() for k in MEI_OUTDOOR_TOKENS):
        fails.append("Scene 1 build_role_prompt missing dynamic outdoor environment")
    if re.search(r"(?i)\binside\b.*\btemple\b|\btemple\s+hall\b", pos1):
        fails.append("Scene 1 build_role_prompt still uses indoor temple default")
    if "golden" not in pos1.lower() and "sunrise" not in pos1.lower() and "alpine" not in pos1.lower():
        fails.append("Scene 1 atmosphere not mirroring philosophical dawn/clarity cues")

    # Weighted picker: forced hook_env wins; otherwise pool membership
    forced = pick_mei_meditation_environment(hook_env="custom storm cliff above clouds")
    if forced != "custom storm cliff above clouds":
        fails.append("pick_mei_meditation_environment ignored forced hook_env")
    picked = {
        pick_mei_meditation_environment(episode_seed=f"seed-{i}", spoken_beat="discipline")
        for i in range(40)
    }
    if not picked.intersection(set(MEI_PRIMARY_ENVIRONMENTS) | set(MEI_SECONDARY_ENVIRONMENTS)):
        fails.append("pick_mei_meditation_environment not drawing from primary/secondary pools")

    setups = {
        pick_mei_frame1_setup(episode_seed=f"ep-{i}", spoken_beat="discipline")
        for i in range(24)
    }
    if len(setups) < 8:
        fails.append(f"Frame 1 setup not diversifying across seeds: {len(setups)} unique triples")
    actions = {a for a, _c, _e in setups}
    cameras = {c for _a, c, _e in setups}
    if len(actions) < 3:
        fails.append(f"Frame 1 actions not rotating: {actions}")
    if len(cameras) < 3:
        fails.append(f"Frame 1 cameras not rotating: {cameras}")
    if not actions.intersection(set(MEI_FRAME1_ACTIONS)):
        fails.append("Frame 1 actions not drawn from MEI_FRAME1_ACTIONS")
    if not cameras.intersection(set(MEI_FRAME1_CAMERAS)):
        fails.append("Frame 1 cameras not drawn from MEI_FRAME1_CAMERAS")
    p_a = compose_scene_01_prompt(episode_seed="alpha-cliff-scroll")
    p_b = compose_scene_01_prompt(episode_seed="beta-bamboo-tea")
    if p_a == p_b:
        fails.append("compose_scene_01_prompt identical across distinct episode seeds")
    if "lotus posture atop" in p_a.lower() and "lotus posture atop" in p_b.lower():
        fails.append("Frame 1 still hard-locks lotus posture across seeds")
    if select_style_references_for_mei_frame(
        [Path(__file__)], act_index=0, episode_seed="x"
    ):
        fails.append("Frame 1 style-reference selector must return empty (no pose bake)")

    pos3, neg3 = build_role_prompt(role="slave", beat="gears", act_index=2, subject="test")
    if "attention monopoly" not in pos3.lower() and "wasteland" not in pos3.lower():
        fails.append("Scene 3 build_role_prompt did not inject post-apocalyptic slaves override")
    if "vr headset" not in pos3.lower() and "propaganda" not in pos3.lower():
        fails.append("Scene 3 refined FLUX prompt missing VR headset / propaganda cues")
    if "graphite" in pos3.lower():
        fails.append("Scene 3 still contains graphite")
    if "firearm" not in neg3.lower() and "handgun" not in neg3.lower():
        fails.append("Scene 3 negative missing firearm ban")

    pos7, _ = build_role_prompt(role="slave", beat="dopamine_trap", act_index=6, subject="test")
    if "computing apparatus" not in pos7.lower() and "hydraulic" not in pos7.lower():
        fails.append("Scene 7 build_role_prompt missing computing-apparatus captive")

    pos8, _ = build_role_prompt(role="slave", beat="gears", act_index=7, subject="test")
    if "samurai" not in pos8.lower() or "katana" not in pos8.lower():
        fails.append("Scene 8 build_role_prompt missing samurai matrix-slice")
    if "graphite" in pos8.lower():
        fails.append("Scene 8 still contains graphite")

    pos_pen, _ = build_role_prompt(role="disciple", beat="break_free", act_index=8, subject="test")
    if "ninja" not in pos_pen.lower() and "warrior" not in pos_pen.lower():
        fails.append("Penultimate build_role_prompt missing ninja-warrior")
    if "gear" not in pos_pen.lower():
        fails.append("Penultimate build_role_prompt missing gears")
    if not re.search(r"(?i)push-?up|balance|horse stance|train", pos_pen):
        fails.append("Penultimate build_role_prompt missing a rigorous drill")
    if "mei" not in pos_pen.lower():
        fails.append("Penultimate missing distant Master Mei observer")
    if re.search(r"(?i)shirtless|bare[- ]chest|muscular warrior", pos_pen):
        fails.append("Penultimate build_role_prompt still shirtless strongman")
    if "graphite" in pos_pen.lower():
        fails.append("Penultimate still contains graphite")

    pos_out, _ = build_role_prompt(role="master", beat="outro", act_index=8, subject="test")
    if "topknot" not in pos_out.lower() or "snow-white" not in pos_out.lower():
        fails.append("Outro build_role_prompt missing snow-white DNA")
    if "headroom" not in pos_out.lower():
        fails.append("Outro missing last-frame headroom lock")
    finals = {
        pick_mei_frame_final_setup(episode_seed=f"fin-{i}", spoken_beat="sovereign")
        for i in range(24)
    }
    if len(finals) < 8:
        fails.append(f"Last-frame setup not diversifying: {len(finals)} unique triples")
    if not {a for a, _c, _e in finals}.intersection(set(MEI_FINAL_ACTIONS)):
        fails.append("Last-frame actions not drawn from MEI_FINAL_ACTIONS")
    if not {c for _a, c, _e in finals}.intersection(set(MEI_FINAL_CAMERAS)):
        fails.append("Last-frame cameras not drawn from MEI_FINAL_CAMERAS")
    f_a = compose_scene_final_prompt(episode_seed="omega-ridge")
    f_b = compose_scene_final_prompt(episode_seed="zeta-bamboo")
    if f_a == f_b:
        fails.append("compose_scene_final_prompt identical across distinct seeds")
    if "looking directly at the camera" in f_a.lower() and "looking directly at the camera" in f_b.lower():
        fails.append("Last frame still hard-locks eyes-on-camera stare")
    if select_style_references_for_mei_frame(
        [Path(__file__)], act_index=7, n_acts=8, episode_seed="x"
    ):
        fails.append("Last-frame style-reference selector must return empty")

    ok, viol, repaired = validate_mei_visual_prompt(
        "Original scene concept: A precise graphite drawing of a man with a handgun",
        act_index=2, n_acts=9, beat="gears",
    )
    if "graphite" in repaired.lower():
        fails.append("QA left graphite in repaired prompt")
    if "attention monopoly" not in repaired.lower() and "vr headset" not in repaired.lower() and "wasteland" not in repaired.lower():
        fails.append("QA Scene 3 repair missing post-apocalyptic slaves injection")

    if abs(float(mei_cfg.BGM_START_TIME) - 0.5) > 1e-6:
        fails.append("BGM_START_TIME not 0.5")
    if abs(float(mei_cfg.ATMOSPHERE_SFX_FADE_IN) - 0.2) > 1e-6:
        fails.append("ATMOSPHERE_SFX_FADE_IN not 0.2")
    if float(mei_cfg.REEL_HOOK_MAX_S) != 8.0:
        fails.append("REEL_HOOK_MAX_S not 8.0")
    if float(mei_cfg.MUSIC_V2_MIN_SECONDS) < 40.0:
        fails.append("MUSIC_V2_MIN_SECONDS < 40")
    if abs(float(mei_cfg.ATMOSPHERE_SFX_VOLUME) - 0.35) > 1e-6:
        fails.append(f"ATMOSPHERE_SFX_VOLUME not 0.35: {mei_cfg.ATMOSPHERE_SFX_VOLUME}")
    if abs(float(mei_cfg.AMBIENT_VOLUME) - 0.24) > 1e-6:
        fails.append(f"AMBIENT_VOLUME (BGM) not 0.24: {mei_cfg.AMBIENT_VOLUME}")
    if float(mei_cfg.IMPACT_SFX_VOLUME) != 0.0:
        fails.append("IMPACT_SFX_VOLUME should be 0.0")
    if str(getattr(mei_cfg, "MUSIC_TRACK_STYLE", "")) != "industrial_cyberpunk_percussion":
        fails.append("MUSIC_TRACK_STYLE not industrial_cyberpunk_percussion")

    if "percussion" not in MUSIC_PROMPT_SYSTEM_DIRECTIVE.lower():
        fails.append("music_prompt system directive missing industrial percussion cues")
    if "150" not in MUSIC_PROMPT_SYSTEM_DIRECTIVE:
        fails.append("music_prompt system directive missing 150-char constraint")
    siege_pos, siege_neg, _, _ = _mei_section_styles()
    if not any("percussion" in s.lower() or "drum" in s.lower() for s in siege_pos):
        fails.append("mei section styles missing industrial percussion")
    if not any("cheerful" in s.lower() or "silent" in s.lower() or "vocal" in s.lower() for s in siege_neg):
        fails.append("mei section negatives missing cheerful/silent-intro ban")
    # Offline fallback path (no network required)
    mp = generate_dynamic_music_prompt(topic="attention economy trap", subject="test")
    if len(mp) < 40:
        fails.append("dynamic music_prompt fallback too short")
    if "drum" not in mp.lower() and "percussion" not in mp.lower() and "bpm" not in mp.lower():
        fails.append("dynamic music_prompt missing industrial percussion cues")

    if fails:
        print("[FAIL]")
        for f in fails:
            print(" -", f)
        return 1

    print("[PASS] Duration profiles 8/9/10 for 90-120s")
    print(f"[PASS] Hook={durs[0]:.1f}s body_avg={body_avg:.1f}s")
    print("[PASS] Scene 1/3/7/penultimate/final RAG + Mei base anchor")
    print("[PASS] Firearm ban + Visual QA gate")
    print("[PASS] Dynamic music_prompt + SFX@0.35 / BGM@0.24 @0.5s (industrial percussion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
