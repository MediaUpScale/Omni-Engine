# -*- coding: utf-8 -*-
"""Rebuild Nazca v02 stills with topic lock and replace the scheduled YouTube upload."""
from __future__ import annotations

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["ACTIVE_PAGE"] = "ancient_knowledge"

TOPIC = (
    "The Nazca Lines — vast geoglyphs only visible and meaningful "
    "from 1,500 feet in the air"
)
# Recovered from 2026-09-08 v02 bucket-plan / act-beat logs (141-word class).
LOCKED_SCRIPT = (
    "The Nazca Plains of Peru hold a profound secret. "
    "Colossal drawings scar miles of desert. "
    "These Nazca Lines depict intricate animals and figures. "
    "Their wonder emerges only from 1,500 feet. "
    "From the ground, they are barely visible trenches. "
    "Perfect figures, like animals, appear. Created from BCE to CE. "
    "They achieved incredible precision on an epic scale. "
    "Some researchers believe they were astronomical calendars. "
    "But how did this ancient culture achieve such aerial design? "
    "Historians debate if observation platforms were used. "
    "One theory proposes rudimentary hot air balloons. "
    "Ancient records, however, offer no clear explanation. "
    "The scale and accuracy suggest advanced understanding. "
    "Understanding of geometry, perspective, ahead of their era. "
    "Could an unknown aerial presence inspire these markings? "
    "Who provided blueprints for these immense desert figures? "
    "The Nazca Lines continue to defy conventional archaeology. "
    "Pondering ancient ingenuity, or something far more."
)
TITLE = "Nazca Lines: Who saw these from the ancient sky?"
CAPTION = (
    "The colossal Nazca Lines in Peru span miles, forming intricate desert "
    "geoglyphs discernible only from high in the air. Mainstream theories "
    "suggest ground-level ritual purposes, yet some researchers highlight "
    "their scale demands an aerial perspective, implying a lost, unrecognized "
    "understanding. This inexplicable requirement challenges current timelines "
    "for human ingenuity and the true capabilities of ancient civilizations. "
    "What do you believe about these incredible desert markings? Follow "
    "Ancient Knowledge for more discoveries."
)
OLD_YT_ID = "8tfBySnnHc0"
OLD_PUBLISH_AT = datetime(2026, 9, 9, 6, 56, 57, tzinfo=timezone.utc)
CTA_TEXT = "Follow Ancient Knowledge for more hidden mysteries."
CLIPS = Path(
    r"G:\My Drive\Z sosFiles\Z_act\@ NETWORK"
    r"\@MEDIAUPSCALE_FACTORY_DYNAMIC_CONTENT"
    r"\Unified Multi-Page Factory\outputs\ancient_knowledge\clips"
)
STEM = "the_nazca_lines_vast_c86fe8_v02"
REEL_NAME = "reel_nazca_lines__who_saw_these_from__v02.mp4"
MAX_WORKERS = 3


def _audio_duration(path: Path) -> float:
    from moviepy import AudioFileClip

    clip = AudioFileClip(str(path))
    try:
        return float(clip.duration or 0.0)
    finally:
        clip.close()


def main() -> int:
    import config as app_config

    app_config.bind_active_page("ancient_knowledge")

    from agents.media.audio_engine import approximate_word_timings
    from agents.media.providers.together_image import (
        JUGGERNAUT_LIGHTNING_FLUX_MODEL,
        TogetherImageAdapter,
    )
    from agents.media.scene_prompt_generator import generate_scene_prompts
    from channel_loader import load_page_context
    from channels_config.shared import mei_chrome
    from core.audio_chunker import chunk_word_timings
    from core.reel_sequence_engine import compile_sequence_reel
    from modules.reel_visual_qa import generate_and_gate

    page_ctx = load_page_context(
        "ancient_knowledge", avatar_mode="OFF", post_format="DYNAMIC_REEL",
    )
    voice = CLIPS / f"{STEM}_v02_voice.mp3"
    narr = CLIPS / f"{STEM}_v02_narration.mp3"
    music = CLIPS / f"{STEM}_v02_music_v2.mp3"
    sfx = CLIPS / f"{STEM}_v02_atmosphere_sfx.mp3"
    reel_path = CLIPS / REEL_NAME
    for required in (voice, narr, music, sfx, reel_path):
        if not required.is_file():
            print(f"[repair] missing {required}", flush=True)
            return 2

    narr_dur = _audio_duration(narr)
    voice_dur = _audio_duration(voice)
    print(
        f"[repair] audio narr={narr_dur:.1f}s voice={voice_dur:.1f}s "
        f"script_words={len(LOCKED_SCRIPT.split())}",
        flush=True,
    )

    timings = approximate_word_timings(LOCKED_SCRIPT, narr_dur)
    chunks = chunk_word_timings(timings)
    if not chunks:
        print("[repair] no audio chunks", flush=True)
        return 3

    style = (page_ctx.illustration_style or "").rstrip(" .")
    scenes = generate_scene_prompts(
        LOCKED_SCRIPT,
        chunks,
        {"topic": TOPIC, "lighting_style": style},
        channel_id="ancient_knowledge",
        atmosphere=style,
        use_llm=True,
    )
    print(
        f"[repair] scenes={len(scenes)} domain={scenes[0].get('domain_id')} "
        f"banned={scenes[0].get('banned_subjects')}",
        flush=True,
    )

    work = (
        Path(app_config.PAGE_OUTPUTS_DIR)
        / "assets"
        / "ep_nazca_repair"
        / "work"
    )
    work.mkdir(parents=True, exist_ok=True)
    adapter = TogetherImageAdapter(model_id=JUGGERNAUT_LIGHTNING_FLUX_MODEL)

    def _one(idx: int) -> tuple[int, Path]:
        scene = scenes[idx]
        stem = f"{STEM}_fix_act{idx + 1:02d}_r01"
        path, extra = generate_and_gate(
            adapter.generate,
            prompt=str(scene.get("image_generation_prompt") or ""),
            chunk_text=str(scene.get("spoken_text") or ""),
            output_stem=stem,
            output_directory=work,
            channel="ancient_knowledge",
            search_dirs=[work],
            generate_kwargs={
                "avatar_mode": "OFF",
                "negative_prompt": str(scene.get("negative_prompt") or "") or None,
            },
            banned_subjects=list(scene.get("banned_subjects") or []),
            domain_anchors=str(scene.get("domain_anchors") or ""),
            prefer_stem=STEM,
        )
        print(
            f"[repair] act {idx + 1}/{len(scenes)} extra={extra} -> {path.name}",
            flush=True,
        )
        return idx, Path(path)

    paths: list[Path | None] = [None] * len(scenes)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(_one, i) for i in range(len(scenes))]
        for fut in as_completed(futs):
            idx, path = fut.result()
            paths[idx] = path
    stills = [p for p in paths if p is not None and p.is_file()]
    if len(stills) != len(scenes):
        print(f"[repair] missing stills {len(stills)}/{len(scenes)}", flush=True)
        return 4

    raw_durs = [c.duration_s for c in chunks]
    scale = narr_dur / (sum(raw_durs) or 1.0)
    act_durs = [d * scale for d in raw_durs]
    cta_hold = max(0.5, voice_dur - narr_dur)
    act_durs[-1] += cta_hold

    backup = reel_path.with_name(reel_path.stem + "_pyramid_bad.mp4")
    if not backup.is_file():
        shutil.copy2(reel_path, backup)
        print(f"[repair] backed up bad reel -> {backup.name}", flush=True)

    logo = ROOT / "channels_config" / "ancient_knowledge" / "logo" / "logo.png"
    cta_wts = approximate_word_timings(CTA_TEXT, max(0.5, voice_dur - narr_dur - 1.0))
    word_timings = list(timings) + [
        (w, s + narr_dur + 1.0, e + narr_dur + 1.0) for w, s, e in cta_wts
    ]

    print("[repair] compiling replacement reel…", flush=True)
    compile_sequence_reel(
        stills,
        TITLE,
        voice_audio=voice,
        ambient_audio=music if music.is_file() else None,
        sfx_loop_audio=sfx if sfx.is_file() else None,
        sfx_loop_volume=getattr(page_ctx, "atmosphere_sfx_volume", 0.35),
        output_path=reel_path,
        target_duration=voice_dur + 1.0,
        act_durations=act_durs,
        strict_act_durations=True,
        word_timings=word_timings,
        font_path=mei_chrome.FONT_PATH,
        overlay_opacity=page_ctx.reel_overlay_opacity,
        enable_hook_text=page_ctx.enable_top_hook_text,
        vignette_strength=page_ctx.vignette_strength,
        grain_intensity=page_ctx.grain_intensity,
        logo_image_path=logo if logo.is_file() else None,
        logo_width_px=page_ctx.logo_width_px,
        logo_y_offset_px=page_ctx.logo_y_offset_px,
        logo_opacity=page_ctx.logo_opacity,
        logo_max_height_px=page_ctx.logo_max_height_px,
        subtitle_fontsize=page_ctx.subtitle_fontsize,
        subtitle_y_position=page_ctx.subtitle_y_position,
        page_id="ancient_knowledge",
        words_per_phrase=mei_chrome.SUBTITLE_WORDS_PER_PHRASE,
        subtitle_fill=tuple(page_ctx.subtitle_fill),
        subtitle_stroke_width=page_ctx.subtitle_stroke_width,
        subtitle_stroke_fill=tuple(page_ctx.subtitle_stroke_fill),
        enable_flicker=page_ctx.enable_flicker,
        enable_light_rays=page_ctx.enable_light_rays,
        enable_dust_particles=page_ctx.enable_dust_particles,
        cta_text=CTA_TEXT,
        cta_start_s=narr_dur + 0.3,
        cta_y_position=page_ctx.cta_subtitle_y_position,
        narration_duration_s=narr_dur,
        cta_visual_gap_s=0.3,
        ambient_volume=page_ctx.ambient_volume,
        tail_pad_s=page_ctx.reel_tail_pad_s,
        duration_override=voice_dur + 2.0,
    )
    print(f"[repair] compiled {reel_path} ({reel_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    from agents.media.b2_client import B2VideoUploader

    uploader = B2VideoUploader()
    try:
        bucket = uploader._bucket()
        uploader._resource().Object(bucket, reel_path.name).delete()
    except Exception as exc:  # noqa: BLE001
        print(f"[repair] B2 delete skipped ({exc})", flush=True)
    b2_url = uploader.upload(reel_path)
    print(f"[repair] B2 {b2_url}", flush=True)

    from agents.posting.youtube_publisher import (
        build_youtube_client_for_page,
        delete_youtube_video,
        fetch_videos_metadata,
        upload_short,
    )

    youtube = build_youtube_client_for_page("ancient_knowledge", enforce_channel=True)
    meta = fetch_videos_metadata(youtube, [OLD_YT_ID]).get(OLD_YT_ID) or {}
    title = str(meta.get("title") or TITLE)
    description = CAPTION
    tags = ["ancient knowledge", "nazca lines", "geoglyphs", "hidden history"]
    print(
        f"[repair] replacing YouTube {OLD_YT_ID} title={title!r} "
        f"publish_at={OLD_PUBLISH_AT.isoformat()}",
        flush=True,
    )
    new_id, url, published = upload_short(
        reel_path,
        title=title,
        description=description,
        tags=tags,
        privacy_status="private",
        publish_at=OLD_PUBLISH_AT,
        page_name="ancient_knowledge",
        youtube=youtube,
        playlist_title="Ancient Mysteries & Forbidden History",
        preserve_title=True,
    )
    print(f"[repair] uploaded replacement {new_id} {url} publish_at={published}", flush=True)
    deleted = delete_youtube_video(youtube, OLD_YT_ID)
    print(f"[repair] deleted old {OLD_YT_ID} ok={deleted}", flush=True)
    print(f"[repair] DONE https://youtu.be/{new_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
