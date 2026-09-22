# -*- coding: utf-8 -*-
"""Cost-effective remux of the 2026-09-06 ancient_knowledge silent batch.

Reuses existing MP4 video (or subject stills for v19/v20), existing music_v2
and atmosphere SFX, and generates ElevenLabs narration + burnt-in subtitles.
Does not call image or music APIs.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")
os.environ["ACTIVE_PAGE"] = "ancient_knowledge"
os.environ.setdefault("ALLOW_GOOGLE_API", "false")

from agents.media.audio_engine import (  # noqa: E402
    _audio_file_duration_s,
    assert_voice_and_subtitles_ready,
    generate_voiceover_with_timestamps,
)
from main import _stitch_audio_sequential  # noqa: E402

CLIPS = Path(
    r"G:\My Drive\Z sosFiles\Z_act\@ NETWORK\@MEDIAUPSCALE_FACTORY_DYNAMIC_CONTENT"
    r"\Unified Multi-Page Factory\outputs\ancient_knowledge\clips"
)
ASSETS = CLIPS.parent / "assets"
CAPTIONS = CLIPS.parent / "last_captions_bundle.txt"
FFMPEG = Path(
    ROOT / ".venv" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries"
    / "ffmpeg-win-x86_64-v7.1.exe"
)
FONT_CANDIDATES = [
    ROOT / "Fonts" / "Montserrat" / "static" / "Montserrat-Bold.ttf",
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\Arial Bold.ttf"),
    Path(r"C:\Windows\Fonts\trebucbd.ttf"),
]
VOICE_ID = "WdZjiN0nNcik2LBjOHiv"
MODEL_ID = "eleven_multilingual_v2"
VOICE_SETTINGS = {
    "stability": 0.80,
    "similarity_boost": 0.85,
    "style": 0.20,
    "use_speaker_boost": True,
    "speed": 1.0,
}
CTA = "Follow Ancient Knowledge for more hidden mysteries."
CTA_STRIP = re.compile(
    r"(?is)\s*(?:share your (?:insights|theories|thoughts|own story)[^.]*\.)?"
    r"\s*follow(?:\s+ancient knowledge)?(?:\s+for more[^.]*\.)?\s*$"
)

# (variant, reel filename, music/sfx stem, youtube id or None)
BATCH: list[tuple[int, str, str, str | None]] = [
    (1, "reel_did_ancient_texts_unveil_a_cosmi_v01.mp4", "advanced_astronomy_the_9dcf03", "JnxfU-u9KAo"),
    (2, "reel_osireion__a_pre_dynastic_enigma__v02.mp4", "the_osireion_at_abydos_and_414449", "CwVSVAgKaiA"),
    (3, "reel_ancient_texts_describe_flight_be_v03.mp4", "ancient_indian_vimana_4e1c59", "s-uZfut4qMc"),
    (4, "reel_abydos__did_ancient_egypt_record_v04.mp4", "ancient_egyptian_glyphs_3e9015", "Z7NiFzSb5qs"),
    (5, "reel_easter_island_s_giants__what_sec_v05.mp4", "easter_island_s_moai_were_39479a", "Wp3R4Ke5dd8"),
    (6, "reel_göbekli_tepe__who_buried_history_v06.mp4", "göbekli_tepe_was_0e1f75", "-OuzV9L0jEE"),
    (7, "reel_el_dorado__was_the_jungle_hiding_v07.mp4", "the_lost_golden_city_of_el_3bd987", "JUXknID2JII"),
    (8, "reel_ancient_geology_whispers_of_atla_v08.mp4", "plato_s_atlantis_the_8a13de", "LqldV4IKLG0"),
    (9, "reel_dogon_knowledge__did_ancient_eye_v09.mp4", "the_sirius_mystery_how_did_d0ea66", "dzrCEAPD8Sw"),
    (10, "reel_mohenjo_daro__what_melted_stone__v10.mp4", "the_ancient_nuclear_blast_ed6b71", "rX3rfnljFbY"),
    (11, "reel_baalbek_s_megaliths__redefining__v11.mp4", "the_1_500_tonne_trilithon_7a5666", None),
    (12, "reel_this_book_s_language_defies_know_v12.mp4", "the_voynich_manuscript_an_035830", None),
    (13, "reel_oak_island_s_pit__what_hidden_di_v13.mp4", "oak_island_s_money_pit_200_c66921", None),
    (14, "reel_prehistoric_art__a_universal_enc_v14.mp4", "prehistoric_cave_paintings_0ebfe8", None),
    (15, "reel_sacsayhuamán_s_stones__a_forgott_v15.mp4", "sacsayhuamán_s_100_tonne_0e9794", None),
    (16, "reel_crystal_skulls__who_carved_them__v16.mp4", "the_crystal_skulls_of_eb02f6", None),
    (17, "reel_egypt_s_ancient_glider__a_forgot_v17.mp4", "the_saqqara_bird_a_2_200_0664ef", None),
    (18, "reel_great_pyramid__did_radar_unveil__v18.mp4", "inside_the_great_pyramid_s_c4466c", None),
    (19, "reel_puma_punku__what_tools_shaped_th_v19.mp4", "puma_punku_s_h_blocks_at_6da374", None),
    (20, "reel_sphinx__what_ancient_water_truly_v20.mp4", "the_sphinx_enclosure_54af68", None),
]


def _font_path() -> Path:
    for cand in FONT_CANDIDATES:
        if cand.is_file():
            return cand
    raise FileNotFoundError("No subtitle font found (Montserrat/Arial Bold).")


def _load_captions() -> dict[int, str]:
    text = CAPTIONS.read_text(encoding="utf-8")
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"^(\d+)\.\s+(.*)$", line.strip())
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


def _clean_script(caption: str) -> str:
    body = CTA_STRIP.sub("", caption or "").strip()
    body = re.sub(r"\s+", " ", body)
    return body


def _find_sidecar(stem: str, variant: int, suffix: str) -> Path | None:
    vv = f"{variant:02d}"
    exact = CLIPS / f"{stem}_v{vv}_v{vv}_{suffix}"
    if exact.is_file():
        return exact
    hits = sorted(CLIPS.glob(f"{stem}*_v{vv}*{suffix}"))
    return hits[-1] if hits else None


def _find_reel(name: str) -> Path | None:
    p = CLIPS / name
    if p.is_file():
        return p
    # Windows may mojibake accented filenames (göbekli / sacsayhuamán)
    stem = name.replace(".mp4", "")
    key = re.sub(r"[^a-z0-9]+", "", stem.lower())
    for cand in CLIPS.glob("reel_*v*.mp4"):
        ck = re.sub(r"[^a-z0-9]+", "", cand.stem.lower())
        if ck == key or cand.name.lower() == name.lower():
            return cand
    return None


def _collect_stills(stem: str) -> list[Path]:
    stills: list[Path] = []
    if not ASSETS.is_dir():
        return stills
    for folder in ASSETS.iterdir():
        if not folder.is_dir():
            continue
        if stem[:20] not in folder.name and folder.name[:20] not in stem:
            continue
        for img in folder.rglob("*"):
            if img.suffix.lower() in {".png", ".jpg", ".jpeg"} and img.stat().st_size > 200_000:
                stills.append(img)
    stills.sort(key=lambda p: p.stat().st_mtime)
    # Prefer unique-ish set, newest last, cap 16
    uniq: list[Path] = []
    seen: set[str] = set()
    for img in stills:
        key = img.name.split("_v")[0] + str(img.stat().st_size)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(img)
    return uniq[-16:] if len(uniq) > 16 else uniq


def _ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _fmt_ass_time(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}:{m:02d}:{sec:05.2f}"


def _write_ass(
    path: Path,
    timings: list[tuple[str, float, float]],
    font: Path,
    *,
    words_per_phrase: int = 4,
) -> None:
    phrases: list[tuple[str, float, float]] = []
    buf: list[str] = []
    start = 0.0
    end = 0.0
    for i, (word, t0, t1) in enumerate(timings):
        if not buf:
            start = float(t0)
        buf.append(word)
        end = float(t1)
        if len(buf) >= words_per_phrase or i == len(timings) - 1:
            phrases.append((" ".join(buf), start, end))
            buf = []
    fontname = font.stem.replace("-", " ")
    # ASS colour is &HAABBGGRR — warm yellow (255, 230, 0)
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: AK,{fontname},56,&H0000E6FF,&H000000FF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,3,0,2,40,40,470,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for text, t0, t1 in phrases:
        if t1 <= t0:
            t1 = t0 + 0.25
        lines.append(
            f"Dialogue: 0,{_fmt_ass_time(t0)},{_fmt_ass_time(t1)},AK,,0,0,0,,"
            f"{_ass_escape(text)}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _run_ffmpeg(args: list[str]) -> None:
    exe = str(FFMPEG if FFMPEG.is_file() else shutil.which("ffmpeg") or "ffmpeg")
    cmd = [exe, "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err}")


def _mix_audio(
    voice: Path,
    music: Path | None,
    sfx: Path | None,
    out: Path,
    total_s: float,
) -> Path:
    # voice 1.0, music 0.22, sfx 0.12 — then loudnorm to -16 LUFS
    inputs = ["-i", str(voice)]
    filters = ["[0:a]volume=1.0,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[v]"]
    labels = ["[v]"]
    idx = 1
    if music and music.is_file():
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        filters.append(
            f"[{idx}:a]volume=0.22,aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo,atrim=0:{total_s:.3f},asetpts=PTS-STARTPTS[m]"
        )
        labels.append("[m]")
        idx += 1
    if sfx and sfx.is_file():
        inputs += ["-stream_loop", "-1", "-i", str(sfx)]
        filters.append(
            f"[{idx}:a]volume=0.12,aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo,atrim=0:{total_s:.3f},asetpts=PTS-STARTPTS[s]"
        )
        labels.append("[s]")
        idx += 1
    n = len(labels)
    if n == 1:
        mix = "[v]loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    else:
        mix = f"{''.join(labels)}amix=inputs={n}:normalize=0:duration=longest,loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    _run_ffmpeg(
        [
            *inputs,
            "-filter_complex",
            ";".join(filters + [mix]),
            "-map",
            "[out]",
            "-t",
            f"{total_s:.3f}",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(out),
        ]
    )
    return out


def _tts_pair(script: str, variant: int, work: Path) -> tuple[Path, list[tuple[str, float, float]], float]:
    narr_out = work / f"v{variant:02d}_narration.mp3"
    cta_out = work / f"v{variant:02d}_cta.mp3"
    voice_out = work / f"v{variant:02d}_voice.mp3"
    timings_json = work / f"v{variant:02d}_timings.json"
    if voice_out.is_file() and voice_out.stat().st_size > 20_000 and timings_json.is_file():
        import json
        raw = json.loads(timings_json.read_text(encoding="utf-8"))
        timings = [(w, float(s), float(e)) for w, s, e in raw]
        print(f"v{variant:02d} reusing cached TTS {voice_out.name}", flush=True)
        return voice_out, timings, _audio_file_duration_s(narr_out) if narr_out.is_file() else 0.0
    narr_path, narr_wts = generate_voiceover_with_timestamps(
        script,
        narr_out,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        speed=1.0,
        voice_settings=VOICE_SETTINGS,
        expressive_mode=False,
        force_elevenlabs=True,
    )
    cta_path, cta_wts = generate_voiceover_with_timestamps(
        CTA,
        cta_out,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        speed=1.0,
        voice_settings=VOICE_SETTINGS,
        expressive_mode=False,
        force_elevenlabs=True,
    )
    stitched = _stitch_audio_sequential(Path(narr_path), Path(cta_path), voice_out, silence_s=1.0)
    narr_dur = _audio_file_duration_s(Path(narr_path))
    offset = narr_dur + 1.0
    timings = list(narr_wts) + [
        (w, s + offset, e + offset) for w, s, e in cta_wts
    ]
    import json
    timings_json.write_text(json.dumps(timings), encoding="utf-8")
    return Path(stitched), timings, narr_dur


def _burn_onto_video(video: Path, audio: Path, ass: Path, out: Path) -> Path:
    # Escape ASS path for ffmpeg on Windows
    ass_esc = ass.as_posix().replace(":", r"\:").replace("'", r"\'")
    tmp = out.with_suffix(".tmp.mp4")
    _run_ffmpeg(
        [
            "-i", str(video),
            "-i", str(audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-vf", f"ass='{ass_esc}'",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            str(tmp),
        ]
    )
    if out.is_file():
        bak = out.with_suffix(".silent_bak.mp4")
        if not bak.is_file():
            shutil.move(str(out), str(bak))
        else:
            out.unlink()
    shutil.move(str(tmp), str(out))
    return out


def _compile_from_stills(
    stills: list[Path],
    voice: Path,
    music: Path | None,
    sfx: Path | None,
    timings: list,
    out: Path,
) -> Path:
    from core.reel_sequence_engine import compile_sequence_reel

    logo = ROOT / "channels_config" / "ancient_knowledge" / "logo" / "logo.png"
    compile_sequence_reel(
        stills,
        "",
        voice_audio=voice,
        ambient_audio=music,
        output_path=out,
        word_timings=timings,
        logo_image_path=logo if logo.is_file() else None,
        page_id="ancient_knowledge",
        subtitle_fontsize=58,
        subtitle_y_position=1570,
        subtitle_stroke_width=3,
        subtitle_stroke_fill=(0, 0, 0),
        subtitle_fill=(255, 204, 0),
        enable_hook_text=False,
        sfx_loop_audio=sfx,
        sfx_loop_volume=0.12,
        ambient_volume=0.22,
        ambient_profile="mystery",
        cta_text=CTA,
        ffmpeg_preset="ultrafast",
        tail_pad_s=1.0,
    )
    return out


def remux_one(variant: int, reel_name: str, stem: str, captions: dict[int, str], work: Path) -> Path:
    caption = captions.get(variant)
    if not caption:
        raise RuntimeError(f"v{variant:02d}: missing caption")
    script = _clean_script(caption)
    if len(script.split()) < 20:
        raise RuntimeError(f"v{variant:02d}: script too short ({len(script.split())} words)")

    print(f"\n=== v{variant:02d} TTS ({len(script.split())} words) ===", flush=True)
    voice, timings, _narr_dur = _tts_pair(script, variant, work)
    assert_voice_and_subtitles_ready(
        voice_audio=voice,
        word_timings=timings,
        page_id="ancient_knowledge",
    )
    voice_dur = _audio_file_duration_s(voice)
    music = _find_sidecar(stem, variant, "music_v2.mp3")
    sfx = _find_sidecar(stem, variant, "atmosphere_sfx.mp3")
    mixed = _mix_audio(voice, music, sfx, work / f"v{variant:02d}_mix.m4a", voice_dur + 1.0)
    ass = work / f"v{variant:02d}.ass"
    _write_ass(ass, timings, _font_path())

    dest = CLIPS / reel_name
    bak = dest.with_suffix(".silent_bak.mp4")
    src = bak if bak.is_file() else _find_reel(reel_name)
    if src is not None and src.is_file():
        print(f"v{variant:02d} remux video={src.name} music={bool(music)} sfx={bool(sfx)}", flush=True)
        return _burn_onto_video(src, mixed, ass, dest)

    stills = _collect_stills(stem)
    if len(stills) < 2:
        raise RuntimeError(
            f"v{variant:02d}: no source MP4 and only {len(stills)} stills for '{stem}'"
        )
    print(f"v{variant:02d} compile from {len(stills)} stills (no existing MP4)", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return _compile_from_stills(stills, voice, music, sfx, timings, dest)


def verify(path: Path) -> None:
    exe = str(FFMPEG if FFMPEG.is_file() else "ffmpeg")
    proc = subprocess.run(
        [exe, "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    blob = (proc.stderr or "") + (proc.stdout or "")
    mean = re.search(r"mean_volume:\s*([-\d.]+)", blob)
    dur = re.search(r"Duration:\s*(\d+:\d+:\d+\.\d+)", blob)
    has_audio = "Audio:" in blob
    print(
        f"VERIFY {path.name} audio={has_audio} dur={dur.group(1) if dur else '?'} "
        f"mean={mean.group(1) if mean else '?'} dB size={path.stat().st_size / 1e6:.1f}MB",
        flush=True,
    )
    if not has_audio:
        raise RuntimeError(f"{path.name} still has no audio stream")
    if mean and float(mean.group(1)) < -28.0:
        raise RuntimeError(
            f"{path.name} still too quiet (mean {mean.group(1)} dB) — audio map likely wrong"
        )


def main() -> int:
    if not CLIPS.is_dir():
        raise SystemExit(f"clips dir missing: {CLIPS}")
    captions = _load_captions()
    work = CLIPS / "_ak_silent_rebuild"
    work.mkdir(parents=True, exist_ok=True)
    done: list[Path] = []
    errors: list[str] = []
    only = {int(x) for x in sys.argv[1:]} if len(sys.argv) > 1 else set()
    for variant, reel_name, stem, _yt in BATCH:
        if only and variant not in only:
            continue
        try:
            out = remux_one(variant, reel_name, stem, captions, work)
            verify(out)
            done.append(out)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"v{variant:02d}: {exc}")
            print(f"FAIL v{variant:02d}: {exc}", flush=True)
    print(f"\nRebuilt {len(done)}/{len(only) or 20}. errors={len(errors)}", flush=True)
    for e in errors:
        print(" ", e)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
