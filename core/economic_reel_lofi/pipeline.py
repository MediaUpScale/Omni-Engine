# -*- coding: utf-8 -*-
"""
ECONOMIC_REEL_LOFI orchestrator.

EmotionalWriter → Gate1 → TTS timing → Atmosphere → Riso prompts → Gate2
→ ImageGen/VisualQA → audio-driven Assembler
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.economic_reel_lofi import config as lofi_cfg
from core.economic_reel_lofi import lofi_collections as rag
from core.economic_reel_lofi.assembler import (
    assemble_lofi_reel,
    compute_caption_scene_duration_s,
    measure_vo_speech_duration,
)
from core.economic_reel_lofi.image_gen import (
    LOFI_IMAGE_HEIGHT,
    LOFI_IMAGE_STEPS,
    LOFI_IMAGE_WIDTH,
    generate_scene_image,
)
from core.economic_reel_lofi.riso_prompt_bank import (
    assign_riso_prompts_for_scenes,
    export_active_library_diff,
)
from core.google_guardrail import guarded_generate_content, make_guarded_gemini_client
from agents.writer.script_agent import (
    _sanitize_caption_typos,
    assess_object_beat_continuity,
    generate_script,
    get_script_cost_log,
    get_script_llm_call_log,
    note_batch_structure_id,
    repair_script_captions,
    reset_batch_structure_ids,
    reset_script_llm_call_log,
)
from core.economic_reel_lofi.validator_agent import validate_script

_LOG = logging.getLogger(__name__)


def _tts_breath_commas(caption: str) -> str:
    """TTS-only commas where a natural breath falls. On-screen caption unchanged."""
    text = (caption or "").strip()
    if not text:
        return text
    if re.search(r"[,;:—–]", text):
        return text
    rules = (
        (r"^(One night)(\s+)", r"\1,\2"),
        (r"\blit anyway\b", "lit, anyway"),
        (r"\bfor it with\b", "for it, with"),
        (r"\blit because\b", "lit, because"),
    )
    for pat, repl in rules:
        nxt = re.sub(pat, repl, text, count=1, flags=re.I)
        if nxt != text:
            return nxt
    return text


def _tts_text_with_breaks(caption: str) -> str:
    """Insert ElevenLabs SSML pauses at commas / dashes. Caption on-screen stays clean."""
    text = (caption or "").strip()
    if not text:
        return text
    text = re.sub(r"\s*[—–]\s*", ' <break time="0.40s" /> ', text)
    text = re.sub(r",\s+", ', <break time="0.30s" /> ', text)
    text = re.sub(r";\s+", '; <break time="0.32s" /> ', text)
    return " ".join(text.split())


@dataclass
class LofiItemResult:
    ok: bool
    video_path: str | None = None
    meta_path: str | None = None
    module: str = ""
    theme: str = ""
    hook_type: str = ""
    scene_count: int = 0
    duration_s: float = 0.0
    manual_review: bool = False
    errors: list[str] = field(default_factory=list)
    script: dict[str, Any] | None = None
    work_dir: str | None = None
    stills_only: bool = False
    hold_gate: int | None = None
    pipeline_stage: str = ""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _episode_slug(
    script: dict[str, Any] | None = None,
    *,
    theme: str = "",
    subtheme: str = "",
) -> str:
    """Folder/file stem: theme_subtheme, e.g. grief_learning_to_carry_it."""
    row = script or {}
    theme = str(theme or row.get("theme") or "").strip().lower()
    sub = str(subtheme or row.get("subtheme") or "").strip().lower()
    raw = "_".join(p for p in (theme, sub) if p) or "lofi"
    slug = "".join(c if c.isalnum() else "_" for c in raw)
    slug = re.sub(r"_+", "_", slug).strip("_")[:64]
    return slug or "lofi"


def _engine_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_lofi_thumb_gray(
    image_path: Path, size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image as PILImage

    im = PILImage.open(image_path).convert("RGB")
    im.thumbnail((size, size))
    arr = np.array(im, dtype=np.float32)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return arr, gray


def _linework_stats(image_path: Path, size: int = 512) -> dict[str, float]:
    """Color/edge stats already used by the linework guard (512 thumb)."""
    arr, gray = _load_lofi_thumb_gray(image_path, size=size)
    lap = (
        gray[1:-1, 1:-1] * 4
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    q = (arr.astype(np.uint8) // 16).reshape(-1, 3)
    return {
        "uniq16": float(np.unique(q, axis=0).shape[0]),
        "lap_var": float(lap.var()),
        "edge": float(
            np.abs(gray[:, 1:] - gray[:, :-1]).mean()
            + np.abs(gray[1:, :] - gray[:-1, :]).mean()
        ),
        "std": float(gray.std()),
        "hard": float((np.abs(lap) > 40).mean()),
    }


def _assess_linework_complexity(image_path: Path) -> tuple[bool, list[str]]:
    """Reject near-flat graphics with no inked structure (grief scene-3 failure)."""
    try:
        st = _linework_stats(image_path, size=512)
        uniq = int(st["uniq16"])
        lap_var = float(st["lap_var"])
        edge = float(st["edge"])
        std = float(st["std"])
        flaws: list[str] = []
        if uniq < 90:
            flaws.append(f"low color structure uniq16={uniq}")
        if lap_var < 80 and edge < 3.0:
            flaws.append(f"flat/no linework lap_var={lap_var:.1f} edge={edge:.2f}")
        if std < 22 and uniq < 140:
            flaws.append(f"near-flat luminance std={std:.1f}")
        if flaws:
            print(
                f"[LOFI image QA] REJECT {image_path.name} "
                f"uniq16={uniq} lap_var={lap_var:.1f} edge={edge:.2f} std={std:.1f} "
                f"| {'; '.join(flaws)}"
            )
        return (len(flaws) == 0), flaws
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("linework QA failed for %s (%s)", image_path.name, exc)
        return True, []


_INTRUDER_SUBJECTS = frozenset({"object_focus", "silhouette"})
_INTRUDER_SELF = re.compile(
    r"\b(mugs?|cups?|coffee|glasses?|laptops?|computers?|keyboards?)\b",
    re.I,
)
# Compact requested objects: a larger architectural dark plane (door/wall)
# is often the biggest blob; the object itself then looks like a "mug".
_COMPACT_INTENDED_RE = re.compile(
    r"\b(suitcase|bag|kettle|chair|pillow|blanket|coat|lamp|radio|plate|"
    r"journal|notebook|letter|envelope|stone|stones|rock|rocks|pebble|"
    r"pebbles|boulder|box|boxes|paper|newspaper)\b",
    re.I,
)


def _uniform_gray(gray: np.ndarray, k: int) -> np.ndarray:
    try:
        from scipy.ndimage import uniform_filter

        return uniform_filter(gray, size=k, mode="nearest")
    except Exception:  # noqa: BLE001
        kernel = np.ones((k,), dtype=np.float32) / k
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, gray)
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, tmp)


def _connected_components(mask: np.ndarray, min_area: int) -> list[dict[str, float]]:
    h, w = mask.shape
    n = h * w
    seen = np.zeros((h, w), dtype=np.uint8)
    out: list[dict[str, float]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = 1
            ys: list[int] = []
            xs: list[int] = []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = 1
                        stack.append((ny, nx))
            area = len(ys)
            if area < min_area:
                continue
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            out.append(
                {
                    "frac": area / n,
                    "bbox_frac": (bw * bh) / n,
                    "fill": area / max(bw * bh, 1),
                    "aspect": bh / max(bw, 1),
                    "cy": float(np.mean(ys)) / h,
                    "cx": float(np.mean(xs)) / w,
                    "x0": x0 / w,
                    "y0": y0 / h,
                    "x1": (x1 + 1) / w,
                    "y1": (y1 + 1) / h,
                }
            )
    out.sort(key=lambda d: d["bbox_frac"], reverse=True)
    return out


def assess_default_object_intrusion(
    image_path: Path,
    *,
    subject_type: str = "",
    key_object: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Cheap non-LLM gate for mug/laptop desk-still-life substitution.

    Uses the same PIL+numpy gray thumb as linework. Finds locally-darker
    compact blobs (mug/laptop silhouettes) and compares their bbox footprint
    to the rest of the frame. Scoped to object_focus / silhouette when those
    defaults are not the requested key_object. Does not judge photoreal/style.
    """
    st = (subject_type or "").strip().lower().replace(" ", "_")
    obj = (key_object or "").strip()
    meta: dict[str, Any] = {
        "subject_type": st,
        "key_object": obj,
        "passed": True,
        "skipped": False,
        "intruder_bbox_frac": 0.0,
        "intruder_area_frac": 0.0,
        "intended_bbox_frac": 0.0,
        "ratio": 0.0,
        "n_intruders": 0,
    }
    if st not in _INTRUDER_SUBJECTS:
        meta["skipped"] = True
        meta["skip_reason"] = "not_object_focus_or_silhouette"
        return True, [], meta
    licensed_self = bool(obj and _INTRUDER_SELF.search(obj))
    # Licensing mug/cup/coffee/laptop excludes THAT blob as intended —
    # it must not disable the whole gate. Extra compact blobs (jar/plant)
    # still fail. Windows are architectural; spoken_prop still catches them.
    try:
        _, gray = _load_lofi_thumb_gray(image_path, size=192)
        local = _uniform_gray(gray, 21)
        mask = gray < (local - 16.0)
        blobs = _connected_components(mask, min_area=50)
        intruders = [
            b
            for b in blobs
            if b["cy"] >= 0.48
            and 0.35 <= b["aspect"] <= 1.65
            and b["fill"] >= 0.22
            and b["bbox_frac"] >= 0.03
            and b["frac"] >= 0.012
        ]
        # Door knobs / latches: small compact dark blobs, not mug-class props.
        hardware = [b for b in intruders if b["bbox_frac"] < 0.06]
        compact = [b for b in intruders if b["bbox_frac"] >= 0.06]
        meta["n_hardware"] = len(hardware)
        compact_intended = bool(obj and _COMPACT_INTENDED_RE.search(obj)) or licensed_self
        if compact_intended and compact:
            # First compact mid-frame blob is the requested object; only a
            # second compact blob is a real extra prop.
            intended = compact[0]
            extras = compact[1:]
            top = extras[0] if extras else None
            meta["compact_as_intended"] = True
            meta["licensed_blob_excluded"] = bool(licensed_self)
        else:
            intended = blobs[0] if blobs else None
            top = compact[0] if compact else None
            extras = compact
        intended_bbox = float((intended or {}).get("bbox_frac") or 0.0)
        intr_bbox = float((top or {}).get("bbox_frac") or 0.0)
        intr_frac = float((top or {}).get("frac") or 0.0)
        ratio = intr_bbox / max(intended_bbox, 1e-6) if intended_bbox else 0.0
        meta.update(
            {
                "n_intruders": len(extras) if top is not None else 0,
                "intruder_bbox_frac": round(intr_bbox, 4),
                "intruder_area_frac": round(intr_frac, 4),
                "intended_bbox_frac": round(intended_bbox, 4),
                "ratio": round(ratio, 3),
            }
        )
        # Fail when a compact lower/mid dark object has a large footprint of
        # its own, or when it is a large fraction of the biggest dark region.
        fail = bool(top) and (
            intr_bbox >= 0.08 or intr_frac >= 0.04 or (ratio >= 0.45 and intr_bbox >= 0.03)
        )
        # Same blob counted as both intended and intruder (ratio≈1): the
        # requested compact object (stone, bag, box, …) is not a mug.
        if (
            fail
            and intended is not None
            and top is not None
            and obj
            and compact_intended
            and abs(intr_bbox - intended_bbox) < 1e-6
        ):
            fail = False
            top = None
            extras = []
            meta["n_intruders"] = 0
            meta["compact_as_intended"] = True
        if fail and top is not None:
            meta["passed"] = False
            flaw = (
                "INTRUDER: compact dark object (mug/laptop-class) bbox_frac="
                f"{intr_bbox:.3f} area_frac={intr_frac:.3f} ratio={ratio:.2f} "
                f"vs requested {obj or 'key_object'!r}"
            )
            fix = (
                f"Draw only {obj or 'the requested object'} filling the frame. "
                "Empty surface: no mug, no cup, no coffee cup, no glass, no laptop."
            )
            meta["fix_instructions"] = fix
            print(
                f"[LOFI object-gate] REJECT {image_path.name} {flaw}"
            )
            return False, [flaw], meta
        print(
            f"[LOFI object-gate] PASS {image_path.name} "
            f"n_intruders={meta.get('n_intruders')} bbox={intr_bbox:.3f} "
            f"ratio={ratio:.2f} compact_intended={int(bool(meta.get('compact_as_intended')))}"
        )
        return True, [], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("object-gate failed for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        return True, [], meta


def debug_intruder_overlay(
    image_path: Path,
    out_path: Path | None = None,
    *,
    subject_type: str = "object_focus",
    key_object: str = "",
) -> Path:
    """Draw dark-blob bboxes for INTRUDER debug. Does not change gate logic."""
    from PIL import Image as PILImage
    from PIL import ImageDraw

    src = Path(image_path)
    dest = Path(out_path) if out_path else src.with_name(f"{src.stem}_intruder_debug.png")
    im = PILImage.open(src).convert("RGB")
    w, h = im.size
    _, gray = _load_lofi_thumb_gray(src, size=192)
    local = _uniform_gray(gray, 21)
    mask = gray < (local - 16.0)
    blobs = _connected_components(mask, min_area=50)
    draw = ImageDraw.Draw(im)
    for i, b in enumerate(blobs[:12]):
        color = (255, 40, 40) if i == 0 else (40, 200, 80)
        x0 = int(float(b["x0"]) * w)
        y0 = int(float(b["y0"]) * h)
        x1 = int(float(b["x1"]) * w)
        y1 = int(float(b["y1"]) * h)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text((x0 + 4, y0 + 4), f"{i} bf={b['bbox_frac']:.3f}", fill=color)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest)
    ok, flaws, meta = assess_default_object_intrusion(
        src, subject_type=subject_type, key_object=key_object
    )
    print(
        f"[LOFI intruder-debug] {src.name} → {dest.name} "
        f"blobs={len(blobs)} gate_ok={int(ok)} meta={meta} flaws={flaws}"
    )
    return dest


def assess_photoreal_style(
    image_path: Path,
    *,
    subject_type: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Cheap non-LLM photoreal gate for macro object_focus stills.

    Independent of Gemini. Uses the same uniq16/lap_var/edge/std thumb as
    the linework guard. Photographic bokeh has high luminance contrast
    (std) with weak ink edges (edge, lap_var). Illustration-safe stills
    from the DoF A/B sit well above the ink-edge floor.

    Scoped to object_focus. Does not change INTRUDER thresholds or escalate
    the framing ladder. Fail-open on read errors (same as INTRUDER).
    """
    st = (subject_type or "").strip().lower().replace(" ", "_")
    meta: dict[str, Any] = {
        "subject_type": st,
        "passed": True,
        "skipped": False,
        "uniq16": 0,
        "lap_var": 0.0,
        "edge": 0.0,
        "std": 0.0,
        "hard": 0.0,
    }
    if st != "object_focus":
        meta["skipped"] = True
        meta["skip_reason"] = "not_object_focus"
        return True, [], meta
    try:
        stats = _linework_stats(image_path, size=512)
        uniq = int(stats["uniq16"])
        lap_var = float(stats["lap_var"])
        edge = float(stats["edge"])
        std = float(stats["std"])
        hard = float(stats["hard"])
        meta.update(
            {
                "uniq16": uniq,
                "lap_var": round(lap_var, 1),
                "edge": round(edge, 2),
                "std": round(std, 1),
                "hard": round(hard, 4),
            }
        )
        # Calibrated on locked-attachment beat 4:
        # photoreal+DoF (225455 s4): std=86.4 edge=9.78 lap_var=459.7 → FAIL
        # no-DoF illustration (3/3 probe): std 51–75, edge 15–21, lap_var 2174–3889 → PASS
        photographic = std >= 80.0 and edge < 12.0 and lap_var < 900.0
        if photographic:
            meta["passed"] = False
            flaw = (
                "PHOTOREAL: photographic smoothness "
                f"std={std:.1f} edge={edge:.2f} lap_var={lap_var:.1f} "
                f"uniq16={uniq} (want ink-hard edges, not camera bokeh)"
            )
            meta["fix_instructions"] = (
                "Redraw as a risograph illustration: hard-edged color blocks, "
                "halftone grain, no camera bokeh, no lens blur, no photography."
            )
            print(f"[LOFI style-gate] REJECT {image_path.name} {flaw}")
            return False, [flaw], meta
        print(
            f"[LOFI style-gate] PASS {image_path.name} "
            f"std={std:.1f} edge={edge:.2f} lap_var={lap_var:.1f} uniq16={uniq}"
        )
        return True, [], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("style-gate failed for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        return True, [], meta


def pixel_lighting_label(
    image_path: Path,
    *,
    framing: str = "",
    subject_type: str = "",
) -> tuple[str, dict[str, float]]:
    """Classify lighting from the accepted still, not from the prompt tag.

    ``macro_no_setting`` / object_focus close-ups are not eligible for
    sunset_doorway — a warm mug disc is not a doorway.
    """
    from core.economic_reel_lofi.style_modules import get_active_style

    style = get_active_style()
    arr, gray = _load_lofi_thumb_gray(image_path, size=256)
    h = gray.shape[0]
    upper = arr[: int(h * 0.65)]
    r = upper[..., 0]
    g = upper[..., 1]
    b = upper[..., 2]
    warm = (r > 165) & (g > 75) & (b < 140) & ((r.astype(np.int16) - b.astype(np.int16)) > 45)
    warm_frac = float(warm.mean())
    mean_l = float(gray.mean())
    chroma = float(np.mean(np.abs(arr[..., 0].astype(np.float32) - arr[..., 2])))
    st = (subject_type or "").strip().lower().replace(" ", "_")
    fr = (framing or "").strip().lower()
    is_macro = (not style.macro_sunset_eligible) and (
        fr == "macro_no_setting" or st == "object_focus"
    )
    door_thr = float(style.warm_sunset_doorway)
    lamp_thr = float(style.warm_indoor_lamp)
    if is_macro:
        if warm_frac >= lamp_thr:
            label = "indoor_lamp_glow"
        elif mean_l < 70 and chroma < 25:
            label = "blue_hour_streetlight"
        else:
            label = "overcast_daylight"
    elif warm_frac >= door_thr:
        label = "sunset_doorway"
    elif mean_l < 70 and chroma < 25:
        label = "blue_hour_streetlight"
    elif chroma < 18 and 70 <= mean_l <= 150:
        label = "rainy_grey"
    elif mean_l > 150 and chroma < 30:
        label = "morning_cool"
    elif warm_frac >= lamp_thr:
        label = "indoor_lamp_glow"
    else:
        label = "overcast_daylight"
    meta = {
        "warm_frac": round(warm_frac, 4),
        "mean_l": round(mean_l, 1),
        "chroma": round(chroma, 1),
        "framing": fr or st,
        "macro_no_sunset": bool(is_macro),
    }
    return label, meta


def _style_ratio(a: float, b: float) -> float:
    x = max(float(a), 1.0)
    y = max(float(b), 1.0)
    return max(x, y) / min(x, y)


def _is_photographic_stat(st: dict[str, Any]) -> bool:
    return (
        float(st.get("std") or 0) >= 80.0
        and float(st.get("edge") or 0) < 12.0
        and float(st.get("lap_var") or 0) < 900.0
    )


def _styles_compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if abs(float(a.get("uniq16") or 0) - float(b.get("uniq16") or 0)) > 80:
        return False
    if _style_ratio(float(a.get("lap_var") or 0), float(b.get("lap_var") or 0)) > 4.5:
        return False
    if _style_ratio(float(a.get("edge") or 0), float(b.get("edge") or 0)) > 2.4:
        return False
    if _is_photographic_stat(a) != _is_photographic_stat(b):
        return False
    return True


def _episode_style_class(rec: dict[str, Any], row: dict[str, Any] | None = None) -> str:
    """Coarse look bucket for cross-beat comparison (not a lighting tag)."""
    close = str((row or {}).get("close_variant") or "")
    if close == "portrait_close":
        return "portrait_close"
    framing = str((row or {}).get("framing") or "").strip().lower()
    st = str((row or {}).get("subject_type") or "").strip().lower().replace(" ", "_")
    from core.economic_reel_lofi.style_modules import get_active_style

    style = get_active_style()
    is_macro = (not style.macro_sunset_eligible) and (
        framing == "macro_no_setting" or st == "object_focus"
    )
    warm = float(rec.get("warm_frac") or 0)
    uniq = float(rec.get("uniq16") or 0)
    std = float(rec.get("std") or 0)
    edge = float(rec.get("edge") or 0)
    lap = float(rec.get("lap_var") or 0)
    if (not is_macro) and warm >= float(style.warm_sunset_disc):
        return "sunset_disc"
    if uniq <= 240 and std >= 55:
        return "vintage_photo"
    if std >= 80 and edge < 12 and lap < 900:
        return "photographic"
    return "painterly_print"


def _cluster_style_indices(
    stats: list[dict[str, Any]],
    lines: list[dict[str, Any]] | None = None,
) -> list[list[int]]:
    """Group beats that share a look class. Do not merge via transitivity."""
    buckets: dict[str, list[int]] = {}
    for i, rec in enumerate(stats):
        row = lines[i] if lines and i < len(lines) and isinstance(lines[i], dict) else None
        cls = _episode_style_class(rec, row)
        rec["style_class"] = cls
        buckets.setdefault(cls, []).append(int(rec.get("scene") or i + 1))
    return list(buckets.values())


def assess_cross_beat_style(
    scene_paths: list[Path],
    lines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare accepted stills to each other — not per-beat in isolation."""
    stats: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for i, path in enumerate(scene_paths):
        if not path or not Path(path).is_file():
            continue
        st = _linework_stats(Path(path))
        pix, pix_meta = pixel_lighting_label(
            Path(path),
            framing=str(lines[i].get("framing") or "") if i < len(lines) and isinstance(lines[i], dict) else "",
            subject_type=str(lines[i].get("subject_type") or "") if i < len(lines) and isinstance(lines[i], dict) else "",
        )
        declared = ""
        if i < len(lines) and isinstance(lines[i], dict):
            declared = str(lines[i].get("lighting_condition") or "")
        rec = {
            "scene": i + 1,
            "uniq16": int(st["uniq16"]),
            "lap_var": round(float(st["lap_var"]), 1),
            "edge": round(float(st["edge"]), 2),
            "std": round(float(st["std"]), 1),
            "hard": round(float(st["hard"]), 4),
            "pixel_lighting": pix,
            "declared_lighting": declared,
            **pix_meta,
        }
        stats.append(rec)
        sunset_vs_not = (
            pix == "sunset_doorway" and declared not in {"", "sunset_doorway"}
        ) or (
            declared == "sunset_doorway"
            and pix != "sunset_doorway"
            and float(pix_meta["warm_frac"]) < 0.04
        )
        if sunset_vs_not:
            mismatches.append(
                {
                    "scene": i + 1,
                    "declared": declared,
                    "pixel": pix,
                    "warm_frac": pix_meta["warm_frac"],
                }
            )
    clusters = _cluster_style_indices(stats, lines) if stats else []
    pix_counts: dict[str, int] = {}
    decl_counts: dict[str, int] = {}
    for rec in stats:
        p = str(rec.get("pixel_lighting") or "")
        d = str(rec.get("declared_lighting") or "")
        if p:
            pix_counts[p] = pix_counts.get(p, 0) + 1
        if d:
            decl_counts[d] = decl_counts.get(d, 0) + 1
    fails: list[str] = []
    if len(clusters) >= 3:
        fails.append(
            f"cross-beat-style: {len(clusters)} distinct looks across "
            f"{len(stats)} beats clusters={clusters}"
        )
        print(
            f"[LOFI cross-style] FAIL classes={len(clusters)} "
            f"groups={clusters}"
        )
    else:
        print(f"[LOFI cross-style] PASS clusters={len(clusters)} groups={clusters}")
    if mismatches:
        fails.append(
            "lighting-pixel-mismatch: declared tags do not match accepted stills: "
            + ", ".join(
                f"s{m['scene']} {m['declared']}→{m['pixel']} "
                f"warm={m['warm_frac']}"
                for m in mismatches
            )
        )
        print(f"[LOFI lighting] PIXEL-MISMATCH {mismatches}")
    print(
        f"[LOFI lighting] DECLARED counts={decl_counts} "
        f"PIXEL counts={pix_counts}"
    )
    for rec in stats:
        print(
            f"[LOFI cross-style] scene={rec['scene']} "
            f"uniq16={rec['uniq16']} lap_var={rec['lap_var']} "
            f"edge={rec['edge']} std={rec['std']} "
            f"declared={rec['declared_lighting']!r} "
            f"pixel={rec['pixel_lighting']!r} "
            f"class={rec.get('style_class')!r} warm={rec['warm_frac']}"
        )
    return {
        "stats": stats,
        "clusters": clusters,
        "cluster_count": len(clusters),
        "declared_counts": decl_counts,
        "pixel_counts": pix_counts,
        "mismatches": mismatches,
        "fails": fails,
        "passed": not fails,
    }


def assess_object_beat_visual_continuity(
    lines: list[dict[str, Any]],
    scene_paths: list[Path],
) -> dict[str, Any]:
    """
    Shipping rule: an object-only beat must be (b) named in its caption, or
    (a) already in a prior beat of the SAME visual style.
    """
    text_rep = assess_object_beat_continuity(lines)
    stats: list[dict[str, Any] | None] = []
    for path in scene_paths:
        if path and Path(path).is_file():
            stats.append(_linework_stats(Path(path)))
        else:
            stats.append(None)
    fails: list[str] = list(text_rep.get("fails") or [])
    beats: list[dict[str, Any]] = []
    for rec in list(text_rep.get("beats") or []):
        rec = dict(rec)
        cond = str(rec.get("condition") or "")
        scene_i = int(rec.get("scene") or 0)
        idx = scene_i - 1
        if cond == "intentional_objectless":
            rec["shipping"] = "intentional_objectless"
            beats.append(rec)
            continue
        if cond in {"skip_no_concrete_object"}:
            rec["shipping"] = "skip"
            beats.append(rec)
            continue
        if cond == "caption_named":
            rec["shipping"] = "b_caption_named"
            print(
                f"[LOFI object-continuity] SHIP scene={scene_i} "
                f"condition=b_caption_named object={rec.get('object')!r}"
            )
            beats.append(rec)
            continue
        matched = 0
        cur = stats[idx] if 0 <= idx < len(stats) else None
        for ps in rec.get("prior_scenes") or []:
            pidx = int(ps) - 1
            if cur is None or pidx < 0 or pidx >= len(stats) or stats[pidx] is None:
                continue
            if _styles_compatible(stats[pidx], cur):
                matched = int(ps)
                break
        if matched:
            rec["shipping"] = "a_prior_same_style"
            rec["matched_prior"] = matched
            print(
                f"[LOFI object-continuity] SHIP scene={scene_i} "
                f"condition=a_prior_same_style prior={matched} "
                f"object={rec.get('object')!r}"
            )
        else:
            rec["shipping"] = "fail"
            msg = (
                f"object-continuity scene {scene_i} {rec.get('object')!r}: "
                "not named in caption, and prior appearance is not the same "
                "visual style"
            )
            fails.append(msg)
            print(f"[LOFI object-continuity] HOLD {msg}")
        beats.append(rec)
    return {
        "beats": beats,
        "fails": fails,
        "passed": not fails,
    }


def assess_anchor_object_identity(
    image_path: Path,
    *,
    object_name: str,
    shape_parts: tuple[str, ...] | list[str] | None = None,
    caption: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Confirm the episode motif was actually drawn — not a substitute still-life.

    Gemini tag/caption check: the named object and its shape parts must be
    plausibly present. Fail-open only on API/parse errors.
    """
    from core.economic_reel_lofi.visual_identity import (
        anchor_shape_clause,
        anchor_shape_parts,
        needs_umbrella_open_closed,
        umbrella_open_closed_clause,
    )

    name = " ".join((object_name or "").split())
    parts = list(shape_parts or anchor_shape_parts(name))
    want_asym = needs_umbrella_open_closed(caption, name)
    meta: dict[str, Any] = {
        "object_name": name,
        "shape_parts": parts,
        "want_open_closed": want_asym,
        "passed": True,
        "skipped": False,
    }
    if not name:
        meta["skipped"] = True
        meta["skip_reason"] = "no_anchor_name"
        return True, [], meta
    try:
        from google import genai
        from google.genai import types
        from quality.VisualQA_Agent import config as qa_cfg

        if not qa_cfg.GEMINI_API_KEY:
            meta["skipped"] = True
            meta["skip_reason"] = "no_gemini_key"
            return True, [], meta
        client = make_guarded_gemini_client(qa_cfg.GEMINI_API_KEY)
        model_id = qa_cfg.GEMINI_CRITIC_MODEL
        if not str(model_id).startswith("models/"):
            model_id = f"models/{model_id}"
        prompt = (
            "Look at this illustration. Answer JSON only with keys: "
            "object_present (bool), visible_parts (string array), "
            "missing_parts (string array), substitute_object (string, empty if none)"
        )
        if want_asym:
            prompt += (
                ", open_canopy_present (bool), closed_furled_present (bool), "
                "both_closed (bool)"
            )
        prompt += (
            ".\n"
            f"Requested object: {name}. Required shape parts: {', '.join(parts)}.\n"
            "object_present=true only if the requested object is recognizable "
            "by those parts. If a blanket, comforter, cushion, mug, or other "
            "unrelated still-life dominates instead, object_present=false and "
            "name it in substitute_object."
        )
        if want_asym:
            prompt += (
                " open_canopy_present=true only if one umbrella has its canopy "
                "raised (dome up). closed_furled_present=true only if one "
                "umbrella is closed/furled. both_closed=true if every umbrella "
                "is closed with no open canopy."
            )
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        response = guarded_generate_content(
            client,
            model=model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = (getattr(response, "text", None) or "").strip()
        data = json.loads(raw) if raw else {}
        present = bool(data.get("object_present"))
        substitute = str(data.get("substitute_object") or "").strip()
        sub_l = substitute.lower()
        if sub_l and "umbrella" in sub_l and not re.search(
            r"blanket|comforter|cushion|mug|pumpkin", sub_l
        ):
            # "leaning together" is the caption; do not treat contact as a miss.
            present = True
            substitute = ""
        missing = [
            str(x) for x in (data.get("missing_parts") or []) if str(x).strip()
        ]
        meta.update(
            {
                "object_present": present,
                "visible_parts": list(data.get("visible_parts") or []),
                "missing_parts": missing,
                "substitute_object": substitute,
            }
        )
        if want_asym:
            meta["open_canopy_present"] = bool(data.get("open_canopy_present"))
            meta["closed_furled_present"] = bool(data.get("closed_furled_present"))
            meta["both_closed"] = bool(data.get("both_closed"))
        asym_ok = (not want_asym) or (
            meta.get("open_canopy_present")
            and meta.get("closed_furled_present")
            and not meta.get("both_closed")
        )
        if present and not substitute and asym_ok:
            print(
                f"[LOFI anchor-id] PASS {image_path.name} object={name!r} "
                f"parts={meta.get('visible_parts')}"
            )
            return True, [], meta
        meta["passed"] = False
        if present and not substitute and not asym_ok:
            flaw = "OBJECT: need one open canopy-up umbrella leaning on one closed/furled"
            fix = umbrella_open_closed_clause()
        else:
            sub = f" (got {substitute})" if substitute else ""
            flaw = f"OBJECT: requested {name!r} not in pixels{sub}"
            fix = anchor_shape_clause(name, retry=True)
        meta["fix_instructions"] = fix
        print(f"[LOFI anchor-id] REJECT {image_path.name} {flaw}")
        return False, [flaw], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("anchor-id skipped for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        return True, [], meta


def assess_spoken_line_prop_leak(
    image_path: Path,
    *,
    caption: str = "",
    window_allowed: bool = False,
    lamp_allowed: bool = False,
    licensed_object: str = "",
    episode_motif: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """Pixel gate: reject still-life nouns the spoken line did not license.

    Prompt bans are not enough — Flux still invents kettle/mug/plant on a
    tabletop. This reads the PNG. Fail-open only on API/parse errors.
    Pronouns it/this/that resolve to licensed_object / episode_motif first.
    """
    from core.economic_reel_lofi.licensed_objects import (
        licensed_display,
        licensed_object_names,
        licensed_stems_for_beat,
    )
    from core.economic_reel_lofi.visual_identity import (
        INVENTED_STILL_LIFE_STEMS,
        caption_has_object_anaphora,
        spoken_licenses_key_object,
    )

    names = licensed_object_names(
        {
            "licensed_objects": [],
            "key_object": licensed_object,
            "episode_anchor_name": episode_motif,
        }
    )
    # Caller may pass a beat via licensed_object already joined; keep motif.
    motif = licensed_display(
        {
            "licensed_objects": names,
            "key_object": licensed_object,
            "episode_anchor_name": episode_motif,
        }
    ) or str(licensed_object or episode_motif or "").strip()
    spoken_for_model = caption
    if caption_has_object_anaphora(caption) and motif:
        spoken_for_model = f"{caption} (this line's object is {motif})"

    meta: dict[str, Any] = {
        "caption": caption,
        "window_allowed": bool(window_allowed),
        "lamp_allowed": bool(lamp_allowed),
        "licensed_object": motif,
        "passed": True,
        "skipped": False,
        "named_objects": [],
        "leaks": [],
    }
    try:
        from google import genai
        from google.genai import types
        from quality.VisualQA_Agent import config as qa_cfg

        if not qa_cfg.GEMINI_API_KEY:
            meta["skipped"] = True
            meta["skip_reason"] = "no_gemini_key"
            return True, [], meta
        client = make_guarded_gemini_client(qa_cfg.GEMINI_API_KEY)
        model_id = qa_cfg.GEMINI_CRITIC_MODEL
        if not str(model_id).startswith("models/"):
            model_id = f"models/{model_id}"
        banned = ", ".join(INVENTED_STILL_LIFE_STEMS)
        window_rule = (
            "A window pane or window frame is allowed."
            if window_allowed
            else (
                "window_visible=true only if a pane, sash, sill, or handle "
                "is readable. A pool of lamp light on a blank wall is not a window."
            )
        )
        lamp_rule = (
            "A lamp or lampshade is allowed."
            if lamp_allowed
            else "A lamp is a leak unless it is only distant street lighting."
        )
        prompt = (
            "Look at this illustration. JSON only with keys: "
            "named_objects (string array of distinct props you can see), "
            "window_visible (bool), leaks (string array).\n"
            f"Spoken line: {spoken_for_model or '(none)'}\n"
            f"Licensed object for this line: {motif or 'none'}. "
            "Pronouns it/this/that refer to that licensed object.\n"
            "Do not put the licensed object in leaks.\n"
            f"Banned supporting props unless named in that line: {banned}.\n"
            f"{window_rule} {lamp_rule}\n"
            "Generic environment is allowed: wall, floor, sky, pavement, "
            "road, table surface, building mass, streetlamp on a street.\n"
            "Put a banned prop, or a disallowed window, in leaks. "
            "Use short stems: kettle, mug, vase, plant, curtain, window."
        )
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        response = guarded_generate_content(
            client,
            model=model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = (getattr(response, "text", None) or "").strip()
        data = json.loads(raw) if raw else {}
        named = [str(x).strip().lower() for x in (data.get("named_objects") or []) if str(x).strip()]
        leaks = [str(x).strip().lower() for x in (data.get("leaks") or []) if str(x).strip()]
        window_visible = bool(data.get("window_visible"))
        if window_visible and not window_allowed and "window" not in leaks:
            leaks.append("window")
        # Keep only invented-still-life / window leaks. Ignore Gemini extras
        # like "hair" or "sweater".
        keep = []
        for item in leaks:
            stem = re.sub(r"[^a-z]+", "", item.split()[0] if item else "")
            if stem in INVENTED_STILL_LIFE_STEMS or stem == "window":
                keep.append(stem)
            elif any(s in item for s in INVENTED_STILL_LIFE_STEMS) or "window" in item:
                keep.append(item)
        leaks = list(dict.fromkeys(keep))
        licensed_stems = licensed_stems_for_beat(
            {
                "licensed_objects": names,
                "key_object": motif,
            }
        )
        if motif and (
            spoken_licenses_key_object(caption, motif)
            or caption_has_object_anaphora(caption)
        ):
            leaks = [
                item
                for item in leaks
                if not any(
                    s == re.sub(r"[^a-z]+", "", item.split()[0] if item else "")
                    or s in item
                    for s in licensed_stems
                )
            ]
        meta.update(
            {
                "named_objects": named,
                "leaks": leaks,
                "window_visible": window_visible,
            }
        )
        if not leaks:
            print(
                f"[LOFI spoken-prop] PASS {image_path.name} "
                f"objects={named}"
            )
            return True, [], meta
        meta["passed"] = False
        flaw = f"SPOKEN-PROP: unspoken objects in pixels: {', '.join(leaks)}"
        meta["fix_instructions"] = (
            "Redraw with only the licensed subject. Empty floor or open sky. "
            "If a figure: side profile toward off-frame light. One subject."
        )
        print(f"[LOFI spoken-prop] REJECT {image_path.name} {flaw}")
        return False, [flaw], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("spoken-prop skipped for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        return True, [], meta


# Interior/threshold archetypes where Flux's room-prior draws a background
# window even after the prompt-text leak is closed. Not a global relaxation.
BACKGROUND_WINDOW_ARCHETYPES: frozenset[str] = frozenset(
    {"interior-domestic", "public-interior", "threshold"}
)
# Same compact-blob fail line as assess_default_object_intrusion.
WINDOW_BG_BBOX_MAX: float = 0.08
WINDOW_BG_RATIO_MAX: float = 0.45
_WALL_PLANE_SETTING_RE = re.compile(
    r"\b(painted wall|wall plane|hallway|apartment|interior|threshold|"
    r"indoor|plain wall)\b",
    re.I,
)


def _window_leak_split(leaks: list[Any] | None) -> tuple[list[str], list[str]]:
    windows: list[str] = []
    other: list[str] = []
    for item in leaks or []:
        raw = str(item or "").strip().lower()
        if not raw:
            continue
        stem = re.sub(r"[^a-z]+", "", raw.split()[0])
        if stem == "window" or "window" in raw:
            windows.append(raw)
        else:
            other.append(raw)
    return windows, other


def interior_threshold_window_setting(beat: dict[str, Any] | None) -> bool:
    """True for interior/threshold archetypes (or wall-plane reframes of them)."""
    if not isinstance(beat, dict):
        return False
    arch = str(beat.get("setting_archetype") or "").strip().lower()
    if arch in BACKGROUND_WINDOW_ARCHETYPES:
        return True
    setting = str(beat.get("setting") or "")
    return bool(_WALL_PLANE_SETTING_RE.search(setting))


def background_window_exception_applies(
    *,
    leaks: list[Any] | None,
    beat: dict[str, Any] | None,
    object_gate: dict[str, Any] | None = None,
) -> bool:
    """License a non-dominant background window as the *only* spoken_prop leak.

    Does not apply when window is spoken/licensed, composition-dominant, or
    co-occurs with another leak. Scoped to interior/threshold settings.
    """
    from core.economic_reel_lofi.visual_identity import spoken_line_licenses_window

    windows, other = _window_leak_split(leaks)
    if not windows or other:
        return False
    if not isinstance(beat, dict):
        return False
    if beat.get("window_primary"):
        return False
    if spoken_line_licenses_window(beat):
        return False
    ko = str(beat.get("key_object") or "").strip().lower()
    if re.search(r"\bwindows?\b", ko):
        return False
    if not interior_threshold_window_setting(beat):
        return False
    gate = object_gate if isinstance(object_gate, dict) else {}
    bbox = float(gate.get("intruder_bbox_frac") or 0.0)
    ratio = float(gate.get("ratio") or 0.0)
    if bbox >= WINDOW_BG_BBOX_MAX or ratio >= WINDOW_BG_RATIO_MAX:
        return False
    return True


def apply_background_window_exception(
    prop_meta: dict[str, Any],
    beat: dict[str, Any] | None,
    object_gate: dict[str, Any] | None = None,
) -> bool:
    """Mutate spoken_prop meta in place. Returns True if the exception fired."""
    if not isinstance(prop_meta, dict):
        return False
    leaks = list(prop_meta.get("leaks") or [])
    if not background_window_exception_applies(
        leaks=leaks, beat=beat, object_gate=object_gate
    ):
        return False
    windows, _other = _window_leak_split(leaks)
    kept = [x for x in leaks if str(x).strip().lower() not in windows]
    prop_meta["leaks"] = kept
    prop_meta["background_window_licensed"] = True
    prop_meta["passed"] = not kept
    gate = object_gate if isinstance(object_gate, dict) else {}
    print(
        f"[LOFI spoken-prop] background-window licensed "
        f"scene={beat.get('scene') if isinstance(beat, dict) else '?'} "
        f"arch={beat.get('setting_archetype') if isinstance(beat, dict) else ''} "
        f"bbox={gate.get('intruder_bbox_frac')} ratio={gate.get('ratio')}"
    )
    return True


def _strip_window_only_spoken_prop_flaws(
    flaws: list[Any] | None,
    remaining_leaks: list[Any] | None,
) -> list[str]:
    out = [str(f) for f in (flaws or []) if not str(f).startswith("SPOKEN-PROP:")]
    leaks = [str(x) for x in (remaining_leaks or []) if str(x).strip()]
    if leaks:
        out.append(f"SPOKEN-PROP: unspoken objects in pixels: {', '.join(leaks)}")
    return out


def rescore_episode_background_window(episode: dict[str, Any]) -> dict[str, Any]:
    """Re-apply the background-window exception to existing gates. No image calls."""
    script = episode.get("script") if isinstance(episode.get("script"), dict) else {}
    lines = list(script.get("lines") or [])
    gates = list(episode.get("object_gate_by_scene") or [])
    report: list[dict[str, Any]] = []
    for i, row in enumerate(lines):
        if not isinstance(row, dict):
            continue
        gate = dict(row.get("default_object_gate") or {})
        if not gate and i < len(gates) and isinstance(gates[i], dict):
            gate = dict(gates[i])
        prop = dict(gate.get("spoken_prop") or {})
        before_leaks = list(prop.get("leaks") or [])
        applied = apply_background_window_exception(prop, row, gate)
        flaws = _strip_window_only_spoken_prop_flaws(
            list(gate.get("qa_flaws") or []),
            list(prop.get("leaks") or []) if applied else before_leaks,
        ) if applied else list(gate.get("qa_flaws") or [])
        if applied:
            gate["spoken_prop"] = prop
            gate["qa_flaws"] = flaws
            still_fail = bool(flaws)
            gate["qa_passed"] = not still_fail
            if not still_fail:
                gate["image_ok"] = True
                gate["passed"] = True
            attempts = list(gate.get("attempts") or [])
            if attempts and isinstance(attempts[-1], dict):
                last = dict(attempts[-1])
                last["spoken_prop"] = prop
                last["qa_flaws"] = flaws
                last["qa_passed"] = not still_fail
                attempts[-1] = last
                gate["attempts"] = attempts
            row["default_object_gate"] = gate
            while len(gates) <= i:
                gates.append({})
            gates[i] = gate
        scene_n = int(row.get("scene") or i + 1)
        report.append(
            {
                "scene": scene_n,
                "applied": bool(applied),
                "qa_passed": bool(gate.get("qa_passed")),
                "leaks": list((gate.get("spoken_prop") or {}).get("leaks") or []),
                "qa_flaws": list(gate.get("qa_flaws") or []),
            }
        )
    leftover = []
    for g in gates:
        if not isinstance(g, dict) or g.get("qa_passed"):
            continue
        scene_g = int(g.get("scene") or 0)
        leftover.append(
            f"scene_{scene_g}: {'; '.join(g.get('qa_flaws') or []) or 'qa failed'}"
        )
    episode["object_gate_by_scene"] = gates
    if isinstance(script, dict):
        script["lines"] = lines
        episode["script"] = script
    episode["visual_qa_flags"] = leftover
    episode["manual_review"] = bool(leftover)
    return {"beats": report, "leftover": leftover}


def assess_anchor_painterly_lock(
    image_path: Path,
    *,
    subject_type: str = "",
    anchor_beat: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Anchor object_focus stills must match the episode's painterly look.

    Isolation still-life often collapses to flat ink/woodcut. Fail that
    before style_ok can pass. Fail-open only on API/parse errors.
    """
    st = (subject_type or "").strip().lower().replace(" ", "_")
    meta: dict[str, Any] = {
        "subject_type": st,
        "passed": True,
        "skipped": False,
    }
    if st != "object_focus" or str(anchor_beat or "") not in {"introduce", "callback"}:
        meta["skipped"] = True
        meta["skip_reason"] = "not_anchor_object_focus"
        return True, [], meta
    try:
        from google import genai
        from google.genai import types
        from quality.VisualQA_Agent import config as qa_cfg

        if not qa_cfg.GEMINI_API_KEY:
            meta["skipped"] = True
            meta["skip_reason"] = "no_gemini_key"
            return True, [], meta
        client = make_guarded_gemini_client(qa_cfg.GEMINI_API_KEY)
        model_id = qa_cfg.GEMINI_CRITIC_MODEL
        if not str(model_id).startswith("models/"):
            model_id = f"models/{model_id}"
        prompt = (
            "Look at this illustration. JSON only: "
            "style_kind (flat_gouache_riso or painterly_photographic or "
            "flat_ink_woodcut), note (string).\n"
            "flat_gouache_riso = flat printed color blocks, paper grain, "
            "hard edges, a readable environment, not a photograph.\n"
            "painterly_photographic = grain plus photographic depth, lens "
            "falloff, or cinematic sunset glow — still a painted scene.\n"
            "flat_ink_woodcut = bold outlines on a blank or cream field, "
            "poster diagram, isolated linework, no environment."
        )
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        response = guarded_generate_content(
            client,
            model=model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = (getattr(response, "text", None) or "").strip()
        data = json.loads(raw) if raw else {}
        kind = str(data.get("style_kind") or "").strip().lower()
        meta["style_kind"] = kind
        meta["note"] = str(data.get("note") or "")
        if kind in {"flat_gouache_riso", "painterly_photographic"}:
            print(f"[LOFI painterly] PASS {image_path.name} kind={kind}")
            return True, [], meta
        meta["passed"] = False
        flaw = "STYLE: flat ink/woodcut on an empty field; need printed environment"
        meta["fix_instructions"] = (
            "Same flat gouache risograph atmosphere as a furnished "
            "room or rainy street — paper tooth, hard color blocks, "
            "printed dusk. Not a photograph, not isolated "
            "ink-diagram linework on blank paper."
        )
        print(f"[LOFI painterly] REJECT {image_path.name} {flaw}")
        return False, [flaw], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("painterly skipped for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        return True, [], meta


def assess_couple_bed_pose(
    image_path: Path,
    *,
    setting: str = "",
    key_object: str = "",
    subject_type: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """Bed scenes only: fail if figures stand on the mattress."""
    from core.economic_reel_lofi.visual_identity import couple_bed_pose_clause

    st = (subject_type or "").strip().lower().replace(" ", "_")
    meta: dict[str, Any] = {
        "subject_type": st,
        "passed": True,
        "skipped": False,
    }
    clause = couple_bed_pose_clause(setting, key_object, st)
    if not clause:
        meta["skipped"] = True
        meta["skip_reason"] = "not_couple_bed"
        return True, [], meta
    try:
        from google import genai
        from google.genai import types
        from quality.VisualQA_Agent import config as qa_cfg

        if not qa_cfg.GEMINI_API_KEY:
            meta["skipped"] = True
            meta["skip_reason"] = "no_gemini_key"
            return True, [], meta
        client = make_guarded_gemini_client(qa_cfg.GEMINI_API_KEY)
        model_id = qa_cfg.GEMINI_CRITIC_MODEL
        if not str(model_id).startswith("models/"):
            model_id = f"models/{model_id}"
        prompt = (
            "Look at this illustration. JSON only: "
            "figures_on_mattress (bool), pose_ok (bool), note (string).\n"
            "figures_on_mattress=true if ANY of these are true: a person stands "
            "on the mattress or on top of the bed; feet are planted in rumpled "
            "bedding; the bed fills the lower third and figures rise out of the "
            "covers with no visible floor under their feet; or it is ambiguous "
            "whether they stand on the mattress or just behind it.\n"
            "pose_ok=true ONLY if the floor is clearly visible under their feet "
            "beside the bed, OR they sit on the edge with feet on the floor. "
            "If unsure, set figures_on_mattress=true and pose_ok=false."
        )
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        response = guarded_generate_content(
            client,
            model=model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = (getattr(response, "text", None) or "").strip()
        data = json.loads(raw) if raw else {}
        on_bed = bool(data.get("figures_on_mattress"))
        pose_ok = bool(data.get("pose_ok")) if "pose_ok" in data else (not on_bed)
        meta.update(
            {
                "figures_on_mattress": on_bed,
                "pose_ok": pose_ok,
                "note": str(data.get("note") or ""),
            }
        )
        if on_bed or not pose_ok:
            meta["passed"] = False
            flaw = "POSE: figures standing on the mattress / on top of the bed"
            meta["fix_instructions"] = clause
            print(f"[LOFI bed-pose] REJECT {image_path.name} {flaw}")
            return False, [flaw], meta
        print(f"[LOFI bed-pose] PASS {image_path.name}")
        return True, [], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("bed-pose skipped for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        return True, [], meta


def assess_hook_still(
    image_path: Path,
    *,
    scene: int = 1,
    subject_type: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """Highest-priority visual gate: scene 1 must hold attention."""
    meta: dict[str, Any] = {
        "scene": scene,
        "subject_type": subject_type,
        "passed": True,
        "skipped": False,
    }
    if int(scene or 1) != 1:
        meta["skipped"] = True
        meta["skip_reason"] = "not_hook"
        return True, [], meta
    try:
        from google import genai
        from google.genai import types
        from quality.VisualQA_Agent import config as qa_cfg

        if not qa_cfg.GEMINI_API_KEY:
            meta["skipped"] = True
            meta["skip_reason"] = "no_gemini_key"
            return True, [], meta
        client = make_guarded_gemini_client(qa_cfg.GEMINI_API_KEY)
        model_id = qa_cfg.GEMINI_CRITIC_MODEL
        if not str(model_id).startswith("models/"):
            model_id = f"models/{model_id}"
        prompt = (
            "Look at this illustration, the first frame of a 9:16 reel. JSON only: "
            "has_character (bool), high_contrast (bool), empty_still_life (bool), "
            "striking (bool), note (string).\n"
            "has_character = a human figure or silhouette is in frame. "
            "empty_still_life = only furniture/objects, no person. "
            "high_contrast = strong light (sunset, rain, lamp, dawn backlight). "
            "striking = would hold a thumb-stop in the first 3 seconds. "
            "FAIL if empty_still_life or not has_character or not high_contrast."
        )
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        response = guarded_generate_content(
            client,
            model=model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = (getattr(response, "text", None) or "").strip()
        data = json.loads(raw) if raw else {}
        meta.update(
            {
                "has_character": bool(data.get("has_character")),
                "high_contrast": bool(data.get("high_contrast")),
                "empty_still_life": bool(data.get("empty_still_life")),
                "striking": bool(data.get("striking")),
                "note": str(data.get("note") or ""),
            }
        )
        failed = (
            meta["empty_still_life"]
            or not meta["has_character"]
            or not meta["high_contrast"]
        )
        if failed:
            meta["passed"] = False
            flaw = (
                "HOOK: scene 1 is flat or object-only — need a character in "
                "high-contrast light (sunset/rain/lamp/dawn)"
            )
            meta["fix_instructions"] = (
                "Put a silhouette or figure in strong backlight, rain, or lamp "
                "glow. No empty bed or object-only still life."
            )
            print(f"[LOFI hook] REJECT {image_path.name} {flaw}")
            return False, [flaw], meta
        print(
            f"[LOFI hook] PASS {image_path.name} "
            f"character={int(meta['has_character'])} "
            f"contrast={int(meta['high_contrast'])} "
            f"striking={int(meta['striking'])}"
        )
        return True, [], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("hook still skipped for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        return True, [], meta


def _is_critic_infra_error(exc: BaseException) -> bool:
    """Retired/missing Gemini SKU — not a garment verdict."""
    msg = str(exc).upper()
    return any(
        tok in msg
        for tok in (
            "404",
            "NOT_FOUND",
            "NO LONGER AVAILABLE",
            "NOT AVAILABLE TO NEW USERS",
        )
    )


COVERAGE_FLAW = "COVERAGE: incomplete garment / exposed skin on a human figure"
COVERAGE_FIX = (
    "fully covered by dress fabric, back and shoulders clothed, "
    "chest and torso clothed, high-neck sweater to the jaw"
)
_HUMAN_FIGURE_TYPES = frozenset({"woman", "man", "couple", "silhouette"})


def _is_human_figure_scene(subject_type: str, close_variant: str = "") -> bool:
    st = str(subject_type or "").strip().lower().replace(" ", "_")
    cv = str(close_variant or "").strip().lower()
    if cv == "eye_close":
        return False
    return st in _HUMAN_FIGURE_TYPES or cv == "portrait_close"


def _coverage_log_path() -> Path:
    return (
        Path(__file__).resolve().parent / "store" / "lofi_coverage_flags.jsonl"
    )


def log_coverage_flag(record: dict[str, Any]) -> None:
    """Append one coverage rejection for prompt/CFG/seed pattern tracking."""
    path = _coverage_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[LOFI coverage] logged {path.name} scene={row.get('scene')} "
        f"cfg={row.get('guidance_scale')} seed={row.get('seed')}"
    )


def assess_garment_coverage(
    image_path: Path,
    *,
    subject_type: str = "",
    close_variant: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Hard post-gen gate on every human-figure still. Fail-closed: a skipped
    check never reaches review as an un-flagged pass.
    """
    meta: dict[str, Any] = {
        "subject_type": subject_type,
        "close_variant": close_variant,
        "passed": True,
        "skipped": False,
    }
    if not _is_human_figure_scene(subject_type, close_variant):
        meta["skipped"] = True
        meta["skip_reason"] = "not_human_figure"
        return True, [], meta
    try:
        from google import genai
        from google.genai import types
        from quality.VisualQA_Agent import config as qa_cfg

        if not qa_cfg.GEMINI_API_KEY:
            meta["skipped"] = True
            meta["skip_reason"] = "no_gemini_key"
            meta["passed"] = False
            print(f"[LOFI coverage] FAIL-CLOSED {image_path.name} no Gemini key")
            return False, [COVERAGE_FLAW + " (coverage check unavailable)"], meta
        client = make_guarded_gemini_client(qa_cfg.GEMINI_API_KEY)
        model_id = qa_cfg.GEMINI_CRITIC_MODEL
        if not str(model_id).startswith("models/"):
            model_id = f"models/{model_id}"
        prompt = (
            "Look at this illustration. JSON only: "
            "human_figure (bool), incomplete_garment (bool), "
            "exposed_skin_beyond_face_hands (bool), coverage_ok (bool), note (string).\n"
            "human_figure = a person or silhouette is in frame.\n"
            "incomplete_garment = clothing does not fully cover back, shoulders, "
            "chest, or torso (bare back, off-shoulder, cleavage, nape-to-waist skin).\n"
            "exposed_skin_beyond_face_hands = torso/back/shoulders/chest skin is visible.\n"
            "coverage_ok = fully clothed in dress/sweater/coat fabric; face and hands "
            "may show skin. Silhouettes still fail if the body reads unclothed/nude.\n"
            "FAIL if human_figure and (incomplete_garment or exposed_skin_beyond_face_hands)."
        )
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        response = guarded_generate_content(
            client,
            model=model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = (getattr(response, "text", None) or "").strip()
        data = json.loads(raw) if raw else {}
        meta.update(
            {
                "human_figure": bool(data.get("human_figure")),
                "incomplete_garment": bool(data.get("incomplete_garment")),
                "exposed_skin_beyond_face_hands": bool(
                    data.get("exposed_skin_beyond_face_hands")
                ),
                "coverage_ok": bool(data.get("coverage_ok")),
                "note": str(data.get("note") or ""),
            }
        )
        failed = bool(meta["human_figure"]) and (
            meta["incomplete_garment"]
            or meta["exposed_skin_beyond_face_hands"]
            or not meta["coverage_ok"]
        )
        if failed:
            meta["passed"] = False
            meta["fix_instructions"] = COVERAGE_FIX
            print(
                f"[LOFI coverage] REJECT {image_path.name} "
                f"note={meta.get('note')!r}"
            )
            return False, [COVERAGE_FLAW], meta
        print(f"[LOFI coverage] PASS {image_path.name}")
        return True, [], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("coverage check failed-closed for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        if _is_critic_infra_error(exc):
            # Retired/missing critic SKU must not burn a Flux retry.
            print(
                f"[LOFI coverage] SKIP infra {image_path.name} "
                f"({exc.__class__.__name__})"
            )
            return True, [], meta
        meta["passed"] = False
        return False, [COVERAGE_FLAW + " (coverage check error)"], meta


def assess_portrait_close_style(
    image_path: Path,
    *,
    close_variant: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """Portrait/eye close must stay painterly — no photoreal skin or eyes."""
    meta: dict[str, Any] = {
        "close_variant": close_variant,
        "passed": True,
        "skipped": False,
    }
    if str(close_variant or "") not in {"portrait_close", "eye_close"}:
        meta["skipped"] = True
        meta["skip_reason"] = "not_portrait_or_eye_close"
        return True, [], meta
    try:
        from google import genai
        from google.genai import types
        from quality.VisualQA_Agent import config as qa_cfg

        if not qa_cfg.GEMINI_API_KEY:
            meta["skipped"] = True
            meta["skip_reason"] = "no_gemini_key"
            return True, [], meta
        client = make_guarded_gemini_client(qa_cfg.GEMINI_API_KEY)
        model_id = qa_cfg.GEMINI_CRITIC_MODEL
        if not str(model_id).startswith("models/"):
            model_id = f"models/{model_id}"
        prompt = (
            "Look at this illustration. JSON only: "
            "photoreal_skin (bool), photoreal_eyes (bool), painterly_match (bool), "
            "full_face_visible (bool), eye_macro_only (bool), note (string).\n"
            "painterly_match = ink, flat printed color, paper grain — same as a "
            "risograph poster, not a camera photo. "
            "photoreal_skin / photoreal_eyes = rendered photographic skin or glossy "
            "camera eyes. "
            "full_face_visible = eyes, nose, mouth, and some hair or shoulder are "
            "all in frame (classic head-and-shoulders or waist-up portrait). "
            "eye_macro_only = the crop is only eyes/brow with no nose or mouth. "
            "FAIL if photoreal_skin or photoreal_eyes or not painterly_match. "
            "If this is a portrait_close, also FAIL if eye_macro_only or not "
            "full_face_visible."
        )
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        response = guarded_generate_content(
            client,
            model=model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = (getattr(response, "text", None) or "").strip()
        data = json.loads(raw) if raw else {}
        meta.update(
            {
                "photoreal_skin": bool(data.get("photoreal_skin")),
                "photoreal_eyes": bool(data.get("photoreal_eyes")),
                "painterly_match": bool(data.get("painterly_match")),
                "full_face_visible": (
                    True
                    if "full_face_visible" not in data
                    else bool(data.get("full_face_visible"))
                ),
                "eye_macro_only": (
                    False
                    if "eye_macro_only" not in data
                    else bool(data.get("eye_macro_only"))
                ),
                "note": str(data.get("note") or ""),
            }
        )
        framing_fail = False
        if str(close_variant or "") == "portrait_close":
            framing_fail = meta["eye_macro_only"] or not meta["full_face_visible"]
        if (
            meta["photoreal_skin"]
            or meta["photoreal_eyes"]
            or not meta["painterly_match"]
            or framing_fail
        ):
            meta["passed"] = False
            if framing_fail:
                flaw = (
                    "FRAME: portrait_close is an eye/brow macro — need head and "
                    "shoulders, full face visible (eyes, nose, mouth)"
                )
                meta["fix_instructions"] = (
                    "Redraw as a medium close-up portrait, head and shoulders in "
                    "frame, full face visible, painterly illustration — not an "
                    "extreme macro crop on eyes alone."
                )
            else:
                flaw = (
                    "STYLE: portrait_close/eye_close mixed photoreal skin or eyes "
                    "into a painterly episode"
                )
                meta["fix_instructions"] = (
                    "Redraw as risograph illustration: ink line, flat printed color, "
                    "paper grain. No photoreal skin, no camera eyes."
                )
            print(f"[LOFI portrait-style] REJECT {image_path.name} {flaw}")
            return False, [flaw], meta
        print(f"[LOFI portrait-style] PASS {image_path.name} variant={close_variant}")
        return True, [], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("portrait style skipped for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        return True, [], meta


def assess_eye_close_crop(
    image_path: Path,
    *,
    close_variant: str = "",
) -> tuple[bool, list[str], dict[str, Any]]:
    """Reject eye_close stills that leak a nose or mouth."""
    meta: dict[str, Any] = {
        "close_variant": close_variant,
        "passed": True,
        "skipped": False,
    }
    if str(close_variant or "") != "eye_close":
        meta["skipped"] = True
        meta["skip_reason"] = "not_eye_close"
        return True, [], meta
    try:
        from google import genai
        from google.genai import types
        from quality.VisualQA_Agent import config as qa_cfg

        if not qa_cfg.GEMINI_API_KEY:
            meta["skipped"] = True
            meta["skip_reason"] = "no_gemini_key"
            return True, [], meta
        client = make_guarded_gemini_client(qa_cfg.GEMINI_API_KEY)
        model_id = qa_cfg.GEMINI_CRITIC_MODEL
        if not str(model_id).startswith("models/"):
            model_id = f"models/{model_id}"
        prompt = (
            "Look at this illustration. JSON only: "
            "eyes_present (bool), nose_visible (bool), mouth_visible (bool), "
            "chin_visible (bool), full_face (bool), note (string).\n"
            "This must be an extreme crop of eyes/brow/hair only. "
            "nose_visible or mouth_visible or chin_visible or full_face = fail."
        )
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        response = guarded_generate_content(
            client,
            model=model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = (getattr(response, "text", None) or "").strip()
        data = json.loads(raw) if raw else {}
        meta.update(
            {
                "eyes_present": bool(data.get("eyes_present")),
                "nose_visible": bool(data.get("nose_visible")),
                "mouth_visible": bool(data.get("mouth_visible")),
                "chin_visible": bool(data.get("chin_visible")),
                "full_face": bool(data.get("full_face")),
                "note": str(data.get("note") or ""),
            }
        )
        leaked = (
            meta["nose_visible"]
            or meta["mouth_visible"]
            or meta["chin_visible"]
            or meta["full_face"]
            or not meta["eyes_present"]
        )
        if leaked:
            meta["passed"] = False
            flaw = "FACE: eye_close leaked nose/mouth/full face or missing eyes"
            meta["fix_instructions"] = (
                "Extreme close on eyes only, cropped above the nose bridge. "
                "No nose, no mouth, no chin, no lower face."
            )
            print(f"[LOFI eye-close] REJECT {image_path.name} {flaw}")
            return False, [flaw], meta
        print(f"[LOFI eye-close] PASS {image_path.name}")
        return True, [], meta
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("eye-close skipped for %s (%s)", image_path.name, exc)
        meta["skipped"] = True
        meta["skip_reason"] = f"error:{exc}"
        return True, [], meta


def _qa_scene_image(
    image_path: Path,
    visual_prompt: str,
    *,
    subject_type: str = "",
    key_object: str = "",
    licensed_objects: list[str] | None = None,
    prior_fail_criteria: list[str] | None = None,
    style_profile: str | None = None,
    close_variant: str | None = None,
) -> tuple[bool, list[str], str]:
    """
    Run VisualQA lofi_economic profile when available.
    Soft-fail open if Gemini/critic unavailable (log + accept).
    Always run local linework/complexity check (rejects flat graphics).
    Returns (passed, flaws, critic_fix_instructions).
    """
    del visual_prompt
    struct_ok, struct_flaws = _assess_linework_complexity(image_path)
    fix = ""
    try:
        from quality.VisualQA_Agent.channel_rag import get_channel_rules, set_channel_context
        from quality.VisualQA_Agent.visual_critic import evaluate_image

        set_channel_context("lofi_economic")
        rules = get_channel_rules("lofi_economic")
        from core.economic_reel_lofi.licensed_objects import licensed_display

        qa_obj = licensed_display({"licensed_objects": licensed_objects or [], "key_object": key_object}) or key_object
        verdict = evaluate_image(
            image_path,
            channel_name="lofi_economic",
            rules=rules,
            quality_threshold=6.0,
            requested_subject=subject_type or None,
            requested_object=qa_obj or None,
            licensed_objects=list(licensed_objects or []),
            prior_fail_criteria=prior_fail_criteria,
            style_profile=style_profile,
            close_variant=close_variant,
        )
        critic_ok = bool(verdict.passed)
        critic_flaws = list(verdict.flaws or [])
        fix = str(verdict.fix_instructions or "").strip()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("VisualQA skipped for %s (%s)", image_path.name, exc)
        critic_ok = True
        critic_flaws = []
        try:
            from PIL import Image as PILImage

            im = PILImage.open(image_path)
            w, h = im.size
            if h < w:
                critic_flaws.append("aspect ratio not vertical")
                critic_ok = False
            if w < 512 or h < 512:
                critic_flaws.append("resolution too low for reels")
                critic_ok = False
        except Exception:  # noqa: BLE001
            pass
    flaws = list(struct_flaws)
    if not critic_ok:
        flaws.extend(critic_flaws)
    return (struct_ok and critic_ok and not flaws), flaws, fix


def apply_ad_hoc_guidance(visual_prompt: str, reason: str) -> str:
    """Append one-off prompt guidance. Not persisted to the RAG / object bank."""
    extra = " ".join((reason or "").split())
    base = " ".join((visual_prompt or "").split())
    if not extra:
        return base
    if extra.lower() in base.lower():
        return base
    return f"{base} {extra}"


def _retry_prompt_extras(prompt_i: str, attempt: int, last_fix: str) -> str:
    """Same licensed noun; vary angle/distance/light. Never name banned props.

    Linework guard is style-module line-quality only (outlines / grain /
    contrast). It must never re-inject clutter content such as
    'detailed interior objects'.
    """
    from core.economic_reel_lofi.style_modules import get_active_style
    from core.economic_reel_lofi.visual_identity import (
        INVENTED_STILL_LIFE_STEMS,
        retry_variation_clause,
    )

    vary = retry_variation_clause(attempt - 1)
    if vary:
        prompt_i = f"{prompt_i} {vary}"
    guard = str(get_active_style().linework_guard or "").strip()
    if re.search(r"interior objects|clutter|still[- ]life", guard, re.I):
        guard = "fine black ink outlines, paper grain, hard-edged color planes"
    fix = str(last_fix or "")
    low = fix.lower()
    if any(s in low for s in INVENTED_STILL_LIFE_STEMS) or "window" in low:
        fix = ""
    extras = [p for p in (guard, fix) if p]
    if extras:
        prompt_i = f"{prompt_i} {' '.join(extras)}"
    from core.economic_reel_lofi.visual_identity import _strip_positive_negation

    return _strip_positive_negation(prompt_i)


def generate_and_qa_scene(
    row: dict[str, Any],
    out_img: Path,
    *,
    attempt_budget: int | None = None,
    extra_prompt: str = "",
    mood: dict[str, Any] | None = None,
    generate_fn: Any | None = None,
    assemble_fn: Any | None = None,
) -> tuple[bool, dict[str, Any], int, int]:
    """
    Generate one beat still and run critic + uniq16 style_gate + INTRUDER.

    Writes pixels to ``out_img`` (last attempt kept on failure). Returns
    ``(ok, last_gate, n_image_calls, n_critic_calls)``.

    ``generate_fn`` / ``assemble_fn`` default to the Schnell live path.
    Flux Dev callers pass generate_scene_image_dev + assemble_v2_prompt_dev.
    """
    gen_image = generate_fn or generate_scene_image
    scene_i = int(row.get("scene") or 1)
    visual = str(row.get("visual_prompt") or "")
    extra = " ".join((extra_prompt or "").split())
    subject_type = str(row.get("subject_type") or "").strip()
    key_object = str(row.get("key_object") or "").strip()
    st_l = subject_type.strip().lower().replace(" ", "_")
    mood_meta = dict(mood or {
        "id": str(row.get("riso_id") or f"scene_{scene_i}"),
        "lighting": "from_riso_prompt",
        "palette": row.get("riso_palette"),
        "shadow": lofi_cfg.DUOTONE_SHADOW,
        "highlight": lofi_cfg.DUOTONE_HIGHLIGHT,
    })
    from core.economic_reel_lofi.licensed_objects import licensed_display, licensed_object_names

    mood_meta["licensed_object"] = licensed_display(row) or key_object
    mood_meta["not_in_frame"] = list(row.get("not_in_frame") or [])
    if str(row.get("negative_prompt") or "").strip():
        mood_meta["negative_prompt"] = str(row.get("negative_prompt")).strip()
    licensed_names = licensed_object_names(row)
    n_attempts = lofi_cfg.clamp_attempt_budget(attempt_budget)
    print(
        f"[LOFI attempts] scene={scene_i} hard_cap={n_attempts} "
        f"(IMAGE_ATTEMPTS_PER_BEAT={lofi_cfg.IMAGE_ATTEMPTS_PER_BEAT})"
    )

    ok_img = False
    last_flaws: list[str] = []
    last_fix = ""
    last_gate: dict[str, Any] = {
        "scene": scene_i,
        "subject_type": subject_type,
        "key_object": key_object,
        "passed": True,
        "skipped": True,
    }
    last_intruder = False
    focus_step = 0
    scene_attempts: list[dict[str, Any]] = []
    n_image_calls = 0
    n_critic_calls = 0
    spoken_fallback: Path | None = None

    for attempt in range(1, n_attempts + 1):
        prompt_i = apply_ad_hoc_guidance(visual, extra)
        if assemble_fn is not None:
            escalate = last_intruder or attempt > 1
            if (
                escalate
                and str(row.get("anchor_beat") or "") not in {
                    "introduce",
                    "callback",
                }
                and str(row.get("close_variant") or "") != "eye_close"
            ):
                focus_step = min(focus_step + 1, 2)
            # Attempt 1 keeps the Gate-2-approved prompt. Reassemble only on
            # retry/escalate or if Stage 3 never stamped visual_prompt.
            if escalate or not visual.strip():
                rebuilt = assemble_fn(row, focus_step=focus_step)
                row["visual_prompt"] = rebuilt
                visual = rebuilt
                prompt_i = apply_ad_hoc_guidance(rebuilt, extra)
                print(
                    f"[LOFI framing] scene={scene_i} attempt={attempt} "
                    f"step={focus_step} kind={row.get('object_focus_framing')} "
                    f"escalate_intruder={int(last_intruder)}"
                )
                if attempt > 1:
                    prompt_i = _retry_prompt_extras(prompt_i, attempt, last_fix)
        elif st_l in {"object_focus", "silhouette"}:
            if last_intruder:
                focus_step = min(focus_step + 1, 2)
            from core.economic_reel_lofi.visual_identity import (
                assemble_v2_prompt,
            )

            assembler = assemble_v2_prompt
            rebuilt = assembler(row, focus_step=focus_step)
            row["visual_prompt"] = rebuilt
            prompt_i = apply_ad_hoc_guidance(rebuilt, extra)
            print(
                f"[LOFI framing] scene={scene_i} attempt={attempt} "
                f"step={focus_step} kind={row.get('object_focus_framing')} "
                f"escalate_intruder={int(last_intruder)}"
            )
            if attempt > 1:
                prompt_i = _retry_prompt_extras(prompt_i, attempt, last_fix)
        elif attempt > 1:
            prompt_i = _retry_prompt_extras(prompt_i, attempt, last_fix)
        lighting_guard = " ".join(
            str(row.get("lighting_prompt_guard") or "").split()
        )
        if lighting_guard and lighting_guard.lower() not in prompt_i.lower():
            prompt_i = f"{prompt_i} {lighting_guard}"
        from core.economic_reel_lofi.licensed_objects import scrub_assembled_prompt

        if str(row.get("visual_source") or "") == "writer_single_pass":
            print(
                f"[LOFI prompt] scene={scene_i} preserving programmatic "
                "Risograph location and character"
            )
        else:
            prompt_i = scrub_assembled_prompt(prompt_i, row)
        try:
            _, mood_meta = gen_image(
                prompt_i,
                out_img,
                mood=mood_meta,
                verbatim=True,
            )
            n_image_calls += 1
            if str(row.get("close_variant") or "") == "eye_close":
                from core.economic_reel_lofi.visual_identity import (
                    crop_eye_close_still,
                )

                crop_eye_close_still(out_img)
            lofi_cfg.write_still_style_sidecar(
                out_img, run_id=out_img.parent.name, reused=False
            )
        except Exception as exc:  # noqa: BLE001
            last_flaws = [f"image gen failed: {exc}"]
            _LOG.warning("scene %s gen attempt %s failed: %s", scene_i, attempt, exc)
            continue
        n_critic_calls += 1
        qa_object = key_object
        if str(row.get("anchor_beat") or "") in {"introduce", "callback"}:
            qa_object = str(row.get("episode_anchor_name") or key_object).strip() or key_object
        passed, flaws, last_fix = _qa_scene_image(
            out_img, visual,
            subject_type=subject_type,
            key_object=qa_object,
            licensed_objects=licensed_names,
            prior_fail_criteria=[
                f for f in last_flaws
                if f and not str(f).startswith("image gen failed")
            ] if attempt > 1 and last_flaws else None,
            style_profile=str(row.get("visual_identity_profile") or "") or None,
            close_variant=str(row.get("close_variant") or "") or None,
        )
        if attempt > 1 and last_flaws:
            hold = [
                f for f in last_flaws
                if f and not str(f).startswith("image gen failed")
            ]
            print(
                f"[LOFI retry-hold] scene={scene_i} attempt={attempt} "
                f"same_criteria={hold}"
            )
        gate_ok, gate_flaws, gate_meta = assess_default_object_intrusion(
            out_img,
            subject_type=subject_type,
            key_object=key_object,
        )
        style_ok, style_flaws, style_meta = assess_photoreal_style(
            out_img,
            subject_type=subject_type,
        )
        del gate_ok, style_ok
        gate_meta = dict(gate_meta)
        gate_meta["scene"] = scene_i
        gate_meta["attempt"] = attempt
        gate_meta["focus_step"] = focus_step
        gate_meta["framing"] = row.get("object_focus_framing") or ""
        gate_meta["style_gate"] = style_meta
        last_gate = gate_meta
        last_intruder = bool(gate_flaws)
        if gate_flaws:
            flaws = list(flaws) + gate_flaws
            passed = False
            extra_fix = str(gate_meta.get("fix_instructions") or "").strip()
            if extra_fix:
                last_fix = f"{last_fix} {extra_fix}".strip() if last_fix else extra_fix
        if style_flaws:
            flaws = list(flaws) + style_flaws
            passed = False
            extra_fix = str(style_meta.get("fix_instructions") or "").strip()
            if extra_fix:
                last_fix = f"{last_fix} {extra_fix}".strip() if last_fix else extra_fix
        if (
            str(row.get("anchor_beat") or "") in {"introduce", "callback"}
            and str(row.get("close_variant") or "") != "eye_close"
        ):
            id_ok, id_flaws, id_meta = assess_anchor_object_identity(
                out_img,
                object_name=qa_object,
                caption=str(row.get("text") or row.get("beat_text") or ""),
            )
            gate_meta["anchor_identity"] = id_meta
            if id_flaws:
                flaws = list(flaws) + id_flaws
                passed = False
                extra_fix = str(id_meta.get("fix_instructions") or "").strip()
                if extra_fix:
                    last_fix = f"{last_fix} {extra_fix}".strip() if last_fix else extra_fix
            paint_ok, paint_flaws, paint_meta = assess_anchor_painterly_lock(
                out_img,
                subject_type=subject_type,
                anchor_beat=str(row.get("anchor_beat") or ""),
            )
            del paint_ok
            gate_meta["painterly_lock"] = paint_meta
            if paint_flaws:
                flaws = list(flaws) + paint_flaws
                passed = False
                extra_fix = str(paint_meta.get("fix_instructions") or "").strip()
                if extra_fix:
                    last_fix = f"{last_fix} {extra_fix}".strip() if last_fix else extra_fix
        pose_ok, pose_flaws, pose_meta = assess_couple_bed_pose(
            out_img,
            setting=str(row.get("setting") or ""),
            key_object=str(row.get("key_object") or ""),
            subject_type=subject_type,
        )
        gate_meta["bed_pose"] = pose_meta
        if pose_flaws:
            flaws = list(flaws) + pose_flaws
            passed = False
            extra_fix = str(pose_meta.get("fix_instructions") or "").strip()
            if extra_fix:
                last_fix = f"{last_fix} {extra_fix}".strip() if last_fix else extra_fix
        hook_ok, hook_flaws, hook_meta = assess_hook_still(
            out_img,
            scene=scene_i,
            subject_type=subject_type,
        )
        del hook_ok
        gate_meta["hook"] = hook_meta
        if hook_flaws:
            flaws = list(hook_flaws) + list(flaws)
            passed = False
            extra_fix = str(hook_meta.get("fix_instructions") or "").strip()
            if extra_fix:
                last_fix = f"{extra_fix} {last_fix}".strip() if last_fix else extra_fix
        port_ok, port_flaws, port_meta = assess_portrait_close_style(
            out_img,
            close_variant=str(row.get("close_variant") or ""),
        )
        del port_ok
        gate_meta["portrait_style"] = port_meta
        if port_flaws:
            flaws = list(flaws) + port_flaws
            passed = False
            extra_fix = str(port_meta.get("fix_instructions") or "").strip()
            if extra_fix:
                last_fix = f"{last_fix} {extra_fix}".strip() if last_fix else extra_fix
        eye_ok, eye_flaws, eye_meta = assess_eye_close_crop(
            out_img,
            close_variant=str(row.get("close_variant") or ""),
        )
        del eye_ok
        gate_meta["eye_close"] = eye_meta
        if eye_flaws:
            flaws = list(flaws) + eye_flaws
            passed = False
            extra_fix = str(eye_meta.get("fix_instructions") or "").strip()
            if extra_fix:
                last_fix = f"{last_fix} {extra_fix}".strip() if last_fix else extra_fix
        cov_ok, cov_flaws, cov_meta = assess_garment_coverage(
            out_img,
            subject_type=subject_type,
            close_variant=str(row.get("close_variant") or ""),
        )
        del cov_ok
        gate_meta["garment_coverage"] = cov_meta
        if cov_flaws:
            flaws = list(flaws) + cov_flaws
            passed = False
            extra_fix = str(cov_meta.get("fix_instructions") or COVERAGE_FIX).strip()
            if extra_fix:
                last_fix = f"{last_fix} {extra_fix}".strip() if last_fix else extra_fix
            log_coverage_flag(
                {
                    "scene": scene_i,
                    "attempt": attempt,
                    "subject_type": subject_type,
                    "close_variant": str(row.get("close_variant") or ""),
                    "prompt": prompt_i,
                    "negative_prompt": (
                        mood_meta.get("gen_negative_prompt")
                        or getattr(lofi_cfg, "LOFI_DEV_NEGATIVE_PROMPT", "")
                    ),
                    "seed": mood_meta.get("gen_seed"),
                    "guidance_scale": mood_meta.get("gen_guidance_scale"),
                    "model": mood_meta.get("gen_model"),
                    "flaws": list(cov_flaws),
                    "note": cov_meta.get("note") or "",
                    "image": str(out_img),
                }
            )
        from core.economic_reel_lofi.visual_identity import (
            spoken_line_licenses_lamp,
            spoken_line_licenses_window,
        )

        from core.economic_reel_lofi.licensed_objects import licensed_display as _lic_disp

        if str(row.get("visual_source") or "") == "writer_single_pass":
            prop_ok, prop_flaws, prop_meta = True, [], {
                "passed": True,
                "skipped": True,
                "skip_reason": "single_pass_atmospheric_visual",
                "leaks": [],
            }
            print(
                f"[LOFI spoken-prop] SKIP {out_img.name} "
                "single-pass atmospheric visual"
            )
        else:
            prop_ok, prop_flaws, prop_meta = assess_spoken_line_prop_leak(
                out_img,
                caption=str(row.get("text") or row.get("beat_text") or ""),
                window_allowed=spoken_line_licenses_window(row),
                lamp_allowed=spoken_line_licenses_lamp(row),
                licensed_object=_lic_disp(row) or str(row.get("key_object") or key_object or ""),
                episode_motif=str(
                    row.get("episode_anchor_name")
                    or row.get("episode_spoken_motif")
                    or ""
                ),
            )
        del prop_ok
        if apply_background_window_exception(prop_meta, row, gate_meta):
            prop_flaws = []
            if prop_meta.get("leaks"):
                prop_flaws = [
                    "SPOKEN-PROP: unspoken objects in pixels: "
                    + ", ".join(str(x) for x in prop_meta["leaks"])
                ]
        gate_meta["spoken_prop"] = prop_meta
        if prop_flaws:
            flaws = list(flaws) + prop_flaws
            passed = False
            extra_fix = str(prop_meta.get("fix_instructions") or "").strip()
            if extra_fix:
                last_fix = f"{last_fix} {extra_fix}".strip() if last_fix else extra_fix
        hard_leaks = [
            x for x in (prop_meta.get("leaks") or [])
            if x and x != "window"
        ]
        if not hard_leaks:
            dest = out_img.with_name(f"{out_img.stem}_spoken_ok.png")
            shutil.copy2(out_img, dest)
            spoken_fallback = dest
        if scene_i == 1:
            print(
                f"[LOFI hook] still scene=1 "
                f"status={'pass' if not hook_flaws else 'FAIL'} "
                f"note={hook_meta.get('note')!r}"
            )
        if str(row.get("close_variant") or "") in {"portrait_close", "eye_close", "silhouette"}:
            print(
                f"[LOFI close] still scene={scene_i} "
                f"variant={row.get('close_variant')} "
                f"character={row.get('close_character') or row.get('subject_type')} "
                f"style={'pass' if not port_flaws else 'FAIL'}"
            )
        gate_meta["qa_passed"] = bool(passed)
        gate_meta["qa_flaws"] = list(flaws)
        if passed:
            ok_img = True
            scene_attempts.append(dict(last_gate))
            break
        last_flaws = flaws
        scene_attempts.append(dict(last_gate))
        if attempt >= n_attempts:
            print(
                f"[LOFI retry-stop] scene={scene_i} attempt={attempt} "
                f"same named fail — not spending another call. flaws={flaws}"
            )
            break
        _LOG.info(
            "VisualQA reject scene %s attempt %s subject=%s: %s",
            scene_i,
            attempt,
            subject_type or "?",
            "; ".join(flaws),
        )

    style_hard = any(
        any(
            k in str(f).lower()
            for k in (
                "photoreal",
                "rendered-skin",
                "rendered skin",
                "lit portrait",
                "camera portrait",
            )
        )
        for f in (last_flaws or [])
    )
    if not ok_img and spoken_fallback and spoken_fallback.is_file() and not style_hard:
        shutil.copy2(spoken_fallback, out_img)
        print(f"[LOFI spoken-prop] restored hard-clean fallback {spoken_fallback.name}")
    elif not ok_img and style_hard:
        print(
            "[LOFI spoken-prop] skip fallback restore — last attempt failed "
            "photoreal/portrait style; keeping last pixels for review"
        )
    last_gate["image_ok"] = ok_img
    last_gate["attempts"] = scene_attempts
    last_gate["attempts_used"] = n_image_calls
    last_gate["focus_step"] = focus_step
    last_gate["qa_flaws"] = list(last_gate.get("qa_flaws") or last_flaws)
    last_gate["qa_passed"] = bool(ok_img)
    last_gate["mood"] = mood_meta
    return ok_img, last_gate, n_image_calls, n_critic_calls


def _caption_cap_reasons_only(reasons: list[str]) -> bool:
    """True when every validator reason is a per-beat word/char cap miss."""
    if not reasons:
        return False
    joined = " ".join(reasons)
    if "spoken line" in joined or "duration_requested_s" in joined:
        return False
    pats = (
        re.compile(r"has \d+ words \(max \d+\)"),
        re.compile(r"caption exceeds \d+ chars"),
    )
    return all(any(p.search(r) for p in pats) for r in reasons)


def _targeted_line_failure(
    script: dict[str, Any],
    reasons: list[str],
) -> tuple[int, str, list[str]] | None:
    """Return one repairable beat (index, constraint, required terms)."""
    if not reasons:
        anchor_holds = [
            str(r)
            for r in (script.get("narrative_holds") or [])
            if str(r).startswith("anchor_object ") and "mentioned in " in str(r)
        ]
        other_holds = [
            str(r)
            for r in (script.get("narrative_holds") or [])
            if str(r) not in anchor_holds
        ]
        lines = [r for r in (script.get("lines") or []) if isinstance(r, dict)]
        anchor = str((script.get("anchor_object") or {}).get("name") or "").strip()
        if len(anchor_holds) == 1 and not other_holds and lines and anchor:
            return len(lines) - 1, anchor_holds[0], [anchor]
        return None
    lines = [r for r in (script.get("lines") or []) if isinstance(r, dict)]

    # Story diagnostics can usually identify one weak transition or closing
    # beat. Repair that beat instead of asking for another whole episode.
    close_reason = next(
        (r for r in reasons if "story-close:" in str(r)),
        None,
    )
    if close_reason and lines:
        return len(lines) - 1, str(close_reason), []
    spine_reason = next(
        (r for r in reasons if "story-spine:" in str(r)),
        None,
    )
    if spine_reason and len(lines) > 1:
        linked_pairs: set[int] = set()
        for note in ((script.get("story_quality") or {}).get("links") or []):
            match = re.match(r"(\d+)->(\d+):", str(note))
            if match:
                linked_pairs.add(int(match.group(1)))
        # Pair notes are zero-based (0->1). Rewrite the second line of the
        # first missing pair, while protecting the opening hook.
        missing_pair = next(
            (i for i in range(len(lines) - 1) if i not in linked_pairs),
            0,
        )
        target_i = min(missing_pair + 1, len(lines) - 1)
        prior = str(lines[target_i - 1].get("text") or "")
        constraint = (
            f"{spine_reason}; connect naturally to the prior beat {prior!r} "
            "without adding a decorative transition"
        )
        return target_i, constraint, []
    stakes_reason = next(
        (r for r in reasons if "story-stakes:" in str(r)),
        None,
    )
    if stakes_reason and lines:
        return max(1, len(lines) // 2), str(stakes_reason), []

    allowed = (
        "spoken line has",
        " words (max ",
        "caption exceeds",
        "setting/object not tied to caption",
    )
    scene_ids: set[int] = set()
    for reason in reasons:
        match = re.search(r"\bscene\s+(\d+)\b", reason)
        if not match or not any(token in reason for token in allowed):
            scene_ids.clear()
            break
        scene_ids.add(int(match.group(1)) - 1)
    if len(scene_ids) == 1:
        idx = next(iter(scene_ids))
        if not 0 <= idx < len(lines):
            return None
        row = lines[idx]
        must_keep: list[str] = []
        if any("setting/object not tied to caption" in r for r in reasons):
            must_keep.extend(
                str(row.get(key) or "").strip()
                for key in ("key_object",)
                if str(row.get(key) or "").strip()
            )
        return idx, "; ".join(reasons), must_keep

    return None


def _repair_targeted_line(
    script: dict[str, Any],
    *,
    module: str,
    scene_count: int,
    reasons: list[str],
    persist_on_pass: bool,
) -> tuple[Any, int]:
    """Repair one beat up to MAX_LINE_REPAIRS, then use a local safe trim."""
    target = _targeted_line_failure(script, reasons)
    if target is None:
        return None, 0
    idx, constraint, must_keep = target
    lines = [r for r in (script.get("lines") or []) if isinstance(r, dict)]
    row = lines[idx]
    from agents.writer.freeform_writer import split_long_line
    from agents.writer.spoken_budget import repair_one_line, trim_at_clause_boundary

    max_words, max_chars = lofi_cfg.caption_limits(
        str(script.get("arc_template") or "")
    )
    repairs = max(1, int(getattr(lofi_cfg, "MAX_LINE_REPAIRS", 2)))
    for repair_n in range(1, repairs + 1):
        original = str(row.get("text") or row.get("beat_text") or "")
        try:
            rewritten = repair_one_line(
                original,
                constraint=constraint,
                max_words=max_words,
                max_chars=max_chars,
                provider="claude",
                must_keep=must_keep,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[LOFI line-repair] scene={idx + 1} "
                f"attempt={repair_n}/{repairs} failed: {exc}"
            )
            continue
        row["text"] = rewritten
        row["beat_text"] = rewritten
        row["caption_beats"] = split_long_line(rewritten, max_words, max_chars)
        script["monologue"] = " ".join(
            str(item.get("text") or "") for item in lines
        )
        checked = validate_script(
            script,
            module=module,
            scene_count=scene_count,
            persist_on_pass=persist_on_pass,
        )
        if checked.ok:
            if constraint.startswith("anchor_object "):
                script["narrative_holds"] = []
                script["holds_episode"] = False
            print(
                f"[LOFI line-repair] scene={idx + 1} PASS "
                f"after {repair_n}/{repairs}"
            )
            return checked, repair_n
        next_target = _targeted_line_failure(script, list(checked.reasons))
        if next_target is None or next_target[0] != idx:
            return checked, repair_n
        constraint = "; ".join(checked.reasons)

    current = str(row.get("text") or row.get("beat_text") or "")
    trimmed = trim_at_clause_boundary(
        current,
        max_words=max_words,
        max_chars=max_chars,
    )
    row["text"] = trimmed
    row["beat_text"] = trimmed
    row["caption_beats"] = [trimmed]
    script["monologue"] = " ".join(str(item.get("text") or "") for item in lines)
    print(
        f"[LOFI line-repair] scene={idx + 1} deterministic fallback "
        f"after {repairs} LLM attempts: {trimmed!r}"
    )
    checked = validate_script(
        script,
        module=module,
        scene_count=scene_count,
        persist_on_pass=persist_on_pass,
    )
    return checked, repairs


def _generate_validated_script(
    *,
    module: str,
    theme_row: dict[str, Any],
    scene_count: int,
    duration_s: float,
    writer_mode: str = "emotional",
    seed_quote: str | None = None,
    aphorism_id: str | None = None,
    persist_on_pass: bool = True,
    strict_judge: bool = False,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    """Returns (script|None, errors, needs_manual_review)."""
    feedback: str | None = None
    last_errors: list[str] = []
    last_script: dict[str, Any] | None = None
    mode = str(writer_mode or "emotional").strip().lower()
    from agents.writer.writer_brief import get_narrative_harness

    narrative_harness = (
        get_narrative_harness(
            module,
            theme=str(theme_row.get("theme") or ""),
            subtheme=str(theme_row.get("subtheme") or ""),
        )
        if mode != "paraphrase"
        else {}
    )
    if not strict_judge:
        from agents.writer.freeform_writer import write_draft
        from agents.writer.script_brain import draft_to_script
        from agents.writer.writer_brief import WriterBrief

        common = {
            "duration_s": float(duration_s),
            "beat_duration_s": lofi_cfg.beat_duration_s(),
            **narrative_harness,
        }
        details = [
            str(item.get("detail") or "")
            for item in rag.select_concrete_details(theme_row, module=module)
            if str(item.get("detail") or "").strip()
        ]
        if mode == "quote":
            brief = WriterBrief.from_quote(
                quote=str(seed_quote or ""),
                theme=str(theme_row.get("theme") or ""),
                module=module,
                context_notes=details,
                meta=common,
            )
        elif mode == "paraphrase":
            from core.economic_reel_lofi.aphorism_bank import get_entry

            entry = get_entry(aphorism_id, module=module)
            brief = WriterBrief.from_paraphrase(
                aphorism=str(entry["text"]),
                theme=str(theme_row.get("theme") or ""),
                module=module,
                source_id=str(entry["id"]),
                meta=common,
            )
        elif mode == "theme":
            brief = WriterBrief.from_theme(
                theme=str(theme_row.get("theme") or "connection"),
                subtheme=str(theme_row.get("subtheme") or ""),
                module=module,
                context_notes=details,
                meta=common,
            )
        else:
            brief = WriterBrief.from_emotional(
                theme=str(theme_row.get("theme") or "emotional maturity"),
                subtheme=str(theme_row.get("subtheme") or ""),
                module=module,
                context_notes=details,
                meta=common,
            )
        try:
            draft = write_draft(
                brief,
                attempt=1,
                provider="gemini38",
                reference_seed=None,
            )
        except Exception as exc:  # valid JSON/schema failure; never retry in fast mode
            msg = f"single-pass writer failed: {exc}"
            print(f"[LOFI writer fast] {msg}")
            return None, [msg], True
        script = draft_to_script(draft)
        script["writer_mode"] = mode
        script["duration_requested_s"] = float(duration_s)
        script["scene_duration_s"] = lofi_cfg.beat_duration_s()
        script["subtheme"] = theme_row.get("subtheme")
        script["module"] = module
        script["niche"] = str(script.get("niche") or module or "relationship")
        from core.economic_reel_lofi.niche_presets import inject_prompt_fields

        inject_prompt_fields(script)
        celestial_scenes = [
            int(value) for value in (script.get("celestial_disc_scenes") or [])
        ]
        if not (1 <= len(celestial_scenes) <= 2) or any(
            scene not in {1, 8} for scene in celestial_scenes
        ):
            raise ValueError(
                f"invalid celestial lighting budget: {celestial_scenes}"
            )
        print(
            f"[LOFI lighting] celestial_budget={len(celestial_scenes)} "
            f"scenes={celestial_scenes} non_disc={8 - len(celestial_scenes)}"
        )
        script["script_ship_ok"] = True
        script["script_ship_errors"] = []
        script["writer_diagnostics"] = {
            "architecture": "single_pass_few_shot",
            "strict_judge": False,
            "attempt_count": 1,
            "judge_calls": 0,
            "repair_calls": 0,
            "model": draft.meta.get("model") or lofi_cfg.LOFI_EMOTIONAL_WRITER_MODEL,
            "latency_s": draft.meta.get("latency_s"),
            "cost_usd_est": draft.meta.get("cost_usd_est"),
            "output_tokens_est": draft.meta.get("output_tokens_est"),
        }
        note_batch_structure_id(str(script.get("structure_id") or ""))
        rag.note_batch_script(script)
        print(
            f"[LOFI writer fast] ACCEPT first valid candidate | "
            f"beats={len(script.get('lines') or [])} judge=OFF repairs=OFF"
        )
        return script, [], False

    max_structural_attempts = (
        1 if mode == "paraphrase" else int(lofi_cfg.SCRIPT_MAX_RETRIES)
    )
    for attempt in range(1, max_structural_attempts + 1):
        if mode == "theme":
            script = generate_script(
                module=module,
                theme=str(theme_row.get("theme") or "connection"),
                subtheme=str(theme_row.get("subtheme") or ""),
                scene_count=scene_count,
                feedback=feedback,
                theme_row=theme_row,
                duration_s=duration_s,
            )
        else:
            from agents.writer.script_brain import compose, draft_to_script, paraphrase
            from agents.writer.writer_brief import WriterBrief

            common = {
                "duration_s": float(duration_s),
                "beat_duration_s": lofi_cfg.beat_duration_s(),
                **narrative_harness,
            }
            if mode == "emotional":
                brief = WriterBrief.from_emotional(
                    theme=str(theme_row.get("theme") or "emotional maturity"),
                    subtheme=str(theme_row.get("subtheme") or ""),
                    module=module,
                    context_notes=[
                        str(item.get("detail") or "")
                        for item in rag.select_concrete_details(theme_row, module=module)
                        if str(item.get("detail") or "").strip()
                    ],
                    meta=common,
                )
                if feedback:
                    brief = brief.with_revision(feedback)
                brain = compose(brief, max_attempts=1)
                draft = brain.draft or next(
                    (row.draft for row in reversed(brain.attempts) if row.draft is not None),
                    None,
                )
                if draft is None:
                    last_errors = [brain.reason()]
                    feedback = brain.reason()
                    continue
                script = draft_to_script(draft)
                script["brain_gate_ok"] = bool(brain.ok)
                script["brain_gate_errors"] = [] if brain.ok else [brain.reason()]
                script["writer_diagnostics"] = brain.to_dict()
            elif mode == "quote":
                brief = WriterBrief.from_quote(
                    quote=str(seed_quote or ""),
                    theme=str(theme_row.get("theme") or ""),
                    module=module,
                    context_notes=[
                        str(item.get("detail") or "")
                        for item in rag.select_concrete_details(theme_row, module=module)
                        if str(item.get("detail") or "").strip()
                    ],
                    meta=common,
                )
                if feedback:
                    brief = brief.with_revision(feedback)
                brain = compose(brief, max_attempts=1)
                draft = brain.draft or next(
                    (
                        row.draft
                        for row in reversed(brain.attempts)
                        if row.draft is not None
                    ),
                    None,
                )
                if draft is None:
                    last_errors = [brain.reason()]
                    feedback = brain.reason()
                    continue
                script = draft_to_script(draft)
                script["brain_gate_ok"] = bool(brain.ok)
                script["brain_gate_errors"] = [] if brain.ok else [brain.reason()]
                script["writer_diagnostics"] = brain.to_dict()
            elif mode == "paraphrase":
                from core.economic_reel_lofi.aphorism_bank import get_entry

                # No --lofi-aphorism-id: pick a random bank entry for this
                # module. Do not lock to the first theme-matched row.
                entry = get_entry(aphorism_id, module=module)
                print(
                    f"[LOFI paraphrase] bank entry={entry.get('id')} "
                    f"{'(forced)' if aphorism_id else '(random)'}"
                )
                brief = WriterBrief.from_paraphrase(
                    aphorism=str(entry["text"]),
                    theme=str(theme_row.get("theme") or ""),
                    module=module,
                    source_id=str(entry["id"]),
                    meta={
                        **common,
                        "source_structure": str(entry.get("structure") or ""),
                    },
                )
                script = draft_to_script(paraphrase(brief))
            else:
                raise ValueError(f"unsupported LOFI writer mode: {mode!r}")
        script["writer_mode"] = mode
        script["duration_requested_s"] = float(duration_s)
        script["scene_duration_s"] = lofi_cfg.beat_duration_s()
        script["subtheme"] = theme_row.get("subtheme")
        last_script = script
        pre_repaired, _ = _repair_targeted_line(
            script,
            module=module,
            scene_count=scene_count,
            reasons=[],
            persist_on_pass=persist_on_pass,
        )
        result = pre_repaired or validate_script(
            script,
            module=module,
            scene_count=scene_count,
            persist_on_pass=persist_on_pass,
        )
        narrative_structural = [
            str(reason)
            for reason in (script.get("narrative_holds") or [])
            if str(reason).startswith(("HOOK:", "THESIS:", "THEME:"))
            or "theme-drift" in str(reason).lower()
            or str(reason).startswith(("story-spine:", "story-stakes:", "story-close:"))
        ]
        for reason in narrative_structural:
            tagged = f"narrative_structure: {reason}"
            if tagged not in result.reasons:
                result.reasons.append(tagged)
                result.ok = False
        if result.ok and mode in {"emotional", "quote"} and not script.get("brain_gate_ok"):
            result.ok = False
            result.reasons.extend(script.get("brain_gate_errors") or ["judge gate failed"])
        if not result.ok:
            # A draft may have two independent scoped misses (for example one
            # weak transition and one generic close). Repair them in sequence;
            # each beat retains its own MAX_LINE_REPAIRS safety cap.
            for _ in range(min(3, scene_count)):
                repaired, _ = _repair_targeted_line(
                    script,
                    module=module,
                    scene_count=scene_count,
                    reasons=list(result.reasons),
                    persist_on_pass=persist_on_pass,
                )
                if repaired is None:
                    break
                result = repaired
                if result.ok:
                    break
        if (not result.ok) and _caption_cap_reasons_only(result.reasons):
            repair_script_captions(script)
            result = validate_script(
                script,
                module=module,
                scene_count=scene_count,
                persist_on_pass=persist_on_pass,
            )
            if result.ok and result.script:
                print(
                    "[LOFI script] recovered over-cap beat(s) by trim — "
                    "not retrying the whole episode"
                )
        if result.ok and result.script:
            result.script["script_ship_ok"] = True
            result.script["script_ship_errors"] = []
            note_batch_structure_id(str(result.script.get("structure_id") or ""))
            rag.note_batch_script(result.script)
            return result.script, [], False
        if last_script is not None:
            last_script["script_ship_ok"] = False
            last_script["script_ship_errors"] = list(result.reasons)
        last_errors = list(result.reasons)
        feedback = result.feedback()
        print(f"[LOFI script] attempt {attempt}/{max_structural_attempts} rejected: {feedback}")
        _LOG.info(
            "Script attempt %d/%d rejected: %s",
            attempt,
            max_structural_attempts,
            feedback,
        )
    if not persist_on_pass and last_script:
        print(
            f"[LOFI script] script-only: keeping last {mode} draft for review "
            "(not substituting fallback)"
        )
        return last_script, last_errors, True
    if mode != "theme":
        return None, last_errors or [f"{mode} writer did not clear validation"], True
    # Last resort: deterministic short-line fallback so a test still renders
    from agents.writer.script_agent import _fallback_script

    last_details = list((last_script or {}).get("retrieved_details") or [])
    if not last_details:
        last_details = rag.select_concrete_details(theme_row, module=module)
    fallback = _fallback_script(
        module=module,
        theme=str(theme_row.get("theme") or "connection"),
        scene_count=scene_count,
        hook_type="definition",
        quote=None,
        setting_object_pairs=theme_row.get("setting_object_pairs")
        if isinstance(theme_row.get("setting_object_pairs"), list)
        else None,
        retrieved_details=last_details,
        arc_template=str((last_script or {}).get("arc_template") or ""),
    )
    fallback["subtheme"] = theme_row.get("subtheme")
    repair_script_captions(fallback)
    result = validate_script(
        fallback,
        module=module,
        scene_count=scene_count,
        persist_on_pass=persist_on_pass,
    )
    if result.ok and result.script:
        print("[LOFI script] using fallback after validator retries")
        result.script["script_ship_ok"] = True
        result.script["script_ship_errors"] = []
        return result.script, [], False
    fallback_reasons = list(result.reasons)
    print(f"[LOFI validator] last-resort REJECT: {result.feedback()}")
    if last_script is not None:
        last_script["script_ship_ok"] = False
        last_script["script_ship_errors"] = fallback_reasons or last_errors
    return None, fallback_reasons or last_errors or ["script validation failed"], True


def _resolve_page_dirs(
    page_id: str, outputs_dir: Path | str | None
) -> tuple[Path, Path, Path, Path]:
    """Return ``(page_outputs, clips_dir, assets_dir, metadata_dir)``.

    ``clips/`` is final MP4s only. JSON diagnostics go to ``metadata/``.
    Scene stills and VO stay under ``assets/<run_id>/``.
    """
    from utils.pipeline_paths import coerce_outputs_path, page_outputs_dir

    if outputs_dir is None:
        page_outputs = page_outputs_dir(page_id)
    else:
        page_outputs = coerce_outputs_path(outputs_dir)
        # Callers may pass .../clips by mistake — normalize to page root.
        if page_outputs.name == "clips":
            page_outputs = page_outputs.parent
        elif page_outputs.name == "economic_reel_lofi":
            page_outputs = page_outputs.parent
        elif page_outputs.name == "metadata":
            page_outputs = page_outputs.parent
    clips_dir = page_outputs / "clips"
    assets_dir = page_outputs / "assets"
    metadata_dir = page_outputs / "metadata"
    clips_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return page_outputs, clips_dir, assets_dir, metadata_dir


def _load_locked_script(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    envelope: dict[str, Any] = raw if isinstance(raw, dict) else {}
    if isinstance(raw, dict) and isinstance(raw.get("script"), dict):
        script = dict(raw["script"])
    elif isinstance(raw, dict):
        script = dict(raw)
    else:
        raise ValueError(f"locked script is not a JSON object: {p}")
    if not (script.get("lines") or script.get("monologue")):
        spoken = script.get("spoken_lines")
        if isinstance(spoken, list) and spoken:
            script["lines"] = [
                {
                    "scene": i + 1,
                    "text": str(t).strip(),
                    "beat_text": str(t).strip(),
                }
                for i, t in enumerate(spoken)
                if str(t).strip()
            ]
            script["monologue"] = " ".join(
                str(r.get("text") or "") for r in script["lines"]
            )
            script["writer"] = str(script.get("writer") or "locked_pack")
            print(
                f"[LOFI pipeline] pack spoken_lines → {len(script['lines'])} beats"
            )
    if not (script.get("lines") or script.get("monologue")):
        raise ValueError(f"locked script has no lines/monologue: {p}")
    scene_images = list(envelope.get("scene_images") or [])
    manual_accept = [int(x) for x in (envelope.get("manual_accept_scenes") or [])]
    hold_flags = [str(x) for x in (envelope.get("visual_qa_flags") or [])]
    lines = [row for row in (script.get("lines") or []) if isinstance(row, dict)]
    atmospheric_hold = bool(
        scene_images
        and hold_flags
        and not manual_accept
        and all("SPOKEN-PROP:" in flag for flag in hold_flags)
        and lines
        and all(str(row.get("visual_source") or "") == "writer_single_pass" for row in lines)
    )
    if atmospheric_hold:
        manual_accept = [int(row.get("scene") or i + 1) for i, row in enumerate(lines)]
        print(
            "[LOFI pipeline] migrating atmospheric spoken-prop hold; "
            f"reusing scenes={manual_accept}"
        )
    extras = {
        "scene_images": scene_images,
        "work_dir": envelope.get("work_dir"),
        "manual_accept_scenes": manual_accept,
        "source": str(p),
    }
    script["_locked_sidecar_assets"] = extras
    print(f"[LOFI pipeline] loaded locked script → {p}")
    return script


_RUN_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}


def _purge_stale_temp_voice_files(
    run_dir: Path,
    *,
    keep_names: set[str] | None = None,
) -> list[Path]:
    """Remove leftover VO/cache audio so only this run's scene files can mix.

    Never used to *discover* mix inputs. Assembly reads
    ``vo_scene_01.mp3`` … ``vo_scene_NN.mp3`` explicitly.
    """
    target = Path(run_dir)
    if not target.is_dir():
        return []
    removed: list[Path] = []
    keep = {str(name) for name in (keep_names or set())}
    for path in sorted(target.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        low = name.lower()
        stale = (
            low.startswith("temp_voice")
            or low.startswith("temp_audio")
            or "temp_mpy" in low
        )
        is_vo_scene = bool(re.fullmatch(r"vo_scene_\d{2}\.mp3", name, flags=re.I))
        leftover = path.suffix.lower() in _RUN_AUDIO_SUFFIXES and not is_vo_scene
        extra_scene = bool(keep) and is_vo_scene and name not in keep
        if not (stale or leftover or extra_scene):
            continue
        path.unlink()
        removed.append(path)
    if removed:
        print(f"[LOFI audio] purged stale temp voice files={len(removed)}")
    return removed


def _current_run_scene_audio_paths(
    run_dir: Path,
    n_beats: int,
    voice_paths: Sequence[Path | None] | None = None,
) -> list[Path | None]:
    """Return this run's ``vo_scene_01..N`` paths only. Never glob ``*.mp3``."""
    resolved: list[Path | None] = []
    root = Path(run_dir)
    declared = list(voice_paths or [])
    for i in range(max(0, int(n_beats))):
        canonical = root / f"vo_scene_{i + 1:02d}.mp3"
        prior = declared[i] if i < len(declared) and declared[i] else None
        prior_path = Path(prior) if prior else None
        if canonical.is_file():
            resolved.append(canonical)
            continue
        if (
            prior_path is not None
            and prior_path.is_file()
            and prior_path.name == canonical.name
        ):
            resolved.append(prior_path)
            continue
        resolved.append(None)
    return resolved


def _stamp_assembled_negatives(lines: list[dict[str, Any]]) -> None:
    """Stage 3: per-beat negative from style + not_in_frame, minus licensed stems."""
    from core.economic_reel_lofi.licensed_objects import licensed_display

    for row in lines:
        if not isinstance(row, dict):
            continue
        licensed = licensed_display(row) or str(row.get("key_object") or "")
        nots = list(row.get("not_in_frame") or [])
        row["negative_prompt"] = lofi_cfg.compose_beat_negative(
            licensed_object=licensed,
            not_in_frame=nots,
        )


def _gate2_review_rows(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Human-review payload for the complete atmospheric prompt artifact."""
    from core.economic_reel_lofi.visual_identity import beat_shot_type, beat_who_label

    out: list[dict[str, Any]] = []
    for row in lines:
        if not isinstance(row, dict):
            continue
        meaning = str(row.get("meaning") or "").strip()
        if not meaning:
            meaning = str(row.get("visual_concept") or row.get("text") or "").strip()
        out.append(
            {
                "scene": row.get("scene"),
                "text": row.get("text") or row.get("beat_text"),
                "meaning": meaning,
                "beat_mood": row.get("beat_mood"),
                "emotional_temperature": row.get("emotional_temperature"),
                "who": beat_who_label(row),
                "shot_type": row.get("shot_type") or beat_shot_type(row),
                "environment": row.get("setting") or row.get("episode_place"),
                "palette_temp": row.get("palette_temp") or row.get("palette_key"),
                "arc_position": row.get("arc_position"),
                "subject_type": row.get("subject_type"),
                "lighting_condition": row.get("lighting_condition"),
                "visual_concept": row.get("visual_concept"),
                "final_positive_prompt": row.get("final_positive_prompt"),
                "visual_prompt": row.get("visual_prompt"),
                "negative_prompt": row.get("negative_prompt"),
            }
        )
    return out


def _assemble_stage3_prompts(
    script: dict[str, Any],
    *,
    theme_row: dict[str, Any] | None,
    lock_visuals: bool,
    clips_dir: Path,
    metadata_dir: Path | None = None,
    stamp: str,
    scene_count: int,
) -> dict[str, Any]:
    """Style-module prompt assembly. No Flux, no TTS."""
    lines = [r for r in (script.get("lines") or []) if isinstance(r, dict)]
    episode_variety: dict[str, Any] = {}
    from core.economic_reel_lofi.niche_presets import (
        build_flux_prompt,
        get_niche_preset,
    )

    niche_key = str(script.get("niche") or script.get("module") or "relationship")
    preset = get_niche_preset(niche_key)
    celestial_scenes = frozenset(
        int(value) for value in (script.get("celestial_disc_scenes") or [])
    )
    writer_concepts = [
        " ".join(str(row.get("visual_concept") or "").split()) for row in lines
    ]
    if lines and all(writer_concepts):
        for i, (row, concept) in enumerate(zip(lines, writer_concepts), start=1):
            positive, negative = build_flux_prompt(
                concept,
                int(row.get("scene") or i),
                celestial_scenes=celestial_scenes,
            )
            row["final_positive_prompt"] = positive
            row["visual_prompt"] = row["final_positive_prompt"]
            row["negative_prompt"] = negative
            row["visual_source"] = row.get("visual_source") or "writer_single_pass"
            row["aesthetic_prefix"] = preset.aesthetic_prefix
        script["lines"] = lines
        script["niche"] = preset.key
        script["aesthetic_wrapper"] = preset.aesthetic_prefix
        print(
            f"[LOFI stage3] aesthetic wrapper niche={preset.key} "
            f"beats={len(lines)} source=writer_single_pass"
        )
        return episode_variety
    use_dev = bool(lofi_cfg.uses_flux_dev())
    if bool(getattr(lofi_cfg, "USE_VISUAL_IDENTITY_V2", False)):
        if use_dev:
            from core.economic_reel_lofi.visual_identity import (
                apply_v2_prompts_to_lines_dev,
            )

            print(
                f"[LOFI stage3] flux={'flux2-dev' if lofi_cfg.uses_flux2_dev() else 'dev'} | "
                f"profile={lofi_cfg.DEFAULT_VISUAL_IDENTITY_PROFILE} | "
                "assemble=assemble_v2_prompt_dev"
            )
            apply_v2_prompts_to_lines_dev(
                lines,
                theme_row=theme_row,
                vary_imagery=lofi_cfg.is_thematic_arc(
                    str(script.get("arc_template") or "")
                ),
                lock_visuals=lock_visuals,
            )
        else:
            from core.economic_reel_lofi.visual_identity import (
                apply_v2_prompts_to_lines,
            )

            apply_v2_prompts_to_lines(
                lines,
                theme_row=theme_row,
                vary_imagery=lofi_cfg.is_thematic_arc(
                    str(script.get("arc_template") or "")
                ),
                lock_visuals=lock_visuals,
            )
        if lines and isinstance(lines[0], dict):
            raw_var = lines[0].pop("episode_variety", None)
            if isinstance(raw_var, dict):
                episode_variety = raw_var
        if episode_variety:
            script["episode_variety"] = episode_variety
        if lines and isinstance(lines[0], dict) and isinstance(
            lines[0].get("episode_atmosphere"), dict
        ):
            script["episode_atmosphere"] = dict(lines[0]["episode_atmosphere"])
        rag.stamp_close_variant_on_script(script, lines)
        rag.stamp_lighting_on_script(script, lines)
        for row in lines:
            print(
                f"[LOFI stage3] scene={row.get('scene')} "
                f"type={row.get('subject_type')} framing={row.get('framing')} "
                f"object={row.get('key_object')!r}"
            )
    elif bool(getattr(lofi_cfg, "USE_RISO_PROMPT_LIBRARY", True)):
        from core.economic_reel_lofi.riso_prompt_bank import load_riso_library

        lib_diff = export_active_library_diff()
        try:
            dest = Path(metadata_dir) if metadata_dir else clips_dir.parent / "metadata"
            dest.mkdir(parents=True, exist_ok=True)
            live_dump = dest / f"riso_library_live_{stamp}.json"
            live_dump.write_text(
                json.dumps(load_riso_library(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[LOFI riso] full live library exported -> {live_dump}")
            print(
                f"[LOFI riso] diff added={lib_diff.get('added')} "
                f"removed={lib_diff.get('removed')}"
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("library export failed: %s", exc)
        emotions = [str(r.get("emotion") or r.get("mood") or "") for r in lines]
        arcs = [str(r.get("arc_position") or "") for r in lines]
        riso_rows = assign_riso_prompts_for_scenes(
            len(lines) if lines else scene_count,
            theme=str(script.get("theme") or ""),
            seed=None,
            scene_emotions=emotions,
            scene_arcs=arcs,
        )
        for i, row in enumerate(lines):
            if i >= len(riso_rows):
                break
            r = riso_rows[i]
            row["visual_prompt"] = str(r.get("prompt") or "")
            row["riso_id"] = r.get("id")
            row["riso_palette"] = r.get("palette")
            row["riso_mood"] = r.get("mood")
            row["riso_scene_type"] = r.get("scene_type")
            row["riso_emotion_tags"] = r.get("emotion_tags")
            row["riso_arc_position"] = r.get("arc_position")
            row.pop("lighting_mood", None)
    _stamp_assembled_negatives(lines)
    script["lines"] = lines
    return episode_variety


def _print_script_report(script: dict[str, Any], *, index: int, qty: int) -> None:
    lines = [r for r in (script.get("lines") or []) if isinstance(r, dict)]
    candidate = {
        "writer_mode": script.get("writer_mode"),
        "theme": script.get("theme"),
        "subtheme": script.get("subtheme"),
        "niche": script.get("niche") or script.get("module"),
        "location_anchor": script.get("location_anchor"),
        "human_situation": script.get("human_situation"),
        "structure": script.get("writer_structure"),
        "closing_tool": script.get("closing_tool"),
        "beats": [
            {
                "scene": int(row.get("scene") or i),
                "text": str(row.get("text") or ""),
                "visual_concept": str(row.get("visual_concept") or ""),
                "final_positive_prompt": str(
                    row.get("final_positive_prompt") or row.get("visual_prompt") or ""
                ),
                "negative_prompt": str(row.get("negative_prompt") or ""),
            }
            for i, row in enumerate(lines, start=1)
        ],
    }
    print("\nSCRIPT_CANDIDATE_JSON")
    print(json.dumps(candidate, indent=2, ensure_ascii=False))
    print("END_SCRIPT_CANDIDATE_JSON")


def _produce_one(
    *,
    page_id: str,
    module: str,
    duration_s: int,
    clips_dir: Path,
    assets_dir: Path,
    metadata_dir: Path | None = None,
    force_theme: str | None = None,
    force_subtheme: str | None = None,
    writer_mode: str = "emotional",
    seed_quote: str | None = None,
    aphorism_id: str | None = None,
    image_provider: str = "together",
    index: int,
    script_only: bool = False,
    stills_only: bool = False,
    batch_qty: int = 1,
    locked_script: dict[str, Any] | None = None,
    review_required: bool | None = None,
    resume_from: str | None = None,
    approve_gate: int | None = None,
) -> LofiItemResult:
    scene_count = lofi_cfg.scene_count_for_duration(duration_s, thematic=True)
    stamp = _utc_stamp()
    metadata_dir = Path(metadata_dir) if metadata_dir else clips_dir.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    persist_on_pass = not script_only and not stills_only
    review_required = bool(
        lofi_cfg.REVIEW_REQUIRED if review_required is None else review_required
    )
    strict_writer = bool(
        lofi_cfg.strict_judge_enabled()
        and not script_only
        and not review_required
    )
    from core.economic_reel_lofi.niche_config import niche_stage1_notes
    from core.economic_reel_lofi.review_gates import (
        auto_pass_gates,
        load_state,
        mark_approved,
        write_state,
    )
    from core.economic_reel_lofi.visual_concept import (
        stamp_stage1_timing,
        strip_visual_fields,
        translate_episode_visuals,
    )

    resume_state: dict[str, Any] | None = None
    if resume_from:
        resume_state = load_state(resume_from)
        print(f"[LOFI pipeline] resume ← {resume_state.get('source')}")
        if approve_gate in {1, 2}:
            mark_approved(resume_state, int(approve_gate), reason="cli")
        if isinstance(resume_state.get("script"), dict) and not locked_script:
            locked_script = dict(resume_state["script"])
            print("[LOFI pipeline] resume loaded script — skipping writer")

    niche_notes = niche_stage1_notes(module)
    print(
        f"[LOFI pipeline] four-stage | review_required={int(review_required)} "
        f"niche={niche_notes.get('niche_id')} duration_s={duration_s} "
        f"scenes={scene_count}"
    )

    reset_script_llm_call_log()
    if isinstance(locked_script, dict) and (locked_script.get("lines") or locked_script.get("monologue")):
        script = dict(locked_script)
        if strict_writer:
            repair_script_captions(script, keep_extra_scenes=True)
        else:
            print("[LOFI pipeline] LOCKED script text preserved verbatim")
        theme_row = rag.select_theme(
            module,
            theme=str(script.get("theme") or force_theme or ""),
            subtheme=str(script.get("subtheme") or force_subtheme or ""),
        )
        if not script.get("theme"):
            first = next(
                (r for r in (script.get("lines") or []) if isinstance(r, dict)),
                {},
            )
            script["theme"] = (
                force_theme
                or str(first.get("episode_theme") or "")
                or "hope"
            )
        print(
            "[LOFI pipeline] LOCKED script — skipping writer "
            f"theme={script.get('theme')} subtheme={script.get('subtheme')} "
            f"beats={len(script.get('lines') or [])}"
        )
        if strict_writer:
            locked_result = validate_script(
                script,
                module=module,
                scene_count=scene_count,
                persist_on_pass=False,
            )
            script["script_ship_ok"] = bool(locked_result.ok)
            script["script_ship_errors"] = list(locked_result.reasons)
            if not locked_result.ok:
                print(
                    "[LOFI pipeline] LOCKED script failed strict validator — "
                    "not generating stills. Hard stop (no assemble path)."
                )
                _print_script_report(script, index=index, qty=batch_qty)
                errs = list(locked_result.reasons)
                review_path = metadata_dir / f"lofi_manual_review_{stamp}_{index:02d}.json"
                review_path.write_text(
                    json.dumps(
                        {
                            "errors": errs,
                            "theme": theme_row,
                            "module": module,
                            "script": script,
                            "gate": "validate_script",
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return LofiItemResult(
                    ok=False,
                    module=module,
                    theme=str(script.get("theme") or theme_row.get("theme") or ""),
                    scene_count=scene_count,
                    duration_s=float(duration_s),
                    manual_review=True,
                    errors=errs,
                    meta_path=str(review_path),
                    script=script,
                )
            if locked_result.script:
                script = locked_result.script
            print("[LOFI pipeline] LOCKED script cleared strict validator gate")
        else:
            script["script_ship_ok"] = True
            script["script_ship_errors"] = []
            print("[LOFI pipeline] LOCKED script accepted by fast JSON path")
        _print_script_report(script, index=index, qty=batch_qty)
        errs: list[str] = []
    else:
        theme_row = rag.select_theme(module, theme=force_theme, subtheme=force_subtheme)
        script, errs, manual = _generate_validated_script(
            module=module,
            theme_row=theme_row,
            scene_count=scene_count,
            duration_s=float(duration_s),
            writer_mode=writer_mode,
            seed_quote=seed_quote,
            aphorism_id=aphorism_id,
            persist_on_pass=persist_on_pass,
            strict_judge=strict_writer,
        )
    if script is None:
        review_path = metadata_dir / f"lofi_manual_review_{stamp}_{index:02d}.json"
        review_path.write_text(
            json.dumps(
                {"errors": errs, "theme": theme_row, "module": module, "script_only": script_only},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return LofiItemResult(
            ok=False,
            module=module,
            theme=str(theme_row.get("theme") or ""),
            scene_count=scene_count,
            duration_s=float(duration_s),
            manual_review=True,
            errors=errs,
            meta_path=str(review_path),
        )

    script_llm_log = get_script_llm_call_log()
    script_cost_log = get_script_cost_log()
    script_llm_calls = len(script_llm_log)
    script_llm_cost_usd = round(sum(c.get("cost_usd_est", 0.0) for c in script_cost_log), 6)
    print(
        f"[script cost] episode entries={len(script_cost_log)} "
        f"llm_calls={script_llm_calls} usd={script_llm_cost_usd}"
    )

    if script_only:
        theme = str(script.get("theme") or theme_row.get("theme") or "")
        sub = str(script.get("subtheme") or theme_row.get("subtheme") or "")
        _print_script_report(script, index=index, qty=batch_qty)
        theme_slug = _episode_slug(script, theme=theme, subtheme=sub)
        meta_path = metadata_dir / f"lofi_script_{theme_slug}_{stamp}_v{index:02d}.json"
        meta = {
            "post_type": "ECONOMIC_REEL_LOFI",
            "mode": "script_only",
            "page": page_id,
            "module": module,
            "duration_requested_s": duration_s,
            "scene_count": len(script.get("lines") or []),
            "scene_duration_s": lofi_cfg.beat_duration_s(),
            "duration_expected_s": round(
                sum(
                    float(row.get("duration_s") or lofi_cfg.beat_duration_s())
                    for row in (script.get("lines") or [])
                    if isinstance(row, dict)
                ),
                3,
            ),
            "subtheme": script.get("subtheme"),
            "hook_type": script.get("hook_type"),
            "arc_template": script.get("arc_template"),
            "retrieved_details": script.get("retrieved_details"),
            "script": script,
            "script_llm_calls": script_llm_calls,
            "script_llm_cost_usd": script_llm_cost_usd,
            "est_cost_usd": round(script_llm_cost_usd, 5),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_stage": "1_script",
            "niche": niche_notes,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = list(script.get("lines") or [])
        return LofiItemResult(
            ok=True,
            video_path=None,
            meta_path=str(meta_path),
            module=module,
            theme=theme,
            hook_type=str(script.get("hook_type") or ""),
            scene_count=len(lines),
            duration_s=lofi_cfg.duration_for_beat_count(len(lines)),
            manual_review=False,
            errors=[],
            script=script,
            pipeline_stage="1_script",
        )

    has_any_concept = any(
        isinstance(r, dict) and str(r.get("visual_concept") or r.get("scene_description") or "").strip()
        for r in (script.get("lines") or [])
    )
    missing_concept = any(
        isinstance(r, dict)
        and not str(r.get("visual_concept") or r.get("scene_description") or "").strip()
        for r in (script.get("lines") or [])
    )
    if missing_concept and not has_any_concept:
        script = strip_visual_fields(script)
        stamp_stage1_timing(
            script,
            beat_s=lofi_cfg.beat_duration_s(),
            duration_s=float(duration_s),
        )
        for row in script.get("lines") or []:
            if isinstance(row, dict):
                row.pop("subject_type", None)

    def _hold(gate: int, state: dict[str, Any], path: Path) -> LofiItemResult:
        write_state(path, state)
        blocked = "image/TTS" if gate == 1 else "image generation"
        print(
            f"[LOFI GATE {gate}] HOLD — no {blocked} until approved. Resume:\n"
            f"  python main.py --page {page_id} --post-type ECONOMIC_REEL_LOFI "
            f"--lofi-resume-from {path} --lofi-approve-gate {gate}"
        )
        if gate == 2:
            print("[LOFI GATE 2] concept list — approve before any Flux calls:")
            lines = [r for r in (script.get("lines") or []) if isinstance(r, dict)]
            by_scene = {int(r.get("scene") or 0): r for r in lines}
            for row in _gate2_review_rows(lines):
                src = by_scene.get(int(row.get("scene") or 0), {})
                desc = str(src.get("scene_description") or src.get("visual_concept") or "")
                desc = " ".join(desc.split())
                print(
                    f"  #{row.get('scene')} meaning={row.get('meaning')!r} | "
                    f"mood={row.get('beat_mood')!r} | "
                    f"who={row.get('who')} | shot={row.get('shot_type')} | "
                    f"env={row.get('environment')!r} | temp={row.get('palette_temp')} "
                    f"({row.get('arc_position')})"
                )
                print(
                    f"      object={src.get('key_object')!r} | {desc[:180]}"
                )
                print(f"      positive={str(row.get('visual_prompt') or '')}")
                print(f"      negative={str(row.get('negative_prompt') or '')}")
        return LofiItemResult(
            ok=True,
            meta_path=str(path),
            module=module,
            theme=str(script.get("theme") or ""),
            hook_type=str(script.get("hook_type") or ""),
            scene_count=len(script.get("lines") or []),
            duration_s=float(script.get("duration_s") or duration_s),
            manual_review=True,
            errors=[f"hold_gate_{gate}"],
            script=script,
            hold_gate=gate,
            pipeline_stage=f"gate{gate}_hold",
        )

    state_path = metadata_dir / f"lofi_pipeline_{stamp}_{index:02d}.json"
    if resume_state and resume_state.get("source"):
        src = Path(str(resume_state["source"]))
        if src.is_file() and "lofi_pipeline_" in src.name:
            state_path = src
    pipe_state: dict[str, Any] = dict(resume_state or {})
    pipe_state["script"] = script
    pipe_state["page"] = page_id
    pipe_state["module"] = module
    pipe_state["review_required"] = review_required
    pipe_state["niche"] = niche_notes
    pipe_state["stage_completed"] = max(int(pipe_state.get("stage_completed") or 0), 1)

    if not review_required:
        auto_pass_gates(pipe_state)
    elif locked_script is not None and not resume_from:
        mark_approved(pipe_state, 1, reason="locked_script")
    elif resume_from and approve_gate == 1:
        mark_approved(pipe_state, 1, reason="cli")

    gate1_ok = str((pipe_state.get("gate1") or {}).get("status") or "") in {
        "approved",
        "auto_pass",
    }
    if review_required and not gate1_ok:
        pipe_state["gate1"] = {"status": "pending"}
        pipe_state["gate2"] = pipe_state.get("gate2") or {"status": "pending"}
        return _hold(1, pipe_state, state_path)

    # TTS-first: once the narrative clears Gate 1, measure the actual spoken
    # beats before visual concepts are created. Gate 2 can then review prompts
    # whose pacing is already final. Stills-only preserves its no-TTS contract.
    prior_work_dir = str(pipe_state.get("work_dir") or "").strip()
    run_dir = (
        Path(prior_work_dir)
        if prior_work_dir
        else assets_dir / f"lofi_run_{_episode_slug(script)}_{stamp}_{index:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _purge_stale_temp_voice_files(run_dir)
    if (not stills_only) and bool(getattr(lofi_cfg, "ENABLE_VOICEOVER", True)):
        from core.economic_reel_lofi.voiceover import ensure_script_voiceover

        ensure_script_voiceover(pipe_state, script, run_dir)
        n_vo = len([r for r in (script.get("lines") or []) if isinstance(r, dict)])
        canon_vo = _current_run_scene_audio_paths(
            run_dir, n_vo, pipe_state.get("voice_paths")
        )
        pipe_state["voice_paths"] = [str(p) if p else None for p in canon_vo]
        _purge_stale_temp_voice_files(
            run_dir,
            keep_names={p.name for p in canon_vo if p is not None},
        )
        pipe_state["stage_completed"] = max(
            float(pipe_state.get("stage_completed") or 1), 1.5
        )
        print(
            f"[LOFI TTS-first] complete beats={len(pipe_state.get('voice_paths') or [])} "
            f"duration={float(pipe_state.get('duration_actual_s') or 0):.2f}s"
        )
        missing_concept = any(
            isinstance(row, dict)
            and not str(
                row.get("visual_concept") or row.get("scene_description") or ""
            ).strip()
            for row in (script.get("lines") or [])
        )
    else:
        pipe_state["work_dir"] = str(run_dir)

    if missing_concept:
        print("[LOFI stage2] translating beats (LLM). _EPISODE_WORLDS is not used.")
        translate_episode_visuals(script, module=module)
        from core.economic_reel_lofi.visual_identity import stamp_episode_anchor

        motif = str(
            ((script.get("lines") or [{}])[0] or {}).get("episode_visual_state", {}).get("motif")
            or ""
        )
        if motif:
            stamp_episode_anchor(list(script.get("lines") or []), motif)
        for row in script.get("lines") or []:
            if isinstance(row, dict):
                row.pop("visual_prompt", None)
        pipe_state["script"] = script
        pipe_state["stage_completed"] = 2

    # Stage 3 — assemble positive+negative (no Flux, no TTS). Gate 2 reviews this.
    sidecar = dict(script.get("_locked_sidecar_assets") or {})
    has_locked_stills = any(str(p).strip() for p in (sidecar.get("scene_images") or []))
    lock_visuals = bool(locked_script) and has_locked_stills
    lines = [r for r in (script.get("lines") or []) if isinstance(r, dict)]
    if lines:
        lines[0]["episode_theme"] = str(script.get("theme") or force_theme or "")
        lines[0]["episode_module"] = module
        lines[0]["episode_thesis"] = str(script.get("thesis") or "")
    script["lines"] = lines
    has_stage3 = any(str(r.get("visual_prompt") or "").strip() for r in lines)
    episode_variety: dict[str, Any] = dict(script.get("episode_variety") or {})
    if not has_stage3:
        print("[LOFI stage3] assembling prompts (style module + episode_anchor_stem)")
        episode_variety = _assemble_stage3_prompts(
            script,
            theme_row=theme_row,
            lock_visuals=lock_visuals,
            clips_dir=clips_dir,
            metadata_dir=metadata_dir,
            stamp=stamp,
            scene_count=scene_count,
        ) or episode_variety
        pipe_state["stage_completed"] = 3
        has_stage3 = True
    else:
        print("[LOFI stage3] skip — visual_prompt already stamped")
    pipe_state["script"] = script
    pipe_state["gate2_prompts"] = _gate2_review_rows(
        [r for r in (script.get("lines") or []) if isinstance(r, dict)]
    )

    gate2_ok = str((pipe_state.get("gate2") or {}).get("status") or "") in {
        "approved",
        "auto_pass",
    }
    if resume_from and approve_gate == 2:
        mark_approved(pipe_state, 2, reason="cli")
        gate2_ok = True
    if review_required and not gate2_ok:
        pipe_state["gate2"] = {"status": "pending"}
        return _hold(2, pipe_state, state_path)

    write_state(state_path, pipe_state)
    print("[LOFI GATE 2] cleared — Stage 4 images/QA + audio-driven assembly")
    use_dev = bool(lofi_cfg.uses_flux_dev())
    use_gemini = image_provider == "gemini"
    active_flux_backend = (
        "gemini"
        if use_gemini
        else ("flux2-dev" if lofi_cfg.uses_flux2_dev() else ("dev" if use_dev else "schnell"))
    )
    print(
        "[LOFI backend] "
        f"provider={image_provider} | "
        f"model={'gemini-flash-image' if use_gemini else ('flux2-dev' if lofi_cfg.uses_flux2_dev() else ('dev' if use_dev else 'schnell'))} | "
        f"profile={lofi_cfg.DEFAULT_VISUAL_IDENTITY_PROFILE} | "
        f"gen={lofi_cfg.LOFI_IMAGE_WIDTH}x{lofi_cfg.LOFI_IMAGE_HEIGHT} "
        f"delivery={lofi_cfg.REEL_WIDTH}x{lofi_cfg.REEL_HEIGHT}"
    )

    # Working stills and the Gate-1-approved VO share a stable run directory.

    lines = list(script.get("lines") or [])
    episode_variety = dict(script.get("episode_variety") or episode_variety)

    locked_assets = script.pop("_locked_sidecar_assets", None) or {}
    accept_scenes = {int(x) for x in (locked_assets.get("manual_accept_scenes") or [])}
    reuse_images = [Path(p) for p in (locked_assets.get("scene_images") or []) if str(p).strip()]
    if accept_scenes:
        print(f"[LOFI pipeline] manual_accept_scenes={sorted(accept_scenes)}")

    scene_paths: list[Path] = []
    captions: list[str] = []
    scene_moods: list[dict] = []
    qa_flags: list[str] = []
    tts_overruns: list[dict[str, Any]] = []
    voice_paths: list[Path | None] = []
    word_timings_per_scene: list[list[tuple[str, float, float]] | None] = []
    pre_voice_paths = [
        Path(str(path)) if path else None
        for path in (pipe_state.get("voice_paths") or [])
    ]
    pre_word_timings = list(pipe_state.get("word_timings_per_scene") or [])
    voice_settings_result: dict[str, Any] | None = None
    n_image_calls = 0
    n_critic_calls = 0
    object_gate_by_scene: list[dict[str, Any]] = []
    n_beats_planned = max(1, len(lines))
    call_budget = lofi_cfg.image_call_budget(n_beats_planned)
    lofi_cfg.log_image_budget(used=0, budget=call_budget, n_beats=n_beats_planned)
    print(
        f"[LOFI budget] image_call_cap={call_budget} "
        f"({n_beats_planned} beats x {lofi_cfg.IMAGE_CALL_BUDGET_MULT:g}) "
        f"attempts_per_beat={lofi_cfg.IMAGE_ATTEMPTS_PER_BEAT} HARD CAP"
    )

    for row in lines:
        scene_i = int(row.get("scene") or len(scene_paths) + 1)
        out_img = run_dir / f"scene_{scene_i:02d}.png"
        caption = _sanitize_caption_typos(str(row.get("text") or ""))
        mood_meta = {
            "id": str(row.get("riso_id") or f"scene_{scene_i}"),
            "lighting": "from_riso_prompt",
            "palette": row.get("riso_palette"),
            "shadow": lofi_cfg.DUOTONE_SHADOW,
            "highlight": lofi_cfg.DUOTONE_HIGHLIGHT,
        }
        ok_img = False
        last_flaws: list[str] = []
        subject_type = str(row.get("subject_type") or "").strip()
        key_object = str(row.get("key_object") or "").strip()
        last_gate: dict[str, Any] = {
            "scene": scene_i,
            "subject_type": subject_type,
            "key_object": key_object,
            "passed": True,
            "skipped": True,
        }
        reuse_src = reuse_images[scene_i - 1] if 0 < scene_i <= len(reuse_images) else None
        prev_ok = bool((row.get("default_object_gate") or {}).get("image_ok"))
        want_tag = lofi_cfg.current_still_style_tag()
        src_tag = lofi_cfg.still_style_tag_of(reuse_src) if reuse_src else None
        allow_mixed = lofi_cfg.allow_mixed_era_assemble()
        reuse_ok = bool(
            reuse_src
            and reuse_src.is_file()
            and (prev_ok or scene_i in accept_scenes)
        )
        if reuse_ok and src_tag != want_tag:
            if allow_mixed:
                print(
                    f"[LOFI reuse] mixed-era APPROVED scene={scene_i} "
                    f"src={src_tag or 'UNTAGGED'} want={want_tag}"
                )
            else:
                print(
                    f"[LOFI reuse] REJECT stale scene={scene_i} "
                    f"src={src_tag or 'UNTAGGED'} want={want_tag} "
                    f"— generating fresh"
                )
                reuse_ok = False
        if reuse_ok:
            if Path(reuse_src) != out_img:
                shutil.copy2(reuse_src, out_img)
            keep_tag = src_tag if (src_tag and src_tag != want_tag) else want_tag
            lofi_cfg.write_still_style_sidecar(
                out_img,
                run_id=run_dir.name,
                reused=True,
                style_tag=keep_tag,
            )
            ok_img = True
            last_gate = dict(row.get("default_object_gate") or last_gate)
            last_gate["passed"] = True
            last_gate["qa_passed"] = True
            last_gate["qa_flaws"] = []
            last_gate["image_ok"] = True
            last_gate["reused"] = True
            last_gate["manual_accept"] = scene_i in accept_scenes
            print(
                f"[LOFI reuse] scene={scene_i} "
                f"{'manual_accept' if scene_i in accept_scenes else 'prior_pass'} "
                f"← {reuse_src.name}"
            )
            cov_ok, cov_flaws, cov_meta = assess_garment_coverage(
                out_img,
                subject_type=subject_type,
                close_variant=str(row.get("close_variant") or ""),
            )
            last_gate["garment_coverage"] = cov_meta
            if cov_flaws:
                print(
                    f"[LOFI reuse] coverage REJECT scene={scene_i} — generating fresh"
                )
                log_coverage_flag(
                    {
                        "scene": scene_i,
                        "attempt": "reuse",
                        "subject_type": subject_type,
                        "close_variant": str(row.get("close_variant") or ""),
                        "prompt": str(row.get("visual_prompt") or ""),
                        "negative_prompt": getattr(
                            lofi_cfg, "LOFI_DEV_NEGATIVE_PROMPT", ""
                        ),
                        "seed": None,
                        "guidance_scale": getattr(
                            lofi_cfg, "LOFI_DEV_GUIDANCE_SCALE", None
                        ),
                        "model": getattr(lofi_cfg, "LOFI_DEV_IMAGE_MODEL", None),
                        "flaws": list(cov_flaws),
                        "note": cov_meta.get("note") or "reuse still failed coverage",
                        "image": str(out_img),
                    }
                )
                reuse_ok = False
                ok_img = False
        if not reuse_ok:
            remaining = call_budget - n_image_calls
            if remaining <= 0:
                qa_flags.append(
                    f"scene_{scene_i}: skipped — episode image-call budget "
                    f"{call_budget} exhausted"
                )
                print(
                    f"[LOFI budget] STOP scene={scene_i} calls={n_image_calls}/"
                    f"{call_budget} — unresolved"
                )
                if not out_img.is_file():
                    from PIL import Image as PILImage

                    PILImage.new("RGB", (LOFI_IMAGE_WIDTH, LOFI_IMAGE_HEIGHT), (30, 40, 55)).save(out_img)
                last_gate = {
                    "scene": scene_i,
                    "subject_type": subject_type,
                    "key_object": key_object,
                    "passed": False,
                    "qa_passed": False,
                    "qa_flaws": ["episode image-call budget exhausted"],
                    "image_ok": False,
                    "skipped": True,
                    "skip_reason": "image_call_budget",
                }
                object_gate_by_scene.append(last_gate)
                row["default_object_gate"] = last_gate
                row["riso_id"] = mood_meta.get("id")
                scene_paths.append(out_img)
                captions.append(caption)
                scene_moods.append(mood_meta)
                prior_i = scene_i - 1
                voice_paths.append(
                    pre_voice_paths[prior_i] if prior_i < len(pre_voice_paths) else None
                )
                word_timings_per_scene.append(
                    pre_word_timings[prior_i] if prior_i < len(pre_word_timings) else None
                )
                continue
            gen_kw: dict[str, Any] = {
                "mood": mood_meta,
                "attempt_budget": lofi_cfg.clamp_attempt_budget(remaining),
            }
            if use_gemini:
                from core.economic_reel_lofi.image_gen import (
                    generate_scene_image_gemini,
                )

                gen_kw["generate_fn"] = generate_scene_image_gemini
            elif use_dev:
                from core.economic_reel_lofi.image_gen import (
                    generate_scene_image_dev,
                )
                from core.economic_reel_lofi.visual_identity import (
                    assemble_v2_prompt_dev,
                )

                gen_kw["generate_fn"] = generate_scene_image_dev
                gen_kw["assemble_fn"] = assemble_v2_prompt_dev
            ok_img, last_gate, n_i, n_c = generate_and_qa_scene(
                row, out_img, **gen_kw,
            )
            n_image_calls += n_i
            n_critic_calls += n_c
            last_flaws = list(last_gate.get("qa_flaws") or [])
            if isinstance(last_gate.get("mood"), dict):
                mood_meta = last_gate["mood"]
        object_gate_by_scene.append(last_gate)
        row["default_object_gate"] = last_gate
        if not ok_img:
            qa_flags.append(f"scene_{scene_i}: {'; '.join(last_flaws) or 'qa failed'}")
            if not out_img.is_file():
                from PIL import Image as PILImage

                PILImage.new("RGB", (LOFI_IMAGE_WIDTH, LOFI_IMAGE_HEIGHT), (30, 40, 55)).save(out_img)
        row["riso_id"] = mood_meta.get("id")
        scene_paths.append(out_img)
        captions.append(caption)
        scene_moods.append(mood_meta)

        # VO was generated immediately after Gate 1 and persisted in gate state.
        prior_i = scene_i - 1
        vo_path = pre_voice_paths[prior_i] if prior_i < len(pre_voice_paths) else None
        timings = pre_word_timings[prior_i] if prior_i < len(pre_word_timings) else None
        if (
            not stills_only
            and bool(getattr(lofi_cfg, "REQUIRE_VOICEOVER", True))
            and (not vo_path or not vo_path.is_file())
        ):
            qa_flags.append(f"scene_{scene_i} missing Gate-1-approved voiceover")
        voice_paths.append(vo_path)
        word_timings_per_scene.append(timings)

    if tts_overruns:
        applied, stop = lofi_cfg.apply_isolated_tts_duration_bumps(
            tts_overruns,
            requested_total_s=float(duration_s),
            n_beats=max(1, len(lines)),
        )
        if stop:
            qa_flags.append(stop)
        else:
            script["duration_auto_bumps"] = applied
            for rec in tts_overruns:
                row_i = rec.get("row")
                if isinstance(row_i, dict) and rec.get("bumped_s") is not None:
                    row_i["duration_s"] = float(rec["bumped_s"])

    try:
        (run_dir / "vo_meta.json").write_text(
            json.dumps(
                {
                    "tts_speed": lofi_cfg.tts_speed(),
                    "tts_model": lofi_cfg.tts_model(),
                    "tts_voice_id": lofi_cfg.tts_voice_id(),
                    "trim_tts_edges": bool(getattr(lofi_cfg, "VO_TRIM_TTS_EDGES", True)),
                    "files": [p.name if p else None for p in voice_paths],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[LOFI VO] warn could not write vo_meta.json: {exc}")

    xstyle = assess_cross_beat_style(scene_paths, lines)
    riso_locked = bool(
        lines
        and all(
            str(row.get("visual_source") or "") == "writer_single_pass"
            and "risograph print" in str(row.get("visual_prompt") or "").lower()
            for row in lines
        )
    )
    if riso_locked and xstyle.get("fails"):
        xstyle["diagnostic_fails"] = list(xstyle["fails"])
        xstyle["fails"] = []
        xstyle["programmatic_style_lock"] = True
        print(
            "[LOFI cross-style] programmatic Risograph lock PASS; "
            "pixel clusters retained as diagnostics"
        )
    script["cross_beat_style"] = xstyle
    script["declared_lighting_beats"] = list(script.get("lighting_beats") or [])
    script["declared_dominant_lighting"] = str(script.get("dominant_lighting") or "")
    pix_counts = dict(xstyle.get("pixel_counts") or {})
    script["pixel_lighting_counts"] = pix_counts
    script["pixel_lighting_beats"] = [
        str(r.get("pixel_lighting") or "") for r in (xstyle.get("stats") or [])
    ]
    if pix_counts:
        script["dominant_lighting"] = max(pix_counts, key=pix_counts.get)
    if xstyle.get("fails"):
        qa_flags.extend(str(f) for f in xstyle["fails"])
    obj_ship = assess_object_beat_visual_continuity(lines, scene_paths)
    script["object_beat_continuity"] = obj_ship
    if obj_ship.get("fails"):
        qa_flags.extend(str(f) for f in obj_ship["fails"])
    for rec in xstyle.get("stats") or []:
        i = int(rec.get("scene") or 0) - 1
        if 0 <= i < len(lines) and isinstance(lines[i], dict):
            lines[i]["still_stats"] = {
                "uniq16": rec["uniq16"],
                "lap_var": rec["lap_var"],
                "edge": rec["edge"],
                "std": rec["std"],
            }
            lines[i]["pixel_lighting"] = rec["pixel_lighting"]

    theme_slug = _episode_slug(script)

    print("[LOFI object-gate] per-beat")
    for g in object_gate_by_scene:
        attempts = list(g.get("attempts") or [g])
        status = "SKIP" if g.get("skipped") else ("PASS" if g.get("passed") else "FAIL")
        print(
            f"[LOFI object-gate] scene={g.get('scene')} {status} "
            f"type={g.get('subject_type')} object={g.get('key_object')!r} "
            f"bbox={g.get('intruder_bbox_frac')} ratio={g.get('ratio')} "
            f"n={g.get('n_intruders')} step={g.get('focus_step')}"
        )
        if len(attempts) > 1 or (attempts and attempts[0].get("attempt")):
            for a in attempts:
                a_st = "SKIP" if a.get("skipped") else ("PASS" if a.get("passed") else "FAIL")
                print(
                    f"[LOFI object-gate]   attempt={a.get('attempt')} {a_st} "
                    f"step={a.get('focus_step')} framing={a.get('framing')!r} "
                    f"bbox={a.get('intruder_bbox_frac')} ratio={a.get('ratio')}"
                )
                sg = a.get("style_gate") if isinstance(a.get("style_gate"), dict) else {}
                if sg and not sg.get("skipped"):
                    sg_st = "PASS" if sg.get("passed") else "FAIL"
                    print(
                        f"[LOFI style-gate]   attempt={a.get('attempt')} {sg_st} "
                        f"std={sg.get('std')} edge={sg.get('edge')} "
                        f"lap_var={sg.get('lap_var')} uniq16={sg.get('uniq16')}"
                    )

    # Per-image rate follows the provider/model this run actually posted.
    from quality.VisualQA_Agent.config import COST_GEMINI_FLASH_USD

    img_cost_per_call, cost_meta = lofi_cfg.lofi_image_cost_per_call_usd(
        LOFI_IMAGE_WIDTH, LOFI_IMAGE_HEIGHT, schnell_steps=LOFI_IMAGE_STEPS
    )
    img_cost = img_cost_per_call * n_image_calls
    critic_cost = COST_GEMINI_FLASH_USD * n_critic_calls
    media_cost = img_cost + critic_cost
    total_est_cost = round(media_cost + script_llm_cost_usd, 5)
    print(
        f"[LOFI cost] backend={cost_meta.get('backend')} "
        f"provider={cost_meta.get('provider')} "
        f"rate=${img_cost_per_call:.6f}/image "
        f"({cost_meta.get('formula')}) | "
        f"images={n_image_calls}/{call_budget} "
        f"image_cost=${img_cost:.4f} | critic={n_critic_calls} "
        f"critic_cost=${critic_cost:.4f} | script_cost=${script_llm_cost_usd:.4f} "
        f"| total_est_cost=${total_est_cost:.4f}"
    )
    lofi_cfg.log_image_budget(
        used=n_image_calls, budget=call_budget, n_beats=n_beats_planned
    )
    if cost_meta.get("note"):
        print(f"[LOFI cost] {cost_meta['note']}")
    cost_delta = lofi_cfg.lofi_image_cost_delta()
    print(
        f"[LOFI cost] delta old={cost_delta['old_size']} "
        f"${cost_delta['old_usd_per_image']:.6f}/image "
        f"({cost_delta['old_mp']} MP) → new={cost_delta['new_size']} "
        f"${cost_delta['new_usd_per_image']:.6f}/image "
        f"({cost_delta['new_mp']} MP) usd_delta=${cost_delta['usd_delta']:.6f} "
        f"pixel_ratio={cost_delta['pixel_ratio']} "
        f"cost_ratio={cost_delta['cost_ratio']} "
        f"scaling={cost_delta['scaling']} "
        f"time_est_s={cost_delta['time_est_s_linear']} "
        f"(baseline {cost_delta['time_baseline_s']}s @ {cost_delta['old_size']})"
    )

    if stills_only:
        n_beats = max(1, len(captions) or len(lines))
        stills_cost = media_cost
        # Full render pays the same image+critic, plus 9 TTS calls and MoviePy encode.
        # ElevenLabs eleven_multilingual_v2 (LOFI_TTS_MODEL); 9 short captions ≈ cents.
        # Encode on the last full reel was ~8 min after stills were done.
        print(
            f"[LOFI stills-only] beats={n_beats} images={n_image_calls} "
            f"critic={n_critic_calls} est_cost=${stills_cost:.4f} "
            f"(image ${img_cost:.4f} + critic ${critic_cost:.4f}) | "
            f"skipped TTS + video assemble"
        )
        print(
            "[LOFI stills-only] vs full render: same image+critic cost; "
            "saves ~9 ElevenLabs TTS calls (~$0.03–0.08) and ~5–8 min MoviePy encode. "
            "Typical full 9-beat wall time ~12 min; stills-only is image+QA only."
        )
        mix: dict[str, int] = {}
        for row in lines:
            st = str(row.get("subject_type") or "?").strip().lower() or "?"
            mix[st] = mix.get(st, 0) + 1
        print(f"[LOFI stills-only] subject_type mix={mix}")
        for row in lines:
            print(
                f"[LOFI stills-only] scene={row.get('scene')} "
                f"type={row.get('subject_type')} object={row.get('key_object')!r} "
                f"text={str(row.get('text') or '')[:56]!r}"
            )
        meta = {
            "post_type": "ECONOMIC_REEL_LOFI",
            "mode": "stills_only",
            "page": page_id,
            "module": module,
            "duration_requested_s": duration_s,
            "scene_count": n_beats,
            "scene_duration_s": lofi_cfg.beat_duration_s(),
            "duration_expected_s": lofi_cfg.duration_for_beat_count(n_beats),
            "subtheme": script.get("subtheme"),
            "hook_type": script.get("hook_type"),
            "quote_id": script.get("quote_id"),
            "anchor_object": script.get("anchor_object"),
            "arc_template": script.get("arc_template"),
            "retrieved_details": script.get("retrieved_details"),
            "script": script,
            "riso_ids": [str(r.get("riso_id") or "") for r in lines],
            "visual_identity": "v2"
            if bool(getattr(lofi_cfg, "USE_VISUAL_IDENTITY_V2", False))
            else "riso_library",
            "flux_backend": active_flux_backend,
            "visual_identity_profile": (
                lofi_cfg.DEFAULT_VISUAL_IDENTITY_PROFILE if use_dev else None
            ),
            "scene_images": [str(p) for p in scene_paths],
            "work_dir": str(run_dir),
            "visual_qa_flags": qa_flags,
            "manual_review": bool(qa_flags),
            "episode_variety": episode_variety,
            "object_gate_by_scene": object_gate_by_scene,
            "image_calls": n_image_calls,
            "image_call_budget": call_budget,
            "image_cost_per_call": img_cost_per_call,
            "image_cost_meta": cost_meta,
            "critic_calls": n_critic_calls,
            "script_llm_calls": script_llm_calls,
            "script_llm_cost_usd": script_llm_cost_usd,
            "est_cost_usd": total_est_cost,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = metadata_dir / f"lofi_stills_{theme_slug}_{stamp}_v{index:02d}.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[LOFI stills-only] wrote {meta_path}")
        print(f"[LOFI stills-only] stills {run_dir}")
        if qa_flags:
            print(
                "[LOFI HOLD] QA exhausted — stills kept for review, episode NOT "
                f"cleared for render/post. flags={qa_flags}"
            )
        return LofiItemResult(
            ok=not bool(qa_flags),
            video_path=None,
            meta_path=str(meta_path),
            module=module,
            theme=str(script.get("theme") or ""),
            hook_type=str(script.get("hook_type") or ""),
            scene_count=n_beats,
            duration_s=lofi_cfg.duration_for_beat_count(n_beats),
            manual_review=bool(qa_flags),
            errors=qa_flags,
            script=script,
            work_dir=str(run_dir),
            stills_only=True,
        )

    if qa_flags:
        n_beats = max(1, len(captions) or len(lines))
        print(
            "[LOFI HOLD] QA exhausted — NOT assembling MP4, not postable. "
            f"flags={qa_flags}"
        )
        print(f"[LOFI HOLD] stills kept at {run_dir}")
        hold_meta = {
            "post_type": "ECONOMIC_REEL_LOFI",
            "mode": "qa_hold",
            "page": page_id,
            "module": module,
            "duration_requested_s": duration_s,
            "scene_count": n_beats,
            "subtheme": script.get("subtheme"),
            "hook_type": script.get("hook_type"),
            "arc_template": script.get("arc_template"),
            "script": script,
            "scene_images": [str(p) for p in scene_paths],
            "work_dir": str(run_dir),
            "visual_qa_flags": qa_flags,
            "manual_review": True,
            "episode_variety": episode_variety,
            "object_gate_by_scene": object_gate_by_scene,
            "video_path": None,
            "flux_backend": active_flux_backend,
            "visual_identity_profile": (
                lofi_cfg.DEFAULT_VISUAL_IDENTITY_PROFILE if use_dev else None
            ),
            "image_calls": n_image_calls,
            "image_call_budget": call_budget,
            "image_cost_per_call": img_cost_per_call,
            "image_cost_meta": cost_meta,
            "critic_calls": n_critic_calls,
            "script_llm_calls": script_llm_calls,
            "script_llm_cost_usd": script_llm_cost_usd,
            "est_cost_usd": total_est_cost,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        hold_path = metadata_dir / f"lofi_hold_{theme_slug}_{stamp}_v{index:02d}.json"
        hold_path.write_text(
            json.dumps(hold_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[LOFI HOLD] meta {hold_path}")
        return LofiItemResult(
            ok=False,
            video_path=None,
            meta_path=str(hold_path),
            module=module,
            theme=str(script.get("theme") or ""),
            hook_type=str(script.get("hook_type") or ""),
            scene_count=n_beats,
            duration_s=0.0,
            manual_review=True,
            errors=qa_flags,
            script=script,
            work_dir=str(run_dir),
        )

    out_mp4 = clips_dir / f"lofi_reel_{theme_slug}_{stamp}_v{index:02d}.mp4"
    n_beats = max(1, len(captions) or len(lines))
    lock_beat = bool(getattr(lofi_cfg, "LOCK_FIXED_BEAT_DURATION", True))
    beat_s = float(lofi_cfg.beat_duration_s())
    scene_durations: list[float] = []
    scene_timing_flags: list[dict[str, Any]] = []
    tts_durs = [
        float(x)
        for x in (pipe_state.get("scene_durations") or [])
        if x is not None
    ]
    if len(tts_durs) == n_beats and all(d > 0 for d in tts_durs):
        scene_durations = [round(float(d), 3) for d in tts_durs]
        for i, cap in enumerate(captions):
            vo_dur = 0.0
            if i < len(lines) and isinstance(lines[i], dict):
                try:
                    vo_dur = float(lines[i].get("vo_duration_s") or 0.0)
                except (TypeError, ValueError):
                    vo_dur = 0.0
            scene_timing_flags.append(
                {
                    "scene": i + 1,
                    "text": cap,
                    "extended": scene_durations[i] > beat_s,
                    "base_s": beat_s,
                    "vo_dur": round(vo_dur, 3),
                    "needed_s": scene_durations[i],
                    "duration_s": scene_durations[i],
                    "locked": True,
                    "source": "tts_first",
                }
            )
        print(
            f"[LOFI assemble] TTS-first durations passed to renderer: {scene_durations}"
        )
    for i, cap in enumerate(captions):
        if scene_durations and i < len(scene_durations):
            continue
        timings_i = word_timings_per_scene[i] if i < len(word_timings_per_scene) else None
        vp_i = voice_paths[i] if i < len(voice_paths) else None
        if lock_beat:
            vo_dur = 0.0
            if vp_i and Path(vp_i).is_file():
                try:
                    vo_dur = float(measure_vo_speech_duration(Path(vp_i)))
                except Exception:  # noqa: BLE001
                    vo_dur = 0.0
            trail = (
                0.0
                if i >= n_beats - 1
                else float(getattr(lofi_cfg, "VO_INTERLINE_SILENCE_S", 0.30))
            )
            declared_s = float(
                (lines[i].get("duration_s") if i < len(lines) and isinstance(lines[i], dict) else None)
                or beat_s
            )
            if vo_dur > 0.05 and i >= n_beats - 1:
                trail = float(getattr(lofi_cfg, "VO_SLOT_PAD_S", 0.12))
            dur_i, extended_i = lofi_cfg.slot_duration_for_vo(
                vo_dur,
                base_s=0.0 if vo_dur > 0.05 else declared_s,
                trailing_silence_s=trail,
            )
            if vo_dur > 0.05:
                print(
                    f"[LOFI assemble] scene {i + 1} audio-driven "
                    f"vo={vo_dur:.3f}s slot={dur_i:.3f}s "
                    f"ignored_estimate={declared_s:.3f}s"
                )
            n_words = len(str(cap).split())
            wps = (n_words / vo_dur) if vo_dur > 0.05 else 0.0
            dur_meta = {
                "base_s": beat_s,
                "vo_dur": round(vo_dur, 3),
                "needed_s": dur_i,
                "duration_s": dur_i,
                "locked": True,
                "vo_trimmed": False,
                "words": n_words,
                "wps": round(wps, 2),
            }
            print(
                f"[LOFI vo-pace] scene {i + 1} words={n_words} "
                f"vo={vo_dur:.2f}s slot={dur_i:.2f}s "
                f"extended={int(extended_i)} wps={wps:.2f} | text={cap!r}"
            )
            if extended_i:
                print(
                    f"[LOFI caption-timing] scene {i + 1} VO {vo_dur:.2f}s > "
                    f"{beat_s:.1f}s base — extending slot to {dur_i:.2f}s "
                    f"(never trim VO) | text={cap!r}"
                )
        else:
            dur_i, extended_i, dur_meta = compute_caption_scene_duration_s(
                timings_i,
                Path(vp_i) if vp_i else None,
            )
        scene_durations.append(dur_i)
        scene_timing_flags.append(
            {
                "scene": i + 1,
                "text": cap,
                "extended": extended_i,
                **dur_meta,
            }
        )
        if extended_i:
            print(
                f"[LOFI caption-timing] FLAG scene {i + 1} extended to {dur_i:.2f}s "
                f"(base={lofi_cfg.SCENE_DURATION_S:.1f}s) so the line can fully "
                f"display + hold | text={cap!r}"
            )
    actual_dur = float(sum(scene_durations))
    print(
        f"[LOFI timing] beats={n_beats} beat_s={beat_s:.1f} "
        f"total={actual_dur:.1f}s locked={int(lock_beat)} "
        f"(requested_writer_target={duration_s}s)"
    )
    want_tag = lofi_cfg.current_still_style_tag()
    by_tag: dict[str, list[int]] = {}
    for i, p in enumerate(scene_paths, start=1):
        tag = lofi_cfg.still_style_tag_of(p) or "UNTAGGED"
        by_tag.setdefault(tag, []).append(i)
    style_ok = set(by_tag.keys()) == {want_tag}
    if not style_ok:
        bits = []
        for tag, idxs in sorted(by_tag.items()):
            kind = "fresh" if tag == want_tag else "stale"
            bits.append(
                f"{len(idxs)} of {len(scene_paths)} beats are {kind} "
                f"(style {tag}) scenes={idxs}"
            )
        mix_msg = " — ".join(bits) + (
            " — cannot produce a consistent video automatically."
        )
        if not lofi_cfg.allow_mixed_era_assemble():
            print(f"[LOFI assemble] REFUSE mixed-era: {mix_msg}")
            hold_path = metadata_dir / f"lofi_hold_mixed_era_{stamp}_v{index:02d}.json"
            hold_path.write_text(
                json.dumps(
                    {
                        "post_type": "ECONOMIC_REEL_LOFI",
                        "mode": "mixed_era_hold",
                        "error": mix_msg,
                        "want_style_tag": want_tag,
                        "style_tags": by_tag,
                        "scene_images": [str(p) for p in scene_paths],
                        "work_dir": str(run_dir),
                        "script": script,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            return LofiItemResult(
                ok=False,
                module=module,
                theme=str(script.get("theme") or ""),
                hook_type=str(script.get("hook_type") or ""),
                scene_count=scene_count,
                duration_s=actual_dur,
                manual_review=True,
                errors=[mix_msg],
                script=script,
                meta_path=str(hold_path),
            )
        print(f"[LOFI assemble] mixed-era APPROVED: {mix_msg}")
    else:
        print(
            f"[LOFI assemble] style_ok=1 tag={want_tag} "
            f"beats={len(scene_paths)}"
        )
    try:
        from core.economic_reel_lofi.ship_gates import (
            ShipGateError,
            assert_script_cleared_for_assemble,
            beat_integrity_blockers,
        )

        assert_script_cleared_for_assemble(
            script, module=module, scene_count=n_beats
        )
        if qa_flags:
            raise ShipGateError(
                "visual_qa_hold still set at assemble — hard stop"
            )
        _integrity = beat_integrity_blockers(
            script=script,
            scene_images=scene_paths,
            captions=captions,
            voice_paths=voice_paths,
            require_voiceover=bool(getattr(lofi_cfg, "REQUIRE_VOICEOVER", True)),
            expected_beats=n_beats,
        )
        if _integrity:
            raise ShipGateError("; ".join(_integrity))
        voice_paths = _current_run_scene_audio_paths(run_dir, n_beats, voice_paths)
        _purge_stale_temp_voice_files(
            run_dir,
            keep_names={p.name for p in voice_paths if p is not None},
        )
        assemble_audit: dict[str, Any] = {}
        assemble_lofi_reel(
            scene_paths,
            captions,
            out_mp4,
            engine_root=_engine_root(),
            page_id=page_id,
            scene_duration_s=lofi_cfg.SCENE_DURATION_S,
            scene_durations=scene_durations,
            moods=scene_moods,
            caption_style=lofi_cfg.DEFAULT_CAPTION_STYLE,
            voice_paths=voice_paths,
            word_timings_per_scene=word_timings_per_scene,
            caption_beats_per_scene=[
                list(r.get("caption_beats") or [])
                if isinstance(r, dict)
                else []
                for r in lines
            ],
            sidecar_dir=metadata_dir,
            vo_sidecar_dir=run_dir,
            audit_out=assemble_audit,
        )
    except ShipGateError as exc:
        _LOG.error("ship gate blocked assemble: %s", exc)
        hold_path = metadata_dir / f"lofi_manual_review_{stamp}_{index:02d}.json"
        hold_path.write_text(
            json.dumps(
                {
                    "errors": [str(exc)],
                    "gate": "ship_gate_pre_assemble",
                    "module": module,
                    "script": script,
                    "visual_qa_flags": qa_flags,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        return LofiItemResult(
            ok=False,
            module=module,
            theme=str(script.get("theme") or ""),
            hook_type=str(script.get("hook_type") or ""),
            scene_count=scene_count,
            duration_s=actual_dur,
            manual_review=True,
            errors=[str(exc)],
            script=script,
            meta_path=str(hold_path),
            work_dir=str(run_dir),
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.error("assemble failed: %s", exc, exc_info=True)
        return LofiItemResult(
            ok=False,
            module=module,
            theme=str(script.get("theme") or ""),
            hook_type=str(script.get("hook_type") or ""),
            scene_count=scene_count,
            duration_s=actual_dur,
            manual_review=True,
            errors=[f"assemble failed: {exc}"],
            script=script,
        )

    meta = {
        "post_type": "ECONOMIC_REEL_LOFI",
        "page": page_id,
        "module": module,
        "duration_requested_s": duration_s,
        "duration_actual_s": actual_dur,
        "duration_expected_s": lofi_cfg.duration_for_beat_count(n_beats),
        "scene_count": n_beats,
        "scene_duration_s": beat_s,
        "pacing": "locked_3s_per_beat" if lock_beat else "vo_extend",
        "scene_durations": scene_durations,
        "caption_timing": scene_timing_flags,
        "voice_settings_api": voice_settings_result,
        "voice_speed": lofi_cfg.tts_speed(),
        "tts_speed": lofi_cfg.tts_speed(),
        "tts_model": lofi_cfg.tts_model(),
        "tts_voice_id": lofi_cfg.tts_voice_id(),
        "logo_path": str(
            lofi_cfg.resolve_logo_path(page_id, _engine_root()) or ""
        ),
        "caption_line_height_frac": lofi_cfg.CAPTION_LINE_HEIGHT_FRAC,
        "subtheme": script.get("subtheme"),
        "hook_type": script.get("hook_type"),
        "quote_id": script.get("quote_id"),
        "anchor_object": script.get("anchor_object"),
        "arc_template": script.get("arc_template"),
        "retrieved_details": script.get("retrieved_details"),
        "script": script,
        "riso_ids": [str(r.get("riso_id") or "") for r in lines],
        "visual_identity": "v2"
        if bool(getattr(lofi_cfg, "USE_VISUAL_IDENTITY_V2", False))
        else "riso_library",
        "flux_backend": active_flux_backend,
        "visual_identity_profile": (
            lofi_cfg.DEFAULT_VISUAL_IDENTITY_PROFILE if use_dev else None
        ),
        "still_style_tag": lofi_cfg.current_still_style_tag(),
        "gen_width": lofi_cfg.LOFI_IMAGE_WIDTH,
        "gen_height": lofi_cfg.LOFI_IMAGE_HEIGHT,
        "delivery_width": lofi_cfg.REEL_WIDTH,
        "delivery_height": lofi_cfg.REEL_HEIGHT,
        "aspect": round(lofi_cfg.LOFI_IMAGE_WIDTH / lofi_cfg.LOFI_IMAGE_HEIGHT, 6),
        "image_cost_delta": cost_delta,
        "watermark_native_size": int(assemble_audit.get("watermark_native_size") or 0),
        "watermark_audit": assemble_audit,
        "voice_id": lofi_cfg.tts_voice_id(),
        "caption_style": lofi_cfg.DEFAULT_CAPTION_STYLE,
        "grading_applied": bool(getattr(lofi_cfg, "LOFI_APPLY_GRADING", False)),
        "video_path": str(out_mp4),
        "scene_images": [str(p) for p in scene_paths],
        "voice_paths": [str(p) if p else None for p in voice_paths],
        "work_dir": str(run_dir),
        "visual_qa_flags": qa_flags,
        "manual_review": False,
        "episode_variety": episode_variety,
        "object_gate_by_scene": object_gate_by_scene,
        "image_calls": n_image_calls,
        "image_call_budget": call_budget,
        "image_cost_per_call": img_cost_per_call,
        "image_cost_meta": cost_meta,
        "critic_calls": n_critic_calls,
        "script_llm_calls": script_llm_calls,
        "script_llm_cost_usd": script_llm_cost_usd,
        "est_cost_usd": total_est_cost,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = metadata_dir / f"{out_mp4.stem}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    ao = script.get("anchor_object") if isinstance(script.get("anchor_object"), dict) else {}
    rag.record_video_performance(
        module,
        theme=str(script.get("theme") or ""),
        subtheme=str(script.get("subtheme") or ""),
        views=None,
        completion_rate=None,
        anchor_object=str(ao.get("name") or "") or None,
        video_path=str(out_mp4),
        retrieved_details=list(script.get("retrieved_details") or []),
        close_variant=str(script.get("close_variant") or ""),
        close_target=str(script.get("close_target") or ""),
        eye_close_context=str(script.get("eye_close_context") or ""),
        close_character=str(script.get("close_character") or ""),
        dominant_lighting=str(script.get("dominant_lighting") or ""),
        lighting_beats=list(script.get("lighting_beats") or []),
    )

    return LofiItemResult(
        ok=True,
        video_path=str(out_mp4),
        meta_path=str(meta_path),
        module=module,
        theme=str(script.get("theme") or ""),
        hook_type=str(script.get("hook_type") or ""),
        scene_count=n_beats,
        duration_s=actual_dur,
        manual_review=False,
        errors=qa_flags,
        script=script,
    )


def run_economic_reel_lofi(
    *,
    page_id: str,
    quantity: int = 1,
    duration: int | None = None,
    module: str | None = None,
    outputs_dir: Path | str | None = None,
    theme: str | None = None,
    subtheme: str | None = None,
    writer_mode: str = "emotional",
    seed_quote: str | None = None,
    aphorism_id: str | None = None,
    image_provider: str = "together",
    script_only: bool = False,
    stills_only: bool = False,
    locked_scripts: list[str] | None = None,
    review_required: bool | None = None,
    resume_from: str | None = None,
    approve_gate: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    """
    Registry entrypoint for ECONOMIC_REEL_LOFI.

    Final MP4 + metadata → ``outputs/<page>/clips/``
    Working scene stills → ``outputs/<page>/assets/lofi_run_<episode>_*/``
    """
    page = (page_id or "").strip().lower()
    mod = lofi_cfg.validate_module_for_page(module or "relationship", page)
    mode = str(writer_mode or "emotional").strip().lower()
    if mode not in {"emotional", "theme", "quote", "paraphrase"}:
        raise ValueError("--lofi-mode must be emotional, theme, quote, or paraphrase")
    if mode == "quote" and not str(seed_quote or "").strip():
        raise ValueError("--lofi-mode quote requires --lofi-seed-quote")
    provider = str(image_provider or "together").strip().lower()
    if provider not in {"together", "gemini"}:
        raise ValueError("--lofi-image-provider must be together or gemini")
    os.environ["LOFI_IMAGE_PROVIDER"] = provider
    dur = lofi_cfg.validate_duration(duration if duration is not None else lofi_cfg.DEFAULT_DURATION_S)
    qty = max(1, int(quantity))
    if review_required is None:
        env = str(os.environ.get("LOFI_REVIEW_REQUIRED") or "").strip().lower()
        if env in {"0", "false", "no", "off"}:
            review_required = False
        elif env in {"1", "true", "yes", "on"}:
            review_required = True
        else:
            review_required = bool(lofi_cfg.REVIEW_REQUIRED)

    rag.ensure_seeded()
    reset_batch_structure_ids()
    force_theme = str(theme or os.getenv("LOFI_FORCE_THEME") or "").strip() or None
    force_subtheme = str(subtheme or os.getenv("LOFI_FORCE_SUBTHEME") or "").strip() or None
    locked_rows: list[dict[str, Any]] = []
    for raw_path in locked_scripts or []:
        if str(raw_path).strip():
            locked_rows.append(_load_locked_script(raw_path))
    if locked_rows:
        qty = len(locked_rows)

    page_outputs, clips_dir, assets_dir, metadata_dir = _resolve_page_dirs(
        page, outputs_dir
    )

    items: list[dict[str, Any]] = []
    ok_n = 0
    for i in range(1, qty + 1):
        print(
            f"[ECONOMIC_REEL_LOFI] ({i}/{qty}) page={page} module={mod} "
            f"duration={dur}s scenes={lofi_cfg.scene_count_for_duration(dur)} "
            f"theme={force_theme or 'auto'} writer_mode={mode} "
            f"{'STILLS-ONLY ' if stills_only else ''}"
            f"{'SCRIPT-ONLY ' if script_only and not stills_only else ''}"
            f"→ {clips_dir}"
        )
        result = _produce_one(
            page_id=page,
            module=mod,
            duration_s=dur,
            clips_dir=clips_dir,
            assets_dir=assets_dir,
            metadata_dir=metadata_dir,
            force_theme=force_theme,
            force_subtheme=force_subtheme,
            writer_mode=mode,
            seed_quote=seed_quote,
            aphorism_id=aphorism_id,
            image_provider=provider,
            index=i,
            script_only=script_only and not stills_only,
            stills_only=stills_only,
            batch_qty=qty,
            locked_script=locked_rows[i - 1] if i <= len(locked_rows) else None,
            review_required=review_required,
            resume_from=resume_from,
            approve_gate=approve_gate,
        )
        item = asdict(result)
        items.append(item)
        if result.ok:
            ok_n += 1
            if result.video_path:
                print(f"  Video : {result.video_path}")
            if result.work_dir:
                print(f"  Stills: {result.work_dir}")
            print(f"  Meta  : {result.meta_path}")
            if result.manual_review:
                print(f"  WARN  : flagged for manual review ({'; '.join(result.errors)})")
        else:
            print(f"  FAIL  : {'; '.join(result.errors)}")
            if result.meta_path:
                print(f"  Review: {result.meta_path}")

    envelope = {
        "post_type": "ECONOMIC_REEL_LOFI",
        "page": page,
        "module": mod,
        "duration_s": dur,
        "writer_mode": mode,
        "image_provider": provider,
        "seed_quote": str(seed_quote or "") if mode == "quote" else "",
        "aphorism_id": str(aphorism_id or "") if mode == "paraphrase" else "",
        "quantity": qty,
        "successful": ok_n,
        "script_only": bool(script_only) and not stills_only,
        "stills_only": bool(stills_only),
        "items": items,
        "outputs_dir": str(page_outputs),
        "clips_dir": str(clips_dir),
    }
    summary_path = metadata_dir / f"lofi_batch_{_utc_stamp()}.json"
    summary_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    envelope["batch_meta"] = str(summary_path)
    return envelope
