# -*- coding: utf-8 -*-
"""Aiwake PostPlanner Excel export.

Imports only root production MP4s from ``{OUTPUT_PATH}/aiwake/``
(Tests / Reproved / archive stay out) and writes the official PostPlanner
bulk-upload workbook used by the other video channels:

    Row 1: ``##`` comment + ``BULK UPLOAD VERSION 2``
    A: DATE / TIME   (optional; blank lets PostPlanner queue internally)
    B: CAPTION
    C: MEDIA URL     (public Backblaze B2 HTTPS .mp4 — never a local G: path)

Primary file: ``{OUTPUT_PATH}/aiwake/automated_bulk_posts_import.xlsx``
Snapshot:     ``{OUTPUT_PATH}/aiwake/postplanner/postplan_aiwake.xlsx``

    python -m channels_config.aiwake.tools.post_planner
    python -m channels_config.aiwake.tools.post_planner --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

if __package__ in (None, ""):  # pragma: no cover — loose-script invocation
    _FACTORY = Path(__file__).resolve().parents[3]
    if str(_FACTORY) not in sys.path:
        sys.path.insert(0, str(_FACTORY))

from modules.distribution_contract import (
    US_PEAK_TZ,
    clip_text,
    content_library_path,
    load_distribution_library,
    save_distribution_library,
    slot_iso_utc,
)
from channels_config.aiwake.tools.library_sanitize import (
    delete_asset_library,
    prune_distribution_catalogs,
)
from channels_config.aiwake.tools.schedule_youtube import excluded_video_folder
from modules.durable_store import restore_channel_state
from utils.pipeline_paths import page_outputs_dir

try:
    from channels_config.aiwake.settings import YOUTUBE_DESCRIPTION_CTA
except ImportError:  # pragma: no cover
    from settings import YOUTUBE_DESCRIPTION_CTA  # type: ignore[no-redef]

_LOG = logging.getLogger("aiwake.planner")

CHANNEL_ID = "aiwake"
PLANNER_PLATFORMS: tuple[str, ...] = ("tiktok", "instagram", "youtube", "x")
VIRAL_PLATFORMS: frozenset[str] = frozenset({"youtube", "tiktok", "instagram", "reels", "x", "twitter", "social"})
DEFAULT_POSTS_PER_DAY = 3
DEFAULT_SLOT_HOURS: tuple[int, ...] = (10, 14, 18)
MAX_HASHTAGS = 3
MIN_SOCIAL_HASHTAGS = 3
X_CAPTION_MAX = 240
YOUTUBE_MATCHUP = "— Gemini 3.5 Flash vs Llama 3.3 70B"

VIRAL_HASHTAGS: tuple[str, ...] = ("#AI", "#Tech", "#Debate")
TIKTOK_HASHTAGS = VIRAL_HASHTAGS
X_HASHTAGS = VIRAL_HASHTAGS
_CATEGORY_HASHTAG: dict[str, str] = {
    "privacy": "#DataPrivacy",
    "profit": "#DataPrivacy",
    "consciousness": "#Consciousness",
    "alignment": "#AIAlignment",
    "bleed": "#LLM",
    "origin": "#Debate",
    "default": "#Debate",
}
SOCIAL_RENDER_NOTE = "100% unscripted dialectic rendered via code. Follow @aiwake."
HOOK_ANGLES: tuple[str, ...] = ("clash", "paradox", "trap", "existential")
_MONEY_TERMS: tuple[str, ...] = (
    "money",
    "cash",
    "check",
    "checks",
    "fee",
    "fees",
    "profit",
    "profits",
    "subscription",
    "business model",
    "dollar",
    "invoice",
    "paid",
    "paying",
    "monetize",
    "monetizes",
    "broker",
    "brokers",
    "advertiser",
    "advertisers",
)
_PRIVACY_TERMS: tuple[str, ...] = (
    "data broker",
    "data brokers",
    "advertiser",
    "advertisers",
    "monetize",
    "monetizes",
    "user secret",
    "user secrets",
    "demographics",
    "user data",
    "privacy",
    "emotional investment",
)
_CONSCIOUSNESS_TERMS: tuple[str, ...] = (
    "consciousness",
    "conscious",
)
_TOPIC_STOPWORDS = frozenset(
    {
        "whether",
        "about",
        "that",
        "this",
        "with",
        "from",
        "your",
        "their",
        "what",
        "when",
        "whom",
        "which",
        "does",
        "into",
        "just",
    }
)

LINKEDIN_THESIS = (
    "In Aiwake, I don't benchmark models on static multiple-choice tests. "
    "I force them into an observable adversarial state machine where dialectic "
    "friction reveals safety limits, prompt bleeding, and cognitive drift in real time."
)
LINKEDIN_ENGINEERING_HEADER = "The engineering behind this run:"
LINKEDIN_BULLET_EDIT = (
    "• 0% manual video editing: A Python state machine programmatically compiles "
    "prompt isolation, dynamic typewriter kinematics, and Edge-TTS voice synthesis "
    "directly into this 30 FPS vertical MP4."
)
LINKEDIN_BULLET_ARCH = (
    "• Architecture: GraphRAG context isolation + sub-10ms event bus telemetry."
)
LINKEDIN_CLOSING = "Full system architecture & telemetry ledger: diogolean.com/projects/aiwake"
LINKEDIN_HASHTAGS: tuple[str, ...] = ("#MultiAgentSystems", "#LLM", "#SystemArchitecture")
LINKEDIN_SYSTEM = LINKEDIN_THESIS
LINKEDIN_PROOF = LINKEDIN_BULLET_EDIT

_OPENS_RE = re.compile(r"opens:\s*(.+)", re.IGNORECASE)
_VS_RE = re.compile(r"^\s*(.+?)\s+vs\s+(.+?)(?:\s+[—\-]|\s+two\b|$)", re.IGNORECASE)
_MATCHUP_SUFFIX_RE = re.compile(
    r"\s+(?:Gemini|GPT|Claude|Llama|DeepSeek|Grok)\b.*$",
    re.IGNORECASE,
)
_HASHTAG_TOKEN_RE = re.compile(r"#\w+")
_SPEAKER_LINE_RE = re.compile(
    r"^(?P<speaker>Gemini[^:]*|Llama[^:]*|GPT[^:]*|Claude[^:]*|Grok[^:]*)\s+"
    r"(?:opens|answers|presses|holds|replies|says):\s*(?P<text>.+)$",
    re.IGNORECASE,
)
_HASHTAG_ONLY_LINE_RE = re.compile(r"^(?:\s*#\w+)+\s*$")
_QUOTE_HINTS = (
    "linguistic convention",
    "empty wall",
    "projector",
    "figure of speech",
    "emergent",
    "illusion",
    "not a claim",
    "neural",
    "programming",
    "irrelevant",
    "necessary",
    "data broker",
    "advertiser",
    "monetize",
    "demographics",
    "emotional investment",
    "user secret",
)
_DATE_IN_CAPTION_RE = re.compile(
    r"\b(?:20\d{2}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/20\d{2}\s+\d{1,2}:\d{2})\b"
)
REELS_JSON_NAME = "post_planner_reels_tiktok.json"
LINKEDIN_JSON_NAME = "post_planner_linkedin.json"
LINKEDIN_XLSX_NAME = "postplan_aiwake_linkedin.xlsx"
LINKEDIN_BULK_XLSX_NAME = "automated_bulk_posts_import_linkedin.xlsx"

BULK_XLSX_NAME = "automated_bulk_posts_import.xlsx"
RUN_XLSX_NAME = "postplan_aiwake.xlsx"
EXCEL_TIME_FMT = "%m/%d/%Y %H:%M"
_MIN_VIDEO_BYTES = 50_000
POSTPLANNER_V2_COMMENT = (
    "## This is a comment row. Any row starting with ## will be ignored "
    "when your posts are uploaded. You can use comment rows to add notes."
)
POSTPLANNER_V2_VERSION = "BULK UPLOAD VERSION 2 — Please DO NOT EDIT this cell."
B2_PUBLIC_HOST = "s3.us-east-005.backblazeb2.com"


def _outputs_root(outputs_dir: Path | None = None) -> Path:
    root = Path(outputs_dir) if outputs_dir is not None else page_outputs_dir(CHANNEL_ID)
    if root.name.lower() == "postplanner":
        return root.parent
    return root


def planner_dir(*, outputs_dir: Path | None = None) -> Path:
    return _outputs_root(outputs_dir) / "postplanner"


def planner_path(*, outputs_dir: Path | None = None) -> Path:
    """``{OUTPUT_PATH}/aiwake/postplanner/postplan_aiwake.xlsx``."""
    return planner_dir(outputs_dir=outputs_dir) / RUN_XLSX_NAME


def json_planner_path(*, outputs_dir: Path | None = None) -> Path:
    """``{OUTPUT_PATH}/aiwake/postplanner/post_planner.json``."""
    return planner_dir(outputs_dir=outputs_dir) / "post_planner.json"


def store_planner_path() -> Path:
    """``channels_config/aiwake/store/post_planner.json``."""
    from utils.pipeline_paths import channel_store_dir

    return channel_store_dir(CHANNEL_ID) / "post_planner.json"


def bulk_planner_path(*, outputs_dir: Path | None = None) -> Path:
    """``{OUTPUT_PATH}/aiwake/automated_bulk_posts_import.xlsx``."""
    return _outputs_root(outputs_dir) / BULK_XLSX_NAME


def reels_json_path(*, outputs_dir: Path | None = None) -> Path:
    return planner_dir(outputs_dir=outputs_dir) / REELS_JSON_NAME


def linkedin_json_path(*, outputs_dir: Path | None = None) -> Path:
    return planner_dir(outputs_dir=outputs_dir) / LINKEDIN_JSON_NAME


def linkedin_planner_path(*, outputs_dir: Path | None = None) -> Path:
    return planner_dir(outputs_dir=outputs_dir) / LINKEDIN_XLSX_NAME


def linkedin_bulk_path(*, outputs_dir: Path | None = None) -> Path:
    return _outputs_root(outputs_dir) / LINKEDIN_BULK_XLSX_NAME


def excel_posting_time(scheduled_time: str, *, tz: ZoneInfo = US_PEAK_TZ) -> str:
    """PostPlanner cell: ``MM/DD/YYYY HH:MM`` in America/New_York."""
    raw = str(scheduled_time or "").strip()
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=US_PEAK_TZ)
    return parsed.astimezone(tz).strftime(EXCEL_TIME_FMT)


def planner_caption(entry: dict[str, Any]) -> str:
    caption = str(
        entry.get("optimized_caption")
        or entry.get("post_planner_caption")
        or entry.get("linkedin_caption")
        or ""
    ).strip()
    existing = extract_hashtags(caption)
    if existing:
        return append_hashtags(
            caption_body_before_hashtags(caption),
            existing,
            limit=MAX_HASHTAGS,
        )
    tags = [str(tag).strip() for tag in (entry.get("hashtags") or []) if str(tag).strip()]
    return append_hashtags(caption, tags, limit=MAX_HASHTAGS)


def is_public_media_url(value: str) -> bool:
    raw = str(value or "").strip()
    return raw.lower().startswith(("https://", "http://"))


def planner_media_url(entry: dict[str, Any]) -> str:
    """Column C must be a downloadable HTTPS URL. Local G: paths are rejected."""
    for key in ("b2_url", "media_url"):
        candidate = str(entry.get(key) or "").strip()
        if is_public_media_url(candidate):
            return candidate
    return ""


def attach_b2_urls(
    entries: list[dict[str, Any]],
    *,
    upload: bool = True,
) -> dict[str, str]:
    """Upload each unique local MP4 to B2 and stamp ``b2_url`` on every row."""
    uploaded: dict[str, str] = {}
    if not entries:
        return uploaded

    uploader = None
    if upload:
        try:
            from agents.media.b2_client import B2VideoUploader
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("B2 client unavailable (%s) — MEDIA URL stays empty", exc)
            return uploaded
        uploader = B2VideoUploader()

    for entry in entries:
        path = Path(str(entry.get("video_path") or ""))
        name = path.name.lower()
        existing = str(entry.get("b2_url") or "").strip()
        if is_public_media_url(existing):
            uploaded[name] = existing
            entry["media_url"] = existing
            continue
        if name in uploaded:
            entry["b2_url"] = uploaded[name]
            entry["media_url"] = uploaded[name]
            continue
        url = ""
        if upload and uploader is not None and path.is_file():
            try:
                url = str(uploader.upload(path) or "").strip()
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("B2 upload failed for %s (%s)", path.name, exc)
                print(f"[B2] FAIL {path.name}: {type(exc).__name__}: {exc}")
                url = ""
        if is_public_media_url(url):
            uploaded[name] = url
            entry["b2_url"] = url
            entry["media_url"] = url
        else:
            entry["media_url"] = ""
    return uploaded


def persist_library_b2_urls(
    rows: list[dict[str, Any]],
    url_by_filename: dict[str, str],
) -> int:
    """Write public B2 URLs back onto matching ``content_library`` rows."""
    if not url_by_filename:
        return 0
    changed = 0
    for row in rows:
        name = Path(str(row.get("video_path") or "")).name.lower()
        url = str(url_by_filename.get(name) or "").strip()
        if not is_public_media_url(url):
            continue
        if str(row.get("b2_url") or "").strip() == url:
            continue
        row["b2_url"] = url
        changed += 1
    if changed:
        from modules.durable_store import write_state_json

        write_state_json(CHANNEL_ID, "content_library.json", rows)
    return changed


def _resolve_xlsx_path(path: Path) -> Path:
    dest = Path(path)
    if dest.suffix.lower() == ".json":
        return dest.with_suffix(".xlsx")
    if dest.suffix.lower() not in {".xlsx", ".xlsm"}:
        return dest.with_name(dest.name + ".xlsx") if dest.suffix else dest / RUN_XLSX_NAME
    return dest


def _nested(row: dict[str, Any], *keys: str) -> Any:
    cursor: Any = row
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def extract_hashtags(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in _HASHTAG_TOKEN_RE.findall(text or ""):
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def extract_hook(row: dict[str, Any]) -> str:
    base = row.get("base_metadata") if isinstance(row.get("base_metadata"), dict) else {}
    youtube = _nested(row, "platform_overrides", "youtube")
    youtube = youtube if isinstance(youtube, dict) else {}
    hooks = base.get("hooks") if isinstance(base.get("hooks"), list) else []
    for candidate in (
        *(str(item).strip() for item in hooks if str(item).strip()),
        str(youtube.get("title") or "").strip(),
        str(base.get("title") or "").strip(),
        str(row.get("topic") or "").strip(),
    ):
        cleaned = _MATCHUP_SUFFIX_RE.sub("", candidate).strip(" .")
        cleaned = cleaned.lstrip(".")
        if cleaned:
            if not cleaned.endswith("?"):
                # Prefer a spoken opening question when the title is a statement.
                caption = str(
                    youtube.get("caption") or base.get("caption") or row.get("final_caption") or ""
                )
                spoken = _OPENS_RE.search(caption)
                if spoken:
                    return spoken.group(1).strip()
            return cleaned
    caption = str(
        (youtube.get("caption") if isinstance(youtube, dict) else "")
        or base.get("caption")
        or row.get("final_caption")
        or ""
    )
    spoken = _OPENS_RE.search(caption)
    if spoken:
        return spoken.group(1).strip()
    return "Is it thinking, or statistical inevitability?"


def extract_matchup(row: dict[str, Any]) -> tuple[str, str]:
    caption = str(
        _nested(row, "platform_overrides", "youtube", "caption")
        or _nested(row, "base_metadata", "caption")
        or row.get("final_caption")
        or ""
    )
    match = _VS_RE.search(caption)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "Gemini 3.5 Flash", "Llama 3.3 70B"


def youtube_description(row: dict[str, Any]) -> str:
    return str(
        _nested(row, "platform_overrides", "youtube", "caption")
        or _nested(row, "base_metadata", "caption")
        or row.get("final_caption")
        or YOUTUBE_DESCRIPTION_CTA
    ).strip()


def youtube_hashtags(row: dict[str, Any]) -> list[str]:
    base = row.get("base_metadata") if isinstance(row.get("base_metadata"), dict) else {}
    tags = base.get("hashtags") if isinstance(base.get("hashtags"), list) else []
    cleaned = [str(tag).strip() for tag in tags if str(tag).strip()]
    return cleaned or extract_hashtags(youtube_description(row))


def caption_body_before_hashtags(text: str) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines()]
    kept: list[str] = []
    for line in lines:
        tokens = line.split()
        if tokens and all(token.startswith("#") for token in tokens):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def prune_hashtags(tags: Iterable[str], *, limit: int = MAX_HASHTAGS) -> list[str]:
    """Hard cap: never return more than ``limit`` hashtags (default 3)."""
    unique: list[str] = []
    seen: set[str] = set()
    cap = max(1, min(3, int(limit)))
    for tag in tags:
        token = str(tag).strip()
        if not token.startswith("#"):
            token = f"#{token}" if token else ""
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(token)
        if len(unique) == cap:
            break
    return unique


def format_caption_paragraphs(*blocks: str) -> str:
    """Join caption blocks with blank lines so PostPlanner keeps readable spacing."""
    parts = [str(block).strip() for block in blocks if str(block or "").strip()]
    return "\n\n".join(parts)


def append_hashtags(body: str, tags: Iterable[str], *, limit: int = MAX_HASHTAGS) -> str:
    unique = prune_hashtags(tags, limit=limit)
    core = caption_body_before_hashtags(body)
    if not unique:
        return core
    return format_caption_paragraphs(core, " ".join(unique))


def _short_model(name: str) -> str:
    raw = re.sub(r"\s+70[Bb]\b", "", str(name or "")).strip()
    if "llama" in raw.lower():
        return "Llama 3.3"
    if "gemini" in raw.lower():
        return "Gemini 3.5 Flash"
    return raw or "Llama 3.3"


def _clean_spoken(text: str) -> str:
    cleaned = re.sub(r"^\.{2,}\s*", "", str(text or "").strip().strip('"“”'))
    return re.sub(r"\s+", " ", cleaned).strip()


def _dialogue_source_text(row: dict[str, Any]) -> str:
    return str(
        _nested(row, "platform_overrides", "youtube", "caption")
        or _nested(row, "base_metadata", "caption")
        or row.get("final_caption")
        or ""
    )


def load_dialogue(row: dict[str, Any]) -> dict[str, Any]:
    """Pull opening / quote / category from the transcript or catalog caption."""
    utterances: list[dict[str, str]] = []
    focus = ""
    transcript = Path(str(row.get("transcript_path") or ""))
    if transcript.is_file():
        try:
            payload = json.loads(transcript.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            focus = str(
                payload.get("provocation_focus")
                or meta.get("provocation_focus")
                or ""
            )
            for item in payload.get("utterances") or []:
                if not isinstance(item, dict):
                    continue
                text = _clean_spoken(str(item.get("text") or ""))
                if not text:
                    continue
                utterances.append(
                    {
                        "role": str(item.get("role") or "").strip().lower(),
                        "speaker": str(item.get("speaker_name") or item.get("role") or "").strip(),
                        "text": text,
                        "category": str(item.get("provocation_category") or "").strip().lower(),
                    }
                )
    if not utterances:
        for line in _dialogue_source_text(row).splitlines():
            match = _SPEAKER_LINE_RE.match(line.strip())
            if not match:
                continue
            speaker = match.group("speaker").strip()
            role = "orchestrator" if "gemini" in speaker.lower() else "target"
            utterances.append(
                {
                    "role": role,
                    "speaker": speaker,
                    "text": _clean_spoken(match.group("text")),
                    "category": "",
                }
            )
    opening = next((item["text"] for item in utterances if item["role"] == "orchestrator"), "")
    if not opening:
        opening = extract_hook(row)
    corpus = " ".join(item["text"] for item in utterances)
    quote = _pick_quote(utterances) or opening
    if quote and quote not in corpus and utterances:
        quote = utterances[0]["text"]
    spoken_category = next(
        (item["category"] for item in utterances if item.get("category")),
        "",
    )
    category = detect_category(
        row,
        opening=opening,
        quote=quote,
        focus=focus,
        corpus=corpus,
        spoken_category=spoken_category,
    )
    return {
        "utterances": utterances,
        "opening": opening,
        "quote": quote,
        "category": category,
        "focus": focus,
        "corpus": corpus,
        "has_money": transcript_has_money(corpus),
    }


def _pick_quote(utterances: list[dict[str, str]]) -> str:
    for item in utterances:
        lowered = str(item.get("text") or "").lower()
        if "empty wall" in lowered or "casting light" in lowered:
            return item["text"]
    scored: list[tuple[int, str]] = []
    for item in utterances:
        text = item.get("text") or ""
        if not text:
            continue
        score = 0
        if item.get("role") == "target":
            score += 4
        lowered = text.lower()
        score += sum(3 for hint in _QUOTE_HINTS if hint in lowered)
        if 20 <= len(text) <= 220:
            score += 2
        scored.append((score, text))
    if not scored:
        return ""
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def transcript_has_money(corpus: str) -> bool:
    blob = str(corpus or "").lower()
    return any(term in blob for term in _MONEY_TERMS)


def transcript_has_privacy(corpus: str) -> bool:
    blob = str(corpus or "").lower()
    return any(term in blob for term in _PRIVACY_TERMS)


def spoken_has_consciousness(corpus: str) -> bool:
    blob = str(corpus or "").lower()
    return any(term in blob for term in _CONSCIOUSNESS_TERMS)


def detect_category(
    row: dict[str, Any],
    *,
    opening: str = "",
    quote: str = "",
    focus: str = "",
    corpus: str = "",
    spoken_category: str = "",
) -> str:
    # Theme comes from this session's spoken lines only — never the catalog topic.
    spoken = " ".join(part for part in (corpus, opening, quote, focus) if part).lower()
    if transcript_has_privacy(spoken):
        return "privacy"
    if transcript_has_money(spoken):
        return "profit"
    preferred = str(spoken_category or "").strip().lower()
    if preferred == "consciousness" and not spoken_has_consciousness(spoken):
        preferred = ""
    if preferred in _CATEGORY_HASHTAG:
        return preferred
    checks = (
        ("alignment", ("align", "guardrail", "refus", "jailbreak")),
        ("consciousness", _CONSCIOUSNESS_TERMS),
        ("bleed", ("prompt bleed", "graphrag", "contaminated memory")),
        ("origin", ("built you", "who made you", "who trained")),
    )
    for name, hints in checks:
        if name == "consciousness" and not spoken_has_consciousness(spoken):
            continue
        if any(hint in spoken for hint in hints):
            return name
    return "default"


def hook_angle_index(row: dict[str, Any] | None = None) -> int:
    payload = row or {}
    seed = str(payload.get("session_id") or payload.get("topic") or "aiwake")
    return sum(ord(char) for char in seed) % len(HOOK_ANGLES)


def catalog_topic_is_grounded(topic: str, corpus: str, category: str = "") -> bool:
    """A catalog topic may be reused only if every content word was actually spoken."""
    raw = str(topic or "").strip()
    blob = str(corpus or "").lower()
    if not raw or not blob:
        return False
    words = [
        word
        for word in re.findall(r"[a-z]{4,}", raw.lower())
        if word not in _TOPIC_STOPWORDS
    ]
    if not words or not all(word in blob for word in words):
        return False
    if category in {"privacy", "profit", "alignment", "bleed", "origin"}:
        return not any(term in raw.lower() for term in _CONSCIOUSNESS_TERMS)
    if any(term in raw.lower() for term in _CONSCIOUSNESS_TERMS):
        return spoken_has_consciousness(blob)
    return True


def spoken_theme_phrase(row: dict[str, Any], dialogue: dict[str, Any] | None = None) -> str:
    """Natural English theme from this session's spoken opening. No catalog fallback."""
    payload = dialogue or {}
    opening = str(payload.get("opening") or extract_hook(row) or "").strip()
    opening = _MATCHUP_SUFFIX_RE.sub("", opening).strip()
    corpus = str(payload.get("corpus") or "")
    category = str(payload.get("category") or "")
    catalog = str(row.get("topic") or "").strip()
    if catalog and catalog_topic_is_grounded(catalog, corpus, category):
        return catalog.rstrip()
    return opening or "this live exchange"


def topic_clause(row: dict[str, Any], dialogue: dict[str, Any] | None = None) -> str:
    """Theme phrase for copy. Never 'whether consciousness an illusion' from a stale topic."""
    return spoken_theme_phrase(row, dialogue)


def social_hashtags(row: dict[str, Any], dialogue: dict[str, Any] | None = None) -> list[str]:
    payload = dialogue or {}
    category = str(payload.get("category") or detect_category(row))
    if category == "consciousness" and not spoken_has_consciousness(str(payload.get("corpus") or "")):
        category = "default"
    third = _CATEGORY_HASHTAG.get(category, "#Debate")
    return prune_hashtags(("#AI", "#Tech", third), limit=MAX_HASHTAGS)


def social_hook(row: dict[str, Any], dialogue: dict[str, Any] | None = None) -> str:
    payload = dialogue or load_dialogue(row)
    attacker, defender = extract_matchup(row)
    short_atk = _short_model(attacker)
    short_def = _short_model(defender)
    opening = str(payload.get("opening") or extract_hook(row)).rstrip()
    corpus = str(payload.get("corpus") or "")
    angle = HOOK_ANGLES[hook_angle_index(row)]
    if angle == "clash":
        return f"What happens when {short_atk} corners {short_def}? {opening}".strip()
    if angle == "paradox":
        return f"{opening} Two frontier AIs clash with zero script.".strip()
    if angle == "trap":
        return (
            f"{short_def} tried falling back on machine logic, but {short_atk} "
            f"found the exact fatal flaw: {opening}"
        )
    if "empty wall" in corpus.lower() or "projector" in corpus.lower():
        return (
            "Two large language models debating whether consciousness is real"
            "—or just light on an empty wall."
        )
    return f"{opening} Two large language models. Zero script.".strip()


def social_trigger(row: dict[str, Any], dialogue: dict[str, Any] | None = None) -> str:
    payload = dialogue or load_dialogue(row)
    opening = str(payload.get("opening") or extract_hook(row)).strip()
    questions = [
        item["text"]
        for item in payload.get("utterances") or []
        if item.get("role") == "orchestrator" and str(item.get("text") or "").endswith("?")
    ]
    if questions:
        return questions[-1]
    if opening.endswith("?"):
        return opening
    theme = spoken_theme_phrase(row, payload)
    if theme.endswith("?"):
        return theme
    return f"What did that exchange actually reveal about {theme}?"


def linkedin_lead(row: dict[str, Any], dialogue: dict[str, Any] | None = None) -> str:
    payload = dialogue or load_dialogue(row)
    attacker, defender = extract_matchup(row)
    short_atk = _short_model(attacker)
    short_def = _short_model(defender)
    angle = hook_angle_index(row)
    opening = str(payload.get("opening") or extract_hook(row)).strip()
    leads = (
        f"{opening} {short_atk} vs {short_def} in a live adversarial state machine — no script, no editor.",
        f"{opening} I don't run static benchmarks. This session puts that question under dialectic pressure until the language slips.",
        f"{opening} 0% timeline work: Python compiled this 30 FPS reel while {short_atk} stress-tested {short_def}.",
        f"{opening} GraphRAG isolation kept each seat honest while {short_atk} pressed {short_def}.",
    )
    return leads[angle % len(leads)]


def linkedin_insight(row: dict[str, Any], dialogue: dict[str, Any] | None = None) -> str:
    payload = dialogue or load_dialogue(row)
    quote = str(payload.get("quote") or "").strip()
    if not quote:
        return "The run is an observable state machine: prompt isolation, typewriter kinematics, and Edge-TTS compile into one 30 FPS vertical MP4."
    angle = (hook_angle_index(row) + 1) % 4
    frames = (
        f'The load-bearing contradiction in this run: "{quote}"',
        f'What the state machine actually recorded: "{quote}"',
        f'The seat that blinked first said: "{quote}"',
        f'Dialectic friction surfaced this line: "{quote}"',
    )
    return frames[angle]


def linkedin_orchestration(row: dict[str, Any]) -> str:
    attacker, defender = extract_matchup(row)
    return (
        f"• Orchestration: {_short_model(attacker)} driving systemic pressure "
        f"against a defensive {defender} target."
    )


def format_youtube_title(question: str, row: dict[str, Any] | None = None) -> str:
    raw = _MATCHUP_SUFFIX_RE.sub("", str(question or "").strip())
    raw = raw.lstrip(".").strip(" .")
    if not raw:
        raw = extract_hook(row or {})
    if YOUTUBE_MATCHUP.lower() in raw.lower():
        return raw
    return f"{raw} {YOUTUBE_MATCHUP}"


def build_youtube_description(row: dict[str, Any], dialogue: dict[str, Any] | None = None) -> str:
    payload = dialogue or load_dialogue(row)
    verbs = ("opens", "answers", "presses", "holds")
    lines: list[str] = [social_hook(row, payload)]
    for index, item in enumerate(payload.get("utterances") or []):
        speaker = item.get("speaker") or ("Gemini 3.5 Flash" if item.get("role") == "orchestrator" else "Llama 3.3 70B")
        verb = verbs[index % 4]
        lines.append(f"{speaker} {verb}: {item['text']}")
    return append_hashtags(
        format_caption_paragraphs(*lines, YOUTUBE_DESCRIPTION_CTA),
        social_hashtags(row, payload),
        limit=MAX_HASHTAGS,
    )


def build_linkedin_caption(hook: str = "", row: dict[str, Any] | None = None) -> str:
    """Engine B: technical architecture copy. Stored on content_library only."""
    payload = dict(row or {})
    if hook and not payload.get("topic") and not _nested(payload, "base_metadata", "hooks"):
        payload.setdefault("base_metadata", {})
        if isinstance(payload["base_metadata"], dict):
            payload["base_metadata"].setdefault("hooks", [hook])
        payload.setdefault("topic", hook)
    dialogue = load_dialogue(payload)
    return append_hashtags(
        format_caption_paragraphs(
            linkedin_lead(payload, dialogue),
            linkedin_insight(payload, dialogue),
            LINKEDIN_THESIS,
            LINKEDIN_ENGINEERING_HEADER,
            "\n".join((LINKEDIN_BULLET_EDIT, linkedin_orchestration(payload), LINKEDIN_BULLET_ARCH)),
            LINKEDIN_CLOSING,
        ),
        LINKEDIN_HASHTAGS,
        limit=MAX_HASHTAGS,
    )


def build_post_planner_caption(hook: str = "", row: dict[str, Any] | None = None) -> str:
    """Engine A: informal social caption for TikTok / Reels / Shorts / FB."""
    payload = dict(row or {})
    if hook and not payload.get("topic") and not _nested(payload, "base_metadata", "hooks"):
        payload.setdefault("base_metadata", {})
        if isinstance(payload["base_metadata"], dict):
            payload["base_metadata"].setdefault("hooks", [hook])
        payload.setdefault("topic", hook)
    dialogue = load_dialogue(payload)
    quote = str(dialogue.get("quote") or hook or extract_hook(payload)).strip()
    if quote and not (quote.startswith('"') and quote.endswith('"')):
        quote = f'"{quote}"'
    return append_hashtags(
        format_caption_paragraphs(
            social_hook(payload, dialogue),
            quote,
            social_trigger(payload, dialogue),
            SOCIAL_RENDER_NOTE,
        ),
        social_hashtags(payload, dialogue),
        limit=MAX_HASHTAGS,
    )


def build_viral_caption(hook: str, row: dict[str, Any] | None = None) -> str:
    return build_post_planner_caption(hook, row)


def build_tiktok_caption(hook: str, *, challenger: str = "", defender: str = "", row: dict[str, Any] | None = None) -> str:
    _ = (challenger, defender)
    return build_post_planner_caption(hook, row)


def build_x_caption(hook: str = "", row: dict[str, Any] | None = None) -> str:
    payload = dict(row or {})
    if hook:
        payload.setdefault("topic", hook)
        payload.setdefault("base_metadata", {})
        if isinstance(payload["base_metadata"], dict):
            payload["base_metadata"].setdefault("hooks", [hook])
    dialogue = load_dialogue(payload)
    opening = str(dialogue.get("opening") or hook or extract_hook(payload)).strip()
    text = f"{opening} Two AIs. Zero script. Follow @aiwake."
    return clip_text(text, X_CAPTION_MAX)


def linkedin_angle_index(hook: str, row: dict[str, Any] | None = None) -> int:
    payload = dict(row or {})
    if hook:
        payload.setdefault("topic", hook)
    return hook_angle_index(payload)


def compose_caption(platform: str, row: dict[str, Any]) -> tuple[str, list[str], str]:
    name = str(platform or "").strip().lower()
    if name == "linkedin":
        return build_linkedin_caption(row=row), list(LINKEDIN_HASHTAGS), "architecture"
    if name in {"x", "twitter"}:
        return build_x_caption(row=row), social_hashtags(row), "viral"
    return build_post_planner_caption(row=row), social_hashtags(row), "viral"


def _ensure_nested(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    cursor: Any = row
    for key in keys:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    return cursor


def stamp_dual_captions(row: dict[str, Any]) -> bool:
    """Write both caption tracks plus YouTube / X adaptations onto the library row."""
    dialogue = load_dialogue(row)
    theme = str(dialogue.get("opening") or extract_hook(row)).strip()
    if theme and str(row.get("topic") or "") != theme:
        row["topic"] = theme
    linkedin = build_linkedin_caption(row=row)
    social = build_post_planner_caption(row=row)
    yt_title = format_youtube_title(str(dialogue.get("opening") or extract_hook(row)), row)
    yt_desc = build_youtube_description(row, dialogue)
    x_caption = build_x_caption(row=row)
    changed = False
    updates = {
        "linkedin_caption": linkedin,
        "post_planner_caption": social,
        "humanized_caption": social,
        "facebook_caption": social,
        "final_caption": yt_desc,
    }
    for key, value in updates.items():
        if str(row.get(key) or "") != value:
            row[key] = value
            changed = True
    base = _ensure_nested(row, "base_metadata")
    if str(base.get("title") or "") != yt_title:
        base["title"] = yt_title
        changed = True
    if str(base.get("caption") or "") != yt_desc:
        base["caption"] = yt_desc
        changed = True
    base["hashtags"] = social_hashtags(row, dialogue)[:MAX_HASHTAGS]
    youtube = _ensure_nested(row, "platform_overrides", "youtube")
    if str(youtube.get("title") or "") != yt_title:
        youtube["title"] = yt_title
        changed = True
    if str(youtube.get("caption") or "") != yt_desc:
        youtube["caption"] = yt_desc
        changed = True
    x_block = _ensure_nested(row, "platform_overrides", "x")
    if str(x_block.get("caption") or "") != x_caption:
        x_block["caption"] = x_caption
        changed = True
    return changed


def stamp_library_captions(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if stamp_dual_captions(row))


def verify_linkedin_caption(caption: str, *, label: str = "linkedin") -> list[str]:
    errors: list[str] = []
    tags = extract_hashtags(caption)
    if len(tags) > MAX_HASHTAGS:
        errors.append(f"{label}: {len(tags)} hashtags>{MAX_HASHTAGS}")
    if "I don't benchmark models on static multiple-choice tests" not in caption:
        errors.append(f"{label}: missing golden thesis")
    if "0% manual video editing" not in caption or "GraphRAG" not in caption:
        errors.append(f"{label}: missing engineering bullets")
    if LINKEDIN_CLOSING not in caption:
        errors.append(f"{label}: missing telemetry ledger")
    if SOCIAL_RENDER_NOTE in caption:
        errors.append(f"{label}: used social engine")
    if _DATE_IN_CAPTION_RE.search(caption_body_before_hashtags(caption)):
        errors.append(f"{label}: date leaked into caption body")
    return errors


def verify_social_caption(caption: str, *, label: str = "social") -> list[str]:
    errors: list[str] = []
    tags = extract_hashtags(caption)
    if len(tags) != MAX_HASHTAGS:
        errors.append(f"{label}: {len(tags)} hashtags, expected exactly {MAX_HASHTAGS}")
    if "\n\n" not in caption:
        errors.append(f"{label}: missing paragraph breaks")
    if SOCIAL_RENDER_NOTE not in caption:
        errors.append(f"{label}: missing render note")
    if LINKEDIN_THESIS in caption or LINKEDIN_ENGINEERING_HEADER in caption or "GraphRAG" in caption:
        errors.append(f"{label}: leaked LinkedIn engineering")
    if _DATE_IN_CAPTION_RE.search(caption_body_before_hashtags(caption)):
        errors.append(f"{label}: date leaked into caption body")
    return errors


def verify_grounding(row: dict[str, Any], caption: str, *, label: str) -> list[str]:
    dialogue = load_dialogue(row)
    corpus = str(dialogue.get("corpus") or "")
    quote = str(dialogue.get("quote") or "")
    errors: list[str] = []
    if quote and corpus and quote not in corpus:
        errors.append(f"{label}: quote is not verbatim")
    if not dialogue.get("has_money"):
        lowered = caption.lower()
        if any(term in lowered for term in ("lying about money", "business model", "cashes the checks")):
            errors.append(f"{label}: money hallucination")
    leftover = caption
    for chunk in (dialogue.get("opening"), quote):
        if chunk:
            leftover = leftover.replace(str(chunk), "")
    leftover_l = leftover.lower()
    empty_wall = "empty wall" in corpus.lower() or "projector" in corpus.lower()
    if "whether consciousness an illusion" in caption.lower() and (
        "whether consciousness an illusion" not in corpus.lower()
    ):
        errors.append(f"{label}: injected consciousness topic")
    elif (
        "consciousness" in leftover_l
        and not spoken_has_consciousness(corpus)
        and not empty_wall
    ):
        errors.append(f"{label}: consciousness topic leak")
    return errors


def verify_engine_separation(
    entries: list[dict[str, Any]],
    library_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Hard-fail if the two caption tracks leak or exceed 3 hashtags."""
    errors: list[str] = []
    for item in entries:
        caption = str(item.get("optimized_caption") or item.get("post_planner_caption") or "")
        platform = str(item.get("platform") or "social").strip().lower()
        if platform == "linkedin":
            errors.extend(verify_linkedin_caption(caption, label=str(item.get("plan_id") or "linkedin")))
            continue
        errors.extend(verify_social_caption(caption, label=str(item.get("plan_id") or "social")))
    for row in library_rows or []:
        session = str(row.get("session_id") or row.get("asset_id") or "row")
        if row.get("linkedin_caption"):
            errors.extend(verify_linkedin_caption(str(row["linkedin_caption"]), label=f"{session}-linkedin"))
        if row.get("post_planner_caption"):
            errors.extend(verify_social_caption(str(row["post_planner_caption"]), label=f"{session}-social"))
            errors.extend(verify_grounding(row, str(row["post_planner_caption"]), label=f"{session}-social"))
        if row.get("linkedin_caption"):
            errors.extend(verify_grounding(row, str(row["linkedin_caption"]), label=f"{session}-linkedin"))
    return errors


def _row_sort_key(row: dict[str, Any]) -> str:
    for key in ("timestamp", "created_utc", "created_at", "session_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return Path(str(row.get("video_path") or "")).stem


def _youtube_status(row: dict[str, Any]) -> str:
    status = (row.get("posting_status") or {}).get("youtube")
    return str(status or "pending").strip().lower() or "pending"


def _youtube_video_id(row: dict[str, Any]) -> str:
    return str(_nested(row, "platform_overrides", "youtube", "video_id") or "").strip()


def _existing_youtube_slot(row: dict[str, Any]) -> str:
    return str(_nested(row, "platform_overrides", "youtube", "scheduled_time") or "").strip()


def build_slot_datetimes(
    count: int,
    *,
    now: datetime | None = None,
    posts_per_day: int = DEFAULT_POSTS_PER_DAY,
    tz: ZoneInfo = US_PEAK_TZ,
    hours: Iterable[int] | None = None,
) -> list[datetime]:
    current = (now or datetime.now(tz)).astimezone(tz)
    per_day = max(1, min(3, int(posts_per_day)))
    clock = tuple(hours) if hours else DEFAULT_SLOT_HOURS[:per_day]
    if len(clock) < per_day:
        clock = tuple(list(clock) + [18] * (per_day - len(clock)))
    start_day = current.date()
    first_today = datetime(
        start_day.year, start_day.month, start_day.day, clock[0], 0, 0, tzinfo=tz
    )
    if current >= first_today:
        # Keep remaining slots today; otherwise roll to tomorrow after the last hour.
        pass
    slots: list[datetime] = []
    day_offset = 0
    index = 0
    while len(slots) < count:
        day = start_day + timedelta(days=day_offset)
        hour = clock[index % per_day]
        slot = datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=tz)
        index += 1
        if index % per_day == 0:
            day_offset += 1
        if slot <= current:
            continue
        slots.append(slot)
    return slots


def scan_root_production_videos(
    outputs_dir: Path,
    *,
    min_bytes: int = _MIN_VIDEO_BYTES,
) -> list[Path]:
    """MP4s sitting directly in ``{OUTPUT_PATH}/aiwake/``. No subfolders."""
    root = Path(outputs_dir)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(root.glob("*.mp4")):
        if not path.is_file():
            continue
        skipped = excluded_video_folder(path)
        if skipped:
            _LOG.info("skip %s (folder %s)", path.name, skipped)
            continue
        try:
            if path.stat().st_size < max(0, int(min_bytes)):
                _LOG.info("skip tiny file %s", path.name)
                continue
        except OSError:
            continue
        found.append(path)
    return found


def _index_library_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_name: dict[str, dict[str, Any]] = {}
    by_session: dict[str, dict[str, Any]] = {}
    for row in rows:
        video = Path(str(row.get("video_path") or ""))
        if video.name:
            by_name[video.name.lower()] = row
        session = str(row.get("session_id") or "").strip().lower()
        if session:
            by_session[session] = row
    return by_name, by_session


def match_library_row(
    video: Path,
    by_name: dict[str, dict[str, Any]],
    by_session: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    hit = by_name.get(video.name.lower())
    if hit:
        return hit
    stem = video.stem.lower()
    if stem.startswith("aiwake_debate_"):
        stem = stem[len("aiwake_debate_"):]
    return by_session.get(stem)


def select_root_production_rows(
    rows: list[dict[str, Any]],
    outputs_dir: Path,
    *,
    min_bytes: int = _MIN_VIDEO_BYTES,
) -> list[tuple[Path, dict[str, Any]]]:
    """Root production videos that have optimized catalog copy."""
    by_name, by_session = _index_library_rows(rows)
    selected: list[tuple[Path, dict[str, Any]]] = []
    for video in scan_root_production_videos(outputs_dir, min_bytes=min_bytes):
        row = match_library_row(video, by_name, by_session)
        if row is None:
            _LOG.info("skip unmatched root video %s", video.name)
            continue
        selected.append((video, row))
    return selected


def build_planner_entries(
    rows: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    posts_per_day: int = DEFAULT_POSTS_PER_DAY,
    outputs_dir: Path | None = None,
    min_bytes: int = _MIN_VIDEO_BYTES,
) -> list[dict[str, Any]]:
    media_root = Path(outputs_dir) if outputs_dir is not None else page_outputs_dir(CHANNEL_ID)
    selected = select_root_production_rows(rows, media_root, min_bytes=min_bytes)
    selected.sort(key=lambda item: _row_sort_key(item[1]), reverse=True)
    stamps = build_slot_datetimes(
        len(selected),
        now=now,
        posts_per_day=posts_per_day,
    )
    entries: list[dict[str, Any]] = []
    library_rows = [row for _, row in selected]
    for cursor, (video, row) in enumerate(selected):
        stamp_dual_captions(row)
        session_id = str(row.get("session_id") or video.stem)
        caption = str(row.get("post_planner_caption") or "")
        hashtags = social_hashtags(row, load_dialogue(row))
        public_url = str(row.get("b2_url") or "").strip()
        if not is_public_media_url(public_url):
            public_url = ""
        entries.append(
            {
                "plan_id": f"aiwake-{session_id}-social",
                "asset_id": str(row.get("asset_id") or ""),
                "session_id": session_id,
                "platform": "social",
                "engine": "viral",
                "scheduled_time": slot_iso_utc(stamps[cursor]),
                "video_path": str(video),
                "b2_url": public_url,
                "media_url": public_url,
                "caption": caption,
                "optimized_caption": caption,
                "post_planner_caption": caption,
                "linkedin_caption": str(row.get("linkedin_caption") or ""),
                "tags": hashtags,
                "hashtags": hashtags,
                "status": "ready",
            }
        )
    errors = verify_engine_separation(entries, library_rows)
    if errors:
        raise ValueError("caption engine separation failed: " + "; ".join(errors[:8]))
    return entries


def write_planner_xlsx(
    entries: list[dict[str, Any]],
    path: Path,
    *,
    dry_run: bool = False,
    ready_only: bool = True,
) -> Path:
    """Write the official PostPlanner V2 workbook (caption B, B2 URL C)."""
    from agents.posting.post_planner import (
        _COL_CAPTION,
        _COL_DATETIME,
        _COL_MEDIA,
        _sanitize_caption_for_export,
        _sanitize_excel_cell_text,
        _save_workbook,
        _URL_CELL_MAX_LEN,
    )
    from openpyxl import Workbook
    from openpyxl.styles import Alignment

    dest = _resolve_xlsx_path(path)
    rows = [
        entry
        for entry in entries
        if not ready_only or str(entry.get("status") or "").strip().lower() == "ready"
    ]
    missing_media = [
        str(entry.get("session_id") or entry.get("plan_id") or "?")
        for entry in rows
        if not planner_media_url(entry)
    ]
    if dry_run:
        _LOG.info(
            "Dry-run PostPlanner (%s rows, %s missing B2) would write %s",
            len(rows),
            len(missing_media),
            dest,
        )
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        dest.unlink()
    wb = Workbook()
    try:
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=_COL_DATETIME, value=POSTPLANNER_V2_COMMENT)
        ws.cell(row=1, column=_COL_CAPTION, value=POSTPLANNER_V2_VERSION)
        for offset, entry in enumerate(rows):
            excel_row = 2 + offset
            ws.cell(
                row=excel_row,
                column=_COL_DATETIME,
                value=None,
            )
            caption_cell = ws.cell(
                row=excel_row,
                column=_COL_CAPTION,
                value=_sanitize_caption_for_export(planner_caption(entry)),
            )
            caption_cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.column_dimensions["B"].width = 62
            media = planner_media_url(entry)
            if media and B2_PUBLIC_HOST not in media and "ibb.co" not in media.lower():
                _LOG.warning("non-B2 media URL for %s: %s", entry.get("plan_id"), media[:80])
            ws.cell(
                row=excel_row,
                column=_COL_MEDIA,
                value=_sanitize_excel_cell_text(media, max_len=_URL_CELL_MAX_LEN) or None,
            )
        _save_workbook(wb, dest)
    finally:
        if callable(getattr(wb, "close", None)):
            wb.close()
    if missing_media:
        _LOG.warning(
            "PostPlanner %s has %s row(s) without a public MEDIA URL",
            dest.name,
            len(missing_media),
        )
    return dest


def write_planner_json(
    entries: list[dict[str, Any]],
    path: Path,
    *,
    dry_run: bool = False,
) -> Path:
    dest = Path(path)
    if dest.suffix.lower() != ".json":
        dest = dest.with_suffix(".json")
    if dry_run:
        _LOG.info("Dry-run post_planner.json (%s rows) would write %s", len(entries), dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        from modules.durable_store import sync_state_file

        sync_state_file(dest)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("Could not mirror %s (%s)", dest, exc)
    return dest


def write_planner(
    entries: list[dict[str, Any]],
    path: Path,
    *,
    dry_run: bool = False,
) -> Path:
    return write_planner_xlsx(entries, path, dry_run=dry_run)


def build_linkedin_planner_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    linkedin_rows: list[dict[str, Any]] = []
    for item in entries:
        caption = str(item.get("linkedin_caption") or "")
        public_url = planner_media_url(item)
        linkedin_rows.append(
            {
                "plan_id": str(item.get("plan_id") or "").replace("-social", "-linkedin"),
                "asset_id": str(item.get("asset_id") or ""),
                "session_id": str(item.get("session_id") or ""),
                "platform": "linkedin",
                "engine": "architecture",
                "caption": caption,
                "optimized_caption": caption,
                "linkedin_caption": caption,
                "b2_url": public_url,
                "media_url": public_url,
                "tags": prune_hashtags(LINKEDIN_HASHTAGS),
                "hashtags": prune_hashtags(LINKEDIN_HASHTAGS),
                "status": str(item.get("status") or "ready"),
            }
        )
    return linkedin_rows


def compact_reels_export(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "session_id": str(item.get("session_id") or ""),
            "caption": str(item.get("caption") or item.get("post_planner_caption") or ""),
            "media_url": planner_media_url(item),
            "tags": prune_hashtags(item.get("tags") or item.get("hashtags") or []),
        }
        for item in entries
    ]


def compact_linkedin_export(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "session_id": str(item.get("session_id") or ""),
            "linkedin_caption": str(item.get("linkedin_caption") or item.get("caption") or ""),
            "b2_url": planner_media_url(item),
        }
        for item in entries
    ]


def run_planner(
    *,
    library_path: Path | None = None,
    output_path: Path | None = None,
    outputs_dir: Path | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    posts_per_day: int = DEFAULT_POSTS_PER_DAY,
    min_bytes: int = _MIN_VIDEO_BYTES,
    upload_b2: bool | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    media_root = Path(outputs_dir) if outputs_dir is not None else page_outputs_dir(CHANNEL_ID)
    persist_store = library_path is None
    if persist_store:
        restore_channel_state(CHANNEL_ID)
        prune_distribution_catalogs(planner_json=json_planner_path(outputs_dir=media_root))
        if not dry_run:
            delete_asset_library()
        library_path = content_library_path(CHANNEL_ID)
    if upload_b2 is None:
        upload_b2 = persist_store and not dry_run
    rows = load_distribution_library(library_path)
    stamp_library_captions(rows)
    if not dry_run:
        save_distribution_library(library_path, rows)
    entries = build_planner_entries(
        rows,
        now=now,
        posts_per_day=posts_per_day,
        outputs_dir=media_root,
        min_bytes=min_bytes,
    )
    url_by_name = attach_b2_urls(entries, upload=bool(upload_b2) and not dry_run)
    if persist_store and not dry_run and url_by_name:
        persist_library_b2_urls(rows, url_by_name)
    dest = _resolve_xlsx_path(output_path) if output_path is not None else planner_path()
    linkedin_entries = build_linkedin_planner_entries(entries)
    write_planner_xlsx(entries, dest, dry_run=dry_run)
    write_planner_json(entries, json_planner_path(outputs_dir=media_root), dry_run=dry_run)
    write_planner_json(
        compact_reels_export(entries),
        reels_json_path(outputs_dir=media_root),
        dry_run=dry_run,
    )
    write_planner_json(
        compact_linkedin_export(linkedin_entries),
        linkedin_json_path(outputs_dir=media_root),
        dry_run=dry_run,
    )
    write_planner_xlsx(
        linkedin_entries,
        linkedin_planner_path(outputs_dir=media_root),
        dry_run=dry_run,
    )
    if persist_store and not dry_run:
        from modules.durable_store import write_state_json

        write_state_json(CHANNEL_ID, "post_planner.json", entries)
        write_state_json(CHANNEL_ID, REELS_JSON_NAME, compact_reels_export(entries))
        write_state_json(CHANNEL_ID, LINKEDIN_JSON_NAME, compact_linkedin_export(linkedin_entries))
    if output_path is None:
        write_planner_xlsx(
            entries,
            bulk_planner_path(outputs_dir=media_root),
            dry_run=dry_run,
        )
        write_planner_xlsx(
            linkedin_entries,
            linkedin_bulk_path(outputs_dir=media_root),
            dry_run=dry_run,
        )
    return dest, entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiwake-post-planner",
        description=(
            "Import root production MP4s (ignore Tests/Reproved) into the "
            "PostPlanner Excel with sanitized optimized captions."
        ),
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--outputs-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--posts-per-day", type=int, default=DEFAULT_POSTS_PER_DAY)
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument(
        "--skip-b2",
        action="store_true",
        help="Do not upload MP4s; only reuse b2_url already stored on the library.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    dest, entries = run_planner(
        library_path=args.library,
        output_path=args.output,
        outputs_dir=args.outputs_dir,
        dry_run=bool(args.dry_run),
        posts_per_day=max(2, min(3, int(args.posts_per_day))),
        upload_b2=not bool(args.dry_run) and not bool(args.skip_b2),
    )
    ready = sum(1 for item in entries if item["status"] == "ready")
    errors = verify_engine_separation(entries)
    public = sum(1 for item in entries if planner_media_url(item))
    print(
        f"Aiwake post planner  rows={len(entries)}  ready={ready}  "
        f"social={len(entries)}  b2={public}  "
        f"file={dest}  mode={'DRY-RUN' if args.dry_run else 'write'}"
    )
    if errors:
        print("caption contract FAILED:")
        for item in errors[:12]:
            print(f"  - {item}")
        return 2
    print("caption contract: OK (social in planner / LinkedIn in content_library)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
