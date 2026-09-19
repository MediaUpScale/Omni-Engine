# -*- coding: utf-8 -*-
"""
Queryable catalog of generated stills for reuse across post types.

Persists to ``channels_config/{channel}/store/asset_library.json`` and
mirrors a copy to ``{OUTPUT_PATH}/{channel}/``. Future reel/video
generators can call ``find_reusable_asset`` before billing a new FLUX call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()

_FACTORY_ROOT = Path(__file__).resolve().parents[1]


def _outputs_root() -> Path:
    from utils.pipeline_paths import outputs_root

    return outputs_root()
_HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_]{1,48})")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]{2,}")
_TITLE_CHUNK_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "into", "over",
    "under", "near", "upon", "your", "their", "them", "they", "what",
    "when", "where", "which", "while", "about", "after", "before", "between",
    "through", "across", "around", "still", "just", "more", "some", "every",
    "very", "also", "only", "than", "then", "once", "have", "been", "were",
    "was", "are", "not", "but", "you", "our", "its", "his", "her", "she",
    "him", "who", "how", "why", "can", "may", "will", "would", "could",
    "should", "must", "ultra", "realistic", "cinematic", "photograph",
    "photography", "image", "scene", "shot", "view", "wide", "high",
    "full", "deep", "rich", "single", "source", "light", "lighting",
    "no", "none", "without", "absolute", "absolutely",
    "yet", "these", "this", "that", "one", "two", "could", "would",
    "discovered", "mainstream", "however", "still", "just", "into",
    "onto", "those", "both", "each", "many", "most", "such",
})

_DOMAIN_TERMS = frozenset({
    "pyramid", "pyramids", "sphinx", "giza", "megalith", "megalithic",
    "baalbek", "trilithon", "stonehenge", "moai", "easter", "island",
    "gobekli", "tepe", "atlantis", "nazca", "puma", "punku", "sacsayhuaman",
    "yonaguni", "dwarka", "antikythera", "vimana", "obelisk", "obelisks",
    "hieroglyph", "hieroglyphs", "sarcophagus", "ziggurat", "dolmen",
    "temple", "temples", "ruin", "ruins", "artifact", "artefact",
    "monument", "monuments", "civilization", "civilisation", "ancient",
    "mystery", "mysteries", "anomalous", "anomaly", "technology",
    "glyph", "glyphs", "inscription", "inscriptions", "quarry",
    "lost", "forbidden", "archaeology", "archaeological", "starfield",
    "torchlight", "moonlight", "crystal", "skull", "skulls", "stone",
    "stones", "block", "blocks", "horizon", "desert", "jungle",
    "submerged", "flood", "pre-flood", "preflood",
})

_CAMEL_RE = re.compile(r"[A-Z][a-z]+|[a-z]+|\d+")
_MAX_TAGS = 24


def _split_camel(token: str) -> list[str]:
    parts = _CAMEL_RE.findall(token.replace("#", ""))
    if len(parts) >= 2:
        return [" ".join(p.lower() for p in parts)]
    return []


def library_path(channel: str) -> Path:
    """Primary: ``channels_config/<channel>/store/asset_library.json``."""
    from utils.pipeline_paths import channel_store_dir

    slug = (channel or "unknown").strip().lower() or "unknown"
    return channel_store_dir(slug) / "asset_library.json"


def extract_hashtags(text: str | None) -> list[str]:
    """Return unique ``#Tag`` tokens from caption/prompt text (original casing)."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _HASHTAG_RE.finditer(text or ""):
        tag = f"#{match.group(1)}"
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            out.append(tag)
    return out


def extract_asset_tags(
    prompt: str = "",
    caption: str = "",
    hashtags: Iterable[str] | None = None,
) -> list[str]:
    """
    Combine prompt keywords, caption entities, and hashtags into a cleaned
    lower-case ``search_tags`` list.
    """
    blob = f"{prompt or ''}\n{caption or ''}"
    found: dict[str, None] = {}

    def _add(raw: str) -> None:
        token = " ".join(str(raw or "").lower().replace("_", " ").split())
        token = token.strip("#").strip()
        if len(token) < 3 or token in _STOPWORDS:
            return
        found.setdefault(token, None)

    for tag in list(hashtags or []) + extract_hashtags(blob):
        _add(str(tag))
        for phrase in _split_camel(str(tag)):
            _add(phrase)

    for match in _TITLE_CHUNK_RE.finditer(caption or ""):
        chunk = match.group(1).strip()
        words = chunk.lower().split()
        if not words or all(w in _STOPWORDS for w in words):
            continue
        if len(words) == 1 and words[0] not in _DOMAIN_TERMS and len(words[0]) < 4:
            continue
        _add(chunk)

    lowered = blob.lower()
    for term in _DOMAIN_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            _add(term)

    for token in _TOKEN_RE.findall(blob):
        cleaned = token.lower().replace("’", "'")
        if cleaned in _DOMAIN_TERMS:
            _add(cleaned)

    return list(found.keys())[:_MAX_TAGS]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _relpath(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(str(path))
    resolved = raw.resolve() if raw.is_absolute() or raw.exists() else raw
    for base in (_outputs_root(), _FACTORY_ROOT):
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _asset_id(channel: str, local_path: str, prompt: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha1(
        f"{channel}|{local_path}|{(prompt or '')[:240]}".encode("utf-8", "replace")
    ).hexdigest()[:10]
    slug = (channel or "asset").strip().lower() or "asset"
    return f"{slug}_{stamp}_{digest}"


def _coerce_library(raw: Any, channel: str) -> dict[str, Any]:
    if isinstance(raw, list):
        return {"channel": channel, "assets": raw}
    if isinstance(raw, dict):
        assets = raw.get("assets")
        if not isinstance(assets, list):
            assets = []
        return {"channel": raw.get("channel") or channel, "assets": assets}
    return {"channel": channel, "assets": []}


def _read_library(path: Path, channel: str) -> dict[str, Any]:
    from modules.durable_store import hydrate_state_file

    path = hydrate_state_file(path)
    if not path.is_file():
        return {"channel": channel, "assets": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("asset_library read failed (%s) — starting empty catalog.", exc)
        return {"channel": channel, "assets": []}
    return _coerce_library(raw, channel)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    from modules.durable_store import sync_state_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    sync_state_file(path)


def load_library(channel: str) -> list[dict[str, Any]]:
    """Return the asset list for *channel* (empty if missing)."""
    path = library_path(channel)
    with _LOCK:
        return list(_read_library(path, channel).get("assets") or [])


def save_library(channel: str, assets: list[dict[str, Any]]) -> Path:
    """Replace the asset list for *channel* and mirror to G:."""
    path = library_path(channel)
    with _LOCK:
        catalog = _read_library(path, channel)
        catalog["channel"] = (channel or "").strip().lower() or "unknown"
        catalog["updated_at"] = _now_iso()
        catalog["assets"] = list(assets)
        _atomic_write(path, catalog)
    return path


def register_generated_asset(
    *,
    channel: str,
    post_type: str,
    local_path: str,
    remote_url: str = "",
    prompt: str = "",
    caption: str = "",
    hashtags: Iterable[str] | None = None,
    platform: str = "facebook",
    video_path: str = "",
    audio_duration_s: float | None = None,
    asset_kind: str = "",
    increment_usage: bool = True,
) -> dict[str, Any] | None:
    """
    Append (or refresh) one approved still/video in ``asset_library.json``.

    When *increment_usage* is False (backfills / catalog refresh) an existing
    row is updated in place and ``usage_count`` is left unchanged.

    Returns the stored record, or ``None`` when there is no local path.
    """
    rel = _relpath(local_path)
    if not rel:
        return None
    tags_src = list(hashtags) if hashtags is not None else extract_hashtags(caption)
    if not tags_src:
        tags_src = extract_hashtags(caption)
    search_tags = extract_asset_tags(prompt, caption, tags_src)
    pt = (post_type or "").upper().strip() or "LONG_CAPTION_IMAGE"
    ch = (channel or "").strip().lower() or "unknown"
    kind = (asset_kind or "").strip().lower()
    if not kind:
        kind = "video" if rel.lower().endswith((".mp4", ".mov", ".webm", ".m4v")) else "still"
    usage_event = {
        "post_type": pt,
        "date": _today(),
        "platform": (platform or "facebook").strip().lower() or "facebook",
    }
    record = {
        "asset_id": _asset_id(ch, rel, prompt),
        "channel": ch,
        "post_type": pt,
        "asset_kind": kind,
        "local_path": rel,
        "video_path": _relpath(video_path) if video_path else "",
        "remote_url": str(remote_url or ""),
        "prompt": prompt or "",
        "caption": caption or "",
        "hashtags": list(tags_src),
        "search_tags": search_tags,
        "audio_duration_s": (
            round(float(audio_duration_s), 3) if audio_duration_s is not None else None
        ),
        "created_at": _now_iso(),
        "usage_count": 1,
        "usage_history": [usage_event],
    }
    path = library_path(ch)
    with _LOCK:
        catalog = _read_library(path, ch)
        assets: list[dict[str, Any]] = list(catalog.get("assets") or [])
        existing = next(
            (a for a in assets if str(a.get("local_path") or "") == rel),
            None,
        )
        if existing is not None:
            if increment_usage:
                existing["usage_count"] = int(existing.get("usage_count") or 0) + 1
                history = list(existing.get("usage_history") or [])
                history.append(usage_event)
                existing["usage_history"] = history
            if remote_url and not existing.get("remote_url"):
                existing["remote_url"] = str(remote_url)
            if prompt and not existing.get("prompt"):
                existing["prompt"] = prompt
            if caption and not existing.get("caption"):
                existing["caption"] = caption
            if video_path and not existing.get("video_path"):
                existing["video_path"] = _relpath(video_path)
            if audio_duration_s is not None and existing.get("audio_duration_s") is None:
                existing["audio_duration_s"] = round(float(audio_duration_s), 3)
            if search_tags:
                merged = list(existing.get("search_tags") or [])
                for tag in search_tags:
                    if tag not in merged:
                        merged.append(tag)
                existing["search_tags"] = merged[:_MAX_TAGS]
            record = existing
        else:
            assets.append(record)
        catalog["channel"] = ch
        catalog["updated_at"] = _now_iso()
        catalog["assets"] = assets
        _atomic_write(path, catalog)
    _LOG.info(
        "asset_library | registered %s | tags=%s | %s",
        record.get("asset_id"), record.get("search_tags"), path,
    )
    return record


def find_reusable_asset(
    channel: str,
    tags: Iterable[str],
    max_usage: int = 2,
) -> dict[str, Any] | None:
    """
    Return the best existing still whose ``search_tags`` overlap *tags*
    and whose ``usage_count`` is below *max_usage*.
    """
    wanted = {
        " ".join(str(t).lower().replace("#", " ").replace("_", " ").split())
        for t in (tags or [])
        if str(t).strip()
    }
    wanted.discard("")
    if not wanted:
        return None
    cap = max(0, int(max_usage))
    best: dict[str, Any] | None = None
    best_score = 0
    for asset in load_library(channel):
        used = int(asset.get("usage_count") or 0)
        if used >= cap:
            continue
        hay = {
            str(t).lower().strip()
            for t in (asset.get("search_tags") or [])
            if str(t).strip()
        }
        overlap = len(wanted & hay)
        if overlap <= 0:
            continue
        # Prefer more tag overlap, then fewer prior uses, then newer records.
        rank = (overlap, -used, str(asset.get("created_at") or ""))
        if best is None or rank > (
            best_score,
            -int(best.get("usage_count") or 0),
            str(best.get("created_at") or ""),
        ):
            best = asset
            best_score = overlap
    return best


register_asset = register_generated_asset
