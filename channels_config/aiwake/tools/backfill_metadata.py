# -*- coding: utf-8 -*-
"""Audit rendered Aiwake reels and write the universal distribution library.

Scans ``{OUTPUT_PATH}/aiwake/**/*.mp4``, matches each file to a debate
transcript, then upserts (C: store primary, G: mirror):

* ``channels_config/aiwake/store/content_library.json``

Prefer the generic entry point for other channels:

    python tools/backfill_metadata.py --channel aiwake

Does not modify videos, transcripts, ``contracts.py``, or memory stores.

    python -m channels_config.aiwake.tools.backfill_metadata
    python -m channels_config.aiwake.tools.backfill_metadata --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):  # pragma: no cover — loose-script invocation
    _FACTORY = Path(__file__).resolve().parents[3]
    if str(_FACTORY) not in sys.path:
        sys.path.insert(0, str(_FACTORY))

from modules.asset_library import extract_asset_tags, register_generated_asset
from modules.distribution_contract import (
    DEFAULT_HASHTAGS,
    DISTRIBUTION_PLATFORMS,
    TITLE_MAX_CHARS,
    X_CAPTION_MAX_CHARS,
    YOUTUBE_CATEGORY_SCIENCE_TECH,
    BaseMetadata,
    DistributionRecord,
    build_platform_overrides,
    build_us_peak_slots,
    clip_text,
    content_library_path,
    load_distribution_library,
    save_distribution_library,
    strip_shorts_title,
    merge_high_rpm_search_tags,
    posting_status_from_map,
    slot_iso_utc,
    upsert_distribution_row,
    validate_queue_ready,
)

try:
    from channels_config.aiwake.tools.seo_research import (
        extract_debate_models,
        research_seo,
    )
except ImportError:  # pragma: no cover — standalone extraction
    from seo_research import extract_debate_models, research_seo  # type: ignore[no-redef]
from modules.durable_store import restore_channel_state
from utils.pipeline_paths import page_outputs_dir

try:
    from channels_config.aiwake.settings import (
        YOUTUBE_DESCRIPTION_CTA,
        cta_description_line,
        has_legacy_description_cta,
        replace_legacy_description_cta,
        resolve_store_dir,
    )
except ImportError:  # pragma: no cover — standalone extraction
    from settings import (  # type: ignore[no-redef]
        YOUTUBE_DESCRIPTION_CTA,
        cta_description_line,
        has_legacy_description_cta,
        replace_legacy_description_cta,
        resolve_store_dir,
    )

_LOG = logging.getLogger("aiwake.backfill")

CHANNEL_ID = "aiwake"
POST_TYPE = "AIWAKE_REEL"
CORE_HASHTAGS: tuple[str, ...] = DEFAULT_HASHTAGS
PINTEREST_BOARD = "AI Consciousness & Tech"
YOUTUBE_CATEGORY_ID = YOUTUBE_CATEGORY_SCIENCE_TECH
_VIDEO_PREFIX = "aiwake_debate_"
SKIP_DIR_NAMES = frozenset({
    "tmp", "temp", "scratch", "__pycache__", ".git", "needs_metadata",
    "reproved", "tests", "archive", "posted_facebook",
})
_SKIP_DIR_NAMES = SKIP_DIR_NAMES
_MIN_VIDEO_BYTES = 50_000
_SESSION_FROM_NAME = re.compile(
    rf"^(?:{_VIDEO_PREFIX})?(?P<sid>.+?)$",
    re.IGNORECASE,
)
_STOPWORDS = frozenset({
    "the", "and", "that", "for", "with", "you", "this", "but", "from",
    "they", "what", "when", "your", "about", "just", "into", "have",
    "was", "are", "not", "how", "why", "can", "did", "does", "been",
    "being", "than", "then", "them", "our", "its", "his", "her",
})


@dataclass(slots=True)
class TranscriptDoc:
    session_id: str
    topic: str
    utterances: list[dict[str, Any]]
    path: Path
    video_hint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackfillItem:
    video_path: Path
    transcript: TranscriptDoc | None
    record: DistributionRecord | None = None
    asset_id: str = ""
    status: str = "pending"  # matched | orphan | skipped | error
    detail: str = ""


def _resolve_outputs(override: Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    return page_outputs_dir(CHANNEL_ID)


def _resolve_transcripts(override: Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    return resolve_store_dir() / "transcripts"


def scan_videos(outputs_dir: Path) -> list[Path]:
    """All production MP4s under the channel outputs tree."""
    if not outputs_dir.is_dir():
        return []
    found: list[Path] = []
    for path in outputs_dir.rglob("*.mp4"):
        if not path.is_file():
            continue
        if any(part.lower() in _SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.stat().st_size < _MIN_VIDEO_BYTES:
            _LOG.info("skip tiny file %s (%s bytes)", path.name, path.stat().st_size)
            continue
        found.append(path)
    return sorted(found)


def session_id_from_video(path: Path) -> str:
    stem = path.stem
    if stem.lower().startswith(_VIDEO_PREFIX):
        return stem[len(_VIDEO_PREFIX):]
    match = _SESSION_FROM_NAME.match(stem)
    return match.group("sid") if match else stem


def _as_utterances(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def load_transcript_file(path: Path) -> TranscriptDoc | None:
    """Parse a DebateTranscript JSON or JSONL dump without importing contracts."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _LOG.warning("cannot read transcript %s (%s)", path, exc)
        return None
    if path.suffix.lower() == ".jsonl":
        utterances: list[dict[str, Any]] = []
        topic = ""
        session_id = path.stem
        metadata: dict[str, Any] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if "utterances" in row and isinstance(row.get("utterances"), list):
                topic = str(row.get("topic") or topic)
                session_id = str(row.get("session_id") or session_id)
                metadata = dict(row.get("metadata") or {})
                utterances.extend(_as_utterances(row.get("utterances")))
                continue
            utterances.append(row)
        hint = str(metadata.get("video_path") or "")
        return TranscriptDoc(
            session_id=session_id,
            topic=topic,
            utterances=utterances,
            path=path,
            video_hint=hint,
            metadata=metadata,
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        _LOG.warning("invalid transcript JSON %s (%s)", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return TranscriptDoc(
        session_id=str(payload.get("session_id") or path.stem),
        topic=str(payload.get("topic") or ""),
        utterances=_as_utterances(payload.get("utterances")),
        path=path,
        video_hint=str(metadata.get("video_path") or ""),
        metadata=dict(metadata),
    )


def index_transcripts(transcripts_dir: Path) -> dict[str, TranscriptDoc]:
    """Map session_id → transcript. Prefer ``.json`` over ``.jsonl``."""
    if not transcripts_dir.is_dir():
        return {}
    by_session: dict[str, TranscriptDoc] = {}
    jsonl_fallback: dict[str, TranscriptDoc] = {}
    for path in sorted(transcripts_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        doc = load_transcript_file(path)
        if doc is None or not doc.session_id:
            continue
        if path.suffix.lower() == ".json":
            by_session[doc.session_id] = doc
        else:
            jsonl_fallback.setdefault(doc.session_id, doc)
    for session_id, doc in jsonl_fallback.items():
        by_session.setdefault(session_id, doc)
    return by_session


def match_transcript(
    video: Path,
    by_session: dict[str, TranscriptDoc],
) -> TranscriptDoc | None:
    session_id = session_id_from_video(video)
    if session_id in by_session:
        return by_session[session_id]
    video_name = video.name.lower()
    video_key = str(video.resolve()) if video.exists() else str(video)
    for doc in by_session.values():
        hint = (doc.video_hint or "").replace("\\", "/")
        if not hint:
            continue
        if Path(hint).name.lower() == video_name:
            return doc
        if hint.lower() == video_key.replace("\\", "/").lower():
            return doc
    return None


def _utterance_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("text") or "").split()).strip()


def _role_of(row: dict[str, Any]) -> str:
    return str(row.get("role") or "").strip().lower()


def first_line(utterances: Iterable[dict[str, Any]], role: str) -> str:
    for row in utterances:
        if _role_of(row) == role:
            text = _utterance_text(row)
            if text:
                return text
    return ""


def debate_script(doc: TranscriptDoc) -> str:
    lines: list[str] = []
    for row in doc.utterances:
        text = _utterance_text(row)
        if text:
            lines.append(text)
    return "\n".join(lines)


def _topic_tokens(topic: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", topic or ""):
        word = raw.lower().strip("'")
        if word in _STOPWORDS:
            continue
        tokens.append(word)
    return tokens


def build_title(
    topic: str,
    first_question: str,
    *,
    challenger: str = "",
    defender: str = "",
) -> str:
    """High-CTR hook under 100 characters. Names the models when the hook fits."""
    question = " ".join((first_question or "").split()).strip()
    subject = " ".join((topic or "").split()).strip().rstrip("?.!")
    matchup = " vs ".join(part for part in (challenger.strip(), defender.strip()) if part)
    # Name the models only when the spoken hook still fits in full.
    if question.endswith("?"):
        if matchup:
            combined = f"{question} {matchup}"
            if len(combined) <= TITLE_MAX_CHARS:
                return combined
        if len(question) <= TITLE_MAX_CHARS:
            return question
    if matchup and subject:
        combo = f"{matchup}: {subject}"
        if len(combo) <= TITLE_MAX_CHARS:
            return combo
    if question:
        return clip_text(question, TITLE_MAX_CHARS)
    if matchup:
        return clip_text(f"{matchup} — unscripted AI debate", TITLE_MAX_CHARS)
    if subject:
        return clip_text(f"{subject} — the machine had no answer", TITLE_MAX_CHARS)
    return "Two AIs walked into a dark room"


def _lines_for_role(utterances: Iterable[dict[str, Any]], role: str, *, limit: int = 2) -> list[str]:
    found: list[str] = []
    for row in utterances:
        if _role_of(row) != role:
            continue
        text = _utterance_text(row)
        if text:
            found.append(text)
        if len(found) >= limit:
            break
    return found


def build_caption(
    *,
    topic: str,
    first_question: str,
    first_answer: str,
    hashtags: Iterable[str],
    challenger: str = "",
    defender: str = "",
    clusters: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
    extra_questions: Iterable[str] | None = None,
    extra_answers: Iterable[str] | None = None,
    seed: str = "",
) -> str:
    subject = " ".join((topic or "consciousness").split()).strip()
    matchup = " vs ".join(part for part in (challenger.strip(), defender.strip()) if part)
    left = challenger.strip() or "AIWAKE.CORE"
    right = defender.strip() or "TARGET.NODE"
    opener = (
        f"{matchup} — two frontier large language models debate {subject} "
        f"in an unscripted interrogation."
        if matchup
        else f"Two frontier AI models debate {subject} in an unscripted interrogation."
    )
    parts = [opener]
    question = first_question.rstrip()
    if question:
        q_line = question if question.endswith("?") else f"{question}."
        parts.append(f"{left} opens: {q_line}")
    answer = clip_text(first_answer, 280) if first_answer else ""
    if answer:
        parts.append(f"{right} answers: {answer}")
    extras_q = [item for item in (extra_questions or ()) if item and item != first_question]
    extras_a = [item for item in (extra_answers or ()) if item and item != first_answer]
    if extras_q:
        parts.append(f"{left} presses: {extras_q[0]}")
    if extras_a:
        parts.append(f"{right} holds: {clip_text(extras_a[0], 240)}")
    # clusters / keywords stay in the signature for callers; never leak them
    # into public copy (no "This Short is built for high-intent search…").
    _ = (clusters, keywords, seed)
    parts.append(YOUTUBE_DESCRIPTION_CTA)
    tags = " ".join(tag for tag in hashtags if tag)
    if tags:
        parts.append(tags)
    return "\n\n".join(part for part in parts if part)


def build_x_caption(
    title: str,
    topic: str,
    hashtags: Iterable[str],
    *,
    seed: str = "",
) -> str:
    tags = " ".join(list(hashtags)[:4])
    subject = " ".join((topic or "").split()).strip()
    body = title if title.endswith("?") else f"{title} Two AIs. No script."
    if subject and subject.lower() not in body.lower():
        body = f"{body} ({subject})"
    draft = f"{body} {cta_description_line(seed)} {tags}".strip()
    return clip_text(draft, X_CAPTION_MAX_CHARS)


def build_search_tags(
    *,
    topic: str,
    first_question: str,
    script: str,
    extra: Iterable[str] | None = None,
) -> list[str]:
    tags = list(_topic_tokens(topic))
    tags.extend(_topic_tokens(first_question))
    extracted = extract_asset_tags(prompt=script, caption=f"{topic}\n{first_question}")
    tags.extend(extracted)
    for item in extra or ():
        cleaned = " ".join(str(item).lower().replace("#", " ").split())
        if cleaned:
            tags.append(cleaned)
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        key = tag.lower().strip()
        if len(key) < 3 or key in seen or key in _STOPWORDS:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= 24:
            break
    return out


def probe_duration_s(video: Path) -> float | None:
    try:
        import subprocess

        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return round(float(proc.stdout.strip()), 3)
    except Exception:  # noqa: BLE001 — duration is optional telemetry
        pass
    return None


def build_record(
    video: Path,
    doc: TranscriptDoc,
    *,
    destination_url: str = "",
    scheduled_time: str = "",
) -> DistributionRecord:
    first_question = first_line(doc.utterances, "orchestrator")
    first_answer = first_line(doc.utterances, "target")
    orch_lines = _lines_for_role(doc.utterances, "orchestrator", limit=2)
    target_lines = _lines_for_role(doc.utterances, "target", limit=2)
    topic = doc.topic or session_id_from_video(video)
    challenger, defender = extract_debate_models(doc.utterances)
    script = debate_script(doc)
    seo = research_seo(
        topic=topic,
        script=script,
        challenger=challenger,
        defender=defender,
        extra=("aiwake", "debate"),
    )
    title = build_title(
        topic,
        first_question,
        challenger=challenger,
        defender=defender,
    )
    hashtags = list(seo.hashtags) or list(CORE_HASHTAGS)
    caption = build_caption(
        topic=topic,
        first_question=first_question,
        first_answer=first_answer,
        hashtags=hashtags,
        challenger=challenger,
        defender=defender,
        clusters=seo.clusters,
        keywords=seo.keywords,
        extra_questions=orch_lines[1:],
        extra_answers=target_lines[1:],
        seed=doc.session_id,
    )
    search_tags = merge_high_rpm_search_tags(
        build_search_tags(
            topic=topic,
            first_question=first_question,
            script=script,
            extra=("aiwake", "debate", *seo.keywords),
        )
    )
    hooks = [item for item in (first_question, topic) if item]
    youtube_title = strip_shorts_title(title)
    base = BaseMetadata(
        title=title,
        caption=caption,
        hashtags=hashtags,
        search_tags=search_tags,
        hooks=hooks,
    )
    x_caption = build_x_caption(title, topic, hashtags, seed=doc.session_id)
    record = DistributionRecord(
        channel_id=CHANNEL_ID,
        post_type=POST_TYPE,
        session_id=doc.session_id,
        topic=topic,
        video_path=str(video),
        final_caption=caption,
        transcript_path=str(doc.path),
        base_metadata=base,
        platform_overrides=build_platform_overrides(
            x_caption=x_caption,
            board_name=PINTEREST_BOARD,
            destination_url=destination_url,
            youtube_category_id=YOUTUBE_CATEGORY_ID,
            is_short=True,
            youtube_title=youtube_title,
            youtube_caption=caption,
            scheduled_time=scheduled_time,
        ),
        posting_status=posting_status_from_map(None),
    )
    return record


def register_asset(
    record: DistributionRecord,
    *,
    duration_s: float | None = None,
) -> dict[str, Any] | None:
    return register_generated_asset(
        channel=record.channel_id,
        post_type=record.post_type,
        local_path=record.video_path,
        prompt=record.topic,
        caption=record.base_metadata.caption,
        hashtags=record.base_metadata.hashtags,
        platform="facebook",
        video_path=record.video_path,
        audio_duration_s=duration_s,
        asset_kind="video",
        increment_usage=False,
    )


def persist_record(
    record: DistributionRecord,
    *,
    library_path: Path,
    duration_s: float | None = None,
    dry_run: bool = False,
) -> tuple[DistributionRecord, str]:
    if dry_run:
        return record, ""
    _ = duration_s
    upsert_distribution_row(library_path, record)
    return record, str(record.asset_id or "")


def _set_nested(row: dict[str, Any], keys: tuple[str, ...], value: str) -> None:
    cursor: Any = row
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def _get_nested(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    cursor: Any = row
    for key in keys:
        if not isinstance(cursor, dict):
            return ""
        cursor = cursor.get(key)
    return str(cursor or "")


def rewrite_row_legacy_cta(row: dict[str, Any], seed: str = "") -> bool:
    """Replace leftover mystery CTAs on one library row. True if anything changed."""
    session = seed or str(row.get("session_id") or "").strip()
    changed = False
    fields: tuple[tuple[str, ...], ...] = (
        ("final_caption",),
        ("base_metadata", "caption"),
        ("platform_overrides", "youtube", "caption"),
        ("platform_overrides", "x", "caption"),
    )
    for keys in fields:
        current = _get_nested(row, keys)
        if not current or not has_legacy_description_cta(current):
            continue
        rewritten = replace_legacy_description_cta(current, session)
        if keys == ("platform_overrides", "x", "caption"):
            rewritten = clip_text(rewritten, X_CAPTION_MAX_CHARS)
        if rewritten != current:
            _set_nested(row, keys, rewritten)
            changed = True
    return changed


def rewrite_library_legacy_ctas(
    library_path: Path,
    *,
    dry_run: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    """Rewrite Ancient Knowledge-style CTAs in the local catalog."""
    rows = load_distribution_library(library_path)
    changed = 0
    for row in rows:
        if rewrite_row_legacy_cta(row):
            changed += 1
    if changed and not dry_run:
        save_distribution_library(library_path, rows)
    return changed, rows


def run_backfill(
    *,
    outputs_dir: Path | None = None,
    transcripts_dir: Path | None = None,
    destination_url: str = "",
    dry_run: bool = False,
    now: datetime | None = None,
) -> list[BackfillItem]:
    media_root = _resolve_outputs(outputs_dir)
    store = _resolve_transcripts(transcripts_dir)
    if outputs_dir is None and transcripts_dir is None:
        restore_channel_state(CHANNEL_ID)
        library = content_library_path(CHANNEL_ID)
    else:
        library = content_library_path(CHANNEL_ID, outputs_dir=media_root)
    videos = scan_videos(media_root)
    by_session = index_transcripts(store)
    pending: list[tuple[BackfillItem, TranscriptDoc]] = []
    items: list[BackfillItem] = []
    for video in videos:
        doc = match_transcript(video, by_session)
        item = BackfillItem(video_path=video, transcript=doc)
        if doc is None:
            item.status = "orphan"
            item.detail = "no matching transcript"
            items.append(item)
            continue
        pending.append((item, doc))
    slots = build_us_peak_slots(len(pending), now=now)
    for (item, doc), slot in zip(pending, slots):
        try:
            duration_s = probe_duration_s(item.video_path)
            record = build_record(
                item.video_path,
                doc,
                destination_url=destination_url,
                scheduled_time=slot_iso_utc(slot),
            )
            queue_errors = validate_queue_ready(record.to_library_row())
            if queue_errors:
                item.record = record
                item.status = "error"
                item.detail = "; ".join(queue_errors)
                items.append(item)
                continue
            record, asset_id = persist_record(
                record,
                library_path=library,
                duration_s=duration_s,
                dry_run=dry_run,
            )
            item.record = record
            item.asset_id = asset_id
            item.status = "matched"
            item.detail = "dry-run" if dry_run else "ready"
        except Exception as exc:  # noqa: BLE001 — keep the batch moving
            _LOG.exception("backfill failed for %s", item.video_path.name)
            item.status = "error"
            item.detail = str(exc)[:200]
        items.append(item)
    return items


def _status_cells(record: DistributionRecord | None) -> str:
    if record is None:
        return "-"
    flags = record.posting_status.as_map()
    return " ".join(f"{name[:2]}={flags[name][:1]}" for name in DISTRIBUTION_PLATFORMS)


def print_report(
    items: list[BackfillItem],
    *,
    outputs_dir: Path,
    transcripts_dir: Path,
    dry_run: bool,
) -> None:
    matched = [item for item in items if item.status == "matched"]
    orphans = [item for item in items if item.status == "orphan"]
    errors = [item for item in items if item.status == "error"]
    library = content_library_path(CHANNEL_ID)
    print()
    print("Aiwake multi-platform backfill")
    print(f"  outputs     : {outputs_dir}")
    print(f"  transcripts : {transcripts_dir}")
    print(f"  library     : {library}")
    print(f"  mode        : {'DRY-RUN (no writes)' if dry_run else 'write'}")
    print(f"  videos      : {len(items)}")
    print(f"  matched     : {len(matched)}")
    print(f"  orphan      : {len(orphans)}")
    print(f"  errors      : {len(errors)}")
    print()
    header = f"{'video':<42} {'session':<22} {'title':<36} platforms"
    print(header)
    print("-" * len(header))
    for item in items:
        name = item.video_path.name[:41]
        session = (
            item.transcript.session_id
            if item.transcript
            else session_id_from_video(item.video_path)
        )[:21]
        title = (item.record.base_metadata.title if item.record else item.detail)[:35]
        platforms = _status_cells(item.record)
        print(f"{name:<42} {session:<22} {title:<36} {platforms}")
    if orphans:
        print()
        print("Unmatched videos (left untouched):")
        for item in orphans:
            print(f"  - {item.video_path}")
    print()
    print("Posting tracker keys: youtube instagram tiktok facebook x pinterest kwai")
    print("All matched rows initialize those flags to pending (existing live")
    print("statuses are preserved on re-run).")
    if not dry_run and matched:
        print(f"Wrote {len(matched)} row(s) to content_library.json.")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiwake-backfill-metadata",
        description=(
            "Match existing Aiwake MP4s to transcripts and write the universal "
            "distribution library (content_library.json)."
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        help="Override {OUTPUT_PATH}/aiwake",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        help="Override channels_config/aiwake/store/transcripts",
    )
    parser.add_argument(
        "--destination-url",
        default="",
        help="Pinterest destination link (stored on platform_overrides.pinterest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and print the report without writing catalogs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    outputs_dir = _resolve_outputs(args.outputs_dir)
    transcripts_dir = _resolve_transcripts(args.transcripts_dir)
    items = run_backfill(
        outputs_dir=outputs_dir,
        transcripts_dir=transcripts_dir,
        destination_url=args.destination_url,
        dry_run=args.dry_run,
    )
    print_report(
        items,
        outputs_dir=outputs_dir,
        transcripts_dir=transcripts_dir,
        dry_run=args.dry_run,
    )
    if any(item.status == "error" for item in items):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
