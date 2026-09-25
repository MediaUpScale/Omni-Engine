# -*- coding: utf-8 -*-
"""Schedule pending Aiwake Shorts on YouTube (ESP-style limit + interval).

Reuses the factory publisher (``agents.posting.youtube_publisher.upload_short``)
and the same per-page OAuth token isolation as Wealth / ESP:

    credentials/tokens/youtube_token_aiwake.json

    python -m channels_config.aiwake.tools.schedule_youtube --limit 10 --interval 24h
    python -m channels_config.aiwake.tools.schedule_youtube --limit 17 --interval 84h --dry-run
    python -m channels_config.aiwake.tools.schedule_youtube --validate-only

Companion tools stay in this package:
    python -m channels_config.aiwake.tools.backfill_metadata
    python -m channels_config.aiwake.tools.sync_youtube_metadata
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover — loose-script invocation
    _FACTORY = Path(__file__).resolve().parents[3]
    if str(_FACTORY) not in sys.path:
        sys.path.insert(0, str(_FACTORY))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
except ImportError:
    pass

from modules.distribution_contract import (
    HIGH_RPM_SEARCH_TAGS,
    US_PEAK_TZ,
    YOUTUBE_CATEGORY_SCIENCE_TECH,
    build_us_peak_slots,
    coerce_scheduled_time,
    content_library_path,
    strip_shorts_title,
    first_us_peak_slot,
    load_distribution_library,
    merge_high_rpm_search_tags,
    slot_iso_utc,
    upsert_distribution_row,
    validate_queue_ready,
)
from modules.durable_store import restore_channel_state
from utils.pipeline_paths import page_outputs_dir

_LOG = logging.getLogger("aiwake.youtube")

CHANNEL_ID = "aiwake"
PUBLISHER_MODULE = "agents.posting.youtube_publisher.upload_short"
YOUTUBE_CATEGORY_ID = YOUTUBE_CATEGORY_SCIENCE_TECH
DEFAULT_LIMIT = 10
DEFAULT_INTERVAL = "24h"
ET = US_PEAK_TZ
PUBLISH_HOUR = 18
PUBLISH_MINUTE = 0


def parse_schedule_interval(
    spec: str | float | int | None,
    *,
    default: str = DEFAULT_INTERVAL,
) -> timedelta:
    from agents.posting.youtube_publisher import parse_interval_spec

    return parse_interval_spec(spec if spec not in (None, "") else default, default_hours=24.0)


def cadence_label(interval: timedelta) -> str:
    hours = interval.total_seconds() / 3600.0
    if abs(hours - 24.0) < 0.01:
        gap = "24h"
    elif hours >= 24 and abs(hours % 24) < 0.01:
        gap = f"{int(hours // 24)}d"
    else:
        gap = f"{hours:g}h"
    return f"18:00 ET then +{gap} (UTC Z)"


@dataclass(slots=True)
class SchedulePlanItem:
    index: int
    row: dict[str, Any]
    session_id: str
    title: str
    description: str
    tags: list[str]
    video_path: Path
    publish_at: datetime
    publish_at_iso: str
    category_id: str = YOUTUBE_CATEGORY_ID
    privacy_status: str = "private"
    status: str = "dry_run"
    youtube_video_id: str = ""
    youtube_url: str = ""
    error: str = ""
    publisher: str = PUBLISHER_MODULE


@dataclass(slots=True)
class ScheduleResult:
    dry_run: bool
    interval: timedelta = field(default_factory=lambda: timedelta(hours=24))
    planned: list[SchedulePlanItem] = field(default_factory=list)
    uploaded: list[SchedulePlanItem] = field(default_factory=list)
    errors: list[SchedulePlanItem] = field(default_factory=list)
    deferred: list[dict[str, str]] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    safety_cap_hit: bool = False


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for noisy in ("googleapiclient.discovery", "google.auth", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def youtube_status(row: dict[str, Any]) -> str:
    status = (row.get("posting_status") or {}).get("youtube")
    return str(status or "pending").strip().lower() or "pending"


def _nested(row: dict[str, Any], *keys: str) -> Any:
    cursor: Any = row
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def row_recency_key(row: dict[str, Any]) -> str:
    """Newest-first sort key: timestamp, then session_id, then filename."""
    for key in ("timestamp", "created_utc", "created_at"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    session = str(row.get("session_id") or "").strip()
    if session:
        return session
    return Path(str(row.get("video_path") or "")).stem


_ALWAYS_SKIP_FOLDERS = frozenset({
    "reproved", "tests", "archive", "needs_metadata", "posted_facebook",
})
_CHANNEL_SKIP_FOLDERS = frozenset({
    "tmp", "temp", "scratch", "__pycache__", ".git",
})


def is_animation_clip(path: Path | str) -> bool:
    """True only for the current broadcast aesthetic."""
    return "animation_clips" in {part.lower() for part in Path(path).parts}


def excluded_video_folder(path: Path | str) -> str:
    """Return the skipped folder name if *path* lives under tests/reproved/etc.

    OS temp roots (pytest ``tmp_path``, ``%TEMP%``) are ignored. ``tmp`` /
    ``temp`` only exclude files under the channel outputs tree.
    """
    parts = [part.lower() for part in Path(path).parts]
    channel_seen = False
    for name in parts:
        if name == CHANNEL_ID:
            channel_seen = True
        if name in _ALWAYS_SKIP_FOLDERS:
            return name
        if name in _CHANNEL_SKIP_FOLDERS and channel_seen:
            return name
    return ""


def select_pending_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    pending = [
        row
        for row in rows
        if youtube_status(row) == "pending"
        and is_animation_clip(str(row.get("video_path") or ""))
        and not excluded_video_folder(str(row.get("video_path") or ""))
    ]
    pending.sort(key=row_recency_key, reverse=True)
    cap = max(0, int(limit))
    return pending[:cap]


def first_publish_slot(*, now: datetime | None = None) -> datetime:
    """Tomorrow at 18:00 America/New_York (15:00 PT → 22:00 UTC in EDT)."""
    return first_us_peak_slot(now=now)


def latest_future_publish_slot(
    rows: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Latest future YouTube ``scheduled_time`` already on the catalog."""
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    latest: datetime | None = None
    for row in rows:
        if youtube_status(row) not in {"scheduled", "posted"}:
            continue
        youtube = _nested(row, "platform_overrides", "youtube")
        youtube = youtube if isinstance(youtube, dict) else {}
        raw = str(youtube.get("scheduled_time") or "").strip()
        if not raw:
            continue
        try:
            slot = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if slot.tzinfo is None:
            slot = slot.replace(tzinfo=timezone.utc)
        slot = slot.astimezone(timezone.utc)
        if slot <= current:
            continue
        if latest is None or slot > latest:
            latest = slot
    return latest


def build_publish_slots(
    count: int,
    *,
    now: datetime | None = None,
    first: datetime | None = None,
    interval: timedelta | None = None,
) -> list[datetime]:
    return build_us_peak_slots(count, now=now, first=first, interval=interval)


def slot_iso_utc_z(slot: datetime) -> str:
    """``2026-09-06T22:00:00Z``."""
    return slot_iso_utc(slot)


def map_youtube_payload(row: dict[str, Any]) -> dict[str, Any]:
    base = row.get("base_metadata") if isinstance(row.get("base_metadata"), dict) else {}
    youtube = _nested(row, "platform_overrides", "youtube")
    youtube = youtube if isinstance(youtube, dict) else {}
    title = strip_shorts_title(
        str(youtube.get("title") or base.get("title") or row.get("topic") or "").strip()
    )
    description = str(
        youtube.get("caption")
        or base.get("caption")
        or row.get("final_caption")
        or ""
    ).strip()
    raw_tags = base.get("search_tags") or row.get("search_tags") or []
    if not isinstance(raw_tags, list):
        raw_tags = []
    tags = merge_high_rpm_search_tags(
        [str(tag).strip() for tag in raw_tags if str(tag).strip()] + list(HIGH_RPM_SEARCH_TAGS)
    )
    category_id = YOUTUBE_CATEGORY_ID
    video_path = Path(str(row.get("video_path") or ""))
    session_id = str(row.get("session_id") or "").strip() or video_path.stem
    return {
        "session_id": session_id,
        "title": title or strip_shorts_title(video_path.stem),
        "description": description,
        "tags": tags,
        "category_id": category_id,
        "video_path": video_path,
    }


def plan_schedule(
    rows: list[dict[str, Any]],
    *,
    limit: int = DEFAULT_LIMIT,
    now: datetime | None = None,
    interval: timedelta | None = None,
) -> tuple[list[SchedulePlanItem], list[dict[str, str]]]:
    """Bind newest pending rows to US-peak slots. Incomplete rows never queue."""
    pending = [row for row in rows if youtube_status(row) == "pending"]
    pending.sort(key=row_recency_key, reverse=True)
    rejected: list[dict[str, str]] = []
    eligible: list[dict[str, Any]] = []
    for row in pending:
        video_path = str(row.get("video_path") or "")
        skipped = excluded_video_folder(video_path)
        if skipped:
            rejected.append({
                "session_id": str(row.get("session_id") or Path(video_path).stem),
                "reason": f"skipped folder '{skipped}' (tests/reproved/archive are never queued)",
            })
            continue
        if not is_animation_clip(video_path):
            rejected.append({
                "session_id": str(row.get("session_id") or Path(video_path).stem),
                "reason": "terminal aesthetic retired; only animation_clips are queued",
            })
            continue
        catalog_errors = validate_queue_ready(row, require_scheduled_time=False)
        if catalog_errors:
            rejected.append({
                "session_id": str(row.get("session_id") or Path(str(row.get("video_path") or "")).stem),
                "reason": "; ".join(catalog_errors),
            })
            continue
        eligible.append(row)
        if len(eligible) >= max(0, int(limit)):
            break
    last = latest_future_publish_slot(rows, now=now)
    first = None
    if last is not None:
        gap = interval if isinstance(interval, timedelta) and interval.total_seconds() > 0 else timedelta(hours=24)
        first = last + gap
    slots = build_publish_slots(len(eligible), now=now, first=first, interval=interval)
    planned: list[SchedulePlanItem] = []
    for index, (row, slot) in enumerate(zip(eligible, slots), start=1):
        payload = map_youtube_payload(row)
        iso = slot_iso_utc(slot)
        gate = validate_queue_ready(row, scheduled_time=iso)
        if gate:
            rejected.append({
                "session_id": str(payload["session_id"]),
                "reason": "; ".join(gate),
            })
            continue
        planned.append(
            SchedulePlanItem(
                index=index,
                row=row,
                session_id=str(payload["session_id"]),
                title=str(payload["title"]),
                description=str(payload["description"]),
                tags=list(payload["tags"]),
                video_path=Path(payload["video_path"]),
                publish_at=slot,
                publish_at_iso=iso,
                category_id=YOUTUBE_CATEGORY_ID,
            )
        )
    return planned, rejected


def _mark_row_scheduled(
    row: dict[str, Any],
    *,
    video_id: str,
    scheduled_time: str,
    title: str,
    description: str,
) -> dict[str, Any]:
    updated = dict(row)
    status = dict(updated.get("posting_status") or {})
    status["youtube"] = "scheduled"
    updated["posting_status"] = status
    overrides = dict(updated.get("platform_overrides") or {})
    youtube = dict(overrides.get("youtube") or {})
    youtube["category_id"] = YOUTUBE_CATEGORY_ID
    youtube["is_short"] = True
    youtube["title"] = strip_shorts_title(str(youtube.get("title") or title))
    youtube["caption"] = str(youtube.get("caption") or description)
    youtube["video_id"] = video_id
    youtube["scheduled_time"] = coerce_scheduled_time(scheduled_time)
    overrides["youtube"] = youtube
    updated["platform_overrides"] = overrides
    return updated


def persist_scheduled(
    library_path: Path,
    item: SchedulePlanItem,
    *,
    video_id: str,
) -> dict[str, Any]:
    updated = _mark_row_scheduled(
        item.row,
        video_id=video_id,
        scheduled_time=item.publish_at_iso,
        title=item.title,
        description=item.description,
    )
    return upsert_distribution_row(library_path, updated)


def _print_preview_table(
    items: list[SchedulePlanItem],
    *,
    dry_run: bool,
    interval: timedelta | None = None,
) -> None:
    sep = "+" + "=" * 72 + "+"
    label = "DRY RUN PREVIEW" if dry_run else "UPLOAD SUMMARY"
    cadence = cadence_label(interval or timedelta(hours=24))
    print()
    print(sep)
    print(f"| {('AIWAKE YOUTUBE SHORTS  |  ' + label):<70} |")
    print(sep)
    print(f"| Publisher module : {PUBLISHER_MODULE:<51} |")
    print(f"| Cadence          : {cadence:<51} |")
    print(sep)
    if not items:
        print(f"| {'No pending Aiwake Shorts in content_library.json.':<70} |")
        print(sep)
        return
    header = f"{'#':>2}  {'session':<22} {'publish_at UTC':<25} title"
    print(header)
    print("-" * 72)
    for item in items:
        print(
            f"{item.index:>2}  {item.session_id[:22]:<22} "
            f"{item.publish_at_iso:<25} {item.title[:40]}"
        )
        print(f"    path      : {item.video_path}")
        print(f"    publisher : {item.publisher}")
        if item.status == "failed":
            print(f"    error     : {item.error}")
        elif item.youtube_url:
            print(f"    youtube   : {item.youtube_url}")
        print()
    print(sep)
    ok = sum(1 for item in items if item.status in {"dry_run", "scheduled"})
    failed = sum(1 for item in items if item.status == "failed")
    print(f"  Total: {len(items)} | Ready/OK: {ok} | Failed: {failed}")
    print(sep)
    print()


def run_schedule(
    *,
    outputs_dir: Path | None = None,
    limit: int = DEFAULT_LIMIT,
    dry_run: bool = False,
    now: datetime | None = None,
    interval: timedelta | str | float | None = None,
    upload_fn=None,
    youtube_client=None,
) -> ScheduleResult:
    """Select newest pending rows and schedule them (or preview)."""
    from agents.posting.youtube_publisher import (
        DailyUploadSafetyGate,
        YouTubeQuotaExceededError,
        build_youtube_client_for_page,
        resolve_youtube_token_path,
        sanitize_youtube_tags,
        upload_short,
    )

    media_root = Path(outputs_dir) if outputs_dir else page_outputs_dir(CHANNEL_ID)
    if outputs_dir is None:
        restore_channel_state(CHANNEL_ID)
        library_path = content_library_path(CHANNEL_ID)
    else:
        library_path = content_library_path(CHANNEL_ID, outputs_dir=media_root)
    rows = load_distribution_library(library_path)
    gap = interval if isinstance(interval, timedelta) else parse_schedule_interval(interval)
    planned, rejected = plan_schedule(rows, limit=limit, now=now, interval=gap)
    result = ScheduleResult(dry_run=dry_run, interval=gap, planned=planned, rejected=rejected)

    token = resolve_youtube_token_path(CHANNEL_ID)
    print(f"[Aiwake] Token -> {token}")
    print(f"[Aiwake] Publisher -> {PUBLISHER_MODULE}")
    print(
        f"[Aiwake] Scheduling {len(planned)} Short(s) | "
        f"newest first | {cadence_label(gap)} | dry_run={dry_run}"
    )
    if rejected:
        print(f"[Aiwake] Rejected {len(rejected)} row(s) missing required metadata:")
        for item in rejected:
            print(f"  - {item['session_id']}: {item['reason']}")
    if not planned:
        print("[Aiwake] Nothing queue-ready - run the metadata backfill first.")
        _print_preview_table(planned, dry_run=dry_run, interval=gap)
        return result

    gate = DailyUploadSafetyGate(limit=limit)
    print(f"[Aiwake] Global upload safety cap this run: {gate.limit}")

    youtube = youtube_client
    uploader = upload_fn or upload_short
    if not dry_run and youtube is None and upload_fn is None:
        youtube = build_youtube_client_for_page(CHANNEL_ID, enforce_channel=True)

    for item in planned:
        if not gate.can_upload():
            result.safety_cap_hit = True
            gate.notify_halt()
            result.deferred.append(
                {"video_path": str(item.video_path), "title": item.title}
            )
            break
        print(
            f"[Aiwake] [{item.index}/{len(planned)}] {item.video_path.name} -> "
            f"{item.publish_at_iso} | {item.title[:70]}"
        )
        if dry_run:
            item.status = "dry_run"
            result.uploaded.append(item)
            gate.record_success()
            continue
        if not item.video_path.is_file():
            item.status = "failed"
            item.error = f"missing file: {item.video_path}"
            result.errors.append(item)
            print(f"[Aiwake] Missing file — left pending: {item.video_path}")
            continue
        try:
            video_id, url, used_slot = uploader(
                video_path=str(item.video_path),
                title=item.title,
                description=item.description,
                tags=sanitize_youtube_tags(item.tags),
                privacy_status="private",
                publish_at=item.publish_at,
                category_id=item.category_id,
                page_name=CHANNEL_ID,
                youtube=youtube,
                skip_playlist=True,
                preserve_title=True,
            )
        except YouTubeQuotaExceededError as exc:
            item.status = "failed"
            item.error = str(exc)
            result.errors.append(item)
            result.safety_cap_hit = True
            gate.notify_halt()
            remaining = [
                later
                for later in planned
                if later.index > item.index
            ]
            result.deferred.extend(
                {"video_path": str(later.video_path), "title": later.title}
                for later in remaining
            )
            print(f"[Aiwake] Quota hit — remaining Shorts left pending. {exc}")
            break
        except Exception as exc:  # noqa: BLE001 — keep the batch moving
            _LOG.exception("Aiwake YouTube upload failed for %s", item.video_path)
            item.status = "failed"
            item.error = str(exc)[:300]
            result.errors.append(item)
            print(f"[Aiwake] FAILED {item.video_path.name}: {exc}")
            continue

        used = used_slot or item.publish_at
        item.publish_at = used
        item.publish_at_iso = slot_iso_utc(used)
        item.youtube_video_id = video_id
        item.youtube_url = url
        item.status = "scheduled"
        persist_scheduled(library_path, item, video_id=video_id)
        result.uploaded.append(item)
        gate.record_success()
        print(
            f"[Aiwake] Library updated | youtube=scheduled | "
            f"id={video_id} | {item.publish_at_iso}"
        )

    _print_preview_table(planned, dry_run=dry_run, interval=gap)
    return result


def validate_library_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Audit required schema fields before any item enters the upload queue."""
    report: list[dict[str, Any]] = []
    for row in rows:
        errors = validate_queue_ready(row)
        session_id = str(row.get("session_id") or Path(str(row.get("video_path") or "")).stem)
        report.append({
            "session_id": session_id,
            "ok": not errors,
            "errors": errors,
            "youtube_status": youtube_status(row),
        })
    return report


def print_validation_report(report: list[dict[str, Any]]) -> None:
    ready = [row for row in report if row["ok"]]
    blocked = [row for row in report if not row["ok"]]
    print()
    print("Aiwake distribution validation")
    print(f"  rows     : {len(report)}")
    print(f"  ready    : {len(ready)}")
    print(f"  blocked  : {len(blocked)}")
    print()
    for row in blocked:
        print(f"  [blocked] {row['session_id']}: {'; '.join(row['errors'])}")
    if not blocked:
        print("  All rows have base_metadata, search_tags, and UTC Z scheduled_time.")
    print()


def run_validate_only(*, outputs_dir: Path | None = None) -> int:
    media_root = Path(outputs_dir) if outputs_dir else page_outputs_dir(CHANNEL_ID)
    if outputs_dir is None:
        restore_channel_state(CHANNEL_ID)
        library_path = content_library_path(CHANNEL_ID)
    else:
        library_path = content_library_path(CHANNEL_ID, outputs_dir=media_root)
    rows = load_distribution_library(library_path)
    report = validate_library_rows(rows)
    print_validation_report(report)
    return 2 if any(not row["ok"] for row in report) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiwake-schedule-youtube",
        description=(
            "Schedule pending Aiwake Shorts via youtube_publisher.upload_short "
            "(private + publishAt). --limit and --interval are dynamic "
            "(ESP-style). Runs live unless --dry-run is passed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m channels_config.aiwake.tools.schedule_youtube --limit 10 --interval 24h\n"
            "  python -m channels_config.aiwake.tools.schedule_youtube --limit 17 --interval 84h --dry-run\n"
            "  python -m channels_config.aiwake.tools.schedule_youtube --validate-only\n"
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        help="Override {OUTPUT_PATH}/aiwake",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="How many newest pending Shorts to schedule this run (default: 10).",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help=(
            "Gap between publishAt slots after the first 18:00 ET peak. "
            "Accepts 24h, 84h, 12h, 2d, or a bare hour count (default: 24h)."
        ),
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=None,
        help="Deprecated alias for --interval (numeric hours only).",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        dest="dry_run",
        action="store_true",
        help="Preview slots without calling the YouTube API.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Audit content_library rows for base_metadata, search_tags, and "
            "youtube.scheduled_time (YYYY-MM-DDTHH:MM:SSZ) without uploading."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(bool(args.verbose))
    if args.validate_only:
        return run_validate_only(outputs_dir=args.outputs_dir)
    interval = args.interval
    if getattr(args, "interval_hours", None) is not None and (
        not str(args.interval or "").strip() or args.interval == DEFAULT_INTERVAL
    ):
        interval = args.interval_hours
    result = run_schedule(
        outputs_dir=args.outputs_dir,
        limit=max(0, int(args.limit)),
        dry_run=bool(args.dry_run),
        interval=interval,
    )
    if result.errors and not result.uploaded:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
