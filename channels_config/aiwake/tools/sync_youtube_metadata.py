# -*- coding: utf-8 -*-
"""Retroactively enrich Aiwake library rows and push metadata to YouTube.

Rewrites local ``content_library.json`` from transcripts (model matchups,
researched descriptions, high-RPM tags), then patches already scheduled
or uploaded videos via ``youtube_publisher.update_video_metadata``.

    python -m channels_config.aiwake.tools.sync_youtube_metadata --dry-run
    python -m channels_config.aiwake.tools.sync_youtube_metadata
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
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
    YOUTUBE_CATEGORY_SCIENCE_TECH,
    content_library_path,
    load_distribution_library,
)
from modules.durable_store import restore_channel_state
from utils.pipeline_paths import page_outputs_dir

try:
    from channels_config.aiwake.settings import (
        cta_description_line,
        has_legacy_description_cta,
        replace_legacy_description_cta,
    )
    from channels_config.aiwake.tools.backfill_metadata import (
        CHANNEL_ID,
        rewrite_library_legacy_ctas,
        run_backfill,
    )
    from channels_config.aiwake.tools.library_sanitize import sanitize_content_library
    from channels_config.aiwake.tools.schedule_youtube import (
        _nested,
        map_youtube_payload,
        youtube_status,
    )
except ImportError:  # pragma: no cover
    from backfill_metadata import (  # type: ignore[no-redef]
        CHANNEL_ID,
        rewrite_library_legacy_ctas,
        run_backfill,
    )
    from library_sanitize import sanitize_content_library  # type: ignore[no-redef]
    from schedule_youtube import (  # type: ignore[no-redef]
        _nested,
        map_youtube_payload,
        youtube_status,
    )
    from settings import (  # type: ignore[no-redef]
        cta_description_line,
        has_legacy_description_cta,
        replace_legacy_description_cta,
    )

_LOG = logging.getLogger("aiwake.sync")

LIVE_YOUTUBE_STATES = frozenset({"scheduled", "posted"})


@dataclass(slots=True)
class SyncItem:
    session_id: str
    video_id: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    status: str = "pending"
    detail: str = ""


@dataclass(slots=True)
class SyncResult:
    dry_run: bool
    enriched: int = 0
    rewritten: int = 0
    pushed: int = 0
    swept: int = 0
    skipped: int = 0
    errors: list[SyncItem] = field(default_factory=list)
    items: list[SyncItem] = field(default_factory=list)


def _youtube_video_id(row: dict[str, Any]) -> str:
    youtube = _nested(row, "platform_overrides", "youtube")
    youtube = youtube if isinstance(youtube, dict) else {}
    return str(youtube.get("video_id") or row.get("youtube_video_id") or "").strip()


def enrich_local_library(
    *,
    outputs_dir: Path | None = None,
    transcripts_dir: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Rebuild every matched catalog row from its transcript."""
    items = run_backfill(
        outputs_dir=outputs_dir,
        transcripts_dir=transcripts_dir,
        dry_run=dry_run,
    )
    return sum(1 for item in items if item.status == "matched")


def iter_live_youtube_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live: list[dict[str, Any]] = []
    for row in rows:
        if youtube_status(row) not in LIVE_YOUTUBE_STATES:
            continue
        if not _youtube_video_id(row):
            continue
        live.append(row)
    return live


def push_youtube_metadata(
    rows: list[dict[str, Any]],
    *,
    dry_run: bool = False,
    youtube_client=None,
    update_fn=None,
) -> SyncResult:
    result = SyncResult(dry_run=dry_run)
    updater = update_fn
    youtube = youtube_client
    for row in iter_live_youtube_rows(rows):
        payload = map_youtube_payload(row)
        item = SyncItem(
            session_id=str(payload["session_id"]),
            video_id=_youtube_video_id(row),
            title=str(payload["title"]),
            description=str(payload["description"]),
            tags=list(payload["tags"]),
        )
        if dry_run:
            item.status = "dry_run"
            item.detail = "would call videos.update"
            result.items.append(item)
            result.pushed += 1
            continue
        try:
            if updater is None:
                from agents.posting.youtube_publisher import (
                    build_youtube_client_for_page,
                    sanitize_youtube_tags,
                    update_video_metadata,
                )

                if youtube is None:
                    youtube = build_youtube_client_for_page(CHANNEL_ID, enforce_channel=True)
                updater = update_video_metadata
                def _bound(video_id, title, description, tags, category_id):
                    return update_video_metadata(
                        youtube,
                        video_id,
                        title=title,
                        description=description,
                        tags=sanitize_youtube_tags(tags),
                        category_id=category_id,
                    )
                updater = _bound
            updater(
                video_id=item.video_id,
                title=item.title,
                description=item.description,
                tags=item.tags,
                category_id=YOUTUBE_CATEGORY_SCIENCE_TECH,
            )
            item.status = "updated"
            item.detail = "videos.update ok"
            result.pushed += 1
        except Exception as exc:  # noqa: BLE001 — keep the batch moving
            _LOG.exception("YouTube metadata push failed for %s", item.video_id)
            item.status = "error"
            item.detail = str(exc)[:240]
            result.errors.append(item)
        result.items.append(item)
    result.skipped = max(0, len(rows) - len(result.items) - len(result.errors))
    return result


def _session_by_video_id(rows: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        video_id = _youtube_video_id(row)
        session_id = str(row.get("session_id") or "").strip()
        if video_id and session_id:
            mapping[video_id] = session_id
    return mapping


def sweep_youtube_legacy_ctas(
    rows: list[dict[str, Any]],
    *,
    dry_run: bool = False,
    youtube_client=None,
    list_fn=None,
    update_fn=None,
) -> SyncResult:
    """Patch leftover mystery CTAs on every scheduled or published upload."""
    result = SyncResult(dry_run=dry_run)
    seeds = _session_by_video_id(rows)
    lister = list_fn
    updater = update_fn
    youtube = youtube_client
    if lister is None:
        from agents.posting.youtube_publisher import (
            build_youtube_client_for_page,
            list_channel_upload_videos,
            update_video_metadata,
        )

        if youtube is None:
            youtube = build_youtube_client_for_page(CHANNEL_ID, enforce_channel=True)
        lister = lambda: list_channel_upload_videos(youtube)
        if updater is None:
            def _bound(video_id, title, description, tags, category_id):
                return update_video_metadata(
                    youtube,
                    video_id,
                    title=title,
                    description=description,
                    tags=tags,
                    category_id=category_id,
                )

            updater = _bound
    videos = lister() if lister is not None else []
    for video in videos or []:
        video_id = str((video or {}).get("video_id") or "").strip()
        description = str((video or {}).get("description") or "")
        title = str((video or {}).get("title") or "")
        if not video_id or not has_legacy_description_cta(description):
            result.skipped += 1
            continue
        seed = seeds.get(video_id) or video_id
        rewritten = replace_legacy_description_cta(description, seed)
        item = SyncItem(
            session_id=seeds.get(video_id, ""),
            video_id=video_id,
            title=title,
            description=rewritten,
        )
        if dry_run:
            item.status = "dry_run"
            item.detail = f"would replace CTA → {cta_description_line(seed)}"
            result.items.append(item)
            result.swept += 1
            continue
        try:
            if updater is None:
                raise RuntimeError("YouTube updater unavailable")
            updater(
                video_id=video_id,
                title=title,
                description=rewritten,
                tags=None,
                category_id=YOUTUBE_CATEGORY_SCIENCE_TECH,
            )
            item.status = "swept"
            item.detail = "legacy CTA replaced on YouTube"
            result.swept += 1
        except Exception as exc:  # noqa: BLE001 — keep the batch moving
            _LOG.exception("YouTube CTA sweep failed for %s", video_id)
            item.status = "error"
            item.detail = str(exc)[:240]
            result.errors.append(item)
        result.items.append(item)
    return result


def run_sync(
    *,
    outputs_dir: Path | None = None,
    transcripts_dir: Path | None = None,
    dry_run: bool = False,
    skip_enrich: bool = False,
    skip_sweep: bool = False,
    youtube_client=None,
    update_fn=None,
    list_fn=None,
) -> SyncResult:
    if outputs_dir is None:
        restore_channel_state(CHANNEL_ID)
    if not skip_enrich:
        enriched = enrich_local_library(
            outputs_dir=outputs_dir,
            transcripts_dir=transcripts_dir,
            dry_run=dry_run,
        )
    else:
        enriched = 0
    media_root = Path(outputs_dir) if outputs_dir else page_outputs_dir(CHANNEL_ID)
    if outputs_dir is None:
        library_path = content_library_path(CHANNEL_ID)
    else:
        library_path = content_library_path(CHANNEL_ID, outputs_dir=media_root)
    rewritten, rows = rewrite_library_legacy_ctas(library_path, dry_run=dry_run)
    sanitized, rows = sanitize_content_library(library_path, dry_run=dry_run)
    rewritten += sanitized
    if not rows:
        rows = load_distribution_library(library_path)
    result = push_youtube_metadata(
        rows,
        dry_run=dry_run,
        youtube_client=youtube_client,
        update_fn=update_fn,
    )
    result.enriched = enriched
    result.rewritten = rewritten
    if not skip_sweep:
        sweep = sweep_youtube_legacy_ctas(
            rows,
            dry_run=dry_run,
            youtube_client=youtube_client,
            list_fn=list_fn,
            update_fn=update_fn,
        )
        result.swept = sweep.swept
        result.items.extend(sweep.items)
        result.errors.extend(sweep.errors)
    return result


def print_sync_report(result: SyncResult) -> None:
    print()
    print("Aiwake YouTube metadata sync")
    print(f"  mode      : {'DRY-RUN' if result.dry_run else 'write'}")
    print(f"  enriched  : {result.enriched}")
    print(f"  rewritten : {result.rewritten}")
    print(f"  pushed    : {result.pushed}")
    print(f"  swept     : {result.swept}")
    print(f"  errors    : {len(result.errors)}")
    for item in result.items:
        print(f"  [{item.status}] {item.session_id} {item.video_id} | {item.title[:60]}")
        if item.status == "error":
            print(f"      {item.detail}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiwake-sync-youtube-metadata",
        description=(
            "Rebuild Aiwake catalog copy from transcripts, then push titles, "
            "descriptions, and tags to already scheduled/uploaded YouTube videos."
        ),
    )
    parser.add_argument("--outputs-dir", type=Path)
    parser.add_argument("--transcripts-dir", type=Path)
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Push current library copy without rebuilding from transcripts.",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Do not list the YouTube uploads playlist for leftover mystery CTAs.",
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
    result = run_sync(
        outputs_dir=args.outputs_dir,
        transcripts_dir=args.transcripts_dir,
        dry_run=bool(args.dry_run),
        skip_enrich=bool(args.skip_enrich),
        skip_sweep=bool(args.skip_sweep),
    )
    print_sync_report(result)
    return 2 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
