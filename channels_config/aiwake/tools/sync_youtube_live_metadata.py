# -*- coding: utf-8 -*-
"""Sanitize Aiwake catalog copy and patch live YouTube Shorts in place.

Reads ``content_library.json``, strips prompt-leakage / keyword-stuffing,
rewrites YouTube descriptions as hook + dialogue + dialectics CTA + hashtags,
then calls ``youtube.videos().update(part="snippet")`` for every row that
already has a ``video_id``. Media files are never re-uploaded.

    python -m channels_config.aiwake.tools.sync_youtube_live_metadata --dry-run
    python scripts/sync_youtube_live_metadata.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover — loose-script invocation
    _FACTORY = Path(__file__).resolve().parents[3]
    if str(_FACTORY) not in sys.path:
        sys.path.insert(0, str(_FACTORY))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
except ImportError:
    pass

from agents.posting.youtube_publisher import (
    build_youtube_client_for_page,
    sanitize_youtube_tags,
    update_video_metadata,
)
from channels_config.aiwake.tools.library_sanitize import (
    CHANNEL_ID,
    sanitize_content_library,
)
from channels_config.aiwake.tools.schedule_youtube import map_youtube_payload
from channels_config.aiwake.tools.sync_youtube_metadata import (
    SyncItem,
    SyncResult,
    _youtube_video_id,
)
from modules.distribution_contract import YOUTUBE_CATEGORY_SCIENCE_TECH
from modules.durable_store import restore_channel_state

_LOG = logging.getLogger("aiwake.live_sync")


def iter_rows_with_video_id(rows: list[dict]) -> list[dict]:
    found: list[dict] = []
    for row in rows:
        if _youtube_video_id(row):
            found.append(row)
    return found


def push_live_youtube_metadata(
    rows: list[dict],
    *,
    dry_run: bool = False,
    youtube_client=None,
    update_fn=None,
) -> SyncResult:
    result = SyncResult(dry_run=dry_run)
    updater = update_fn
    youtube = youtube_client
    for row in iter_rows_with_video_id(rows):
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
            item.detail = "would call videos.update(part=snippet)"
            result.items.append(item)
            result.pushed += 1
            continue
        try:
            if updater is None:
                if youtube is None:
                    youtube = build_youtube_client_for_page(CHANNEL_ID, enforce_channel=True)

                def _bound(video_id, title, description, tags, category_id, client=youtube):
                    return update_video_metadata(
                        client,
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
            _LOG.exception("YouTube live metadata push failed for %s", item.video_id)
            item.status = "error"
            item.detail = str(exc)[:240]
            result.errors.append(item)
        result.items.append(item)
    result.skipped = max(0, len(rows) - len(result.items))
    return result


def run_live_sync(
    *,
    library_path: Path | None = None,
    dry_run: bool = False,
    skip_sanitize: bool = False,
    youtube_client=None,
    update_fn=None,
) -> tuple[int, SyncResult]:
    if library_path is None:
        restore_channel_state(CHANNEL_ID)
    if skip_sanitize:
        from modules.distribution_contract import (
            content_library_path,
            load_distribution_library,
        )

        path = library_path
        if path is None:
            path = content_library_path(CHANNEL_ID)
        changed = 0
        rows = load_distribution_library(path)
    else:
        changed, rows = sanitize_content_library(library_path, dry_run=dry_run)
    from channels_config.aiwake.tools.post_planner import stamp_library_captions
    from modules.distribution_contract import save_distribution_library

    stamped = stamp_library_captions(rows)
    changed += stamped
    if stamped and not dry_run:
        path = library_path
        if path is None:
            from modules.distribution_contract import content_library_path

            path = content_library_path(CHANNEL_ID)
        save_distribution_library(path, rows)
    result = push_live_youtube_metadata(
        rows,
        dry_run=dry_run,
        youtube_client=youtube_client,
        update_fn=update_fn,
    )
    return changed, result


def print_report(changed: int, result: SyncResult) -> None:
    print()
    print("Aiwake live YouTube metadata sync")
    print(f"  mode      : {'DRY-RUN' if result.dry_run else 'write'}")
    print(f"  sanitized : {changed}")
    print(f"  pushed    : {result.pushed}")
    print(f"  errors    : {len(result.errors)}")
    for item in result.items:
        print(f"  [{item.status}] {item.session_id} {item.video_id} | {item.title[:60]}")
        if item.status == "error":
            print(f"      {item.detail}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sync-youtube-live-metadata",
        description=(
            "Sanitize content_library.json and patch titles/descriptions on "
            "already-uploaded Aiwake Shorts via videos.update."
        ),
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument(
        "--skip-sanitize",
        action="store_true",
        help="Push current library copy without rewriting local rows.",
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
    changed, result = run_live_sync(
        library_path=args.library,
        dry_run=bool(args.dry_run),
        skip_sanitize=bool(args.skip_sanitize),
    )
    print_report(changed, result)
    return 2 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
