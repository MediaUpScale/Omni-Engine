# -*- coding: utf-8 -*-
"""
agents/posting/facebook_scheduler/reels_scheduler.py
=====================================
Automate **Reels** scheduling on Meta Business Suite via the dedicated
Reel composer endpoint.

Architecture
------------
Built on ``media_scheduler_base.UniversalComposerScheduler``:

* ``LocalMediaQueue`` — scan ``outputs/<channel>/clips/*.mp4``, track state in
  ``facebook_history.json``, move completed files to ``clips/posted_facebook/``
* Video → ``https://business.facebook.com/latest/reels_composer``
  (never the Universal photo Create-post flow)
* Photo schedulers (future) → Universal ``/latest/`` Create post
* CDP ``DOM.setFileInputFiles`` for large uploads (no OS focus steal)
* Dynamic interval — first: ``now + random(25–60) min``; later: ``last + interval + random(10–30) min``

Usage
-----
    # Dry-run (scan queue, print plan — no browser clicks)
    python -m facebook_scheduler.reels_scheduler --channel master_mei --dry-run

    # Any folder of MP4s (captions from <folder>/asset_library.json)
    python -m facebook_scheduler.reels_scheduler --folder "D:/clips" --dry-run
    python -m facebook_scheduler.reels_scheduler --folder "D:/clips" --modelCTA
    python -m facebook_scheduler.reels_scheduler --folder "D:/clips" --modelHashtag

    # Live schedule (attach to Dolphin CDP / running Business Suite tab)
    python -m facebook_scheduler.reels_scheduler --channel master_mei

    # Limit batch size
    python -m facebook_scheduler.reels_scheduler --channel master_mei --max 3

Requirements
------------
* Meta Business Suite UI language = **English**
* Dolphin{anty} (or Chrome) profile already open on business.facebook.com
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure project root is importable when run as a script
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.posting.facebook_scheduler import config
from agents.posting.facebook_scheduler.facebook_scheduler import attach_to_dolphin_profile
from agents.posting.facebook_scheduler.logger import get_logger
from agents.posting.facebook_scheduler.media_scheduler_base import (
    REELS_COMPOSER_URL,
    MediaItem,
    UniversalComposerScheduler,
    is_video_file,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page

_log = get_logger(__name__)

# Explicit wait after MP4 attach — Meta finishes Reel composer processing.
_REEL_CONVERSION_WAIT_MS = 10_000


class ReelsScheduler(UniversalComposerScheduler):
    """
    Schedule local ``.mp4`` Reels via the dedicated Reel composer.

    Flow (English UI)
    -----------------
    1. ``prepare_composer_for_item`` →
       ``https://business.facebook.com/latest/reels_composer``
       (dismisses ``Can't Read Files`` first).
    2. Wait for ``input[type=file]``; CDP ``DOM.setFileInputFiles``.
    3. Fill caption (``contenteditable`` / ``textarea``) without OS focus.
    4. Advance wizard → Schedule option → datetime → footer Schedule.

    Queue integrity
    ---------------
    ``facebook_history.json`` / ``posted_facebook/`` updates happen ONLY in
    ``run()`` after ``schedule_item()`` returns successfully. Upload failures
    raise and leave the file pending for retry.
    """

    format_type = "reel"
    next_clicks = 2  # Reel optimization: Next → Next → schedule screen

    def select_format(self) -> None:
        """No-op — Reel composer URL already selects the Reel product surface."""
        _log.info(
            "Reel composer endpoint active (%s) — no format card click.",
            REELS_COMPOSER_URL,
        )

    def prepare_composer_for_item(self, item: MediaItem) -> None:
        """
        Always hard-reload the dedicated Reel composer for every queue item.

        Meta leaves a success / post-schedule UI that hides "Add video";
        soft same-URL navigation is not enough between batch items.
        """
        if self.dry_run:
            return
        if not is_video_file(item.path):
            _log.warning(
                "Non-video file in ReelsScheduler queue (%s) — "
                "delegating to base photo/universal routing.",
                item.path.name,
            )
            super().prepare_composer_for_item(item)
            return

        self._dismiss_composer_error_modals()
        self._hard_reload_reels_composer()

    def reset_page(self) -> None:
        """Hard-reload Reel composer after errors so Add video remounts."""
        _log.info("Resetting Reel composer after error...")
        if self.dry_run:
            return
        try:
            for _ in range(3):
                self.page.keyboard.press("Escape")
                self.page.wait_for_timeout(400)
        except Exception:
            pass
        try:
            self._hard_reload_reels_composer()
        except Exception as exc:
            _log.warning("Reel composer hard reload during reset failed: %s", exc)

    def on_media_uploaded(self, item: MediaItem, *, is_video: bool) -> None:
        """After CDP attach, wait for Reel composer processing."""
        if not is_video:
            return
        _log.info(
            "Waiting %dms for Reel composer upload processing...",
            _REEL_CONVERSION_WAIT_MS,
        )
        self.page.wait_for_timeout(_REEL_CONVERSION_WAIT_MS)

    def prepare_caption(self, item: MediaItem) -> str:
        """
        Network-first caption routing (no truncation for library captions).

        ``facebook_caption`` → ``caption`` → ``final_caption`` /
        ``humanized_caption`` → sidecar → pool. Missing library → empty
        caption (post without title/description).
        """
        from agents.posting.facebook_scheduler.media_scheduler_base import LocalMediaQueue

        meta = item.metadata or {}
        allow_fallback = getattr(self.queue, "allow_stem_fallback", True)
        text = (item.caption or "").strip()
        source = getattr(item, "caption_source", "") or ""
        if not text:
            text, source = LocalMediaQueue.resolve_caption(
                item.path,
                meta,
                allow_fallback=allow_fallback,
            )
        item.caption = text
        item.caption_source = source
        if source == "fallback" and text:
            text = LocalMediaQueue.shorten_caption(text)
        _log.info(
            "Caption source=%s (%d chars): %r",
            source,
            len(text),
            text[:100],
        )
        return text

    def fill_caption(self, caption: str) -> None:
        """
        Reel composer caption — delegates to base background-safe fill
        (``contenteditable`` / ``textarea`` / ``role=textbox``).
        """
        super().fill_caption(caption)

    def advance_composer(self) -> None:
        """Wait for upload, advance Reel wizard to the schedule screen."""
        super().advance_composer()


# ===========================================================================
# CLI
# ===========================================================================

def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="reels_scheduler",
        description=(
            "Schedule local Reels (.mp4) on Meta Business Suite "
            "using the dedicated reels_composer endpoint. "
            "Pass --folder for any directory, or --channel for outputs/<channel>/clips."
        ),
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--channel",
        help='Channel folder name under outputs/ (e.g. "master_mei").',
    )
    src.add_argument(
        "--folder",
        help="Any folder of .mp4 files. Captions from <folder>/asset_library.json.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Scan queue and print plan without browser actions or state writes.",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Schedule at most N reels in this run.",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=4.0,
        metavar="HOURS",
        help="Hours between scheduled reels (default: 4). Example: --interval 6",
    )
    ap.add_argument(
        "--cdp",
        default="",
        help="CDP endpoint or port (default: auto-detect Dolphin / config).",
    )
    ap.add_argument(
        "--no-move",
        action="store_true",
        default=False,
        help="Keep files in place after success (still write facebook_history.json).",
    )
    model_lib = ap.add_mutually_exclusive_group()
    model_lib.add_argument(
        "--modelCTA",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="TRUE",
        dest="model_cta",
        help=(
            "If <folder>/asset_library.json is missing, create one with the "
            "LADA model CTA pool. Example: --modelCTA or --modelCTA true"
        ),
    )
    model_lib.add_argument(
        "--modelHashtag",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="TRUE",
        dest="model_hashtag",
        help=(
            "If <folder>/asset_library.json is missing, create one with a "
            "brand-safe 2-hashtag pool. Example: --modelHashtag or --modelHashtag true"
        ),
    )
    return ap


def _build_queue(args: argparse.Namespace):
    from agents.posting.facebook_scheduler.media_scheduler_base import LocalMediaQueue

    if args.folder:
        return LocalMediaQueue.from_folder(
            args.folder,
            extensions=(".mp4",),
            move_on_success=not args.no_move,
            interval_hours=args.interval,
        )
    return LocalMediaQueue(
        args.channel,
        media_subdir="clips",
        extensions=(".mp4",),
        move_on_success=not args.no_move,
        interval_hours=args.interval,
    )


def _maybe_write_model_library(args: argparse.Namespace, queue) -> None:
    if args.model_cta:
        flag, writer = "--modelCTA", queue.ensure_model_cta_library
    elif args.model_hashtag:
        flag, writer = "--modelHashtag", queue.ensure_model_hashtag_library
    else:
        return
    if not args.folder:
        _log.warning("%s only applies with --folder; ignoring.", flag)
        return
    created = writer()
    if created:
        print(f"[reels_scheduler] wrote {flag} library -> {created}")
        return
    if queue.asset_library_path.is_file():
        print(f"[reels_scheduler] {flag}: existing library kept ({queue.asset_library_path})")
    else:
        print(f"[reels_scheduler] {flag}: no library written")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    queue = _build_queue(args)
    _maybe_write_model_library(args, queue)
    pending = queue.scan_pending(format_type="reel")
    if args.max is not None:
        pending = pending[: max(0, args.max)]

    source_label = args.folder or args.channel
    print(
        f"[reels_scheduler] source={source_label!r} pending={len(pending)} "
        f"interval={args.interval:g}h composer={REELS_COMPOSER_URL}"
    )
    if args.dry_run:
        cursor = queue.next_schedule_datetime()
        print(f"[dry-run] first slot ≈ {cursor.strftime('%Y-%m-%d %H:%M')}")
        for i, item in enumerate(pending, 1):
            cap = (item.caption or "").replace("\n", " ")
            preview = f"  caption={cap!r}" if cap else "  caption=<empty>"
            print(f"  {i:>3}  {item.filename}{preview}")
        return 0

    from playwright.sync_api import sync_playwright

    cdp_port: int | None = None
    if args.cdp:
        try:
            cdp_port = int(args.cdp.split(":")[-1].strip("/"))
        except ValueError:
            cdp_port = None

    with sync_playwright() as pw:
        _ctx, page = attach_to_dolphin_profile(pw, port=cdp_port)
        scheduler = ReelsScheduler(
            page,
            queue.channel_name,
            dry_run=False,
            move_on_success=not args.no_move,
            max_items=args.max,
            media_dir=args.folder or None,
            interval_hours=args.interval,
        )
        stats = scheduler.run()
        print(
            f"[reels_scheduler] done scheduled={stats['scheduled']} "
            f"failed={stats['failed']} skipped={stats.get('skipped', 0)}"
        )
        return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
