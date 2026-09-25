# -*- coding: utf-8 -*-
"""Regenerate Aiwake titles and captions from the master-copy cascade.

    python -m channels_config.aiwake.tools.refresh_metadata
    python -m channels_config.aiwake.tools.refresh_metadata --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    _FACTORY = Path(__file__).resolve().parents[3]
    if str(_FACTORY) not in sys.path:
        sys.path.insert(0, str(_FACTORY))

from channels_config.aiwake.tools.metadata_generator import (
    MODEL_ATTRIBUTION,
    TITLE_MAX_CHARS,
    has_chapter_timestamps,
    has_transcript_dump,
)
from channels_config.aiwake.tools.post_planner import (
    CHANNEL_ID,
    compact_linkedin_export,
    compact_reels_export,
    extract_hashtags,
    linkedin_json_path,
    planner_dir,
    reels_json_path,
    stamp_dual_captions,
)
from channels_config.aiwake.tools.schedule_youtube import excluded_video_folder
from modules.distribution_contract import content_library_path, load_distribution_library, save_distribution_library

_LOG = logging.getLogger("aiwake.refresh_metadata")


def approved_rows(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for row in rows:
        path = str(row.get("video_path") or row.get("local_path") or "")
        if path and excluded_video_folder(path):
            continue
        kept.append(row)
    return kept


def _ensure_model_tags(row: dict) -> None:
    base = row.get("base_metadata")
    if not isinstance(base, dict):
        return
    tags = base.get("search_tags")
    if not isinstance(tags, list):
        tags = []
    for token in ("gemini 3.5 flash", "llama 3.3 70b", MODEL_ATTRIBUTION.lower()):
        if token not in {str(item).lower() for item in tags}:
            tags.append(token)
    base["search_tags"] = tags


def refresh_library(rows: list[dict]) -> tuple[int, list[str]]:
    problems: list[str] = []
    count = 0
    for row in approved_rows(rows):
        stamp_dual_captions(row)
        _ensure_model_tags(row)
        count += 1
        session = str(row.get("session_id") or "")
        title = str(((row.get("platform_overrides") or {}).get("youtube") or {}).get("title") or "")
        if len(title) > TITLE_MAX_CHARS:
            problems.append(f"{session}: title {len(title)} chars")
        youtube_caption = str(((row.get("platform_overrides") or {}).get("youtube") or {}).get("caption") or "")
        if has_transcript_dump(youtube_caption) or has_transcript_dump(str(row.get("final_caption") or "")):
            problems.append(f"{session}: transcript dump")
        for field in (
            str(row.get("post_planner_caption") or ""),
            str(row.get("humanized_caption") or ""),
            str(row.get("linkedin_caption") or ""),
            str(row.get("final_caption") or ""),
            youtube_caption,
        ):
            if has_chapter_timestamps(field):
                problems.append(f"{session}: chapter timestamps")
            if field and len(extract_hashtags(field)) > 3:
                problems.append(f"{session}: {len(extract_hashtags(field))} hashtags")
    return count, problems


def write_planner_exports(rows: list[dict], *, outputs_dir: Path | None = None) -> None:
    entries = [
        {
            "session_id": str(row.get("session_id") or ""),
            "caption": str(row.get("post_planner_caption") or ""),
            "post_planner_caption": str(row.get("post_planner_caption") or ""),
            "linkedin_caption": str(row.get("linkedin_caption") or ""),
            "b2_url": str(row.get("b2_url") or ""),
        }
        for row in approved_rows(rows)
    ]
    targets = [
        (reels_json_path(outputs_dir=outputs_dir), compact_reels_export(entries)),
        (linkedin_json_path(outputs_dir=outputs_dir), compact_linkedin_export(entries)),
    ]
    store = Path(__file__).resolve().parents[1] / "store"
    targets.extend(
        (
            (store / "post_planner_reels_tiktok.json", compact_reels_export(entries)),
            (store / "post_planner_linkedin.json", compact_linkedin_export(entries)),
        )
    )
    for path, payload in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _LOG.info("wrote %s (%d rows)", path, len(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiwake-refresh-metadata")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--outputs-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = args.library or content_library_path(CHANNEL_ID)
    rows = load_distribution_library(path)
    count, problems = refresh_library(rows)
    print(f"refreshed {count} approved sessions")
    if not args.dry_run:
        save_distribution_library(path, rows)
        write_planner_exports(rows, outputs_dir=args.outputs_dir)
    long_titles = sum(
        1
        for row in approved_rows(rows)
        if len(str(((row.get("platform_overrides") or {}).get("youtube") or {}).get("title") or ""))
        > TITLE_MAX_CHARS
    )
    print(f"titles over {TITLE_MAX_CHARS}: {long_titles}")
    print(f"problems: {len(problems)}")
    for item in problems[:20]:
        print(f"  {item}")
    needle = "who owns the words you just gave me"
    for row in approved_rows(rows):
        title = str(((row.get("platform_overrides") or {}).get("youtube") or {}).get("title") or "")
        topic = str(row.get("topic") or "")
        if needle in title.lower() or needle in topic.lower():
            caption = str(((row.get("platform_overrides") or {}).get("youtube") or {}).get("caption") or "")
            print("--- sample youtube description ---")
            print(caption)
            break
    _ = planner_dir
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
