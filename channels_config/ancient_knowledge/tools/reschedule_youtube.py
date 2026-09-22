# -*- coding: utf-8 -*-
"""Replace the 10 silent scheduled Ancient Knowledge YouTube uploads.

1. Pull title/description from the live scheduled IDs.
2. Postpone those silent videos so they cannot go public.
3. Upload remuxed MP4s into the original slots (quota-safe: leftovers queued).
4. Queue remuxed v11–v20 for the next quota window.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")

from channels_config.ancient_knowledge.tools.rebuild_silent_reels import (
    BATCH,
    CLIPS,
    _clean_script,
    _find_reel,
    _load_captions,
)
from agents.posting.youtube_publisher import (
    build_youtube_client_for_page,
    get_next_publish_slot,
    delete_youtube_video,
    queue_pending_upload,
    update_video_publish_at,
    upload_short,
    YouTubeQuotaExceededError,
)

PAGE = "ancient_knowledge"
PLAYLIST = "Ancient Mysteries & Forbidden History"
CTA = "Follow Ancient Knowledge for more hidden mysteries."


def _title_from_caption(caption: str) -> str:
    first = re.split(r"[.!?]", caption or "", maxsplit=1)[0].strip()
    return (first or "Ancient Knowledge")[:100]


def _description(caption: str) -> str:
    body = _clean_script(caption)
    return f"{body}\n\n{CTA}\n\n#AncientKnowledge #AncientMysteries #History #Shorts"


def _fetch_video(youtube, video_id: str) -> dict:
    resp = youtube.videos().list(part="snippet,status", id=video_id).execute()
    items = resp.get("items") or []
    return items[0] if items else {}


def main() -> int:
    captions = _load_captions()
    youtube = build_youtube_client_for_page(PAGE)
    uploaded: list[str] = []
    queued: list[str] = []
    postponed: list[str] = []

    # --- 1. Postpone the 10 silent videos already on YouTube ---
    first_ten = [row for row in BATCH if row[3]]
    original_slots: dict[int, datetime] = {}
    meta_by_variant: dict[int, dict] = {}
    for variant, _name, _stem, vid in first_ten:
        info = _fetch_video(youtube, vid)
        if not info:
            print(f"v{variant:02d} {vid} not found — skip postpone")
            continue
        status = info.get("status") or {}
        snippet = info.get("snippet") or {}
        raw_pa = status.get("publishAt")
        if raw_pa:
            pa = datetime.fromisoformat(raw_pa.replace("Z", "+00:00"))
            original_slots[variant] = pa
        meta_by_variant[variant] = {
            "title": snippet.get("title") or _title_from_caption(captions.get(variant, "")),
            "description": snippet.get("description") or _description(captions.get(variant, "")),
            "tags": snippet.get("tags") or ["ancient knowledge", "ancient mysteries", "shorts"],
        }
        far = datetime.now(timezone.utc) + timedelta(days=365)
        try:
            update_video_publish_at(youtube, vid, far, privacy_status="private")
            postponed.append(vid)
            print(f"POSTPONED silent {vid} (v{variant:02d}) → {far.isoformat()}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARN postpone {vid}: {exc}")
            try:
                delete_youtube_video(youtube, vid)
                postponed.append(vid)
                print(f"DELETED silent {vid} (v{variant:02d})")
            except Exception as exc2:  # noqa: BLE001
                print(f"FAIL isolate {vid}: {exc2}")

    # --- 2. Upload remuxed v01–v10 into original slots ---
    for variant, reel_name, _stem, _old_id in first_ten:
        src = _find_reel(reel_name)
        if src is None or not src.is_file():
            print(f"MISSING remux v{variant:02d} {reel_name}")
            continue
        meta = meta_by_variant.get(variant) or {
            "title": _title_from_caption(captions.get(variant, "")),
            "description": _description(captions.get(variant, "")),
            "tags": ["ancient knowledge", "ancient mysteries", "shorts"],
        }
        publish_at = original_slots.get(variant)
        now = datetime.now(timezone.utc)
        if publish_at is None or publish_at <= now + timedelta(minutes=15):
            publish_at = get_next_publish_slot(youtube, interval_hours=12.0)
        try:
            vid, url, when = upload_short(
                src,
                meta["title"],
                description=meta["description"],
                tags=meta["tags"],
                privacy_status="private",
                publish_at=publish_at,
                page_name=PAGE,
                youtube=youtube,
                playlist_title=PLAYLIST,
            )
            uploaded.append(f"v{variant:02d} {url} @ {when}")
            print(f"UPLOADED remux v{variant:02d} → {url} @ {when}")
        except YouTubeQuotaExceededError as exc:
            print(f"QUOTA at v{variant:02d}: {exc}")
            queue_pending_upload(
                page_name=PAGE,
                video_path=src,
                title=meta["title"],
                description=meta["description"],
                tags=meta["tags"],
                publish_at=publish_at,
                playlist_title=PLAYLIST,
                reason="ak_silent_rebuild_quota",
            )
            queued.append(f"v{variant:02d}")
            # remaining first-ten also queue
            for v2, n2, _s2, _i2 in first_ten:
                if v2 <= variant:
                    continue
                src2 = _find_reel(n2)
                if src2 is None:
                    continue
                meta2 = meta_by_variant.get(v2) or {
                    "title": _title_from_caption(captions.get(v2, "")),
                    "description": _description(captions.get(v2, "")),
                    "tags": ["ancient knowledge", "ancient mysteries", "shorts"],
                }
                pa2 = original_slots.get(v2) or (publish_at + timedelta(hours=12 * (v2 - variant)))
                queue_pending_upload(
                    page_name=PAGE,
                    video_path=src2,
                    title=meta2["title"],
                    description=meta2["description"],
                    tags=meta2["tags"],
                    publish_at=pa2,
                    playlist_title=PLAYLIST,
                    reason="ak_silent_rebuild_quota",
                )
                queued.append(f"v{v2:02d}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL upload v{variant:02d}: {exc}")
            queue_pending_upload(
                page_name=PAGE,
                video_path=src,
                title=meta["title"],
                description=meta["description"],
                tags=meta["tags"],
                publish_at=publish_at,
                playlist_title=PLAYLIST,
                reason="ak_silent_rebuild_error",
            )
            queued.append(f"v{variant:02d}")

    # --- 3. Queue remuxed v11–v20 (never reached YouTube) ---
    last_slot = datetime.now(timezone.utc) + timedelta(hours=12)
    if original_slots:
        last_slot = max(original_slots.values())
    for i, (variant, reel_name, _stem, _yt) in enumerate(
        [row for row in BATCH if row[0] >= 11]
    ):
        src = _find_reel(reel_name)
        if src is None or not src.is_file():
            # v15/v06 may have mojibake names
            src = CLIPS / reel_name
        if not src.is_file():
            print(f"MISSING remux v{variant:02d} {reel_name}")
            continue
        pa = last_slot + timedelta(hours=12 * (i + 1))
        queue_pending_upload(
            page_name=PAGE,
            video_path=src,
            title=_title_from_caption(captions.get(variant, "")),
            description=_description(captions.get(variant, "")),
            tags=["ancient knowledge", "ancient mysteries", "shorts", "history"],
            publish_at=pa,
            playlist_title=PLAYLIST,
            reason="ak_silent_rebuild_last10",
        )
        queued.append(f"v{variant:02d}")
        print(f"QUEUED v{variant:02d} → {src.name} @ {pa.isoformat()}")

    print("\n=== YouTube summary ===")
    print(f"postponed silent originals: {len(postponed)}")
    print(f"uploaded remuxes: {len(uploaded)}")
    for row in uploaded:
        print(" ", row)
    print(f"queued: {len(queued)} → {queued}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
