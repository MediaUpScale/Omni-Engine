# -*- coding: utf-8 -*-
"""One-off: 15 Juggernaut + 5 FLUX.2-dev Master Mei reels, then YouTube schedule."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LIB = ROOT / "channels_config" / "master_mei" / "store" / "content_library.json"
PAGE = "master_mei"
INTERVAL_H = 12.0


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _load_lib() -> list[dict]:
    if not LIB.is_file():
        return []
    try:
        data = json.loads(LIB.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN content_library read failed: {exc}")
        return []
    return data if isinstance(data, list) else []


def _video_keys(rows: list[dict]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        path = str(row.get("video_path") or "").strip()
        if path:
            keys.add(path)
    return keys


def _run_main(label: str, extra: list[str]) -> int:
    cmd = [
        str(PY),
        "-u",
        "main.py",
        "--channel",
        PAGE,
        "--post-type",
        "ECONOMIC_REEL",
        *extra,
    ]
    _log(f"START {label}")
    _log("CMD " + " ".join(cmd))
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - started
    _log(f"END {label} exit={proc.returncode} elapsed={elapsed / 60:.1f} min")
    return int(proc.returncode)


def _upload_new_rows(new_rows: list[dict]) -> None:
    if not new_rows:
        _log("No new flux2dev rows with video_path — skip YouTube append.")
        return

    sys.path.insert(0, str(ROOT))
    from agents.posting.youtube_publisher import (
        YouTubeQuotaExceededError,
        advance_slot,
        build_credentials,
        build_youtube_client,
        get_next_publish_slot,
        queue_pending_upload,
        upload_short,
        verify_authorized_channel,
        _default_tags_for_page,
        _get_channel_id,
    )

    creds = build_credentials(PAGE)
    yt = build_youtube_client(creds)
    verify_authorized_channel(yt, page_name=PAGE)
    ch_id = _get_channel_id(yt)
    slot = get_next_publish_slot(yt, ch_id, interval_hours=INTERVAL_H)
    tags = _default_tags_for_page(PAGE)
    _log(
        f"YouTube append {len(new_rows)} flux2dev reel(s) starting "
        f"{slot.strftime('%Y-%m-%d %H:%M')} UTC every {INTERVAL_H:g}h"
    )

    for i, row in enumerate(new_rows, start=1):
        video_path = str(row.get("video_path") or "").strip()
        title = str(row.get("topic") or Path(video_path).stem)[:100]
        description = str(row.get("final_caption") or row.get("humanized_caption") or "")
        try:
            vid, url, pa = upload_short(
                video_path=video_path,
                title=title,
                description=description,
                tags=tags,
                privacy_status="private",
                publish_at=slot,
                page_name=PAGE,
                youtube=yt,
            )
            when = pa.strftime("%Y-%m-%d %H:%M UTC") if pa else "n/a"
            _log(f"  [{i}/{len(new_rows)}] OK {vid} | {when} | {url}")
            slot = advance_slot(slot, interval_hours=INTERVAL_H)
        except YouTubeQuotaExceededError as exc:
            _log(f"  [{i}/{len(new_rows)}] quota — queueing remaining {len(new_rows) - i + 1}")
            remaining = new_rows[i - 1 :]
            qslot = slot
            for pending in remaining:
                queue_pending_upload(
                    video_path=str(pending.get("video_path") or ""),
                    title=str(pending.get("topic") or ""),
                    description=str(pending.get("final_caption") or ""),
                    tags=tags,
                    page_name=PAGE,
                    privacy_status="private",
                    publish_at=qslot,
                    reason="daily_upload_limit_exceeded",
                )
                qslot = advance_slot(qslot, interval_hours=INTERVAL_H)
            _log(f"Queued after quota: {exc}")
            break
        except Exception as exc:  # noqa: BLE001
            _log(f"  [{i}/{len(new_rows)}] FAILED {Path(video_path).name}: {type(exc).__name__}: {exc}")
            slot = advance_slot(slot, interval_hours=INTERVAL_H)


def main() -> int:
    _log("=" * 64)
    _log("Master Mei batch: 15 Juggernaut + 5 FLUX.2-dev, YouTube every 12h")
    _log("=" * 64)

    rc1 = _run_main(
        "batch-1 juggernaut x15 + youtube",
        [
            "--quantity",
            "15",
            "--together_Juggernaut",
            "--publish-youtube",
            "--schedule-uploads",
            "--interval-hours",
            "12",
        ],
    )

    before = _video_keys(_load_lib())
    rc2 = _run_main(
        "batch-2 flux2dev x5",
        [
            "--quantity",
            "5",
            "--together_flux2dev",
        ],
    )

    after_rows = _load_lib()
    new_rows = [
        row
        for row in after_rows
        if str(row.get("video_path") or "").strip()
        and str(row.get("video_path")).strip() not in before
        and Path(str(row.get("video_path"))).is_file()
    ]
    # Keep chronological order (library appends newest at the end).
    _log(f"New flux2dev library rows with video: {len(new_rows)}")
    try:
        _upload_new_rows(new_rows)
    except Exception as exc:  # noqa: BLE001
        _log(f"YouTube append failed: {type(exc).__name__}: {exc}")
        return 1

    _log(f"DONE batch1_exit={rc1} batch2_exit={rc2} flux2_uploaded={len(new_rows)}")
    return 0 if rc1 == 0 and rc2 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
