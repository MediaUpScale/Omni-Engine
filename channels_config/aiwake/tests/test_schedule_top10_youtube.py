# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from channels_config.aiwake.tools.schedule_youtube import (
    ET,
    PUBLISHER_MODULE,
    build_parser,
    build_publish_slots,
    first_publish_slot,
    map_youtube_payload,
    parse_schedule_interval,
    persist_scheduled,
    plan_schedule,
    run_schedule,
    select_pending_rows,
    slot_iso_utc_z,
    validate_library_rows,
)
from modules.distribution_contract import load_distribution_library, validate_queue_ready


def _row(
    *,
    session: str,
    timestamp: str,
    youtube: str = "pending",
    title: str = "",
    yt_title: str = "",
    caption: str = "Base caption",
    yt_caption: str = "",
    path: str = "",
) -> dict:
    return {
        "session_id": session,
        "timestamp": timestamp,
        "topic": "Who built you?",
        "video_path": path or f"/outputs/aiwake/animation_clips/aiwake_battle_{session}.mp4",
        "base_metadata": {
            "title": title or f"Base {session}",
            "caption": caption,
            "hashtags": ["#aiwake", "#futuretech", "#artificialintelligence", "#tech"],
            "search_tags": ["consciousness", "aiwake", "futuretech", "artificialintelligence", "tech"],
        },
        "platform_overrides": {
            "youtube": {
                "category_id": "28",
                "is_short": True,
                "title": yt_title,
                "caption": yt_caption,
                "scheduled_time": "2026-09-06T22:00:00Z",
            }
        },
        "posting_status": {
            "youtube": youtube,
            "instagram": "pending",
            "tiktok": "pending",
            "facebook": "pending",
            "x": "pending",
            "pinterest": "pending",
            "kwai": "pending",
        },
    }


def test_plan_skips_tests_and_reproved_folders() -> None:
    ok = _row(
        session="ok",
        timestamp="2026-09-05T10:00:00+00:00",
        path="/outputs/aiwake/animation_clips/aiwake_battle_ok.mp4",
    )
    reproved = _row(
        session="bad-reproved",
        timestamp="2026-09-06T10:00:00+00:00",
        path="/outputs/aiwake/reproved/aiwake_debate_bad.mp4",
    )
    tests = _row(
        session="bad-tests",
        timestamp="2026-09-07T10:00:00+00:00",
        path="/outputs/aiwake/tests/aiwake_debate_test.mp4",
    )
    scratch = _row(
        session="bad-tmp",
        timestamp="2026-09-08T10:00:00+00:00",
        path="/outputs/aiwake/tmp/aiwake_debate_tmp.mp4",
    )
    planned, rejected = plan_schedule([ok, reproved, tests, scratch], limit=10)
    assert [item.session_id for item in planned] == ["ok"]
    reasons = " ".join(item["reason"] for item in rejected)
    assert "reproved" in reasons
    assert "tests" in reasons
    assert "tmp" in reasons
    picked = select_pending_rows([ok, reproved, tests, scratch], limit=10)
    assert [row["session_id"] for row in picked] == ["ok"]


def test_plan_rejects_retired_terminal_aesthetic() -> None:
    terminal = _row(
        session="terminal",
        timestamp="2026-09-09T10:00:00+00:00",
        path="/outputs/aiwake/aiwake_debate_terminal.mp4",
    )
    animation = _row(
        session="animation",
        timestamp="2026-09-08T10:00:00+00:00",
    )
    planned, rejected = plan_schedule([terminal, animation], limit=10)
    assert [item.session_id for item in planned] == ["animation"]
    assert rejected[0]["session_id"] == "terminal"
    assert "animation_clips" in rejected[0]["reason"]
    assert select_pending_rows([terminal, animation], limit=10) == [animation]


def test_select_pending_newest_first_top_10() -> None:
    rows = [
        _row(session="old", timestamp="2026-09-01T10:00:00+00:00", youtube="pending"),
        _row(session="mid", timestamp="2026-09-03T10:00:00+00:00", youtube="pending"),
        _row(session="new", timestamp="2026-09-05T10:00:00+00:00", youtube="pending"),
        _row(session="done", timestamp="2026-09-06T10:00:00+00:00", youtube="scheduled"),
    ]
    picked = select_pending_rows(rows, limit=2)
    assert [row["session_id"] for row in picked] == ["new", "mid"]


def test_first_slot_is_tomorrow_1800_et_as_utc_z() -> None:
    now = datetime(2026, 9, 5, 15, 13, tzinfo=ET)
    slot = first_publish_slot(now=now)
    assert slot.tzinfo is not None
    assert slot.hour == 18
    assert slot_iso_utc_z(slot) == "2026-09-06T22:00:00Z"
    series = build_publish_slots(3, now=now)
    assert [slot_iso_utc_z(item) for item in series] == [
        "2026-09-06T22:00:00Z",
        "2026-09-07T22:00:00Z",
        "2026-09-08T22:00:00Z",
    ]
    wide = build_publish_slots(2, now=now, interval=timedelta(hours=84))
    assert [slot_iso_utc_z(item) for item in wide] == [
        "2026-09-06T22:00:00Z",
        "2026-09-10T10:00:00Z",
    ]


def test_payload_prefers_youtube_overrides() -> None:
    row = _row(
        session="sess1",
        timestamp="2026-09-05T10:00:00+00:00",
        title="Base hook",
        yt_title="Override hook",
        caption="Base caption body",
        yt_caption="YouTube description",
    )
    payload = map_youtube_payload(row)
    assert payload["title"] == "Override hook"
    assert payload["description"] == "YouTube description"
    assert "consciousness" in payload["tags"]
    assert "futuretech" in payload["tags"]
    assert "artificialintelligence" in payload["tags"]
    assert "tech" in payload["tags"]
    assert payload["category_id"] == "28"


def test_payload_strips_shorts_hashtag() -> None:
    row = _row(
        session="sess3",
        timestamp="2026-09-05T10:00:00+00:00",
        title="Base hook #Shorts",
        yt_title="Override hook #Shorts",
    )
    payload = map_youtube_payload(row)
    assert payload["title"] == "Override hook"
    assert "#Shorts" not in payload["title"]


def test_payload_falls_back_to_base_metadata() -> None:
    row = _row(
        session="sess2",
        timestamp="2026-09-05T10:00:00+00:00",
        title="Base hook",
        caption="Base caption body",
    )
    payload = map_youtube_payload(row)
    assert payload["title"] == "Base hook"
    assert payload["description"] == "Base caption body"


def test_plan_continues_after_existing_future_slots() -> None:
    already = _row(
        session="live",
        timestamp="2026-09-08T00:00:00+00:00",
        youtube="scheduled",
    )
    already["platform_overrides"]["youtube"]["scheduled_time"] = "2026-09-15T22:00:00Z"
    pending = _row(session="next", timestamp="2026-09-10T00:00:00+00:00")
    planned, rejected = plan_schedule(
        [already, pending],
        limit=10,
        now=datetime(2026, 9, 11, 12, 0, tzinfo=ET),
    )
    assert rejected == []
    assert [item.session_id for item in planned] == ["next"]
    assert planned[0].publish_at_iso == "2026-09-16T22:00:00Z"


def test_plan_binds_newest_to_earliest_slot() -> None:
    rows = [
        _row(session="older", timestamp="2026-09-01T00:00:00+00:00"),
        _row(session="newer", timestamp="2026-09-05T00:00:00+00:00"),
    ]
    planned, rejected = plan_schedule(rows, limit=10, now=datetime(2026, 9, 5, 12, 0, tzinfo=ET))
    assert rejected == []
    assert planned[0].session_id == "newer"
    assert planned[0].publish_at_iso == "2026-09-06T22:00:00Z"
    assert planned[1].session_id == "older"
    assert planned[1].publish_at_iso == "2026-09-07T22:00:00Z"
    assert planned[0].privacy_status == "private"
    assert planned[0].publisher == PUBLISHER_MODULE
    assert "#Shorts" not in planned[0].title


def test_persist_scheduled_updates_library(tmp_path: Path) -> None:
    library = tmp_path / "content_library.json"
    row = _row(session="sess1", timestamp="2026-09-05T10:00:00+00:00")
    library.write_text("[" + __import__("json").dumps(row) + "]\n", encoding="utf-8")
    planned, rejected = plan_schedule(
        [row],
        limit=1,
        now=datetime(2026, 9, 5, 12, 0, tzinfo=ET),
    )
    assert rejected == []
    persist_scheduled(library, planned[0], video_id="abc123xyz")
    saved = load_distribution_library(library)
    assert saved[0]["posting_status"]["youtube"] == "scheduled"
    youtube = saved[0]["platform_overrides"]["youtube"]
    assert youtube["video_id"] == "abc123xyz"
    assert youtube["scheduled_time"] == "2026-09-06T22:00:00Z"
    assert youtube["is_short"] is True
    assert youtube["category_id"] == "28"
    assert "#Shorts" not in youtube["title"]


def test_cli_defaults_to_execute() -> None:
    args = build_parser().parse_args([])
    assert args.dry_run is False
    assert args.validate_only is False
    assert args.interval == "24h"
    args_iv = build_parser().parse_args(["--limit", "17", "--interval", "84h", "--dry-run"])
    assert args_iv.limit == 17
    assert args_iv.interval == "84h"
    assert args_iv.dry_run is True
    assert parse_schedule_interval("84h") == timedelta(hours=84)
    args_dry = build_parser().parse_args(["--dry-run"])
    assert args_dry.dry_run is True
    args_n = build_parser().parse_args(["-n"])
    assert args_n.dry_run is True
    args_limit = build_parser().parse_args(["--limit", "5"])
    assert args_limit.dry_run is False
    assert args_limit.limit == 5


def test_execute_uses_publisher_wrapper(tmp_path: Path) -> None:
    library = tmp_path / "content_library.json"
    video = tmp_path / "animation_clips" / "aiwake_battle_sess1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"0" * 8)
    row = _row(
        session="sess1",
        timestamp="2026-09-05T10:00:00+00:00",
        path=str(video),
    )
    library.write_text("[" + __import__("json").dumps(row) + "]\n", encoding="utf-8")
    calls: list[dict] = []

    def _fake_upload(**kwargs):
        calls.append(kwargs)
        return "vid001", "https://youtu.be/vid001", kwargs["publish_at"]

    result = run_schedule(
        outputs_dir=tmp_path,
        limit=10,
        dry_run=False,
        now=datetime(2026, 9, 5, 12, 0, tzinfo=ET),
        upload_fn=_fake_upload,
        youtube_client=object(),
    )
    assert len(calls) == 1
    assert calls[0]["privacy_status"] == "private"
    assert calls[0]["category_id"] == "28"
    assert calls[0]["page_name"] == "aiwake"
    assert calls[0]["skip_playlist"] is True
    assert calls[0]["preserve_title"] is True
    assert result.uploaded[0].status == "scheduled"
    saved = load_distribution_library(library)
    assert saved[0]["posting_status"]["youtube"] == "scheduled"
    assert saved[0]["platform_overrides"]["youtube"]["video_id"] == "vid001"
    assert saved[0]["platform_overrides"]["youtube"]["scheduled_time"] == "2026-09-06T22:00:00Z"
    assert "#Shorts" not in calls[0]["title"]


def test_plan_rejects_incomplete_rows() -> None:
    incomplete = _row(session="broken", timestamp="2026-09-05T10:00:00+00:00")
    incomplete["base_metadata"]["search_tags"] = []
    complete = _row(session="ok", timestamp="2026-09-04T10:00:00+00:00")
    planned, rejected = plan_schedule([incomplete, complete], limit=10)
    assert [item.session_id for item in planned] == ["ok"]
    assert rejected[0]["session_id"] == "broken"
    assert "search_tags" in rejected[0]["reason"]


def test_validate_library_rows_requires_utc_z_slot() -> None:
    good = _row(session="ok", timestamp="2026-09-05T10:00:00+00:00")
    bad = _row(session="bad", timestamp="2026-09-05T10:00:00+00:00")
    bad["platform_overrides"]["youtube"]["scheduled_time"] = "2026-09-06T18:00:00-03:00"
    report = validate_library_rows([good, bad])
    assert report[0]["ok"] is True
    assert report[1]["ok"] is False
    assert any("scheduled_time" in err for err in report[1]["errors"])
    assert validate_queue_ready(good) == []
