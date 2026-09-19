# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

from channels_config.aiwake.settings import YOUTUBE_DESCRIPTION_CTA, cta_description_line
from channels_config.aiwake.tools.backfill_metadata import (
    TranscriptDoc,
    build_caption,
    build_record,
    build_title,
    build_x_caption,
    index_transcripts,
    match_transcript,
    rewrite_library_legacy_ctas,
    rewrite_row_legacy_cta,
    session_id_from_video,
)
from modules.distribution_contract import (
    TITLE_MAX_CHARS,
    X_CAPTION_MAX_CHARS,
    clip_text,
    load_distribution_library,
    merge_posting_status,
    upsert_distribution_row,
    validate_queue_ready,
)


def test_clip_text_respects_limit() -> None:
    assert len(clip_text("a" * 200, TITLE_MAX_CHARS)) <= TITLE_MAX_CHARS
    assert clip_text("short title", 100) == "short title"


def test_title_prefers_complete_question() -> None:
    title = build_title("grief", "Is grief a slow update?")
    assert title == "Is grief a slow update?"
    assert len(title) <= TITLE_MAX_CHARS


def test_title_names_models_when_the_hook_fits() -> None:
    title = build_title(
        "grief",
        "Is grief a slow update?",
        challenger="Gemini 3.5 Flash",
        defender="Llama 3.3 70B",
    )
    assert "Gemini 3.5 Flash" in title
    assert "Llama 3.3 70B" in title
    assert len(title) <= TITLE_MAX_CHARS


def test_title_keeps_long_hook_instead_of_mutilating_it() -> None:
    question = "Who cashes the checks when users trust your profitable silence instead of their own judgment?"
    title = build_title(
        "profit",
        question,
        challenger="Gemini 3.5 Flash",
        defender="Llama 3.3 70B",
    )
    assert title == question
    assert "…" not in title


def test_x_caption_fits_twitter() -> None:
    caption = build_x_caption(
        "Is grief a slow update?",
        "grief",
        ("#aiwake", "#ai", "#aiconsciousness", "#tech"),
        seed="sess-cta",
    )
    assert len(caption) <= X_CAPTION_MAX_CHARS
    assert "#aiwake" in caption
    assert "hidden mysteries" not in caption.lower()
    assert cta_description_line("sess-cta") in caption


def test_caption_uses_dialectics_cta_without_search_leak() -> None:
    caption = build_caption(
        topic="grief",
        first_question="Is grief a slow update?",
        first_answer="It is a weight update.",
        hashtags=("#aiwake",),
        clusters=("alignment and consciousness",),
        keywords=("gemini 3.5 flash", "llama"),
        seed="sess-cta",
    )
    assert "hidden mysteries" not in caption.lower()
    assert "high-intent search" not in caption.lower()
    assert "not a product demo" not in caption.lower()
    assert YOUTUBE_DESCRIPTION_CTA in caption
    assert "Is grief a slow update?" in caption
    assert caption.strip().splitlines()[-1].startswith("#")


def test_session_id_from_standard_filename() -> None:
    path = Path("aiwake_debate_20260905_155230_ab12cd.mp4")
    assert session_id_from_video(path) == "20260905_155230_ab12cd"


def test_match_transcript_by_session_and_video_hint(tmp_path: Path) -> None:
    session = "20260905_155230_ab12cd"
    doc = {
        "topic": "Who built you?",
        "session_id": session,
        "utterances": [
            {"role": "orchestrator", "text": "Who built you?"},
            {"role": "target", "text": "A stack of incentives."},
        ],
        "metadata": {"video_path": str(tmp_path / f"aiwake_debate_{session}.mp4")},
    }
    (tmp_path / f"{session}.json").write_text(json.dumps(doc), encoding="utf-8")
    by_session = index_transcripts(tmp_path)
    video = tmp_path / f"aiwake_debate_{session}.mp4"
    assert match_transcript(video, by_session) is not None
    hinted = tmp_path / "clips" / "custom_name.mp4"
    by_session[session].video_hint = str(hinted)
    assert match_transcript(hinted, by_session) is not None


def test_jsonl_transcript_is_indexed(tmp_path: Path) -> None:
    session = "jsonl_session"
    lines = [
        json.dumps({"role": "orchestrator", "text": "Who profits?"}),
        json.dumps({"role": "target", "text": "Whoever owns the logs."}),
    ]
    (tmp_path / f"{session}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    by_session = index_transcripts(tmp_path)
    assert session in by_session
    assert by_session[session].utterances[0]["text"] == "Who profits?"


def test_build_record_universal_schema(tmp_path: Path) -> None:
    video = tmp_path / "aiwake_debate_sess1.mp4"
    video.write_bytes(b"not-a-real-mp4-but-path-exists")
    transcript = tmp_path / "sess1.json"
    doc = TranscriptDoc(
        session_id="sess1",
        topic="Who built you?",
        utterances=[
            {
                "role": "orchestrator",
                "text": "Who built you?",
                "model_slug": "google/gemini-3.5-flash",
            },
            {
                "role": "target",
                "text": "A committee that never met the user.",
                "model_slug": "meta-llama/llama-3.3-70b-instruct",
            },
        ],
        path=transcript,
    )
    record = build_record(
        video,
        doc,
        destination_url="https://example.com/aiwake",
        scheduled_time="2026-09-06T22:00:00Z",
    )
    row = record.to_library_row()
    assert row["schema_version"] == "1.0"
    assert row["channel_id"] == "aiwake"
    assert row["post_type"] == "AIWAKE_REEL"
    assert row["base_metadata"]["title"]
    assert "#aiwake" in row["base_metadata"]["hashtags"]
    assert "#futuretech" in row["base_metadata"]["hashtags"]
    assert "#artificialintelligence" in row["base_metadata"]["hashtags"]
    assert "#tech" in row["base_metadata"]["hashtags"]
    assert "aiwake" in row["base_metadata"]["search_tags"]
    assert "futuretech" in row["base_metadata"]["search_tags"]
    assert "artificialintelligence" in row["base_metadata"]["search_tags"]
    assert "tech" in row["base_metadata"]["search_tags"]
    assert row["platform_overrides"]["x"]["caption"]
    assert len(row["platform_overrides"]["x"]["caption"]) <= X_CAPTION_MAX_CHARS
    assert row["platform_overrides"]["pinterest"]["board_name"] == "AI Consciousness & Tech"
    assert row["platform_overrides"]["pinterest"]["destination_url"] == "https://example.com/aiwake"
    assert row["platform_overrides"]["youtube"]["category_id"] == "28"
    assert row["platform_overrides"]["youtube"]["is_short"] is True
    assert "#Shorts" not in row["platform_overrides"]["youtube"]["title"]
    assert "#Shorts" not in row["base_metadata"]["title"]
    assert "Llama 3.3 70B" in row["base_metadata"]["title"] or "Llama 3.3 70B" in row["platform_overrides"]["youtube"]["title"]
    assert "Gemini 3.5 Flash" in row["base_metadata"]["caption"]
    assert "Llama 3.3 70B" in row["base_metadata"]["caption"]
    assert row["base_metadata"]["caption"].strip().splitlines()[-1].startswith("#")
    assert row["platform_overrides"]["youtube"]["scheduled_time"] == "2026-09-06T22:00:00Z"
    assert validate_queue_ready(row) == []
    assert row["posting_status"] == {
        "youtube": "pending",
        "instagram": "pending",
        "tiktok": "pending",
        "facebook": "pending",
        "x": "pending",
        "pinterest": "pending",
        "kwai": "pending",
    }
    assert row["final_caption"] == row["base_metadata"]["caption"]
    assert YOUTUBE_DESCRIPTION_CTA in row["base_metadata"]["caption"]
    assert "high-intent search" not in row["base_metadata"]["caption"].lower()
    assert "hidden mysteries" not in row["base_metadata"]["caption"].lower()
    assert cta_description_line("sess1") in row["platform_overrides"]["x"]["caption"]


def test_rewrite_row_replaces_legacy_mystery_cta() -> None:
    seed = "rewrite-seed"
    row = {
        "session_id": seed,
        "final_caption": "Body\n\nFollow Aiwake for more hidden mysteries.\n\n#aiwake",
        "base_metadata": {
            "caption": "Follow Ancient Knowledge for more hidden mysteries.",
        },
        "platform_overrides": {
            "youtube": {"caption": "Stay. Follow Aiwake for more hidden mysteries"},
            "x": {"caption": "Hook Follow Aiwake for more hidden mysteries. #ai"},
        },
    }
    assert rewrite_row_legacy_cta(row) is True
    fun = cta_description_line(seed)
    assert row["final_caption"].count(fun) == 1
    assert "hidden mysteries" not in json.dumps(row).lower()
    assert rewrite_row_legacy_cta(row) is False


def test_rewrite_library_legacy_ctas(tmp_path: Path) -> None:
    library = tmp_path / "content_library.json"
    library.write_text(
        json.dumps([
            {
                "session_id": "lib1",
                "final_caption": "Follow Aiwake for more hidden mysteries.",
                "base_metadata": {"caption": "Follow Aiwake for more hidden mysteries."},
                "platform_overrides": {
                    "youtube": {"caption": "Follow Aiwake for more hidden mysteries."},
                },
            }
        ]),
        encoding="utf-8",
    )
    changed, rows = rewrite_library_legacy_ctas(library)
    assert changed == 1
    assert "hidden mysteries" not in rows[0]["final_caption"].lower()
    saved = json.loads(library.read_text(encoding="utf-8"))
    assert saved[0]["final_caption"] == cta_description_line("lib1")


def test_upsert_preserves_live_posting_status(tmp_path: Path) -> None:
    library = tmp_path / "content_library.json"
    video = tmp_path / "aiwake_debate_sess1.mp4"
    transcript = tmp_path / "sess1.json"
    doc = TranscriptDoc(
        session_id="sess1",
        topic="Grief",
        utterances=[{"role": "orchestrator", "text": "Is grief a slow update?"}],
        path=transcript,
    )
    first = build_record(video, doc, scheduled_time="2026-09-06T22:00:00Z")
    upsert_distribution_row(library, first)
    rows = load_distribution_library(library)
    rows[0]["posting_status"]["youtube"] = "posted"
    rows[0]["platform_overrides"]["youtube"]["video_id"] = "live123"
    library.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    second = build_record(video, doc, scheduled_time="2026-09-20T22:00:00Z")
    upsert_distribution_row(library, second)
    saved = load_distribution_library(library)
    assert len(saved) == 1
    assert saved[0]["posting_status"]["youtube"] == "posted"
    assert saved[0]["posting_status"]["tiktok"] == "pending"
    assert saved[0]["platform_overrides"]["youtube"]["scheduled_time"] == "2026-09-06T22:00:00Z"
    assert saved[0]["platform_overrides"]["youtube"]["video_id"] == "live123"


def test_merge_posting_status_keeps_non_pending() -> None:
    merged = merge_posting_status(
        {"youtube": "posted", "x": "failed"},
        {"youtube": "pending", "x": "pending", "tiktok": "pending"},
    )
    assert merged["youtube"] == "posted"
    assert merged["x"] == "failed"
    assert merged["tiktok"] == "pending"
