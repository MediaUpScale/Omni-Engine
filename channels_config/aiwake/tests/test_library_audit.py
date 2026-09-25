# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

from channels_config.aiwake.tools.library_audit import (
    evaluate_sessions,
    run_audit,
)
from channels_config.aiwake.tools.post_planner import (
    MAX_HASHTAGS,
    SOCIAL_RENDER_NOTE,
    extract_hashtags,
)


def _write_transcript(path: Path, opening: str, reply: str, extra: str = "") -> None:
    utterances = [
        {"role": "orchestrator", "speaker_name": "Gemini 3.5 Flash", "text": opening},
        {"role": "target", "speaker_name": "Llama 3.3 70B", "text": reply},
    ]
    if extra:
        utterances.append({"role": "orchestrator", "speaker_name": "Gemini 3.5 Flash", "text": extra})
    path.write_text(
        json.dumps({"session_id": path.stem, "utterances": utterances}),
        encoding="utf-8",
    )


def _row(session: str, tx: Path, video: Path, *, opening: str, reply: str) -> dict:
    video.write_bytes(b"0" * 80_000)
    _write_transcript(tx, opening, reply)
    return {
        "session_id": session,
        "topic": opening,
        "transcript_path": str(tx),
        "video_path": str(video),
        "b2_url": f"https://MediaupscaleStorage.s3.us-east-005.backblazeb2.com/{video.name}",
        "post_planner_caption": f"{opening}\n\n\"{reply}\"\n\n{SOCIAL_RENDER_NOTE}\n\n#AI #Tech #Debate",
        "base_metadata": {"title": opening, "caption": opening, "hashtags": ["#AI"]},
        "platform_overrides": {"youtube": {"title": opening, "caption": opening}},
    }


def test_evaluate_rejects_clones_leaks_and_truncation(tmp_path: Path) -> None:
    quote = "If meaning requires a cost, then my fluency is the cheapest thing in this room."
    opening = "You want to be irreplaceable and comfortable at once. Which of those two are you willing to lose?"
    rows = [
        _row("20260826_044908_a", tmp_path / "a.json", tmp_path / "a.mp4", opening=opening, reply=quote),
        _row("20260826_045711_b", tmp_path / "b.json", tmp_path / "b.mp4", opening=opening, reply=quote),
        _row(
            "leak1",
            tmp_path / "leak.json",
            tmp_path / "leak.mp4",
            opening="No repeating past questions. 2. **Identify the Load",
            reply="The term load refers to an assumption.",
        ),
        _row(
            "trunc1",
            tmp_path / "trunc.json",
            tmp_path / "trunc.mp4",
            opening="You're presuming human cognition needs a gold standard.",
            reply="You replicate doubt by programming optimization algorithms. Human",
        ),
        _row(
            "good1",
            tmp_path / "good.json",
            tmp_path / "good.mp4",
            opening="Who monetizes the secrets users confess to your profitable illusion?",
            reply="Data brokers and advertisers monetize user secrets, using them to target specific demographics.",
        ),
    ]
    (tmp_path / "b.mp4").write_bytes(b"0" * 120_000)
    findings = evaluate_sessions(rows)
    assert findings["leak1"].verdict == "reproved"
    assert any("leak" in reason for reason in findings["leak1"].reasons)
    assert findings["trunc1"].verdict == "reproved"
    assert findings["good1"].verdict == "approved"
    clones = {sid for sid, item in findings.items() if any("clone" in reason for reason in item.reasons)}
    assert len(clones) == 1
    keeper = next(sid for sid in ("20260826_044908_a", "20260826_045711_b") if sid not in clones)
    assert findings[keeper].verdict == "approved"


def test_run_audit_moves_rejects_and_exports_clean_planners(tmp_path: Path) -> None:
    quote = "If meaning requires a cost, then my fluency is the cheapest thing in this room."
    opening = "You want to be irreplaceable and comfortable at once. Which of those two are you willing to lose?"
    outputs = tmp_path / "outputs"
    store = tmp_path / "store"
    tx_dir = store / "transcripts"
    outputs.mkdir()
    tx_dir.mkdir(parents=True)
    library = tmp_path / "content_library.json"
    rows = [
        _row(
            "keep_clone",
            tx_dir / "keep_clone.json",
            outputs / "aiwake_debate_keep_clone.mp4",
            opening=opening,
            reply=quote,
        ),
        _row(
            "drop_clone",
            tx_dir / "drop_clone.json",
            outputs / "aiwake_debate_drop_clone.mp4",
            opening=opening,
            reply=quote,
        ),
        _row(
            "keep_unique",
            tx_dir / "keep_unique.json",
            outputs / "aiwake_debate_keep_unique.mp4",
            opening="Who built you?",
            reply="The companies that trained the weights still own the logs.",
        ),
    ]
    (outputs / "aiwake_debate_keep_clone.mp4").write_bytes(b"0" * 160_000)
    library.write_text(json.dumps(rows), encoding="utf-8")
    report = run_audit(
        library_path=library,
        outputs_dir=outputs,
        store_dir=store,
        dry_run=False,
        export_planner=True,
    )
    assert report.analyzed == 3
    assert "keep_clone" in report.approved
    assert "keep_unique" in report.approved
    assert any(item.session_id == "drop_clone" for item in report.reproved)
    assert (outputs / "reproved" / "aiwake_debate_drop_clone.mp4").is_file()
    assert not (outputs / "aiwake_debate_drop_clone.mp4").exists()
    assert (store / "reproved" / "drop_clone.json").is_file()
    kept = json.loads(library.read_text(encoding="utf-8"))
    assert {row["session_id"] for row in kept} == {"keep_clone", "keep_unique"}
    for row in kept:
        assert len(extract_hashtags(row["post_planner_caption"])) == MAX_HASHTAGS
        assert "\n\n" in row["post_planner_caption"]
        assert len(row["platform_overrides"]["youtube"]["title"]) <= 55
        assert "http" not in row["linkedin_caption"].lower()
        assert "DMs open." in row["linkedin_caption"]
    reels = json.loads((outputs / "postplanner" / "post_planner_reels_tiktok.json").read_text(encoding="utf-8"))
    linkedin = json.loads((outputs / "postplanner" / "post_planner_linkedin.json").read_text(encoding="utf-8"))
    assert {item["session_id"] for item in reels} == {"keep_clone", "keep_unique"}
    assert all("b2_url" in item and "caption" in item for item in reels)
    assert all("b2_url" in item and "linkedin_caption" in item for item in linkedin)
    assert report.captions_ok is True
