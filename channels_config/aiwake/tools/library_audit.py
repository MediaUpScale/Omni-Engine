# -*- coding: utf-8 -*-
"""Triage Aiwake sessions: keep unique top-tier rows, move rejects to reproved/.

    python -m channels_config.aiwake.tools.library_audit
    python -m channels_config.aiwake.tools.library_audit --dry-run
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):  # pragma: no cover — loose-script invocation
    _FACTORY = Path(__file__).resolve().parents[3]
    if str(_FACTORY) not in sys.path:
        sys.path.insert(0, str(_FACTORY))

from channels_config.aiwake.contracts import (
    _LEADING_LEAK,
    _LEAK_SENTENCE,
    has_word_index_leak,
)
from channels_config.aiwake.tools.library_sanitize import has_prompt_leakage
from channels_config.aiwake.tools.post_planner import (
    CHANNEL_ID,
    MAX_HASHTAGS,
    extract_hashtags,
    load_dialogue,
    run_planner,
    spoken_has_consciousness,
    stamp_library_captions,
    transcript_has_privacy,
    verify_engine_separation,
)
from modules.distribution_contract import (
    content_library_path,
    load_distribution_library,
    save_distribution_library,
)
from modules.durable_store import restore_channel_state
from utils.pipeline_paths import channel_store_dir, page_outputs_dir

_LOG = logging.getLogger("aiwake.audit")

_LEAK_HINTS = (
    "no repeating past questions",
    "identify the load",
    "load-bearing contradiction instruction",
    "hard output contract",
    "output only the raw",
    "never output rules",
    "internal context",
    "complexity filter",
    "trivial metrics are banned",
    "high-intent search",
    "not a product demo",
)
_ABRUPT_LAST = frozenset(
    {"human", "the", "a", "an", "and", "or", "of", "to", "for", "with", "co"}
)
_INJECTED_THEME_RE = re.compile(
    r"whether consciousness(?:\s+is)?\s+an illusion|"
    r"corners \w+ on whether consciousness",
    re.IGNORECASE,
)


@dataclass
class Finding:
    session_id: str
    verdict: str = "approved"
    reasons: list[str] = field(default_factory=list)
    keeper: str = ""

    def reject(self, reason: str) -> None:
        if reason and reason not in self.reasons:
            self.reasons.append(reason)
        self.verdict = "reproved"


@dataclass
class AuditReport:
    analyzed: int = 0
    approved: list[str] = field(default_factory=list)
    reproved: list[Finding] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)
    captions_ok: bool = True
    caption_errors: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "Aiwake library audit",
            f"  analyzed: {self.analyzed}",
            f"  approved: {len(self.approved)}",
            f"  reproved: {len(self.reproved)}",
        ]
        for item in self.reproved:
            why = "; ".join(item.reasons) or "unspecified"
            extra = f" (kept {item.keeper})" if item.keeper else ""
            lines.append(f"    - {item.session_id}: {why}{extra}")
        lines.append(
            "  caption contract: "
            + ("OK (<=3 hashtags, native English, paragraph breaks)" if self.captions_ok else "FAILED")
        )
        for err in self.caption_errors[:12]:
            lines.append(f"    - {err}")
        return "\n".join(lines)


def session_id_of(row: dict[str, Any]) -> str:
    return str(row.get("session_id") or "").strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _copy_blob(row: dict[str, Any], dialogue: dict[str, Any]) -> str:
    parts = [
        str(row.get("topic") or ""),
        str(row.get("post_planner_caption") or ""),
        str(row.get("humanized_caption") or ""),
        str(row.get("linkedin_caption") or ""),
        str(row.get("final_caption") or ""),
        dialogue.get("opening") or "",
        dialogue.get("quote") or "",
        dialogue.get("corpus") or "",
    ]
    base = row.get("base_metadata") if isinstance(row.get("base_metadata"), dict) else {}
    youtube = ((row.get("platform_overrides") or {}).get("youtube") or {}) if isinstance(row.get("platform_overrides"), dict) else {}
    parts.extend(
        [
            str(base.get("title") or ""),
            str(base.get("caption") or ""),
            str(youtube.get("title") or ""),
            str(youtube.get("caption") or ""),
        ]
    )
    return "\n".join(parts)


def has_system_artifact(text: str) -> bool:
    blob = str(text or "")
    if not blob.strip():
        return False
    if has_prompt_leakage(blob) or has_word_index_leak(blob):
        return True
    if _LEADING_LEAK.search(blob) or _LEAK_SENTENCE.search(blob):
        return True
    lowered = blob.lower()
    return any(hint in lowered for hint in _LEAK_HINTS)


def is_truncated_line(text: str) -> bool:
    raw = str(text or "").strip().strip('"“”')
    if not raw:
        return False
    if raw.endswith(("...", "…", "?", "!", ".", '"', "”", ")")):
        return False
    last = raw.split()[-1].strip(",;:")
    if last.lower() in _ABRUPT_LAST:
        return True
    if len(last) <= 3 and last.isalpha() and len(raw.split()) >= 4:
        return True
    if last[:1].isupper() and last.lower() not in {"i"} and not raw.endswith("?"):
        return True
    return False


def has_context_drift(row: dict[str, Any], dialogue: dict[str, Any]) -> bool:
    caption = str(row.get("post_planner_caption") or row.get("humanized_caption") or "")
    corpus = str(dialogue.get("corpus") or "")
    if not caption or not corpus:
        return False
    if _INJECTED_THEME_RE.search(caption) and "whether consciousness an illusion" not in corpus.lower():
        return True
    leftover = caption
    for chunk in (dialogue.get("opening"), dialogue.get("quote")):
        if chunk:
            leftover = leftover.replace(str(chunk), "")
    if (
        "consciousness" in leftover.lower()
        and not spoken_has_consciousness(corpus)
        and transcript_has_privacy(corpus)
    ):
        return True
    return False


def dialogue_keys(dialogue: dict[str, Any]) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = []
    utts = [_norm(item.get("text") or "") for item in dialogue.get("utterances") or []]
    utts = [item for item in utts if item]
    if len(utts) >= 2:
        keys.append(("pair", utts[0], utts[1]))
    corpus = _norm(dialogue.get("corpus") or "")
    if len(corpus) >= 40:
        keys.append(("corpus", corpus))
    return keys


def quality_score(row: dict[str, Any], dialogue: dict[str, Any]) -> tuple:
    video = Path(str(row.get("video_path") or ""))
    size = 0
    if video.is_file():
        try:
            size = int(video.stat().st_size)
        except OSError:
            size = 0
    complete = 1
    if is_truncated_line(str(dialogue.get("quote") or "")) or is_truncated_line(str(dialogue.get("opening") or "")):
        complete = 0
    clean = 0 if has_system_artifact(_copy_blob(row, dialogue)) else 1
    return (
        clean,
        complete,
        len(dialogue.get("utterances") or []),
        len(str(dialogue.get("corpus") or "")),
        size,
        session_id_of(row),
    )


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a

    def groups(self) -> list[list[str]]:
        buckets: dict[str, list[str]] = {}
        for item in self.parent:
            buckets.setdefault(self.find(item), []).append(item)
        return list(buckets.values())


def evaluate_sessions(rows: list[dict[str, Any]]) -> dict[str, Finding]:
    findings: dict[str, Finding] = {}
    dialogues: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = session_id_of(row) or f"row-{len(findings)}"
        dialogue = load_dialogue(row)
        dialogues[sid] = dialogue
        finding = Finding(session_id=sid)
        blob = _copy_blob(row, dialogue)
        if has_system_artifact(blob):
            finding.reject("prompt leakage / system artifact")
        if is_truncated_line(str(dialogue.get("quote") or "")) or is_truncated_line(
            str(dialogue.get("opening") or "")
        ):
            finding.reject("truncated quote or broken output")
        if has_context_drift(row, dialogue):
            finding.reject("context drift / hallucinated theme")
        findings[sid] = finding

    uf = _UnionFind(findings)
    buckets: dict[tuple[str, ...], list[str]] = {}
    for sid, dialogue in dialogues.items():
        for key in dialogue_keys(dialogue):
            buckets.setdefault(key, []).append(sid)
    for members in buckets.values():
        unique = list(dict.fromkeys(members))
        if len(unique) < 2:
            continue
        root = unique[0]
        for other in unique[1:]:
            uf.union(root, other)

    by_sid = {session_id_of(row): row for row in rows if session_id_of(row)}
    for group in uf.groups():
        if len(group) < 2:
            continue
        ranked = sorted(
            group,
            key=lambda sid: quality_score(by_sid[sid], dialogues[sid]),
            reverse=True,
        )
        keeper = ranked[0]
        for sid in ranked[1:]:
            findings[sid].reject(f"semantic clone of {keeper}")
            findings[sid].keeper = keeper
    return findings


def _reproved_dir(root: Path) -> Path:
    dest = Path(root) / "reproved"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _move_file(src: Path, dest_dir: Path) -> Path | None:
    if not src or not src.exists() or not src.is_file():
        return None
    if any(part.lower() == "reproved" for part in src.parts):
        return src
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() == src.resolve():
        return dest
    if dest.exists():
        dest = dest_dir / f"{src.stem}__audit{src.suffix}"
    shutil.move(str(src), str(dest))
    _LOG.info("moved %s → %s", src, dest)
    return dest


def session_asset_paths(row: dict[str, Any], *, outputs_dir: Path, store_dir: Path) -> list[Path]:
    sid = session_id_of(row)
    paths: list[Path] = []
    for key in ("video_path", "local_path", "transcript_path"):
        raw = str(row.get(key) or "").strip()
        if raw:
            paths.append(Path(raw))
    overrides = row.get("platform_overrides")
    if isinstance(overrides, dict):
        youtube = overrides.get("youtube")
        if isinstance(youtube, dict) and youtube.get("video_path"):
            paths.append(Path(str(youtube["video_path"])))
    if sid:
        paths.append(store_dir / "transcripts" / f"{sid}.json")
        if outputs_dir.is_dir():
            paths.extend(outputs_dir.glob(f"*{sid}*.mp4"))
            paths.extend(outputs_dir.glob(f"*{sid}*.json"))
        tx_root = store_dir / "transcripts"
        if tx_root.is_dir():
            paths.extend(tx_root.glob(f"*{sid}*.json"))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def move_reproved_assets(
    rows: list[dict[str, Any]],
    *,
    outputs_dir: Path,
    store_dir: Path,
    dry_run: bool = False,
) -> list[str]:
    media_dest = _reproved_dir(outputs_dir)
    store_dest = _reproved_dir(store_dir)
    moved: list[str] = []
    for row in rows:
        for src in session_asset_paths(row, outputs_dir=outputs_dir, store_dir=store_dir):
            dest_dir = store_dest if src.suffix.lower() == ".json" else media_dest
            if dry_run:
                if src.is_file() and not any(part.lower() == "reproved" for part in src.parts):
                    moved.append(str(src))
                continue
            dest = _move_file(src, dest_dir)
            if dest is not None:
                moved.append(str(dest))
    return moved


def _remember_approved_rows(rows: list[dict[str, Any]]) -> None:
    """Write approved debate footprints into Supermemory / the local ledger."""
    try:
        from channels_config.aiwake.tools.supermemory_bridge import remember_approved_session
    except Exception as exc:  # noqa: BLE001 — memory is optional
        _LOG.warning("supermemory bridge unavailable (%s)", exc)
        return
    for row in rows:
        try:
            dialogue = load_dialogue(row)
            remember_approved_session(
                session_id_of(row),
                str(row.get("topic") or dialogue.get("opening") or ""),
                str(dialogue.get("opening") or ""),
                str(dialogue.get("quote") or ""),
            )
        except Exception as exc:  # noqa: BLE001 — one bad row must not abort the audit
            _LOG.warning("could not remember session %s (%s)", session_id_of(row), exc)


def run_audit(
    *,
    library_path: Path | None = None,
    outputs_dir: Path | None = None,
    store_dir: Path | None = None,
    dry_run: bool = False,
    export_planner: bool = True,
) -> AuditReport:
    persist_store = library_path is None
    if persist_store:
        restore_channel_state(CHANNEL_ID)
    path = Path(library_path) if library_path is not None else content_library_path(CHANNEL_ID)
    media_root = Path(outputs_dir) if outputs_dir is not None else page_outputs_dir(CHANNEL_ID)
    store_root = Path(store_dir) if store_dir is not None else channel_store_dir(CHANNEL_ID)
    rows = load_distribution_library(path)
    findings = evaluate_sessions(rows)
    report = AuditReport(analyzed=len(rows))
    approved_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in rows:
        sid = session_id_of(row)
        finding = findings.get(sid) or Finding(session_id=sid)
        if finding.reasons:
            finding.verdict = "reproved"
            report.reproved.append(finding)
            rejected_rows.append(row)
        else:
            report.approved.append(sid)
            approved_rows.append(row)

    report.moved = move_reproved_assets(
        rejected_rows,
        outputs_dir=media_root,
        store_dir=store_root,
        dry_run=dry_run,
    )
    if not dry_run:
        stamp_library_captions(approved_rows)
        save_distribution_library(path, approved_rows)
        if persist_store:
            _remember_approved_rows(approved_rows)
        if export_planner:
            from channels_config.aiwake.tools.post_planner import (
                LINKEDIN_JSON_NAME,
                REELS_JSON_NAME,
                build_linkedin_planner_entries,
                compact_linkedin_export,
                compact_reels_export,
            )
            from modules.durable_store import write_state_json

            _dest, entries = run_planner(
                library_path=path,
                outputs_dir=media_root,
                upload_b2=False,
            )
            if persist_store:
                linkedin_entries = build_linkedin_planner_entries(entries)
                write_state_json(CHANNEL_ID, "post_planner.json", entries)
                write_state_json(CHANNEL_ID, REELS_JSON_NAME, compact_reels_export(entries))
                write_state_json(CHANNEL_ID, LINKEDIN_JSON_NAME, compact_linkedin_export(linkedin_entries))
        report.caption_errors = verify_engine_separation([], approved_rows)
        report.captions_ok = not report.caption_errors
        for row in approved_rows:
            for field in (
                str(row.get("post_planner_caption") or ""),
                str(row.get("humanized_caption") or ""),
                str(row.get("linkedin_caption") or ""),
                str(row.get("final_caption") or ""),
                str(((row.get("platform_overrides") or {}).get("youtube") or {}).get("caption") or ""),
            ):
                if field and len(extract_hashtags(field)) > MAX_HASHTAGS:
                    report.caption_errors.append(f"{session_id_of(row)}: >{MAX_HASHTAGS} hashtags")
                    report.captions_ok = False
            from channels_config.aiwake.tools.metadata_generator import TITLE_MAX_CHARS, has_chapter_timestamps

            title = str(((row.get("platform_overrides") or {}).get("youtube") or {}).get("title") or "")
            if title and len(title) > TITLE_MAX_CHARS:
                report.caption_errors.append(f"{session_id_of(row)}: title over {TITLE_MAX_CHARS} chars")
                report.captions_ok = False
            yt_caption = str(((row.get("platform_overrides") or {}).get("youtube") or {}).get("caption") or "")
            if yt_caption and has_chapter_timestamps(yt_caption):
                report.caption_errors.append(f"{session_id_of(row)}: chapter timestamps")
                report.captions_ok = False
    else:
        report.captions_ok = True
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiwake-library-audit",
        description="Triage Aiwake sessions, move rejects to reproved/, regenerate planners.",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--outputs-dir", type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    report = run_audit(
        library_path=args.library,
        outputs_dir=args.outputs_dir,
        store_dir=args.store_dir,
        dry_run=bool(args.dry_run),
        export_planner=not bool(args.skip_export),
    )
    print(report.render())
    print(f"mode={'DRY-RUN' if args.dry_run else 'write'}")
    if not report.captions_ok and not args.dry_run:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
