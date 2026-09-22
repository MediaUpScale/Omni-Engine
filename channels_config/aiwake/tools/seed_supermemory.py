# -*- coding: utf-8 -*-
"""One-time ingest of approved Aiwake sessions into local Supermemory.

    python -m channels_config.aiwake.tools.seed_supermemory
    python -m channels_config.aiwake.tools.seed_supermemory --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):  # pragma: no cover — loose-script invocation
    _FACTORY = Path(__file__).resolve().parents[3]
    if str(_FACTORY) not in sys.path:
        sys.path.insert(0, str(_FACTORY))

from channels_config.aiwake.tools.post_planner import (
    CHANNEL_ID,
    REELS_JSON_NAME,
    extract_hook,
    load_dialogue,
)
from channels_config.aiwake.tools.supermemory_bridge import (
    GOLDEN_RULE,
    HISTORY_CONTAINER,
    RULES_CONTAINER,
    SupermemoryBridge,
    session_memory_text,
)
from modules.distribution_contract import content_library_path, load_distribution_library
from modules.durable_store import read_state_json, restore_channel_state

_LOG = logging.getLogger("aiwake.seed_supermemory")


def _session_id(row: dict[str, Any]) -> str:
    return str(row.get("session_id") or "").strip()


def _quote_from_caption(caption: str) -> str:
    for block in str(caption or "").split("\n\n"):
        text = block.strip().strip('"“”')
        if text.startswith('"') or (block.strip().startswith('"') or block.strip().startswith("“")):
            return text
        if block.strip().startswith('"') or block.strip().startswith("“"):
            return text
    return ""


def footprint_from_row(row: dict[str, Any]) -> dict[str, str]:
    """Topic / opening hook / core quote for one approved session."""
    dialogue = load_dialogue(row)
    topic = str(row.get("topic") or "").strip() or str(dialogue.get("opening") or "").strip()
    hook = str(dialogue.get("opening") or extract_hook(row) or "").strip()
    quote = str(dialogue.get("quote") or "").strip()
    if not quote:
        quote = _quote_from_caption(
            str(row.get("caption") or row.get("post_planner_caption") or row.get("humanized_caption") or "")
        )
    if not topic:
        topic = hook
    return {
        "session_id": _session_id(row),
        "topic": topic,
        "hook": hook,
        "quote": quote,
    }


def _unique_footprints(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = footprint_from_row(row)
        key = item["session_id"] or item["topic"]
        if not key or key in seen:
            continue
        if not item["topic"] and not item["hook"]:
            continue
        seen.add(key)
        out.append(item)
    return out


def load_approved_sessions(
    *,
    library_path: Path | None = None,
) -> list[dict[str, str]]:
    restore_channel_state(CHANNEL_ID)
    path = Path(library_path) if library_path is not None else content_library_path(CHANNEL_ID)
    rows = load_distribution_library(path)
    footprints = _unique_footprints(rows)
    if footprints:
        return footprints
    planner = read_state_json(CHANNEL_ID, REELS_JSON_NAME, default=[])
    if not isinstance(planner, list):
        planner = []
    return _unique_footprints(item for item in planner if isinstance(item, dict))


def existing_history_ids(bridge: SupermemoryBridge) -> set[str]:
    ids = {str(row.get("session_id") or "") for row in bridge.ledger_sessions() if row.get("session_id")}
    client = bridge._client_or_none()  # noqa: SLF001 — seed-only inventory
    if client is None:
        return ids
    try:
        page = 1
        while page <= 20:
            listed = client.documents.list(
                container_tags=[HISTORY_CONTAINER],
                limit=200,
                page=page,
            )
            memories = getattr(listed, "memories", None) or []
            if not memories:
                break
            for item in memories:
                meta = getattr(item, "metadata", None) or {}
                if isinstance(meta, dict) and meta.get("session_id"):
                    ids.add(str(meta["session_id"]))
            pagination = getattr(listed, "pagination", None)
            total_pages = int(getattr(pagination, "total_pages", 1) or 1) if pagination else 1
            if page >= total_pages:
                break
            page += 1
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("could not list existing Supermemory documents (%s)", exc)
    return ids


def seed_supermemory(
    *,
    library_path: Path | None = None,
    dry_run: bool = False,
    force_offline: bool = False,
) -> dict[str, int]:
    sessions = load_approved_sessions(library_path=library_path)
    bridge = SupermemoryBridge(force_offline=force_offline)
    already = existing_history_ids(bridge)
    stats = {
        "approved_sessions": len(sessions),
        "history_seeded": 0,
        "history_skipped": 0,
        "history_failed": 0,
        "rule_seeded": 0,
        "remote_active": int(bridge.is_active),
    }

    for item in sessions:
        sid = item["session_id"]
        if dry_run:
            stats["history_seeded"] += 1
            continue
        already_remote = bool(sid and sid in already)
        wrote_ledger = bridge.remember_approved_session(
            sid,
            item["topic"],
            item["hook"],
            item["quote"],
            remote=bridge.is_active and not already_remote,
        )
        if already_remote:
            stats["history_skipped"] += 1
            continue
        if wrote_ledger:
            stats["history_seeded"] += 1
            if sid:
                already.add(sid)
        else:
            stats["history_failed"] += 1

    if not dry_run and bridge.remember_system_rule(GOLDEN_RULE):
        stats["rule_seeded"] = 1
    elif dry_run:
        stats["rule_seeded"] = 1

    return stats


def render_report(stats: dict[str, int]) -> str:
    total = int(stats["history_seeded"]) + int(stats["rule_seeded"])
    host = "localhost:6767" if stats.get("remote_active") else "local JSON ledger"
    return "\n".join(
        [
            "[OmniEngine] Supermemory seed complete",
            f"  approved sessions read : {stats['approved_sessions']}",
            f"  history ingested       : {stats['history_seeded']}",
            f"  history already present: {stats['history_skipped']}",
            f"  history failed         : {stats['history_failed']}",
            f"  system rules ingested  : {stats['rule_seeded']}",
            f"  container              : {HISTORY_CONTAINER} + {RULES_CONTAINER}",
            f"  target                 : {host}",
            f"Total memories successfully seeded: {total}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed local Supermemory from approved Aiwake sessions.")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Write history_seeds.json only")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    stats = seed_supermemory(
        library_path=args.library,
        dry_run=bool(args.dry_run),
        force_offline=bool(args.offline),
    )
    print(render_report(stats))
    if stats["approved_sessions"] <= 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
