# -*- coding: utf-8 -*-
"""Strip prompt-leakage / SEO stuffing from Aiwake catalog copy.

Public YouTube descriptions are rebuilt as:

    Hook + unscripted dialogue + dialectics CTA + clean hashtags

    python -m channels_config.aiwake.tools.library_sanitize
    python -m channels_config.aiwake.tools.library_sanitize --dry-run
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover — loose-script invocation
    _FACTORY = Path(__file__).resolve().parents[3]
    if str(_FACTORY) not in sys.path:
        sys.path.insert(0, str(_FACTORY))

import json

from modules.asset_library import library_path as asset_library_path
from modules.distribution_contract import (
    X_CAPTION_MAX_CHARS,
    clip_text,
    content_library_path,
    load_distribution_library,
    save_distribution_library,
    strip_shorts_title,
)
from modules.durable_store import restore_channel_state
from utils.pipeline_paths import page_outputs_dir

try:
    from channels_config.aiwake.settings import YOUTUBE_DESCRIPTION_CTA
except ImportError:  # pragma: no cover
    from settings import YOUTUBE_DESCRIPTION_CTA  # type: ignore[no-redef]

_LOG = logging.getLogger("aiwake.sanitize")

CHANNEL_ID = "aiwake"

_LEAK_LINE_RE = re.compile(
    r"^(?:This Short is built for high-intent search\b.*"
    r"|The exchange treats model weights, inference, and who owns the stack\b.*)$",
    re.IGNORECASE,
)
_LEAK_INLINE_RE = re.compile(
    r"This Short is built for high-intent search around[^.]*\.[^.]*not a product demo\.?",
    re.IGNORECASE,
)
_NOISY_CTA_RE = re.compile(
    r"Follow (?:Ancient Knowledge)\b[^\n]*"
    r"|Follow Aiwake(?:\s+for more hidden mysteries\.?|\.[^\n]*)",
    re.IGNORECASE,
)
_LEADING_ELLIPSIS_RE = re.compile(r"^\.{2,}\s*")
_HASHTAG_LINE_RE = re.compile(r"^#\w")
_INVALID_PATH_SEGMENTS = frozenset({
    "pytest",
    "appdata",
    "temp",
    "tests",
    "reproved",
    "needs_metadata",
    "archive",
    "__pycache__",
    "posted_facebook",
})
_INVALID_PATH_FRAGMENTS = (
    "pytest-of-",
    "pytest_",
    "/pytest/",
    "\\pytest\\",
    ".pytest_cache",
)

_DESCRIPTION_FIELDS: tuple[tuple[str, ...], ...] = (
    ("final_caption",),
    ("humanized_caption",),
    ("facebook_caption",),
    ("base_metadata", "caption"),
    ("platform_overrides", "youtube", "caption"),
)
_X_CAPTION_KEYS = ("platform_overrides", "x", "caption")
_TITLE_FIELDS: tuple[tuple[str, ...], ...] = (
    ("base_metadata", "title"),
    ("platform_overrides", "youtube", "title"),
)


def has_prompt_leakage(text: str) -> bool:
    blob = text or ""
    if _LEAK_LINE_RE.search(blob) or _LEAK_INLINE_RE.search(blob):
        return True
    lowered = blob.lower()
    return "high-intent search around" in lowered or "not a product demo" in lowered


def sanitize_title(title: str) -> str:
    cleaned = strip_shorts_title(_LEADING_ELLIPSIS_RE.sub("", str(title or "").strip()))
    return " ".join(cleaned.split()).strip()


def _is_hashtag_block(paragraph: str) -> bool:
    tokens = paragraph.split()
    return bool(tokens) and all(token.startswith("#") for token in tokens)


def _is_leak_paragraph(paragraph: str) -> bool:
    if has_prompt_leakage(paragraph):
        return True
    return all(_LEAK_LINE_RE.match(line.strip()) for line in paragraph.splitlines() if line.strip())


def _is_noisy_cta(paragraph: str) -> bool:
    text = " ".join(paragraph.split()).strip()
    if not text or text == YOUTUBE_DESCRIPTION_CTA:
        return False
    return bool(_NOISY_CTA_RE.fullmatch(text) or _NOISY_CTA_RE.search(text))


def sanitize_youtube_description(text: str) -> str:
    """Hook + dialogue + dialectics CTA + hashtags. Leak / stuffing paragraphs drop."""
    raw = str(text or "").replace("\r\n", "\n").strip()
    if not raw:
        return YOUTUBE_DESCRIPTION_CTA
    raw = _LEAK_INLINE_RE.sub("", raw)
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", raw) if block.strip()]
    kept: list[str] = []
    hashtags = ""
    for block in paragraphs:
        compact = " ".join(block.split())
        if _is_leak_paragraph(compact) or _is_leak_paragraph(block):
            continue
        if _is_hashtag_block(compact) or _HASHTAG_LINE_RE.match(compact):
            hashtags = compact
            continue
        if _is_noisy_cta(compact):
            continue
        if compact == YOUTUBE_DESCRIPTION_CTA:
            continue
        kept.append(block.strip())
    parts = [part for part in kept if part]
    parts.append(YOUTUBE_DESCRIPTION_CTA)
    if hashtags:
        parts.append(hashtags)
    return "\n\n".join(parts)


def sanitize_x_caption(text: str) -> str:
    """Drop leak sentences; keep a short X body and clip to 280."""
    raw = _LEAK_INLINE_RE.sub("", str(text or ""))
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or _LEAK_LINE_RE.match(stripped) or has_prompt_leakage(stripped):
            continue
        lines.append(stripped)
    cleaned = " ".join(lines).strip()
    cleaned = _NOISY_CTA_RE.sub("", cleaned)
    cleaned = " ".join(cleaned.split())
    return clip_text(cleaned, X_CAPTION_MAX_CHARS)


def _get_nested(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    cursor: Any = row
    for key in keys:
        if not isinstance(cursor, dict):
            return ""
        cursor = cursor.get(key)
    return str(cursor or "")


def _set_nested(row: dict[str, Any], keys: tuple[str, ...], value: str) -> None:
    cursor: Any = row
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def sanitize_library_row(row: dict[str, Any]) -> bool:
    """Rewrite one catalog row. True if anything changed."""
    changed = False
    for keys in _DESCRIPTION_FIELDS:
        current = _get_nested(row, keys)
        if not current:
            continue
        rewritten = sanitize_youtube_description(current)
        if rewritten != current:
            _set_nested(row, keys, rewritten)
            changed = True
    current_x = _get_nested(row, _X_CAPTION_KEYS)
    if current_x:
        rewritten_x = sanitize_x_caption(current_x)
        if rewritten_x != current_x:
            _set_nested(row, _X_CAPTION_KEYS, rewritten_x)
            changed = True
    for keys in _TITLE_FIELDS:
        current = _get_nested(row, keys)
        if not current:
            continue
        rewritten = sanitize_title(current)
        if rewritten != current:
            _set_nested(row, keys, rewritten)
            changed = True
    return changed


def production_output_roots() -> list[Path]:
    roots: list[Path] = []
    factory = page_outputs_dir(CHANNEL_ID)
    if factory.is_dir():
        roots.append(factory.resolve())
    local = Path(__file__).resolve().parents[1] / "outputs"
    if local.is_dir():
        resolved = local.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def is_ephemeral_test_path(path: str | Path) -> bool:
    """True for pytest / AppData / Temp / Tests / Reproved paths."""
    raw = str(path or "").strip()
    if not raw:
        return True
    lowered = raw.replace("\\", "/").lower()
    if any(fragment in lowered for fragment in _INVALID_PATH_FRAGMENTS):
        return True
    for part in Path(raw).parts:
        token = part.lower()
        if token in _INVALID_PATH_SEGMENTS or token.startswith("pytest-"):
            return True
    return False


def resolve_root_production_mp4(
    path: str | Path,
    *,
    roots: list[Path] | None = None,
) -> Path | None:
    """Return a verified root or animation-clips MP4, else None."""
    raw = str(path or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.suffix.lower() != ".mp4":
        return None
    bases = [
        Path(root).resolve()
        for root in (roots if roots is not None else production_output_roots())
    ]
    checks: list[Path] = []
    if candidate.is_absolute():
        checks.append(candidate)
    else:
        from utils.pipeline_paths import outputs_root

        checks.append(outputs_root() / candidate)
        for root in bases:
            checks.append(root / candidate.name)
            checks.append(root / candidate)
    seen: set[Path] = set()
    for item in checks:
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file() or resolved.suffix.lower() != ".mp4":
            continue
        for base in bases:
            if resolved.parent == base:
                return resolved
            if (
                resolved.parent == base / "animation_clips"
                and resolved.name.lower().startswith("aiwake_")
            ):
                return resolved
    return None


def _row_media_paths(row: dict[str, Any]) -> list[str]:
    paths = [
        str(row.get("video_path") or "").strip(),
        str(row.get("local_path") or "").strip(),
    ]
    youtube = row.get("platform_overrides")
    if isinstance(youtube, dict):
        block = youtube.get("youtube")
        if isinstance(block, dict):
            paths.append(str(block.get("video_path") or "").strip())
    return [item for item in paths if item]


def is_production_library_row(
    row: dict[str, Any],
    *,
    roots: list[Path] | None = None,
) -> bool:
    media = _row_media_paths(row)
    if not media:
        return False
    return any(resolve_root_production_mp4(path, roots=roots) for path in media)


def prune_content_library(
    library_path: Path | None = None,
    *,
    dry_run: bool = False,
    roots: list[Path] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    path = library_path or content_library_path(CHANNEL_ID)
    rows = load_distribution_library(path)
    kept = [row for row in rows if is_production_library_row(row, roots=roots)]
    removed = len(rows) - len(kept)
    if removed and not dry_run:
        save_distribution_library(path, kept)
        _LOG.info("Pruned %s content_library row(s) → %s", removed, path)
    return removed, kept


def delete_asset_library() -> list[Path]:
    """content_library.json is the only canonical registry — drop asset_library."""
    from utils.pipeline_paths import channel_store_dir, page_outputs_dir

    candidates = [
        asset_library_path(CHANNEL_ID),
        channel_store_dir(CHANNEL_ID) / "asset_library.json",
        page_outputs_dir(CHANNEL_ID) / "asset_library.json",
        page_outputs_dir(CHANNEL_ID) / "store" / "asset_library.json",
    ]
    removed: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            path.unlink()
            removed.append(path)
            _LOG.info("Deleted asset_library %s", path)
    return removed


def prune_asset_library(
    *,
    dry_run: bool = False,
    roots: list[Path] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    _ = roots
    if dry_run:
        path = asset_library_path(CHANNEL_ID)
        if not path.is_file():
            return 0, []
        return 1, []
    removed = delete_asset_library()
    return len(removed), []


def prune_planner_json(
    path: Path,
    *,
    dry_run: bool = False,
    roots: list[Path] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    if not path.is_file():
        return 0, []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, []
    rows = raw if isinstance(raw, list) else []
    kept = [row for row in rows if isinstance(row, dict) and is_production_library_row(row, roots=roots)]
    removed = len(rows) - len(kept)
    if removed and not dry_run:
        path.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _LOG.info("Pruned %s post_planner row(s) → %s", removed, path)
    return removed, kept


def prune_distribution_catalogs(
    *,
    dry_run: bool = False,
    planner_json: Path | None = None,
) -> dict[str, int]:
    restore_channel_state(CHANNEL_ID)
    roots = production_output_roots()
    content_removed, _ = prune_content_library(dry_run=dry_run, roots=roots)
    asset_removed, _ = prune_asset_library(dry_run=dry_run, roots=roots)
    planner_removed = 0
    if planner_json is not None:
        planner_removed, _ = prune_planner_json(planner_json, dry_run=dry_run, roots=roots)
    return {
        "content_library": content_removed,
        "asset_library": asset_removed,
        "post_planner": planner_removed,
    }


def sanitize_content_library(
    library_path: Path | None = None,
    *,
    dry_run: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    path = library_path or content_library_path(CHANNEL_ID)
    rows = load_distribution_library(path)
    changed = 0
    for row in rows:
        if sanitize_library_row(row):
            changed += 1
    if changed and not dry_run:
        save_distribution_library(path, rows)
        _LOG.info("Sanitized %s row(s) → %s", changed, path)
    return changed, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiwake-sanitize-library",
        description="Remove prompt-leakage and SEO stuffing from content_library.json.",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--skip-prune", action="store_true")
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
    if args.library is None:
        restore_channel_state(CHANNEL_ID)
    if not args.skip_prune:
        pruned = prune_distribution_catalogs(dry_run=bool(args.dry_run))
        print(
            "Aiwake library prune  "
            f"content={pruned['content_library']}  "
            f"assets_deleted={pruned['asset_library']}  "
            f"planner={pruned['post_planner']}"
        )
    changed, rows = sanitize_content_library(args.library, dry_run=bool(args.dry_run))
    mode = "DRY-RUN" if args.dry_run else "write"
    print(f"Aiwake library sanitize  mode={mode}  rows={len(rows)}  changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
