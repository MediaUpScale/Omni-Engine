# -*- coding: utf-8 -*-
"""
Forensic auditor for local Google / Gemini API spend.

Scans factory run journals and CostTracker JSON from yesterday + today,
aggregates by channel (anna_protocol vs ancient_knowledge), flags retries /
429s / runaway output tokens, and prints a cost table.

Usage
-----
    python -m tools.audit_google_costs
    python -m tools.audit_google_costs --peak-start "2026-09-05 21:00" --peak-end "2026-09-06 02:00"
    python -m tools.audit_google_costs --logs-dir "G:/.../outputs/logs"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

# Local pricing (keep in sync with core/google_guardrail.py — no import required).
FLASH_INPUT_USD_PER_1M = 0.075
FLASH_OUTPUT_USD_PER_1M = 0.30
IMAGE_FLASH_1K_USD = 0.005
IMAGE_1K_USD = 0.03
IMAGE_2K_USD = 0.134

_GOOGLE_NEEDLES = (
    "gemini",
    "google-generativeai",
    "google.genai",
    "vertex",
    "imagen",
    "generativelanguage.googleapis",
    "google_api",
    "google api",
    "flash-image",
    "pro-image",
    "generate_content",
    "generatecontent",
    "gemini-2.5-flash",
    "gemini-3-pro-image",
    "gemini_flash_call",
    "google_api |",
)

_CHANNEL_RE = re.compile(
    r"(?:page|channel)\s*[=:]\s*([a-z0-9_]+)",
    re.IGNORECASE,
)
_BEGIN_RE = re.compile(
    r"ENGINE RUN BEGIN\s*\|\s*page=([a-z0-9_]+)",
    re.IGNORECASE,
)
_MODEL_RE = re.compile(
    r"(?:model|head|research|image)\s*[=:`]\s*[`']?(models/)?(gemini[-a-z0-9.]+|imagen[-a-z0-9.]+)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"(?:in(?:put)?[_ ]?tok(?:ens)?|in)=(?P<inn>\d+).{0,40}?"
    r"(?:out(?:put)?[_ ]?tok(?:ens)?|out)=(?P<out>\d+)",
    re.IGNORECASE,
)
_COST_RE = re.compile(r"cost=\$?(?P<usd>\d+(?:\.\d+)?)", re.IGNORECASE)
_RETRY_RE = re.compile(
    r"(?:retry|attempt)\s*[= ]\s*(?P<n>\d+)\s*/\s*(?P<m>\d+)",
    re.IGNORECASE,
)
_TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d+)?)"
)
_GOOGLE_API_LINE = re.compile(
    r"GOOGLE_API\s*\|\s*channel=(?P<ch>\S+)\s*\|\s*src=(?P<src>\S+)\s*\|\s*"
    r"model=(?P<model>\S+)\s*\|\s*kind=(?P<kind>\S+)\s*\|\s*"
    r"calls=(?P<calls>\d+)\s*\|\s*retries=(?P<retries>\d+)\s*\|\s*"
    r"in=(?P<inn>\d+)\s*\|\s*out=(?P<out>\d+)"
    r"(?:\s*\|\s*images=(?P<images>\d+))?"
    r".*?cost=\$(?P<cost>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_FLASH_CALL_RE = re.compile(
    r"GEMINI_FLASH_CALL\s*\|\s*n=(?P<n>\d+)\s*task=(?P<task>\S+)",
    re.IGNORECASE,
)
_GEMINI_OK_RE = re.compile(
    r"Gemini OK\s*\|\s*model=(?P<model>\S+)\s*\|\s*attempt=(?P<attempt>\d+)",
    re.IGNORECASE,
)
_IMAGE_OK_RE = re.compile(
    r"Gemini image OK\s*\|\s*model=(?P<model>\S+)",
    re.IGNORECASE,
)

_ANOMALY_429 = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|RATE_LIMIT|TOO MANY REQUESTS", re.I)
_ANOMALY_503 = re.compile(r"\b503\b|UNAVAILABLE|overloaded|high demand", re.I)
_ANOMALY_RETRY = re.compile(r"retry|backing off|backoff|attempt \d", re.I)

_FOCUS_CHANNELS = ("anna_protocol", "ancient_knowledge")


@dataclass
class CallEvent:
    ts: datetime | None
    channel: str
    source: str
    model: str
    kind: str
    calls: int = 1
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0
    cost_usd: float = 0.0
    path: str = ""
    raw: str = ""
    anomaly: str = ""


@dataclass
class ChannelAgg:
    calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0
    cost_usd: float = 0.0
    events: list[CallEvent] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)


def _parse_ts(raw: str) -> datetime | None:
    raw = raw.strip().replace(",", ".")
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(raw[:26], fmt) if "." in raw else datetime.strptime(raw[:19], fmt)
        except ValueError:
            continue
    return None


def _default_window() -> tuple[datetime, datetime]:
    """Yesterday + today (local)."""
    now = datetime.now()
    start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start, end


def _default_peak() -> tuple[datetime, datetime]:
    """21:00 yesterday → 02:00 today (local), matching the billed window."""
    now = datetime.now()
    yest = now - timedelta(days=1)
    start = yest.replace(hour=21, minute=0, second=0, microsecond=0)
    end = now.replace(hour=2, minute=0, second=0, microsecond=0)
    if end <= start:
        end = start + timedelta(hours=5)
    return start, end


def _looks_google(line: str) -> bool:
    low = line.lower()
    return any(n in low for n in _GOOGLE_NEEDLES)


def _infer_kind(model: str, line: str) -> str:
    blob = f"{model} {line}".lower()
    if "imagen" in blob or "flash-image" in blob or "pro-image" in blob:
        return "image"
    if "gemini image ok" in blob or "image 429" in blob or "image 503" in blob:
        return "image"
    return "text"


def _infer_source(line: str, logger_name: str = "") -> str:
    if logger_name and logger_name not in {"root", ""}:
        return logger_name
    for needle, label in (
        ("caption_engine", "caption_engine"),
        ("subject_brain", "subject_brain"),
        ("batch_planner", "batch_planner"),
        ("visual_critic", "visual_critic"),
        ("visual_inspector", "visual_inspector"),
        ("image_provider", "image_provider"),
        ("gemini_utils", "gemini_utils"),
        ("orchestrator.llm", "orchestrator.llm"),
        ("generate_sequence_voiceover", "caption_engine.generate_sequence_voiceover"),
        ("synthesize_facts", "caption_engine.synthesize_facts"),
        ("GEMINI_FLASH_CALL", "config.note_gemini_flash_call"),
        ("Gemini image", "image_provider"),
        ("Gemini OK", "gemini_utils.generate_content_with_model_fallback"),
        ("DYNAMIC_STYLE_ANCHOR", "channel_loader"),
        ("CostTracker", "core.cost_tracker"),
    ):
        if needle.lower() in line.lower():
            return label
    return logger_name or "unknown"


def _estimate_cost(model: str, kind: str, inn: int, out: int, images: int) -> float:
    if images > 0:
        low = (model or "").lower()
        if "pro" in low or "imagen" in low:
            unit = IMAGE_1K_USD
        else:
            unit = IMAGE_FLASH_1K_USD
        return unit * images
    if kind == "image":
        return 0.0
    if inn <= 0 and out <= 0:
        return 0.0
    return (inn * FLASH_INPUT_USD_PER_1M + out * FLASH_OUTPUT_USD_PER_1M) / 1_000_000.0


def _split_log_line(line: str) -> tuple[datetime | None, str, str, str]:
    """Return (ts, level, logger_name, message) for ``asctime | LEVEL | name | msg``."""
    parts = [p.strip() for p in line.split("|", 3)]
    ts = None
    level = ""
    name = ""
    msg = line
    if parts:
        ts = _parse_ts(parts[0])
        if ts and len(parts) >= 4:
            level, name, msg = parts[1], parts[2], parts[3]
        elif ts and len(parts) == 3:
            level, msg = parts[1], parts[2]
    return ts, level, name, msg


def parse_log_line(line: str, *, channel_hint: str, path: str) -> CallEvent | None:
    if not _looks_google(line):
        return None
    ts, _level, logger_name, msg = _split_log_line(line)
    channel = channel_hint
    m_ch = _BEGIN_RE.search(line) or _CHANNEL_RE.search(line)
    if m_ch:
        channel = m_ch.group(1).strip().lower()

    gapi = _GOOGLE_API_LINE.search(line)
    if gapi:
        inn = int(gapi.group("inn"))
        out = int(gapi.group("out"))
        images = int(gapi.group("images") or 0)
        model = gapi.group("model")
        kind = gapi.group("kind")
        cost = float(gapi.group("cost"))
        return CallEvent(
            ts=ts,
            channel=(gapi.group("ch") or channel or "unknown").lower(),
            source=gapi.group("src"),
            model=model,
            kind=kind,
            calls=int(gapi.group("calls") or 1),
            retries=int(gapi.group("retries") or 0),
            input_tokens=inn,
            output_tokens=out,
            images=images,
            cost_usd=cost,
            path=path,
            raw=line.strip(),
        )

    model = ""
    m_model = _MODEL_RE.search(line)
    if m_model:
        model = f"models/{m_model.group(2)}" if m_model.group(2) else m_model.group(0)
    flash = _FLASH_CALL_RE.search(line)
    if flash and not model:
        model = flash.group("task") if "gemini" in flash.group("task").lower() else "models/gemini-2.5-flash"
    ok = _GEMINI_OK_RE.search(line)
    if ok:
        model = ok.group("model")
    img_ok = _IMAGE_OK_RE.search(line)
    if img_ok:
        model = img_ok.group("model")

    inn = out = 0
    m_tok = _TOKEN_RE.search(line)
    if m_tok:
        inn = int(m_tok.group("inn"))
        out = int(m_tok.group("out"))

    retries = 0
    m_ret = _RETRY_RE.search(line)
    if m_ret:
        retries = max(0, int(m_ret.group("n")) - 1)

    kind = _infer_kind(model, line)
    low = line.lower()
    images = 1 if kind == "image" and (img_ok or "image ok" in low or "imagen" in low) else 0
    if kind == "image" and not images and ("generate" in low or "image " in line):
        images = 1 if "OK" in line else 0

    # Adapter/routing banners mention a cost estimate but are not billed calls.
    banner = "adapter" in low or "together default" in low or "gemini native" in low
    if banner and "image ok" not in low and "gemini ok" not in low:
        return None

    cost = 0.0
    m_cost = _COST_RE.search(line)
    if m_cost and (img_ok or ok or images or inn or out):
        cost = float(m_cost.group("usd"))
    elif inn or out or images:
        cost = _estimate_cost(model, kind, inn, out, images)

    anomaly = ""
    if _ANOMALY_429.search(line):
        anomaly = "429/RESOURCE_EXHAUSTED"
        retries = max(retries, 1)
    elif _ANOMALY_503.search(line):
        anomaly = "503/UNAVAILABLE"
        retries = max(retries, 1)
    elif retries and _ANOMALY_RETRY.search(line):
        anomaly = f"retry/{retries}"

    # Skip handshake / client-connect noise unless it is an error/retry.
    if not (flash or ok or img_ok or gapi or m_tok or anomaly or m_cost):
        if "models.list" in low or "handshake" in low or "client connected" in low:
            return None
        if "planned image" in low or "captionengine online" in low:
            return None
        if not any(x in low for x in ("generate", "call", "retry", "429", "503", "token", "cost")):
            return None

    return CallEvent(
        ts=ts,
        channel=channel or "unknown",
        source=_infer_source(line, logger_name),
        model=model or "models/gemini-2.5-flash",
        kind=kind,
        calls=1,
        retries=retries,
        input_tokens=inn,
        output_tokens=out,
        images=images,
        cost_usd=cost,
        path=path,
        raw=line.strip(),
        anomaly=anomaly,
    )


def _iter_log_files(roots: Iterable[Path], start: datetime, end: datetime) -> list[Path]:
    found: list[Path] = []
    stamps = {
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        (start - timedelta(days=1)).strftime("%Y%m%d"),
    }
    for root in roots:
        if not root or not root.exists():
            continue
        if root.is_file():
            found.append(root)
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if p.suffix.lower() not in {".log", ".json", ".txt"}:
                continue
            # Skip giant render sidecars accidentally dropped in library/.
            try:
                if p.stat().st_size > 20_000_000:
                    continue
            except OSError:
                continue
            if any(s in name for s in stamps) or name.startswith("run_") or name.startswith("cost_"):
                found.append(p)
            elif "log" in name or "cost" in name:
                try:
                    mtime = datetime.fromtimestamp(p.stat().st_mtime)
                except OSError:
                    continue
                if start - timedelta(days=1) <= mtime <= end + timedelta(days=1):
                    found.append(p)
    # de-dupe
    uniq: dict[str, Path] = {}
    for p in found:
        uniq[str(p.resolve())] = p
    return sorted(uniq.values(), key=lambda x: x.name)


def parse_cost_json(path: Path) -> list[CallEvent]:
    events: list[CallEvent] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return events
    rows: list[Any]
    page = "unknown"
    if isinstance(data, dict):
        page = str(data.get("page_id") or data.get("channel") or "unknown")
        rows = list(data.get("breakdown") or data.get("entries") or [])
        if not rows and "events" in data:
            rows = list(data.get("events") or [])
    elif isinstance(data, list):
        rows = data
    else:
        return events
    for row in rows:
        if not isinstance(row, dict):
            continue
        op = str(row.get("operation") or row.get("kind") or "")
        model_key = str(row.get("model_key") or row.get("model") or "")
        low = f"{op} {model_key}".lower()
        if "gemini" not in low and "google" not in low and "imagen" not in low:
            if "text_generation" not in low:
                continue
            if "gemini" not in model_key.lower() and "text_gemini" not in model_key.lower():
                continue
        ts = None
        raw_ts = row.get("ts") or row.get("tracked_at") or ""
        if raw_ts:
            ts = _parse_ts(str(raw_ts).replace("T", " ").split("+")[0].split("Z")[0])
        units = float(row.get("units") or 0)
        cost = float(row.get("cost_usd") or 0)
        kind = "image" if "image" in low else "text"
        inn = int(units) if kind == "text" else 0
        images = int(units) if kind == "image" else 0
        model = model_key if "gemini" in model_key.lower() else "models/gemini-2.5-flash"
        if kind == "text" and cost <= 0 and inn:
            # CostTracker historically billed in+out at the input rate only.
            cost = (inn * FLASH_INPUT_USD_PER_1M) / 1_000_000.0
        if kind == "image" and cost <= 0 and images:
            cost = _estimate_cost(model, "image", 0, 0, images)
        events.append(
            CallEvent(
                ts=ts,
                channel=page,
                source="cost_tracker",
                model=model,
                kind=kind,
                calls=max(1, images or 1),
                input_tokens=inn if kind == "text" else 0,
                images=images,
                cost_usd=cost,
                path=str(path),
                raw=json.dumps(row, ensure_ascii=False)[:240],
            )
        )
    return events


def discover_roots(extra: list[Path] | None = None) -> list[Path]:
    roots: list[Path] = []
    extra = extra or []
    env_out = (os.getenv("OUTPUT_PATH") or os.getenv("OUTPUTS_DIR") or "").strip()
    if not env_out:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip().startswith("OUTPUT_PATH="):
                    env_out = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if env_out:
        out = Path(env_out)
        roots.extend(
            [
                out / "logs",
                out / "anna_protocol" / "library",
                out / "ancient_knowledge" / "library",
            ]
        )
        # Other channel cost ledgers (narrow — library/ only, never the media tree).
        if out.is_dir():
            for child in out.iterdir():
                lib = child / "library"
                if child.is_dir() and lib.is_dir():
                    roots.append(lib)
    repo = Path(__file__).resolve().parents[1]
    roots.extend(
        [
            repo / "outputs" / "logs",
            repo / "core" / "economic_reel_lofi" / "store",
        ]
    )
    roots.extend(extra)
    return roots


def scan(
    *,
    window_start: datetime,
    window_end: datetime,
    extra_dirs: list[Path] | None = None,
) -> list[CallEvent]:
    events: list[CallEvent] = []
    files = _iter_log_files(discover_roots(extra_dirs), window_start, window_end)
    for path in files:
        if path.suffix.lower() == ".json":
            if "cost" not in path.name.lower() and "gemini" not in path.name.lower():
                continue
            for ev in parse_cost_json(path):
                if ev.ts is None or window_start <= ev.ts <= window_end:
                    events.append(ev)
            continue
        channel_hint = "unknown"
        for part in path.parts:
            if part.lower() in _FOCUS_CHANNELS or part.lower() in {
                "wonder_feed",
                "master_mei",
                "momma_circle",
                "down_dirty",
                "endless_summer_paradise",
            }:
                channel_hint = part.lower()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            begin = _BEGIN_RE.search(line)
            if begin:
                channel_hint = begin.group(1).strip().lower()
            ev = parse_log_line(line, channel_hint=channel_hint, path=str(path))
            if ev is None:
                continue
            if ev.ts is not None and not (window_start <= ev.ts <= window_end):
                continue
            events.append(ev)
    return events


def aggregate(events: list[CallEvent]) -> dict[str, ChannelAgg]:
    out: dict[str, ChannelAgg] = defaultdict(ChannelAgg)
    for ev in events:
        bucket = out[ev.channel or "unknown"]
        bucket.calls += ev.calls
        bucket.retries += ev.retries
        bucket.input_tokens += ev.input_tokens
        bucket.output_tokens += ev.output_tokens
        bucket.images += ev.images
        bucket.cost_usd += ev.cost_usd
        bucket.events.append(ev)
        if ev.anomaly:
            bucket.anomalies.append(f"{ev.ts} {ev.source} {ev.anomaly} {ev.model}")
        if ev.output_tokens >= 8000:
            bucket.anomalies.append(
                f"{ev.ts} {ev.source} MASSIVE_OUTPUT tokens={ev.output_tokens} model={ev.model}"
            )
        if ev.retries >= 2:
            bucket.anomalies.append(
                f"{ev.ts} {ev.source} HIGH_RETRY retries={ev.retries} model={ev.model}"
            )
    return dict(out)


def _fmt_ts(ts: datetime | None) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "---------- --------"


def print_table(events: list[CallEvent]) -> None:
    header = (
        f"{'Timestamp':<19} | {'Channel':<20} | {'Script/Function':<36} | "
        f"{'Model Used':<32} | {'Calls/Retries':>13} | {'In Tok':>8} | "
        f"{'Out Tok':>8} | {'Est. Cost (USD)':>15}"
    )
    print(header)
    print("-" * len(header))
    rows = sorted(events, key=lambda e: (e.ts or datetime.min, e.channel, e.source))
    for ev in rows:
        cr = f"{ev.calls}/{ev.retries}"
        print(
            f"{_fmt_ts(ev.ts):<19} | {ev.channel:<20} | {ev.source:<36} | "
            f"{ev.model[:32]:<32} | {cr:>13} | {ev.input_tokens:>8} | "
            f"{ev.output_tokens:>8} | ${ev.cost_usd:>14.6f}"
        )


def print_channel_summary(aggs: dict[str, ChannelAgg], peak: list[CallEvent]) -> None:
    print("\n=== CHANNEL TOTALS (yesterday + today) ===")
    print(
        f"{'Channel':<22} {'Calls':>7} {'Retries':>8} {'In Tok':>10} "
        f"{'Out Tok':>10} {'Images':>7} {'Est. USD':>12}"
    )
    print("-" * 80)
    focus = list(_FOCUS_CHANNELS) + [c for c in sorted(aggs) if c not in _FOCUS_CHANNELS]
    for name in focus:
        agg = aggs.get(name)
        if not agg:
            print(f"{name:<22} {0:>7} {0:>8} {0:>10} {0:>10} {0:>7} ${0:>11.6f}")
            continue
        print(
            f"{name:<22} {agg.calls:>7} {agg.retries:>8} {agg.input_tokens:>10} "
            f"{agg.output_tokens:>10} {agg.images:>7} ${agg.cost_usd:>11.6f}"
        )
    grand = sum(a.cost_usd for a in aggs.values())
    print("-" * 80)
    print(f"{'ALL CHANNELS':<22} {sum(a.calls for a in aggs.values()):>7} "
          f"{sum(a.retries for a in aggs.values()):>8} "
          f"{sum(a.input_tokens for a in aggs.values()):>10} "
          f"{sum(a.output_tokens for a in aggs.values()):>10} "
          f"{sum(a.images for a in aggs.values()):>7} ${grand:>11.6f}")

    print("\n=== PEAK WINDOW (21:00 yesterday -> 02:00 today, local) ===")
    if not peak:
        print("No Google/Gemini events parsed inside the peak window.")
    else:
        peak_aggs = aggregate(peak)
        for name, agg in sorted(peak_aggs.items(), key=lambda kv: -kv[1].cost_usd):
            print(
                f"  {name}: {agg.calls} calls / {agg.retries} retries / "
                f"{agg.images} images / {agg.input_tokens} in / {agg.output_tokens} out / "
                f"${agg.cost_usd:.6f}"
            )

    print("\n=== ANOMALIES (retries, 429, 503, huge outputs) ===")
    any_anom = False
    for name, agg in sorted(aggs.items()):
        if not agg.anomalies:
            continue
        any_anom = True
        print(f"\n[{name}] {len(agg.anomalies)} flag(s)")
        for row in agg.anomalies[:40]:
            print(f"  - {row}")
        if len(agg.anomalies) > 40:
            print(f"  ... {len(agg.anomalies) - 40} more")
    if not any_anom:
        print("None detected in parsed lines. Token counts are often missing from older logs;")
        print("cost JSON (CostTracker) bills Gemini text at the blended $0.075/1M rate.")

    print("\n=== ATTRIBUTION HINT ===")
    anna = aggs.get("anna_protocol")
    ak = aggs.get("ancient_knowledge")
    anna_cost = anna.cost_usd if anna else 0.0
    ak_cost = ak.cost_usd if ak else 0.0
    anna_img = anna.images if anna else 0
    ak_img = ak.images if ak else 0
    if anna_img and not ak_img:
        print("Likely source: anna_protocol - Gemini image / likeness calls present.")
    elif ak_cost > anna_cost * 2 and (ak.calls if ak else 0) > (anna.calls if anna else 0):
        print("Likely source: ancient_knowledge - heavier Gemini text volume.")
    elif anna_cost > ak_cost:
        print("Likely source: anna_protocol (higher estimated Gemini spend).")
    elif ak_cost > anna_cost:
        print("Likely source: ancient_knowledge (higher estimated Gemini spend).")
    else:
        print("Inconclusive from local logs - check Google AI Studio usage by API method / model.")
    print(
        "Note: CostTracker historically priced Gemini text at $0.075/1M for in+out "
        "combined (under-counts output, which bills at $0.30/1M). Confirmed "
        "'Gemini image OK' lines on anna_protocol using gemini-3-pro-image-preview "
        "bill ~$0.134/image (2K). ancient_knowledge reel stills use Together FLUX "
        "(not Gemini image) unless Visual QA was enabled."
    )


def _parse_cli_dt(raw: str) -> datetime:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise SystemExit(f"Cannot parse datetime: {raw!r} (use YYYY-MM-DD HH:MM)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit local Google/Gemini API spend from run logs.")
    parser.add_argument("--logs-dir", action="append", default=[], help="Extra log/JSON directories")
    parser.add_argument("--since", default="", help="Window start (YYYY-MM-DD HH:MM), default yesterday 00:00")
    parser.add_argument("--until", default="", help="Window end (YYYY-MM-DD HH:MM), default today 23:59")
    parser.add_argument("--peak-start", default="", help="Peak window start, default yesterday 21:00")
    parser.add_argument("--peak-end", default="", help="Peak window end, default today 02:00")
    args = parser.parse_args(argv)

    win_s, win_e = _default_window()
    peak_s, peak_e = _default_peak()
    if args.since:
        win_s = _parse_cli_dt(args.since)
    if args.until:
        win_e = _parse_cli_dt(args.until)
    if args.peak_start:
        peak_s = _parse_cli_dt(args.peak_start)
    if args.peak_end:
        peak_e = _parse_cli_dt(args.peak_end)

    extra = [Path(p) for p in args.logs_dir]
    print(f"Audit window : {win_s} -> {win_e} (local)")
    print(f"Peak window  : {peak_s} -> {peak_e} (local)")
    print(f"Extra roots  : {extra or '(none)'}")
    print()

    events = scan(window_start=win_s, window_end=win_e, extra_dirs=extra)
    if not events:
        print("No Google/Gemini log events found for yesterday/today.")
        print("Searched OUTPUT_PATH/logs, per-channel library/cost_*.json, and repo fallbacks.")
        return 0

    billed = [e for e in events if e.cost_usd > 0 or e.anomaly or e.images or e.input_tokens or e.output_tokens]
    print(f"Parsed {len(events)} Google-related lines; showing {len(billed)} billed/token/anomaly rows.\n")
    print_table(billed if billed else events)
    aggs = aggregate(events)
    peak_events = [
        e for e in events
        if e.ts is not None and peak_s <= e.ts <= peak_e
    ]
    print_channel_summary(aggs, peak_events)
    return 0


if __name__ == "__main__":
    sys.exit(main())
