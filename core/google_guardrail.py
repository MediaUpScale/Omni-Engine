# -*- coding: utf-8 -*-
"""
Emergency Google / Gemini billing circuit-breaker.

Wraps every billed Google API call with:

1. ``ALLOW_GOOGLE_API`` kill-switch (false → hard abort, no request is sent).
2. Per-process cost tracker with a hard USD cap (``MAX_GOOGLE_COST_PER_RUN_USD``).
3. Strict retry budget (``MAX_GOOGLE_RETRIES``, default 2) — no unbounded backoff.

Pricing used for estimates (USD, official Gemini list as of 2026-09):

* ``gemini-2.5-flash`` text: $0.075 / 1M input tokens, $0.30 / 1M output tokens
* ``gemini-*-flash-image``: $0.03 / image (1K)
* ``gemini-*-pro-image`` / Imagen: $0.134 / image (2K standard)

Environment
-----------
ALLOW_GOOGLE_API=true|false          default true if unset (``.env`` may force false)
MAX_GOOGLE_COST_PER_RUN_USD=0.50     hard abort once this run's estimate is exceeded
MAX_GOOGLE_RETRIES=2                 additional attempts after the first failure
"""
from __future__ import annotations

import inspect
import logging
import os
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Official Gemini 2.5 Flash list prices (USD / 1M tokens)
FLASH_INPUT_USD_PER_1M: float = 0.075
FLASH_OUTPUT_USD_PER_1M: float = 0.30

# Official Gemini / Imagen image list prices (USD / image) — 1K is the factory default
IMAGE_FLASH_1K_USD: float = 0.005
IMAGE_1K_USD: float = 0.03          # Pro / Imagen at 1K
IMAGE_2K_USD: float = 0.134         # Pro 2K (opt-in only; never the default)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class GoogleAPIBlockedError(RuntimeError):
    """Raised when ``ALLOW_GOOGLE_API`` is false — no request is sent."""


class GoogleBudgetExceededError(RuntimeError):
    """Raised when this process would exceed ``MAX_GOOGLE_COST_PER_RUN_USD``."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUE:
        return True
    if val in _FALSE:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def is_google_api_allowed() -> bool:
    """Live read of ``ALLOW_GOOGLE_API``. Unset → True (tests / CI stay unblocked)."""
    return _env_bool("ALLOW_GOOGLE_API", True)


def max_google_cost_usd() -> float:
    return max(0.0, _env_float("MAX_GOOGLE_COST_PER_RUN_USD", 0.50))


def max_google_retries() -> int:
    """Additional attempts after the first failure (default 2 → 3 total tries)."""
    return max(0, _env_int("MAX_GOOGLE_RETRIES", 2))


def max_google_attempts() -> int:
    """Total tries = 1 initial + ``MAX_GOOGLE_RETRIES``."""
    return 1 + max_google_retries()


def active_channel() -> str:
    env = (os.getenv("ACTIVE_PAGE") or "").strip()
    if env:
        return env
    try:
        import config as app_config

        return str(getattr(app_config, "ACTIVE_PAGE", "") or "unknown")
    except Exception:
        return "unknown"


def assert_google_allowed(*, context: str = "") -> None:
    """Hard kill-switch. Call before any Google client construction or request."""
    if is_google_api_allowed():
        return
    where = context or _caller_name()
    channel = active_channel()
    msg = (
        f"GOOGLE API KILL-SWITCH | ALLOW_GOOGLE_API=false | "
        f"channel={channel} | src={where} | no request sent"
    )
    logger.error(msg)
    raise GoogleAPIBlockedError(msg)


def _caller_name(depth: int = 2) -> str:
    try:
        frame = inspect.stack()[depth]
        mod = frame.filename.replace("\\", "/").rsplit("/", 1)[-1]
        return f"{mod}:{frame.function}"
    except Exception:
        return "unknown"


def classify_model(model_id: str | None) -> str:
    low = (model_id or "").lower()
    if "imagen" in low:
        return "imagen"
    if "image" in low and "pro" in low:
        return "gemini-pro-image"
    if "image" in low:
        return "gemini-flash-image"
    if "pro" in low and "flash" not in low:
        return "gemini-pro-text"
    return "gemini-flash-text"


def is_image_model(model_id: str | None) -> bool:
    kind = classify_model(model_id)
    return kind in {"imagen", "gemini-pro-image", "gemini-flash-image"}


def _configured_image_size() -> str:
    raw = (os.getenv("GEMINI_IMAGE_SIZE") or "").strip().upper()
    if raw in {"1K", "2K"}:
        return raw
    try:
        import config as app_config

        val = str(getattr(app_config, "GEMINI_IMAGE_SIZE", "1K") or "1K").strip().upper()
        return val if val in {"1K", "2K"} else "1K"
    except Exception:
        return "1K"


def image_unit_cost_usd(model_id: str | None) -> float:
    kind = classify_model(model_id)
    size = _configured_image_size()
    if kind in {"imagen", "gemini-pro-image"}:
        return IMAGE_2K_USD if size == "2K" else IMAGE_1K_USD
    if kind == "gemini-flash-image":
        return IMAGE_FLASH_1K_USD
    return 0.0


def text_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    model_id: str | None = None,
) -> float:
    del model_id  # Flash rates apply to the factory's Gemini text path.
    inp = max(0, int(input_tokens or 0))
    out = max(0, int(output_tokens or 0))
    return (inp * FLASH_INPUT_USD_PER_1M + out * FLASH_OUTPUT_USD_PER_1M) / 1_000_000.0


def estimate_call_cost_usd(
    model_id: str | None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    images: int = 0,
    kind: str | None = None,
) -> float:
    resolved = (kind or "").strip().lower()
    if not resolved:
        resolved = "image" if is_image_model(model_id) else "text"
    if resolved in {"image", "vision-image"}:
        n = max(1, int(images or 1))
        return image_unit_cost_usd(model_id) * n
    return text_cost_usd(input_tokens, output_tokens, model_id=model_id)


def estimate_contents_tokens(contents: Any) -> int:
    """Rough input-token estimate (chars / 4) for preflight reserves."""
    return max(1, _contents_char_count(contents) // 4)


def _contents_char_count(contents: Any) -> int:
    if contents is None:
        return 0
    if isinstance(contents, str):
        return len(contents)
    if isinstance(contents, (bytes, bytearray)):
        return len(contents)
    if isinstance(contents, dict):
        return sum(_contents_char_count(v) for v in contents.values())
    if isinstance(contents, (list, tuple)):
        return sum(_contents_char_count(item) for item in contents)
    text = getattr(contents, "text", None)
    if isinstance(text, str):
        return len(text)
    if callable(text):
        try:
            val = text()
            if isinstance(val, str):
                return len(val)
        except Exception:
            pass
    try:
        return len(str(contents))
    except Exception:
        return 0


def extract_usage_tokens(
    response: Any,
    *,
    fallback_input: int = 0,
    fallback_output: int = 0,
) -> tuple[int, int]:
    """Read ``usage_metadata``; fall back to char/4 estimates."""
    prompt_tok = 0
    completion_tok = 0
    um = getattr(response, "usage_metadata", None)
    if um is None and isinstance(response, dict):
        um = response.get("usage_metadata")
    if isinstance(um, dict):
        prompt_tok = int(float(um.get("prompt_token_count") or 0))
        completion_tok = int(float(um.get("candidates_token_count") or 0))
        if completion_tok <= 0:
            completion_tok = int(float(um.get("output_token_count") or 0))
    elif um is not None:
        prompt_tok = int(float(getattr(um, "prompt_token_count", 0) or 0))
        completion_tok = int(float(getattr(um, "candidates_token_count", 0) or 0))
        if completion_tok <= 0:
            completion_tok = int(float(getattr(um, "output_token_count", 0) or 0))
    if prompt_tok <= 0:
        prompt_tok = max(0, int(fallback_input or 0))
    if completion_tok <= 0:
        text_attr = getattr(response, "text", None) if response is not None else None
        raw = ""
        if callable(text_attr):
            try:
                raw = str(text_attr() or "")
            except Exception:
                raw = ""
        elif text_attr:
            raw = str(text_attr)
        completion_tok = max(0, int(fallback_output or 0))
        if completion_tok <= 0 and raw:
            completion_tok = max(1, len(raw) // 4)
    return prompt_tok, completion_tok


@dataclass
class GoogleCostRecord:
    ts: str
    channel: str
    source: str
    model: str
    kind: str
    input_tokens: int
    output_tokens: int
    images: int
    retries: int
    cost_usd: float
    status: str


@dataclass
class GoogleCostTracker:
    """Process-wide singleton ledger. Thread-safe."""

    _total_usd: float = 0.0
    _calls: int = 0
    _retries: int = 0
    _images: int = 0
    _input_tokens: int = 0
    _output_tokens: int = 0
    _entries: list[GoogleCostRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "channel": active_channel(),
                "total_usd": round(self._total_usd, 6),
                "cap_usd": max_google_cost_usd(),
                "calls": self._calls,
                "retries": self._retries,
                "images": self._images,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "entries": [e.__dict__ for e in self._entries],
            }

    def reset(self) -> None:
        with self._lock:
            self._total_usd = 0.0
            self._calls = 0
            self._retries = 0
            self._images = 0
            self._input_tokens = 0
            self._output_tokens = 0
            self._entries.clear()

    def preflight(
        self,
        estimated_usd: float,
        *,
        model: str = "",
        source: str = "",
        kind: str = "text",
    ) -> None:
        """Abort *before* the HTTP request if this call would breach the cap."""
        assert_google_allowed(context=source or "google_guardrail.preflight")
        cap = max_google_cost_usd()
        extra = max(0.0, float(estimated_usd or 0.0))
        with self._lock:
            projected = self._total_usd + extra
            current = self._total_usd
        if projected > cap + 1e-12:
            self._raise_budget(current, extra, cap, model=model, source=source, kind=kind)

    def record(
        self,
        *,
        model: str,
        kind: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        images: int = 0,
        retries: int = 0,
        source: str = "",
        status: str = "ok",
        cost_usd: float | None = None,
    ) -> float:
        """Charge the ledger *after* a request (or a billed failed attempt)."""
        cost = (
            float(cost_usd)
            if cost_usd is not None
            else estimate_call_cost_usd(
                model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                images=images,
                kind=kind,
            )
        )
        cost = max(0.0, cost)
        cap = max_google_cost_usd()
        rec = GoogleCostRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            channel=active_channel(),
            source=source or _caller_name(),
            model=str(model or "unknown"),
            kind=str(kind or "text"),
            input_tokens=max(0, int(input_tokens or 0)),
            output_tokens=max(0, int(output_tokens or 0)),
            images=max(0, int(images or 0)),
            retries=max(0, int(retries or 0)),
            cost_usd=round(cost, 8),
            status=status,
        )
        with self._lock:
            self._entries.append(rec)
            self._total_usd += cost
            self._calls += 1
            self._retries += rec.retries
            self._images += rec.images
            self._input_tokens += rec.input_tokens
            self._output_tokens += rec.output_tokens
            total = self._total_usd
        logger.info(
            "GOOGLE_API | channel=%s | src=%s | model=%s | kind=%s | "
            "calls=1 | retries=%d | in=%d | out=%d | images=%d | "
            "cost=$%.6f | run_total=$%.6f | cap=$%.2f | status=%s",
            rec.channel,
            rec.source,
            rec.model,
            rec.kind,
            rec.retries,
            rec.input_tokens,
            rec.output_tokens,
            rec.images,
            rec.cost_usd,
            total,
            cap,
            rec.status,
        )
        if total > cap + 1e-12:
            self._raise_budget(total, 0.0, cap, model=model, source=source, kind=kind)
        return cost

    def _raise_budget(
        self,
        current: float,
        extra: float,
        cap: float,
        *,
        model: str,
        source: str,
        kind: str,
    ) -> None:
        stack = "".join(traceback.format_stack(limit=12))
        msg = (
            f"GOOGLE BUDGET HARD CAP | channel={active_channel()} | "
            f"src={source or _caller_name()} | model={model} | kind={kind} | "
            f"spent=${current:.6f} | next=${extra:.6f} | cap=${cap:.2f} USD | "
            f"ABORTING to prevent further Gemini billing"
        )
        logger.error(msg)
        logger.error("GOOGLE BUDGET STACK:\n%s", stack)
        raise GoogleBudgetExceededError(msg)


_TRACKER = GoogleCostTracker()
_TRACKER_LOCK = threading.Lock()


def get_google_cost_tracker() -> GoogleCostTracker:
    return _TRACKER


def reset_google_cost_tracker() -> None:
    """Test / new-process helper."""
    _TRACKER.reset()


def make_guarded_gemini_client(api_key: str, **kwargs: Any) -> Any:
    """``genai.Client`` factory that honours the kill-switch."""
    assert_google_allowed(context="make_guarded_gemini_client")
    from google import genai

    return genai.Client(api_key=api_key, **kwargs)


def guarded_generate_content(
    client: Any,
    model: str | None = None,
    contents: Any = None,
    config: Any = None,
    *,
    kind: str | None = None,
    source: str = "",
    **kwargs: Any,
) -> Any:
    """
    Single billed ``generate_content`` with kill-switch, preflight cap, and
    a hard retry ceiling. Prefer this over ``client.models.generate_content``.
    """
    src = source or _caller_name()
    model_id = str(model or kwargs.pop("model", "") or "unknown")
    body = contents if contents is not None else kwargs.pop("contents", None)
    cfg = config if config is not None else kwargs.get("config")
    resolved_kind = kind or ("image" if is_image_model(model_id) else "text")
    tracker = get_google_cost_tracker()
    in_est = estimate_contents_tokens(body)
    out_reserve = 0 if resolved_kind == "image" else 512
    img_n = 1 if resolved_kind == "image" else 0
    est = estimate_call_cost_usd(
        model_id,
        input_tokens=in_est,
        output_tokens=out_reserve,
        images=img_n,
        kind=resolved_kind,
    )
    last_exc: BaseException | None = None
    attempts = max_google_attempts()
    for attempt in range(attempts):
        tracker.preflight(est, model=model_id, source=src, kind=resolved_kind)
        try:
            call_kwargs: dict[str, Any] = {"model": model_id, "contents": body}
            if cfg is not None:
                call_kwargs["config"] = cfg
            call_kwargs.update(kwargs)
            response = client.models.generate_content(**call_kwargs)
            in_tok, out_tok = extract_usage_tokens(response, fallback_input=in_est)
            tracker.record(
                model=model_id,
                kind=resolved_kind,
                input_tokens=in_tok,
                output_tokens=out_tok,
                images=img_n,
                retries=attempt,
                source=src,
                status="ok",
            )
            return response
        except (GoogleAPIBlockedError, GoogleBudgetExceededError):
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 >= attempts:
                logger.error(
                    "GOOGLE_API FAIL | src=%s | model=%s | retries_exhausted=%d | %s",
                    src, model_id, attempt, exc,
                )
                raise
            logger.warning(
                "GOOGLE_API retry %d/%d | src=%s | model=%s | %s",
                attempt + 1, max_google_retries(), src, model_id, exc,
            )
    if last_exc:
        raise last_exc
    raise RuntimeError("guarded_generate_content: no response and no exception")
