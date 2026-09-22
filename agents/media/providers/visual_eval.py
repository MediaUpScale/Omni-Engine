# -*- coding: utf-8 -*-
"""
Modular Visual Evaluation Provider.

Used by the Visual Control Agent (sequence inspector). The brain is decoupled
from any single vendor: swap the backend via ``VISUAL_EVAL_PROVIDER`` /
``VISUAL_EVAL_MODEL`` (OpenRouter Qwen VL, DeepSeek VL, Gemini, …).

Default: OpenRouter OpenAI-compatible client → ``qwen/qwen3.5-flash-02-23``.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

_LOG = logging.getLogger(__name__)

OPENROUTER_DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_VISION_MODEL: str = "qwen/qwen3.5-flash-02-23"
DEFAULT_OPENROUTER_VISION_FALLBACK: str = "qwen/qwen3.5-9b"
DEFAULT_GEMINI_VISION_MODEL: str = "models/gemini-2.5-flash"


@dataclass(frozen=True)
class VisualEvalResult:
    """Raw model text plus the backend that produced it."""

    text: str
    provider: str
    model_id: str


class VisualEvalProvider(Protocol):
    """Vision LLM that inspects a labeled image sequence and returns text."""

    name: str
    model_id: str

    def evaluate(
        self,
        prompt: str,
        image_paths: Sequence[Path],
        *,
        labels: Sequence[str] | None = None,
    ) -> VisualEvalResult:
        ...


def _cfg(name: str, default: str | None = None) -> str:
    try:
        import config as app_config

        val = getattr(app_config, name, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    except Exception:  # noqa: BLE001
        pass
    env = (os.getenv(name) or "").strip()
    return env or (default or "")


def _eval_timeout_s(default: float = 10.0) -> float:
    raw = _cfg("VISUAL_EVAL_TIMEOUT_S")
    try:
        val = float(raw) if raw else default
    except (TypeError, ValueError):
        val = default
    return max(1.0, min(val, 30.0))


def _engine_debug() -> bool:
    raw = (os.getenv("ENGINE_DEBUG") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    try:
        import config as app_config

        return bool(getattr(app_config, "ENGINE_DEBUG", False))
    except Exception:  # noqa: BLE001
        return False


def _is_model_unavailable_error(exc: BaseException) -> bool:
    """True for 404 / missing-endpoint / unknown-model failures."""
    code = getattr(exc, "status_code", None)
    if code is None:
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None)
    if code in (404, 400):
        return True
    msg = str(exc).lower()
    needles = (
        "404",
        "not found",
        "no endpoints",
        "no endpoint",
        "unknown model",
        "invalid model",
        "model does not exist",
        "is not a valid model",
    )
    return any(n in msg for n in needles)


def _read_data_url(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("VisualEval skip unreadable %s: %s", path.name, exc)
        return None
    if not raw:
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


class OpenRouterVisualEval:
    """OpenAI-compatible chat completions via OpenRouter."""

    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.model_id = (
            model_id
            or _cfg("VISUAL_EVAL_MODEL", DEFAULT_OPENROUTER_VISION_MODEL)
            or DEFAULT_OPENROUTER_VISION_MODEL
        )
        self.api_key = (api_key or _cfg("OPENROUTER_API_KEY") or "").strip()
        self.base_url = (
            base_url or _cfg("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL)
            or OPENROUTER_DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout_s = float(timeout_s) if timeout_s is not None else _eval_timeout_s()
        self._client = None

    def _model_chain(self) -> list[str]:
        fallback = (
            _cfg("VISUAL_EVAL_MODEL_FALLBACK", DEFAULT_OPENROUTER_VISION_FALLBACK)
            or DEFAULT_OPENROUTER_VISION_FALLBACK
        )
        chain: list[str] = []
        for mid in (self.model_id, fallback, DEFAULT_OPENROUTER_VISION_MODEL, DEFAULT_OPENROUTER_VISION_FALLBACK):
            name = (mid or "").strip()
            if name and name not in chain:
                chain.append(name)
        return chain

    def _client_or_raise(self):
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing — visual eval disabled.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
                default_headers={
                    "HTTP-Referer": "https://omni-engine.local",
                    "X-Title": "omni-engine-visual-eval",
                },
            )
        return self._client

    def evaluate(
        self,
        prompt: str,
        image_paths: Sequence[Path],
        *,
        labels: Sequence[str] | None = None,
    ) -> VisualEvalResult:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for i, path in enumerate(image_paths):
            p = Path(path)
            if not p.is_file():
                continue
            data_url = _read_data_url(p)
            if not data_url:
                continue
            label = ""
            if labels and i < len(labels) and labels[i]:
                label = str(labels[i])
            else:
                label = f"FRAME {i + 1}"
            content.append({"type": "text", "text": f"{label}:"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        client = self._client_or_raise()
        last_exc: BaseException | None = None
        started = time.monotonic()
        budget = max(1.0, float(self.timeout_s))
        for mid in self._model_chain():
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0.3:
                last_exc = TimeoutError(
                    f"OpenRouter visual eval exceeded {budget:.0f}s budget"
                )
                break
            req_timeout = max(1.0, min(remaining, budget))
            tagged = client
            try:
                tagged = client.with_options(timeout=req_timeout)
            except Exception:  # noqa: BLE001
                tagged = client
            kwargs: dict = {
                "model": mid,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "max_tokens": 2048,
            }
            try:
                try:
                    resp = tagged.chat.completions.create(
                        **kwargs,
                        response_format={"type": "json_object"},
                    )
                except Exception as fmt_exc:
                    if _is_model_unavailable_error(fmt_exc):
                        raise
                    resp = tagged.chat.completions.create(**kwargs)
                text = ""
                try:
                    text = (resp.choices[0].message.content or "").strip()
                except Exception:  # noqa: BLE001
                    text = ""
                if text:
                    self.model_id = mid
                    return VisualEvalResult(text=text, provider=self.name, model_id=mid)
                last_exc = RuntimeError(
                    f"OpenRouter visual eval returned empty text | model={mid}"
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_model_unavailable_error(exc):
                    if _engine_debug():
                        _LOG.warning("OpenRouter visual eval 404/unavailable | model=%s | %s", mid, exc)
                    else:
                        _LOG.debug("OpenRouter visual eval fallback | model=%s | %s", mid, exc)
                    continue
                if _engine_debug():
                    _LOG.warning("OpenRouter visual eval failed | model=%s | %s", mid, exc)
                else:
                    _LOG.debug("OpenRouter visual eval failed | model=%s | %s", mid, exc)
                continue
        raise RuntimeError(
            f"OpenRouter visual eval exhausted models {self._model_chain()}: {last_exc}"
        ) from last_exc


class GeminiVisualEval:
    """Optional Gemini vision backend (swap-in, not the default)."""

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.model_id = (
            model_id
            or _cfg("VISUAL_EVAL_MODEL")
            or DEFAULT_GEMINI_VISION_MODEL
        )
        if self.model_id.startswith("qwen/") or self.model_id.startswith("deepseek/"):
            self.model_id = DEFAULT_GEMINI_VISION_MODEL
        self.api_key = (
            api_key
            or _cfg("GEMINI_API_KEY")
            or _cfg("GOOGLE_API_KEY")
            or ""
        ).strip()

    def evaluate(
        self,
        prompt: str,
        image_paths: Sequence[Path],
        *,
        labels: Sequence[str] | None = None,
    ) -> VisualEvalResult:
        from google.genai import types
        from core.google_guardrail import guarded_generate_content, make_guarded_gemini_client

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY missing — Gemini visual eval disabled.")

        parts: list = [types.Part.from_text(text=prompt)]
        for i, path in enumerate(image_paths):
            p = Path(path)
            if not p.is_file():
                continue
            try:
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                label = (
                    str(labels[i])
                    if labels and i < len(labels) and labels[i]
                    else f"FRAME {i + 1}"
                )
                parts.append(types.Part.from_text(text=f"{label}:"))
                parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("Gemini visual eval skip %s: %s", p.name, exc)

        client = make_guarded_gemini_client(self.api_key)
        mid = self.model_id
        if not mid.startswith("models/") and "gemini" in mid.lower():
            mid = f"models/{mid}"
        resp = guarded_generate_content(
            client,
            model=mid,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
            source="visual_eval.GeminiVisualEval",
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            raise RuntimeError(f"Gemini visual eval returned empty text | model={mid}")
        return VisualEvalResult(text=text, provider=self.name, model_id=mid)


def resolve_visual_eval_provider(
    *,
    provider: str | None = None,
    model_id: str | None = None,
) -> VisualEvalProvider | None:
    """
    Build the active visual evaluator.

    ``provider``: ``openrouter`` (default) | ``gemini`` | ``auto``.
    ``auto`` tries OpenRouter first, then Gemini.
    Returns None when no usable backend is configured (caller uses phash).
    """
    name = (provider or _cfg("VISUAL_EVAL_PROVIDER", "openrouter") or "openrouter").strip().lower()
    model = (model_id or _cfg("VISUAL_EVAL_MODEL") or "").strip() or None

    def _openrouter() -> OpenRouterVisualEval | None:
        or_key = _cfg("OPENROUTER_API_KEY")
        if not or_key:
            return None
        return OpenRouterVisualEval(api_key=or_key, model_id=model)

    def _gemini() -> GeminiVisualEval | None:
        gem_key = _cfg("GEMINI_API_KEY") or _cfg("GOOGLE_API_KEY")
        if not gem_key:
            return None
        try:
            from core.google_guardrail import is_google_api_allowed

            if not is_google_api_allowed():
                return None
        except Exception:  # noqa: BLE001
            pass
        return GeminiVisualEval(api_key=gem_key, model_id=model)

    if name in ("openrouter", "qwen", "deepseek"):
        backend = _openrouter()
        if backend is None:
            msg = "VisualEval | provider=%s requested but OPENROUTER_API_KEY missing"
            if _engine_debug():
                _LOG.warning(msg, name)
            else:
                _LOG.debug(msg, name)
        return backend
    if name == "gemini":
        return _gemini()
    if name == "auto":
        return _openrouter() or _gemini()
    if _engine_debug():
        _LOG.warning("VisualEval | unknown provider=%s — no backend", name)
    else:
        _LOG.debug("VisualEval | unknown provider=%s — no backend", name)
    return None
