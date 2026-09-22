# -*- coding: utf-8 -*-
"""Direct Gemini Developer API provider for explicit per-seat opt-in."""
from __future__ import annotations

import time
from typing import Any, Sequence

try:
    from ..contracts import ChatMessage
    from ..settings import GoogleConfig, ModelSpec
    from .base import LLMError, LLMProvider, LLMResponse, ReasoningEffort
    from .llm_factory import register_provider
except ImportError:  # pragma: no cover — standalone extraction
    from contracts import ChatMessage  # type: ignore[no-redef]
    from models.base import (  # type: ignore[no-redef]
        LLMError,
        LLMProvider,
        LLMResponse,
        ReasoningEffort,
    )
    from models.llm_factory import register_provider  # type: ignore[no-redef]
    from settings import GoogleConfig, ModelSpec  # type: ignore[no-redef]


def _google_model_id(model: str) -> str:
    raw = (model or "").strip().removeprefix("models/")
    if raw.lower().startswith("google/"):
        raw = raw.split("/", 1)[1]
    return raw


def _is_flash_text_model(model: str) -> bool:
    lowered = _google_model_id(model).lower()
    excluded = ("image", "audio", "tts", "live", "embedding", "imagen")
    return "flash" in lowered and not any(marker in lowered for marker in excluded)


def _finish_reason(candidate: Any) -> str:
    value = getattr(candidate, "finish_reason", None)
    if value is None:
        return "stop"
    name = getattr(value, "name", None)
    raw = str(name or value).rsplit(".", 1)[-1].strip().lower()
    return {"max_tokens": "max_tokens", "stop": "stop"}.get(raw, raw or "stop")


@register_provider
class GoogleProvider(LLMProvider):
    """Text completion through Google's official ``google-genai`` SDK."""

    registry_name = "google"
    requires_api_key = True

    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: str | None = None,
        google_config: GoogleConfig | None = None,
        client: Any | None = None,
    ) -> None:
        normalised = spec.model_copy(update={"model": _google_model_id(spec.model)})
        super().__init__(normalised, api_key=api_key)
        self.google_config = google_config or GoogleConfig()
        if self.google_config.free_tier_only and not _is_flash_text_model(self.spec.model):
            raise LLMError(
                self.registry_name,
                self.spec.model,
                "google.free_tier_only permits Flash text models only; "
                "choose a gemini-*-flash model or explicitly disable that policy",
            )
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                self.registry_name,
                self.spec.model,
                "google-genai is not installed",
            ) from exc
        from core.google_guardrail import make_guarded_gemini_client

        self._client = make_guarded_gemini_client(
            self.api_key,
            http_options={"api_version": self.google_config.api_version},
        )
        return self._client

    @staticmethod
    def _wire_messages(messages: Sequence[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(message.content for message in messages if message.role == "system")
        contents: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})
        return system, contents

    def _dispatch(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
    ) -> LLMResponse:
        del reasoning_effort  # OpenRouter-only request control for now.
        try:
            from google.genai import types  # noqa: PLC0415

            system, contents = self._wire_messages(messages)
            started = time.perf_counter()
            from core.google_guardrail import guarded_generate_content

            response = guarded_generate_content(
                self._get_client(),
                model=self.spec.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system or None,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
                source="aiwake.GoogleProvider._dispatch",
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — SDK errors vary by release
            raise LLMError(self.registry_name, self.spec.model, f"request failed: {exc}") from exc

        candidates = list(getattr(response, "candidates", None) or [])
        candidate = candidates[0] if candidates else None
        try:
            text = str(response.text or "").strip()
        except Exception:  # pragma: no cover — SDK raises when no text part exists
            text = ""
        usage = getattr(response, "usage_metadata", None)
        finish = _finish_reason(candidate)

        return LLMResponse(
            text=text,
            model=str(getattr(response, "model_version", None) or self.spec.model),
            provider=self.registry_name,
            latency_ms=latency_ms,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            finish_reason=finish,
            raw={
                "finish_reason": finish,
                "response_id": str(getattr(response, "response_id", "") or ""),
            },
        )

    def complete(self, messages: Sequence[ChatMessage], **kwargs: Any) -> LLMResponse:
        kwargs.setdefault("max_retries", self.google_config.max_retries)
        kwargs.setdefault("backoff_s", self.google_config.backoff_s)
        return super().complete(messages, **kwargs)


__all__ = ["GoogleProvider"]
