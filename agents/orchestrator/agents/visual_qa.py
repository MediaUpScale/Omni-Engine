# -*- coding: utf-8 -*-
"""VisualQAAgent — Vision LLM validation node wrapping VisualQA_Agent."""
from __future__ import annotations

import json
import logging
import mimetypes
import threading
from pathlib import Path
from typing import Sequence

from agents.orchestrator.agents.base import BaseAgent
from agents.orchestrator.criteria import (
    CGI_HARD_CAP,
    MASTER_CRITERIA,
    QUALITY_THRESHOLD,
)
from agents.orchestrator.state import PipelineState, QAStatus, QAVerdict

_LOG = logging.getLogger(__name__)

_CGI_FLAW_NEEDLES = (
    "3d render", "cgi", "video game", "plastic", "unreal engine",
    "cartoon", "corrupt render", "broken render", "floating artifact",
)

# Avoid a second Vision LLM call for the same still within one process.
_VERDICT_CACHE: dict[tuple[str, int, int], QAVerdict] = {}
_VERDICT_CACHE_LOCK = threading.Lock()
_VERDICT_CACHE_MAX = 64


class VisualQAAgent(BaseAgent):
    """
    End-of-pipeline critic.

    Accepts an image path + master criteria, runs a Vision LLM, and returns
    ``is_approved`` + ``feedback_string`` on ``state.qa_verdict``.
    """

    name = "visual_qa"

    def __init__(
        self,
        *,
        master_criteria: Sequence[str] | None = None,
        quality_threshold: float | None = None,
        channel_name: str = "ancient_knowledge",
    ) -> None:
        self.master_criteria = tuple(master_criteria or MASTER_CRITERIA)
        self.quality_threshold = float(
            quality_threshold if quality_threshold is not None else QUALITY_THRESHOLD
        )
        self.channel_name = channel_name

    def run(self, state: PipelineState) -> PipelineState:
        state.last_node = self.name
        self._bind_cost_tracker(state)
        image_path = state.latest_image_path
        if not image_path or not Path(image_path).is_file():
            verdict = QAVerdict(
                is_approved=False,
                feedback_string=(
                    "No generated image to judge. Regenerate with a cinematic "
                    "atmospheric still of the scene — ruins, artefacts, or "
                    "structures are all valid."
                ),
                score=0.0,
                flaws=["missing_image"],
                criteria_checked=list(self.master_criteria),
            )
            return self._apply(state, verdict)

        try:
            verdict = self.evaluate(image_path, self.master_criteria)
        except Exception as exc:  # noqa: BLE001
            _LOG.error("VisualQA evaluate failed (%s)", exc, exc_info=True)
            state.record_error(self.name, str(exc), attempt=state.retry_count)
            verdict = QAVerdict(
                is_approved=False,
                feedback_string=(
                    f"Critic error ({type(exc).__name__}). Rewrite with a clearer "
                    "subject and stronger atmosphere — avoid corrupt or toy-CGI stills."
                ),
                score=0.0,
                flaws=["critic_exception"],
                criteria_checked=list(self.master_criteria),
            )
        return self._apply(state, verdict)

    def evaluate(
        self,
        image_path: str | Path,
        master_criteria: Sequence[str] | None = None,
    ) -> QAVerdict:
        """
        Vision LLM judgement.

        Prefers the existing ``VisualQA_Agent.visual_critic.evaluate_image``
        (channel DNA + Gemini Vision). Falls back to a direct Gemini Vision
        call against ``master_criteria`` if the critic package is unavailable.
        """
        criteria = tuple(master_criteria or self.master_criteria)
        path = Path(image_path)
        cache_key: tuple[str, int, int] | None = None
        try:
            st = path.stat()
            cache_key = (str(path.resolve()), int(st.st_size), int(st.st_mtime_ns))
        except OSError:
            cache_key = None
        if cache_key is not None:
            with _VERDICT_CACHE_LOCK:
                cached = _VERDICT_CACHE.get(cache_key)
            if cached is not None:
                _LOG.info(
                    "VisualQA cache hit | %s | score=%.2f approved=%s flaws=%s",
                    path.name, cached.score, cached.is_approved, cached.flaws,
                )
                return cached

        wrapped = self._evaluate_via_visualqa_package(path, criteria)
        verdict = wrapped if wrapped is not None else self._evaluate_direct_vision(path, criteria)
        if cache_key is not None:
            with _VERDICT_CACHE_LOCK:
                if len(_VERDICT_CACHE) >= _VERDICT_CACHE_MAX:
                    _VERDICT_CACHE.pop(next(iter(_VERDICT_CACHE)))
                _VERDICT_CACHE[cache_key] = verdict
        return verdict

    def _evaluate_via_visualqa_package(
        self, path: Path, criteria: tuple[str, ...],
    ) -> QAVerdict | None:
        try:
            from quality.VisualQA_Agent.channel_rag import (
                get_channel_rules,
                seed_default_channels,
                set_channel_context,
            )
            from quality.VisualQA_Agent.visual_critic import evaluate_image
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("VisualQA_Agent import skipped (%s) — direct vision.", exc)
            return None

        try:
            seed_default_channels(force=False)
            set_channel_context(self.channel_name)
            rules = dict(get_channel_rules(self.channel_name))
        except Exception:
            rules = {
                "forbidden_tokens": ["CGI", "3D render", "illustration", "flat lighting"],
                "mandatory_elements": list(criteria),
                "lighting_style": "dynamic cinematic chiaroscuro, volumetric shafts",
                "target_audience_rules": "ancient mystery documentary stills",
                "critic_profile": "ancient_mystery",
            }
        else:
            rules["critic_profile"] = rules.get("critic_profile") or "ancient_mystery"
            extra = list(rules.get("mandatory_elements") or [])
            for item in criteria:
                if item not in extra:
                    extra.append(item)
            rules["mandatory_elements"] = extra

        critic = evaluate_image(
            path,
            channel_name=self.channel_name,
            rules=rules,
            quality_threshold=self.quality_threshold,
        )
        feedback = (getattr(critic, "fix_instructions", "") or "").strip()
        if not critic.passed and not feedback:
            feedback = "; ".join(critic.flaws) or (
                "Rejected for a severe visual flaw (anatomy, floating artifact, "
                "or corrupt render). Rewrite the subject clearly; fantasy ruins are allowed."
            )
        return QAVerdict(
            is_approved=bool(critic.passed),
            feedback_string=feedback,
            score=float(critic.score),
            flaws=list(critic.flaws or []),
            criteria_checked=list(criteria),
        )

    def _evaluate_direct_vision(
        self, path: Path, criteria: tuple[str, ...],
    ) -> QAVerdict:
        """Self-contained Gemini Vision call — used if VisualQA_Agent is unavailable."""
        import config as app_config
        from google import genai
        from google.genai import types

        key = getattr(app_config, "GEMINI_API_KEY", None)
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing — VisualQAAgent cannot run.")

        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/png"
        criteria_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
        instruction = f"""
You are the Art Director for the Ancient Knowledge documentary channel.

Judge IMAGE 1 PIXELS against these MASTER CRITERIA:
{criteria_block}

ACCEPT: clean high-atmosphere FLUX renders, including digitally polished stills
and fantasy/surreal ancient ruins, artefacts, or structures. Do NOT require
35mm film grain. Do NOT require a real-world monument unless the prompt named one.

HARD FAIL (score must be < {self.quality_threshold}) only for severe visual flaws:
- severe anatomical deformations
- unreadable floating artifacts
- corrupt or broken renders

PASS if score >= {self.quality_threshold} AND none of those severe flaws are present.

Return JSON:
  score (float 0-10),
  passed (boolean),
  flaws (list of strings),
  fix_instructions (actionable camera/lighting rewrite for FLUX; empty if passed).
""".strip()

        from core.google_guardrail import guarded_generate_content, make_guarded_gemini_client

        client = make_guarded_gemini_client(key)
        from quality.VisualQA_Agent import config as vqa_config

        model_id = vqa_config.GEMINI_CRITIC_MODEL
        if not model_id.startswith("models/"):
            model_id = f"models/{model_id}"
        # Downscale + JPEG so the direct path matches the capped payload too.
        try:
            from quality.VisualQA_Agent.visual_critic import _bytes_to_jpeg_part

            part = _bytes_to_jpeg_part(path.read_bytes())
        except Exception:  # noqa: BLE001
            part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
        contents = [
            "IMAGE 1 — CANDIDATE TO JUDGE:",
            part,
            instruction,
        ]
        response = guarded_generate_content(
            client,
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.15,
            ),
            source="visual_qa._evaluate_direct_vision",
        )
        text = (getattr(response, "text", None) or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        data = json.loads(text)
        score = round(float(data.get("score") or 0.0), 2)
        flaws = [str(x) for x in (data.get("flaws") or [])]
        passed = bool(data.get("passed")) and score >= self.quality_threshold
        blob = " ".join(flaws).lower() + " " + str(data.get("fix_instructions") or "").lower()
        if any(n in blob for n in _CGI_FLAW_NEEDLES) and score >= CGI_HARD_CAP:
            score = round(CGI_HARD_CAP - 0.5, 2)
            passed = False
        feedback = str(data.get("fix_instructions") or "").strip()
        if not passed and not feedback:
            feedback = "; ".join(flaws) or "Change camera angle and lighting; kill CGI look."
        return QAVerdict(
            is_approved=passed,
            feedback_string=feedback,
            score=score,
            flaws=flaws,
            criteria_checked=list(criteria),
        )

    @staticmethod
    def _apply(state: PipelineState, verdict: QAVerdict) -> PipelineState:
        state.qa_verdict = verdict
        state.qa_feedback = "" if verdict.is_approved else verdict.feedback_string
        state.qa_status = QAStatus.APPROVED if verdict.is_approved else QAStatus.REJECTED
        _LOG.info(
            "VisualQA | #%s | approved=%s score=%.2f | flaws=%s | %s",
            state.post_index + 1,
            verdict.is_approved,
            verdict.score,
            verdict.flaws or ["(none)"],
            (verdict.feedback_string or "")[:160],
        )
        # The critic is a Gemini Vision round-trip; charge a conservative text
        # call so rejection loops still show up in the cost ledger.
        tracker = getattr(state, "cost_tracker", None)
        if tracker is not None:
            try:
                tracker.track_text("text_gemini_flash", char_count=600)
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("VisualQA cost skipped (%s)", exc)
        return state
