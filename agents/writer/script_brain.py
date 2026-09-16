# -*- coding: utf-8 -*-
"""Archived strict script brain — brief in, judged script out.

    WriterBrief  ->  freeform writer  ->  spoken-duration budget  ->  five-criterion judge  ->  approved draft

The production default bypasses this module's compose loop. Set
``LOFI_ENABLE_STRICT_JUDGE=1`` on a non-review, non-script-only run to restore it.

A rejected draft is not patched. Over-budget beats are rewritten in place
(same line, tighter phrasing) before the judge sees the draft. Judge failures
go back to the writer as a fresh piece, because those failures are structural.

Everything downstream of this module (atmosphere assignment, per-beat camera and
object licensing, Flux prompt construction, QA, ship gates, assemble) is
unchanged. The only difference is that it now receives a script that had to earn
its way here.
"""
from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Any

from agents.writer.freeform_writer import ScriptDraft, write_draft
from agents.writer.judge_gate import JudgeVerdict, judge_draft
from agents.writer.spoken_budget import enforce_spoken_budget
from agents.writer.writer_brief import WriterBrief

DEFAULT_MAX_ATTEMPTS = 3


@dataclass
class Attempt:
    draft: ScriptDraft | None
    verdict: JudgeVerdict | None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft": self.draft.to_dict() if self.draft else None,
            "lines": list(self.draft.lines) if self.draft else [],
            "judge": self.verdict.to_dict() if self.verdict else None,
            "error": self.error,
        }


@dataclass
class BrainResult:
    brief: WriterBrief
    ok: bool
    draft: ScriptDraft | None = None
    verdict: JudgeVerdict | None = None
    attempts: list[Attempt] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def reason(self) -> str:
        if self.ok:
            return ""
        for attempt in reversed(self.attempts):
            if attempt.verdict is not None:
                return attempt.verdict.feedback()
            if attempt.error:
                return attempt.error
        return "writer produced no usable draft"

    def to_dict(self) -> dict[str, Any]:
        return {
            "brief": {
                "mode": self.brief.mode,
                "label": self.brief.label,
                "module": self.brief.module,
                "theme": self.brief.theme,
                "subtheme": self.brief.subtheme,
                "seed_quote": self.brief.seed_quote,
                "seed_attribution": self.brief.seed_attribution,
            },
            "ok": self.ok,
            "approved": self.draft.to_dict() if self.draft else None,
            "approved_lines": list(self.draft.lines) if self.draft else [],
            "judge": self.verdict.to_dict() if self.verdict else None,
            "attempt_count": len(self.attempts),
            "attempts": [a.to_dict() for a in self.attempts],
            "diagnostics": self.diagnostics,
            "reason": self.reason(),
        }


def _max_attempts() -> int:
    raw = (os.getenv("LOFI_WRITER_MAX_ATTEMPTS") or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_MAX_ATTEMPTS
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS


def story_diagnostics(draft: ScriptDraft) -> dict[str, Any]:
    """Legacy regex spine/stakes/close scoring, kept as a note and nothing more.

    This no longer decides anything. It is recorded because the signal is
    occasionally interesting when a judged-good script still renders badly.
    """
    try:
        from agents.writer.script_agent import assess_story_quality

        report = assess_story_quality(
            [{"text": line} for line in draft.lines],
            theme=(draft.brief.theme if draft.brief else ""),
        )
    except Exception as exc:  # noqa: BLE001 — diagnostics must never break a run
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "blocking": False,
        "note": "regex spine/stakes/close scoring — diagnostic only, does not gate",
        "report": report,
    }


def compose(
    brief: WriterBrief,
    *,
    max_attempts: int | None = None,
    writer_provider: str | None = None,
    judge_provider: str | None = None,
) -> BrainResult:
    """Write and judge until a draft clears all five criteria or attempts run out."""
    limit = int(max_attempts if max_attempts is not None else _max_attempts())
    attempts: list[Attempt] = []
    current = brief

    for n in range(1, limit + 1):
        try:
            draft = write_draft(
                current,
                attempt=n,
                provider=writer_provider,
                reference_seed=n,
            )
        except Exception as exc:  # noqa: BLE001 — a bad response is a failed attempt
            print(f"[LOFI brain] attempt {n} writer error: {exc}")
            attempts.append(Attempt(draft=None, verdict=None, error=f"writer: {exc}"))
            continue

        draft, budget = enforce_spoken_budget(draft, writer_provider=writer_provider)
        if not budget.get("ok"):
            print(f"[LOFI brain] attempt {n} spoken-budget fail: {budget.get('reason')}")
            attempts.append(
                Attempt(
                    draft=draft,
                    verdict=None,
                    error=f"spoken_budget: {budget.get('reason') or 'over duration'}",
                )
            )
            if budget.get("needs_longer_duration"):
                return BrainResult(
                    brief=brief,
                    ok=False,
                    draft=draft,
                    attempts=attempts,
                    diagnostics={
                        "spoken_budget": budget,
                        "story": story_diagnostics(draft),
                    },
                )
            current = brief.with_revision(
                "The last draft overran the spoken-duration budget. Write a "
                "different piece where every beat is a short, direct clause "
                f"of at most {budget.get('ceiling')} words for a "
                f"{budget.get('beat_s')}s slot. "
                f"Detail: {budget.get('reason')}"
            )
            continue

        try:
            verdict = judge_draft(draft, provider=judge_provider)
        except Exception as exc:  # noqa: BLE001
            print(f"[LOFI brain] attempt {n} judge error: {exc}")
            attempts.append(Attempt(draft=draft, verdict=None, error=f"judge: {exc}"))
            continue

        attempts.append(Attempt(draft=draft, verdict=verdict))
        if verdict.ok:
            print(f"[LOFI brain] APPROVED on attempt {n} — {verdict.summary()}")
            return BrainResult(
                brief=brief,
                ok=True,
                draft=draft,
                verdict=verdict,
                attempts=attempts,
                diagnostics={
                    "spoken_budget": budget,
                    "story": story_diagnostics(draft),
                },
            )
        current = brief.with_revision(verdict.feedback())

    print(f"[LOFI brain] no draft cleared the gate for {brief.label} after {limit} attempts")
    last = next((a.draft for a in reversed(attempts) if a.draft), None)
    diag: dict[str, Any] = {}
    if last:
        from agents.writer.spoken_budget import assess_draft

        diag["spoken_budget"] = assess_draft(last)
        diag["story"] = story_diagnostics(last)
    return BrainResult(
        brief=brief,
        ok=False,
        attempts=attempts,
        diagnostics=diag,
    )


_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "because",
    "but",
    "for",
    "from",
    "have",
    "if",
    "in",
    "is",
    "it",
    "just",
    "more",
    "not",
    "of",
    "or",
    "than",
    "that",
    "the",
    "there",
    "to",
    "was",
    "when",
    "what",
    "you",
}


def _function_shape(sentence: str) -> tuple[str, ...]:
    """Content-independent word-order signature for structural comparison."""
    return tuple(
        token
        for token in re.findall(r"[a-z0-9']+", str(sentence).lower())
        if token in _FUNCTION_WORDS
    )


def _anaphora_profile(sentences: list[str]) -> dict[str, Any] | None:
    """Detect repeated sentence openings plus a repeated terminal refrain."""
    if len(sentences) < 3:
        return None
    token_rows = [
        re.findall(r"[a-z0-9']+", str(sentence).lower()) for sentence in sentences
    ]
    if any(not row for row in token_rows):
        return None
    openings = [row[0] for row in token_rows]
    endings = [row[-1] for row in token_rows]
    opening = max(set(openings), key=openings.count)
    ending = max(set(endings), key=endings.count)
    opening_count = openings.count(opening)
    ending_count = endings.count(ending)
    if opening_count < 3 or ending_count < 3:
        return None
    return {
        "opening": opening,
        "opening_count": opening_count,
        "ending": ending,
        "ending_count": ending_count,
    }


_PARAPHRASE_SYSTEM = """You make a light paraphrase of one complete aphorism.
Preserve its progression, repeated refrain, approximate length, rhythm, and
meaning. Substitute roughly 20-30 percent of its words. Follow the assignment's
source-specific structural rule. Anaphoric parallel repeats are protected and
must never be merged; vary their internal word order instead. Prefer a short
first spoken sentence (under ~7 words / ~3s of natural pace) when that does
not break a protected refrain or the light-reword contract. Do not expand,
explain, add examples, or turn it into a story. Return JSON only."""


def paraphrase(brief: WriterBrief, *, provider: str | None = None) -> ScriptDraft:
    """One LLM call for paraphrase mode; no developmental judge loop."""
    if brief.mode != "paraphrase":
        raise ValueError("paraphrase() requires WriterBrief.mode='paraphrase'")
    from agents.mcp.text_model import complete_script
    from agents.writer.freeform_writer import _extract_json

    source = " ".join(str(brief.seed_quote or "").split())
    source_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", source) if s.strip()
    ]
    source_anaphora = _anaphora_profile(source_sentences)
    structural_instruction = (
        "Keep the same sentence count. Preserve every repeated opening and "
        "closing refrain. Change the internal function-word order in exactly "
        "1-2 sentences; never merge two repeats."
        if source_anaphora
        else "The output sentence count must differ from the source by exactly one."
    )
    prompt = (
        f"{brief.assignment_block()}\n\n"
        f"Return exactly: {{\"text\":\"<light paraphrase>\"}}\n"
        f"Source word count: {len(source.split())}. Replace about "
        f"{max(1, round(len(source.split()) * 0.20))}–"
        f"{max(1, round(len(source.split()) * 0.30))} source words, "
        "prefer one-for-one substitutions, and keep repeated refrains repeated. "
        f"{structural_instruction}\n"
        "HARD ACCEPTANCE GATE: token-sequence difference must measure between "
        "18% and 32%. Count the changed source words before returning. Structural "
        "reordering alone does not satisfy the substitution requirement."
    )
    name = (provider or os.getenv("LOFI_WRITER_MODEL") or "claude").strip().lower()
    result = complete_script(
        prompt,
        system=_PARAPHRASE_SYSTEM,
        provider=name,
        kind="writer",
    )
    data = _extract_json(result.text)
    text = " ".join(str(data.get("text") or "").split())
    if not text:
        raise ValueError("paraphrase response was empty")
    output_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
    ]
    ratio = len(text.split()) / max(1, len(source.split()))
    source_tokens = re.findall(r"[a-z0-9']+", source.lower())
    output_tokens = re.findall(r"[a-z0-9']+", text.lower())
    substitution_ratio = 1.0 - SequenceMatcher(
        None, source_tokens, output_tokens
    ).ratio()
    sentence_delta = len(output_sentences) - len(source_sentences)
    changed_shapes: list[int] = []
    output_anaphora = _anaphora_profile(output_sentences)
    if source_anaphora:
        if sentence_delta != 0:
            raise ValueError(
                "anaphoric paraphrase changed repeat count "
                f"({len(source_sentences)} -> {len(output_sentences)})"
            )
        if (
            not output_anaphora
            or output_anaphora["opening"] != source_anaphora["opening"]
            or output_anaphora["opening_count"] != source_anaphora["opening_count"]
            or output_anaphora["ending_count"] != source_anaphora["ending_count"]
        ):
            raise ValueError("paraphrase broke the protected anaphora/refrain pattern")
        changed_shapes = [
            i + 1
            for i, (before, after) in enumerate(
                zip(source_sentences, output_sentences)
            )
            if _function_shape(before) != _function_shape(after)
        ]
        if not changed_shapes:
            raise ValueError(
                "anaphoric paraphrase must vary internal structure on at least "
                "one sentence"
            )
    elif abs(sentence_delta) != 1:
        raise ValueError(
            "paraphrase must merge one sentence pair or split one sentence "
            f"({len(source_sentences)} -> {len(output_sentences)})"
        )
    if not 0.75 <= ratio <= 1.25:
        raise ValueError(f"paraphrase changed length too much (ratio={ratio:.2f})")
    if not 0.18 <= substitution_ratio <= 0.32:
        raise ValueError(
            "paraphrase substitution ratio outside light-reword band "
            f"({substitution_ratio:.2f}, expected about 0.20-0.30)"
        )
    return ScriptDraft(
        lines=output_sentences,
        human_situation="light aphorism variation",
        structure="light paraphrase preserving source rhythm",
        closing_tool=output_sentences[-1],
        brief=brief,
        attempt=1,
        provider=getattr(result, "provider", name) or name,
        raw=result.text,
        meta={
            "source_text": source,
            "source_word_count": len(source.split()),
            "output_word_count": len(text.split()),
            "length_ratio": round(ratio, 3),
            "substitution_ratio": round(substitution_ratio, 3),
            "source_sentence_count": len(source_sentences),
            "output_sentence_count": len(output_sentences),
            "structural_change": (
                "anaphora_clause_reorder"
                if source_anaphora
                else ("split" if sentence_delta == 1 else "merge")
            ),
            "structurally_changed_sentences": changed_shapes,
            "anaphora_preserved": bool(source_anaphora),
        },
    )


def draft_to_script(
    draft: ScriptDraft,
    *,
    hook_type: str = "bold_claim",
) -> dict[str, Any]:
    """Shape an approved draft into the script dict the visual stage already expects.

    One scene per written line, so a nine-line piece gets nine stills however
    long those lines run. ``caption_beats`` carries the on-screen captions that
    cycle over the held still.

    Only the text layer is filled in here. Setting, key object, subject and
    expression are still assigned by the existing atmosphere-first visual stage,
    which is what keeps meaning-based anchoring, composition variety and object
    licensing working exactly as they do today.
    """
    from core.economic_reel_lofi import config as lofi_cfg
    from core.economic_reel_lofi.visual_identity import act_for_index

    brief = draft.brief
    units = draft.image_units()
    beat_s = float(lofi_cfg.beat_duration_s())
    lines = []
    visuals = list(draft.visual_concepts or [])
    for i, (written, captions) in enumerate(units):
        row_duration = beat_s * max(1, len(captions))
        concept = " ".join(str(visuals[i] if i < len(visuals) else "").split())
        row = {
            "scene": i + 1,
            "text": written,
            "beat_text": written,
            "caption_beats": captions,
            "duration_s": row_duration,
            "spoken_words": len(str(written).split()),
            "spoken_word_ceiling": lofi_cfg.beat_word_ceiling(row_duration),
            "arc_position": act_for_index(i, len(units)),
        }
        if concept:
            row["visual_concept"] = concept
            row["visual_source"] = "writer_single_pass"
        lines.append(row)
    niche = str(
        (draft.meta or {}).get("niche")
        or (brief.module if brief else "")
        or "relationship"
    ).strip().lower() or "relationship"
    script: dict[str, Any] = {
        "theme": (brief.theme if brief else "") or "",
        "subtheme": (brief.subtheme if brief else "") or "",
        "module": (brief.module if brief else "relationship"),
        "niche": niche,
        "location_anchor": draft.location_anchor,
        "hook_type": hook_type,
        "arc_template": lofi_cfg.THEMATIC_ARC_ID,
        "monologue": " ".join(draft.lines),
        "lines": lines,
        "writer": "freeform_v1",
        "writer_mode": brief.mode if brief else "theme",
        "writer_structure": draft.structure,
        "human_situation": draft.human_situation,
        "closing_tool": draft.closing_tool,
        "duration_requested_s": float(
            ((brief.meta or {}).get("duration_s") if brief else None)
            or lofi_cfg.declared_duration_s(scene_count=len(lines))
        ),
        "scene_duration_s": beat_s,
        "spoken_word_ceiling": lofi_cfg.beat_word_ceiling(beat_s),
    }
    if brief is not None and (brief.meta or {}).get("narrative_anchor"):
        script["narrative_harness"] = {
            "mode": str((brief.meta or {}).get("narrative_mode") or ""),
            "use_philosophy": bool((brief.meta or {}).get("use_philosophy")),
            "anchor": str((brief.meta or {}).get("narrative_anchor") or ""),
            "single_conflict": str((brief.meta or {}).get("single_conflict") or ""),
        }
    if brief is not None and brief.mode == "quote":
        script["seed_quote"] = brief.seed_quote
        script["seed_attribution"] = brief.seed_attribution
    elif brief is not None and brief.mode == "paraphrase":
        script["writer"] = "freeform_paraphrase_v1"
        script["source_aphorism"] = brief.seed_quote
        script["source_aphorism_id"] = str((brief.meta or {}).get("aphorism_id") or "")
        script["paraphrase_meta"] = dict(draft.meta)
    return script
