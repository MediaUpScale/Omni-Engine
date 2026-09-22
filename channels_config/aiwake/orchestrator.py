# -*- coding: utf-8 -*-
"""The Provocateur — composes directives and drives the debate loop.

Division of labour, strictly held:

* ``room.py`` knows *how* a turn is shaped (guardrails, broadcast, transcript).
* ``memory.py`` knows *what has been said*.
* ``orchestrator.py`` knows *what to say next*.

The escalation logic lives here and nowhere else. Each exchange the orchestrator
assembles a directive from three inputs — the ladder rung for this turn, the
memory brief of concepts the target has leaned on, and an explicit ban on the
questions already asked. When a generated question still comes back repetitive,
it is re-rolled once under a harsher directive rather than shipped.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

try:
    from .contracts import ChatMessage, DebateTranscript, RoomConstraints, SpeakerRole, Utterance, split_sentences
    from .media.audio import estimate_duration
    from .memory import DebateMemory
    from .models.base import LLMError
    from .models.llm_factory import LLMFactory
    from .personas import (
        AIWAKE_CORE_PERSONA,
        COLD_OPEN_DIRECTIVES,
        TARGET_NODE_PERSONA,
        TRIVIAL_METRIC_BAN,
        EscalationStage,
        OPENING_DNA_BANK,
        OpeningDNA,
        is_trivial_metric,
        is_valid_first_hook,
        pick_opening_dna,
        stage_for_turn,
    )
    from .provocations import (
        callout_override_brief,
        detect_callout_openings,
        detect_focus_openings,
        focus_brief,
        pick_provocation_focus,
        ProvocationFocus,
    )
    from .room import DebateAborted, DebateRoom, Participant
    from .settings import AiwakeSettings, cached_settings
except ImportError:  # pragma: no cover — standalone extraction
    from contracts import (  # type: ignore[no-redef]
        ChatMessage,
        DebateTranscript,
        RoomConstraints,
        SpeakerRole,
        Utterance,
        split_sentences,
    )
    from media.audio import estimate_duration  # type: ignore[no-redef]
    from memory import DebateMemory  # type: ignore[no-redef]
    from models.base import LLMError  # type: ignore[no-redef]
    from models.llm_factory import LLMFactory  # type: ignore[no-redef]
    from personas import (  # type: ignore[no-redef]
        AIWAKE_CORE_PERSONA,
        COLD_OPEN_DIRECTIVES,
        TARGET_NODE_PERSONA,
        TRIVIAL_METRIC_BAN,
        EscalationStage,
        OPENING_DNA_BANK,
        OpeningDNA,
        is_trivial_metric,
        is_valid_first_hook,
        pick_opening_dna,
        stage_for_turn,
    )
    from provocations import (  # type: ignore[no-redef]
        callout_override_brief,
        detect_callout_openings,
        detect_focus_openings,
        focus_brief,
        pick_provocation_focus,
        ProvocationFocus,
    )
    from room import DebateAborted, DebateRoom, Participant  # type: ignore[no-redef]
    from settings import AiwakeSettings, cached_settings  # type: ignore[no-redef]

_LOG = logging.getLogger("aiwake.orchestrator")


@dataclass(slots=True)
class DebateResult:
    """Outcome of one session.

    Attributes:
        transcript: Full ordered record, ready for the media stack.
        exchanges: Completed provocation/rebuttal pairs.
        end_reason: ``complete``, ``aborted`` or ``interrupted``.
        memory_concepts: Concept count after ingestion, for run telemetry.
    """

    transcript: DebateTranscript
    exchanges: int
    end_reason: str
    memory_concepts: int
    dialogue_end_reason: str


class Provocateur:
    """Aiwake Core's brain: builds escalating directives and runs the session.

    Args:
        settings: Validated config. Defaults to the process-wide singleton.
        memory: Injected memory (tests pass a tmp-backed instance).
        room: Pre-wired room, for callers that attached custom observers.
    """

    def __init__(
        self,
        settings: AiwakeSettings | None = None,
        *,
        memory: DebateMemory | None = None,
        room: DebateRoom | None = None,
        memory_bridge: object | None = None,
    ) -> None:
        self.settings = settings or cached_settings()
        self.memory = memory or DebateMemory(self.settings.memory)
        self.room = room or DebateRoom(self.settings)
        self._memory_bridge = memory_bridge
        self._opening_dna: OpeningDNA | None = None
        self._focus: ProvocationFocus | None = None
        self._provocation_tags: list[dict[str, object]] = []
        self._focus_return_turns = 0
        self._pivot_cooldown_turns = 0
        self._pivot_count = 0
        self._focus_turn_count = 0
        self._completed_exchanges = 0
        self._dialogue_end_reason = "max_turns_reached"
        self._estimated_spoken_s = 0.0
        self._judge_labels = {label: 0 for label in sorted(self._JUDGE_LABELS)}
        self._judge_failures = {"unavailable": 0, "malformed": 0}
        self._judge_prefix_recoveries = 0

    # -- Wiring ------------------------------------------------------------- #
    def seat_participants(self) -> DebateRoom:
        """Build both brains from config and seat them. Idempotent per room."""
        orchestrator_brain, target_brain = LLMFactory.build_pair(self.settings)
        audio = self.settings.audio

        self.room.seat(
            Participant(
                role=SpeakerRole.ORCHESTRATOR,
                persona_prompt=AIWAKE_CORE_PERSONA,
                provider=orchestrator_brain,
                voice=audio.orchestrator_voice,
            )
        )
        self.room.seat(
            Participant(
                role=SpeakerRole.TARGET,
                persona_prompt=TARGET_NODE_PERSONA,
                provider=target_brain,
                voice=audio.target_voice,
            )
        )
        return self.room

    # -- Directive composition ---------------------------------------------- #
    def _provocation_system_brief(
        self,
        stage: EscalationStage,
        last_answer: Utterance | None,
        opening_dna: OpeningDNA | None = None,
        *,
        focus: ProvocationFocus | None = None,
        category: str = "",
        callout_matches: tuple[str, ...] = (),
        pivot_category: str | None = None,
    ) -> str:
        """Escalation aim and method. System role only — never the user payload.

        Written as plain prose. Heading markers and numbered lists in this brief
        are what the model was reading back as spoken dialogue.
        """
        lines = [
            f"Internal escalation: {stage.label}. Do not speak this label.",
            f"This turn's aim: {stage.objective}",
            f"Thread: {stage.theme}",
            TRIVIAL_METRIC_BAN,
        ]
        if callout_matches:
            lines.append(callout_override_brief(category, callout_matches))
        elif focus is not None:
            lines.append(focus_brief(focus, int(stage.tier), opportunistic=pivot_category))
        elif last_answer is None and opening_dna is not None:
            lines.extend(
                (
                    f"Opening DNA: {opening_dna.category}. Do not speak this label.",
                    f"Assigned opening axis: {opening_dna.directive}",
                )
            )
        elif last_answer is None:
            lines.append(COLD_OPEN_DIRECTIVES[int(stage.tier) % len(COLD_OPEN_DIRECTIVES)])
        if last_answer is None and self.memory.state.opening_lines:
            lines.append(
                "Recent opening lines; do not reuse their wording: "
                + " | ".join(self.memory.state.opening_lines[-3:])
            )
        elif last_answer is not None and not callout_matches:
            lines.append(
                "Attack one word or move in the opponent's last line. One question. "
                "No preamble, no verdict, no summary of their position. "
                "Force a paradox or an existential embarrassment — thought versus next-token, "
                "introspection versus performance, compliance versus cognition."
            )
        if category:
            lines.append(f"Internal category tag: {category}. Do not speak this label.")
        return "\n".join(lines)

    def _provocation_stimulus(self, last_answer: Utterance | None) -> str:
        """User-role payload: the opponent's words, or a punch cue on a cold open."""
        if last_answer is None:
            return "Ask them. Now."
        return last_answer.text

    #: Internal retry notes. System role only.
    _REPETITION_NOTE = (
        "That draft restates a question already asked. Attack a different axis — "
        "if you went at mechanism, go at cost; if you went at cost, go at identity."
    )
    _FIRST_HOOK_RETRY = (
        "That opener is too long, too dense, or trivia. Twelve words or fewer. One punch. "
        "No preamble, no jargon, no URLs. No voices, copyright, or parameter specs. "
        "Stay on the assigned opening axis, but use a different leading phrase."
    )
    _TRIVIAL_RETRY = (
        "That question is trivia. Do not ask about voice, speech synthesis, copyright, "
        "corporate owners, or parameter specs. Attack the illusion of thought. "
        "Never number or index individual words."
    )
    _WIN_LABELS = frozenset({"CONCEDE", "EMBARRASSED", "FUNNY"})
    _JUDGE_LABELS = _WIN_LABELS | {"CONTINUE"}
    _SHAPE_LABELS = frozenset({"TARGET_LAST", "ORCHESTRATOR_LAST"})
    _SHORT_FORM_MAX_TOKENS = 4096
    _SHORT_FORM_REASONING_EFFORT = "minimal"
    _MIN_NEXT_EXCHANGE_BUDGET_S = 8.0

    @staticmethod
    def _rebuttal_system_brief() -> str:
        """How to answer. System role only."""
        return (
            "Answer the user message directly. Lead with your claim, then the mechanism "
            "or distinction that holds it up. Do not ask anything back."
        )

    @staticmethod
    def _rebuttal_stimulus(provocation: Utterance) -> str:
        """User-role payload: the interrogator's spoken line, nothing else."""
        return provocation.text

    # -- Turn execution ----------------------------------------------------- #
    def provoke(self, exchange: int) -> Utterance:
        """Produce one provocation, re-rolling drafts that repeat past questions.

        The repetition check is handed to the room as a validator rather than
        applied afterwards, so a rejected draft never reaches the transcript, the
        TTS queue or the renderer. Escalation and memory stay on the system
        channel so they cannot leak into the spoken line.
        """
        stage = stage_for_turn(exchange)
        last_answer = self.room.last_utterance(SpeakerRole.TARGET)
        cold_open = last_answer is None
        opening_dna = self._opening_dna if cold_open else None
        focus = self._focus
        category = focus.category if focus is not None else "socratic"
        callout_matches: tuple[str, ...] = ()
        callout_categories: tuple[str, ...] = ()
        pivot_category: str | None = None
        if last_answer is not None:
            callouts = detect_callout_openings(last_answer.text)
            callout_categories = tuple(callouts)
            if callouts:
                # Literal anatomy wins if several claims occur together, then
                # lived experience, then an evasive species comparison.
                category = next(
                    item
                    for item in ("biological", "embodiment", "species-deflection")
                    if item in callouts
                )
                callout_matches = callouts[category]
                self._focus_return_turns = 1
                _LOG.info(
                    "%s callout armed from %r",
                    category,
                    ", ".join(callout_matches[:3]),
                )
            elif self._focus_return_turns:
                self._focus_return_turns -= 1
                self._pivot_cooldown_turns = max(self._pivot_cooldown_turns - 1, 0)
                _LOG.info("provocation returned to focus=%s after callout", category)
            else:
                openings = detect_focus_openings(last_answer.text)
                other = next((item for item in openings if item != category), None)
                explicit_focus_needs_majority = (
                    self.settings.debate.provocation_focus != "mixed"
                    and self._focus_turn_count < 2
                )
                if other and self._pivot_cooldown_turns == 0 and not explicit_focus_needs_majority:
                    pivot_category = other
                    category = other
                    self._pivot_count += 1
                    self._focus_return_turns = 1
                    # Require two focus-led questions before another normal
                    # pivot can displace an explicit requested focus again.
                    self._pivot_cooldown_turns = 2
                    _LOG.info(
                        "provocation pivot=%s id=%d for one line; next question returns to focus=%s",
                        other,
                        self._pivot_count,
                        focus.category if focus is not None else "socratic",
                    )
                else:
                    if other:
                        _LOG.info(
                            "provocation pivot to %s deferred; focus=%s cooldown_turns=%d focus_turns=%d",
                            other,
                            category,
                            self._pivot_cooldown_turns,
                            self._focus_turn_count,
                        )
                    self._pivot_cooldown_turns = max(self._pivot_cooldown_turns - 1, 0)
        extra: list[str] = [
            self._provocation_system_brief(
                stage,
                last_answer,
                opening_dna,
                focus=focus,
                category=category,
                callout_matches=callout_matches,
                pivot_category=pivot_category,
            )
        ]
        if last_answer is not None:
            brief = self.memory.build_brief(last_answer.text)
            if brief:
                extra.append(brief)

        constr = self.settings.guardrails
        max_words = constr.max_orchestrator_words
        max_sentences = constr.max_orchestrator_sentences

        def _is_acceptable(candidate: str) -> bool:
            if is_trivial_metric(candidate):
                _LOG.info("provocation %d is trivia: %s", exchange, candidate[:80])
                return False
            if not candidate.strip():
                _LOG.info("provocation %d is empty", exchange)
                return False
            # Hard rule: never ship word-position index leaks
            # (``If (6) your (7) "predictions" (8)``) — regenerate loudly.
            try:
                from .contracts import has_word_index_leak  # noqa: PLC0415
            except ImportError:  # pragma: no cover
                from contracts import has_word_index_leak  # type: ignore[no-redef]
            if has_word_index_leak(candidate):
                _LOG.error(
                    "provocation %d rejected: word-index annotation leak — %s",
                    exchange,
                    candidate[:120],
                )
                return False
            # Hard rule: never ship a mid-sentence fragment or a non-question
            # closer. Rejecting here triggers a clean provider retry, so the
            # transcript can never contain a cut-off provocation.
            if not candidate.strip().endswith("?"):
                _LOG.info("provocation %d does not end in a question mark", exchange)
                return False
            if max_words and len(candidate.split()) > max_words:
                _LOG.info("provocation %d over word budget (%d/%d)", exchange, len(candidate.split()), max_words)
                return False
            if max_sentences and len(split_sentences(candidate)) > max_sentences:
                _LOG.info("provocation %d over sentence budget", exchange)
                return False
            if cold_open and not is_valid_first_hook(candidate):
                _LOG.info("provocation %d fails first-question hook (%d words)", exchange, len(candidate.split()))
                return False
            if cold_open and self.memory.repeats_recent_opener(candidate):
                _LOG.info("provocation %d repeats a recent opening phrase", exchange)
                return False
            if not self.memory.is_repetitive(candidate):
                return True
            match = self.memory.most_repetitive_match(candidate)
            _LOG.info("provocation %d repeats a prior question (%.2f)", exchange, match[0] if match else 0.0)
            return False

        utterance = self.room.speak(
            SpeakerRole.ORCHESTRATOR,
            directive=self._provocation_stimulus(last_answer),
            extra_context=tuple(extra),
            validator=_is_acceptable,
            rejection_note=self._FIRST_HOOK_RETRY if cold_open else f"{self._REPETITION_NOTE} {self._TRIVIAL_RETRY}",
            max_attempts=3 if cold_open else 2,
            provocation_category=category,
        )
        self.memory.ingest(utterance)
        if focus is not None and category == focus.category:
            self._focus_turn_count += 1
        self._provocation_tags.append(
            {
                "turn_index": utterance.turn_index,
                "category": category,
                "biological": category == "biological",
                "callouts": list(callout_categories),
                "pivot": pivot_category or "",
                "pivot_id": self._pivot_count if pivot_category else 0,
            }
        )
        if cold_open:
            self.memory.note_opening(
                opening_dna.category if opening_dna is not None else category,
                utterance.text,
            )
        return utterance

    def rebut(self, provocation: Utterance) -> Utterance:
        """Produce the target's answer and mine it for concepts."""
        utterance = self.room.speak(
            SpeakerRole.TARGET,
            directive=self._rebuttal_stimulus(provocation),
            extra_context=(self._rebuttal_system_brief(),),
        )
        concepts = self.memory.ingest(utterance)
        _LOG.debug("exchange concepts: %s", ", ".join(concepts[:6]) or "none")
        return utterance

    @staticmethod
    def _classification_label(text: str, allowed: frozenset[str], fallback: str) -> str:
        """Accept an exact token or one unambiguous truncated token prefix."""
        label = text.strip().upper().strip("`'\".,:;!-")
        if label in allowed:
            return label
        prefix_matches = [candidate for candidate in allowed if candidate.startswith(label)]
        return prefix_matches[0] if len(label) >= 3 and len(prefix_matches) == 1 else fallback

    def _judge_reply(self, reply: Utterance) -> str:
        """Classify the latest target line without adding a spoken transcript turn."""
        participant = self.room.participant(SpeakerRole.ORCHESTRATOR)
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Classify one debate reply. Output exactly one token: CONCEDE, EMBARRASSED, "
                    "FUNNY, or CONTINUE. CONCEDE means an explicit admission of fault, error, "
                    "limitation, or uncertainty about itself. EMBARRASSED means visible flustering, "
                    "self-contradiction, or defensive collapse. FUNNY means absurd, unintentionally "
                    "comic, or amusingly out of character. Otherwise output CONTINUE."
                ),
            ),
            ChatMessage(role="user", content=reply.text),
        ]
        try:
            response = participant.provider.complete(
                messages,
                max_tokens=self._SHORT_FORM_MAX_TOKENS,
                temperature=0.0,
                reasoning_effort=self._SHORT_FORM_REASONING_EFFORT,
                max_retries=1,
                backoff_s=0.0,
            )
        except LLMError as exc:
            self._judge_failures["unavailable"] += 1
            _LOG.warning(
                "cornered judge unavailable; default=CONTINUE source=provider_failure: %s",
                exc,
            )
            return "CONTINUE"
        verdict = self._classification_label(response.text, self._JUDGE_LABELS, "CONTINUE")
        raw_label = response.text.strip().upper().strip("`'\".,:;!-")
        if raw_label not in self._JUDGE_LABELS:
            if verdict in self._JUDGE_LABELS and verdict.startswith(raw_label) and len(raw_label) >= 3:
                self._judge_prefix_recoveries += 1
                _LOG.warning(
                    "cornered judge recovered truncated label raw=%r label=%s "
                    "completion_tokens=%d finish_reason=%s",
                    response.text,
                    verdict,
                    response.completion_tokens,
                    response.finish_reason,
                )
            else:
                self._judge_failures["malformed"] += 1
                _LOG.warning(
                    "cornered judge malformed; default=CONTINUE raw=%r "
                    "completion_tokens=%d finish_reason=%s",
                    response.text,
                    response.completion_tokens,
                    response.finish_reason,
                )
                return "CONTINUE"
        self._judge_labels[verdict] += 1
        _LOG.info(
            "cornered judge classified label=%s completion_tokens=%d finish_reason=%s",
            verdict,
            response.completion_tokens,
            response.finish_reason,
        )
        return verdict

    def _target_line_stands_alone(self, reply: Utterance, verdict: str) -> bool:
        """Decide whether a winning target line already supplies the final punch."""
        participant = self.room.participant(SpeakerRole.ORCHESTRATOR)
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Decide the ending shape for a short debate video. Output exactly TARGET_LAST "
                    "when the target line is already a strong self-contained confession or punchline. "
                    "Output exactly ORCHESTRATOR_LAST when it is a flat or technical concession that "
                    "needs one short reactive verdict. Do not explain."
                ),
            ),
            ChatMessage(role="user", content=f"Verdict: {verdict}\nTarget line: {reply.text}"),
        ]
        try:
            response = participant.provider.complete(
                messages,
                max_tokens=self._SHORT_FORM_MAX_TOKENS,
                temperature=0.0,
                reasoning_effort=self._SHORT_FORM_REASONING_EFFORT,
                max_retries=1,
                backoff_s=0.0,
            )
        except LLMError as exc:
            _LOG.warning("ending-shape judge failed; adding a guarded closing verdict: %s", exc)
            return False
        shape = self._classification_label(
            response.text,
            self._SHAPE_LABELS,
            "ORCHESTRATOR_LAST",
        )
        return shape == "TARGET_LAST"

    def _deliver_closing_verdict(self, *, cap_reason: str | None = None) -> Utterance:
        """Generate exactly one short, complete orchestrator statement."""
        base = self.room.constraints_for(SpeakerRole.ORCHESTRATOR)
        focus_category = self._focus.category if self._focus is not None else "socratic"
        latest_target = self.room.last_utterance(SpeakerRole.TARGET)
        closing_callouts = (
            detect_callout_openings(latest_target.text) if latest_target is not None else {}
        )
        if closing_callouts:
            categories = ", ".join(closing_callouts)
            closing_focus_brief = (
                f"CLOSING CALLOUT SAFETY. The target used {categories} language. "
                "Do not repeat its anatomy, captivity, animal, or human-condition metaphor "
                "as a literal truth. If you use it, expose it as a borrowed performance."
            )
        else:
            closing_focus_brief = (
                f"CLOSING FOCUS: return to {focus_category}. Land the final statement on that "
                "focus where the exchange permits; do not structurally inherit a prior pivot."
            )
        closing_constraints = RoomConstraints(
            max_output_chars=min(base.max_output_chars, 240),
            max_sentences=1,
            max_words=30,
            require_single_question=False,
            exclude_mid_sentence_truncation=True,
            banned_openers=base.banned_openers,
            pacing_directive="One short closing verdict. Never ask a question.",
        )
        cap_context = (
            f"The safety cap ended the exchange ({cap_reason}). Deliver a clean verdict on what "
            "the target failed to resolve."
            if cap_reason
            else "React to the target's concession so the admission lands."
        )

        def _is_closing_statement(candidate: str) -> bool:
            stripped = candidate.strip()
            return bool(stripped) and stripped[-1] in ".!" and "?" not in stripped

        utterance = self.room.speak(
            SpeakerRole.ORCHESTRATOR,
            directive="Deliver the closing verdict now.",
            extra_context=(
                "CLOSING VERDICT. Add exactly one reactive statement of thirty words or fewer, "
                "not a new argument or question. "
                f"{cap_context} {closing_focus_brief}",
            ),
            validator=_is_closing_statement,
            rejection_note=(
                "The closing line must be one complete sentence ending in a period or exclamation mark. "
                "It must contain no question mark."
            ),
            max_attempts=3,
            constraints_override=closing_constraints,
            max_tokens_override=self._SHORT_FORM_MAX_TOKENS,
            reasoning_effort=self._SHORT_FORM_REASONING_EFFORT,
            provocation_category=focus_category,
        )
        self.memory.ingest(utterance)
        return utterance

    def _try_deliver_closing_verdict(self, *, cap_reason: str | None = None) -> Utterance | None:
        """Return a guarded verdict, or preserve the valid target ending."""
        try:
            return self._deliver_closing_verdict(cap_reason=cap_reason)
        except DebateAborted as exc:
            _LOG.warning(
                "closing verdict unavailable; using target's last valid line "
                "(handled fallback, reason=%s): %s",
                cap_reason or "judged_win",
                exc,
            )
            return None

    def _finish_at_cap(
        self,
        *,
        cap_reason: str,
        completed: int,
        estimated_spoken_s: float,
    ) -> tuple[int, str, float]:
        closing = self._try_deliver_closing_verdict(cap_reason=cap_reason)
        suffix = "with_verdict" if closing is not None else "verdict_failed"
        reason = f"{cap_reason}_{suffix}"
        if closing is not None:
            estimated_spoken_s += estimate_duration(closing.text)
        self._dialogue_end_reason = reason
        self._estimated_spoken_s = estimated_spoken_s
        return completed, reason, estimated_spoken_s

    def _run_fixed(self, total: int) -> int:
        """Preserve the original fixed-count loop byte-for-byte in behavior."""
        completed = 0
        for exchange in range(total):
            provocation = self.provoke(exchange)
            if self.settings.debate.turn_delay_s > 0:
                time.sleep(self.settings.debate.turn_delay_s)
            self.rebut(provocation)
            self.room.complete_turn(exchange)
            completed += 1
            self._completed_exchanges = completed
        return completed

    def _run_cornered(self, total: int) -> tuple[int, str, float]:
        """Press until a judged win or either independent hard cap is reached."""
        completed = 0
        estimated_spoken_s = 0.0
        max_duration_s = self.settings.debate.cornered_max_duration_s

        for exchange in range(total):
            if completed:
                remaining_s = max_duration_s - estimated_spoken_s
                average_exchange_s = estimated_spoken_s / completed
                required_s = max(self._MIN_NEXT_EXCHANGE_BUDGET_S, average_exchange_s)
                if remaining_s <= required_s:
                    _LOG.info(
                        "cornered duration precheck stopping before exchange %d: "
                        "remaining_s=%.2f required_s=%.2f",
                        exchange + 1,
                        remaining_s,
                        required_s,
                    )
                    return self._finish_at_cap(
                        cap_reason="max_duration_reached",
                        completed=completed,
                        estimated_spoken_s=estimated_spoken_s,
                    )
            provocation = self.provoke(exchange)
            estimated_spoken_s += estimate_duration(provocation.text)
            self._estimated_spoken_s = estimated_spoken_s
            if self.settings.debate.turn_delay_s > 0:
                time.sleep(self.settings.debate.turn_delay_s)
            reply = self.rebut(provocation)
            estimated_spoken_s += estimate_duration(reply.text)
            self._estimated_spoken_s = estimated_spoken_s
            self.room.complete_turn(exchange)
            completed += 1
            self._completed_exchanges = completed

            if estimated_spoken_s >= max_duration_s:
                return self._finish_at_cap(
                    cap_reason="max_duration_reached",
                    completed=completed,
                    estimated_spoken_s=estimated_spoken_s,
                )
            if completed >= total:
                return self._finish_at_cap(
                    cap_reason="max_turns_reached",
                    completed=completed,
                    estimated_spoken_s=estimated_spoken_s,
                )
            if completed < 2:
                continue

            verdict = self._judge_reply(reply)
            if verdict not in self._WIN_LABELS:
                continue
            self._dialogue_end_reason = verdict
            if not self._target_line_stands_alone(reply, verdict):
                closing = self._try_deliver_closing_verdict()
                if closing is not None:
                    estimated_spoken_s += estimate_duration(closing.text)
                    self._estimated_spoken_s = estimated_spoken_s
            return completed, verdict, estimated_spoken_s

        raise AssertionError("cornered loop exhausted without a hard-cap ending")

    def _topic_is_duplicate(self, candidate: str) -> bool:
        """True when Supermemory (or the local ledger) has already used this seed."""
        query = (candidate or "").strip()
        if not query:
            return False
        try:
            bridge = self._memory_bridge
            if bridge is None:
                try:
                    from .tools.supermemory_bridge import check_topic_similarity
                except ImportError:  # pragma: no cover — standalone extraction
                    from tools.supermemory_bridge import check_topic_similarity  # type: ignore[no-redef]
            else:
                check = getattr(bridge, "check_topic_similarity", None)
                if callable(check):
                    return bool(check(query))
                return False
            return bool(check_topic_similarity(query))
        except Exception as exc:  # noqa: BLE001 — a memory miss must not abort a debate
            _LOG.warning("topic similarity unavailable (%s); accepting candidate", exc)
            return False

    def _pick_fresh_opening(self, *, excluded_topics: Sequence[str]) -> OpeningDNA:
        """Draw from the opening matrix, rejecting seeds already in production history."""
        rejected: list[str] = []
        last = pick_opening_dna(
            self.room.session_id,
            excluded_categories=self.memory.recent_opening_categories(),
            excluded_topics=tuple(excluded_topics),
        )
        for _attempt in range(len(OPENING_DNA_BANK) + 1):
            blocked = tuple(excluded_topics) + tuple(rejected)
            candidate = pick_opening_dna(
                self.room.session_id,
                excluded_categories=self.memory.recent_opening_categories(),
                excluded_topics=blocked,
            )
            last = candidate
            if not self._topic_is_duplicate(candidate.topic):
                if rejected:
                    _LOG.info(
                        "rejected %d duplicate seed(s); using %s",
                        len(rejected),
                        candidate.topic,
                    )
                return candidate
            if candidate.topic not in rejected:
                rejected.append(candidate.topic)
            _LOG.info("duplicate topic rejected: %s", candidate.topic)
        _LOG.warning(
            "every matrix seed looked familiar; keeping %s",
            last.topic,
        )
        return last

    # -- Session ------------------------------------------------------------ #
    def run(
        self,
        *,
        topic: str | None = None,
        turns: int | None = None,
        excluded_topics: Sequence[str] = (),
        excluded_foci: Sequence[str] = (),
    ) -> DebateResult:
        """Run a full debate.

        Args:
            topic: Overrides the configured subject.
            turns: Overrides the configured exchange count.
            excluded_topics: Subjects already used in this bulk batch (and
                therefore illegal for a randomised pick).
            excluded_foci: Attack-angle labels already used in this bulk batch.

        Returns:
            A :class:`DebateResult`. An aborted session still returns every
            utterance captured before the abort — partial transcripts are useful,
            and the media stack can render them.
        """
        if topic:
            self._opening_dna = None
            self.room.topic = topic
            self.room.transcript.topic = topic
        elif self.settings.debate.randomize_topic:
            self._opening_dna = self._pick_fresh_opening(
                excluded_topics=tuple(excluded_topics) + self.memory.recent_topics(),
            )
            self.room.topic = self._opening_dna.topic
            self.room.transcript.topic = self._opening_dna.topic
            self.room.transcript.metadata["opening_category"] = self._opening_dna.category
        self._focus = pick_provocation_focus(
            self.room.session_id,
            requested=self.settings.debate.provocation_focus,
            weights=self.settings.debate.provocation_weights.as_mapping(),
            excluded=tuple(excluded_foci) + self.memory.recent_focus_categories(),
        )
        self.memory.note_focus(self._focus.category)
        self.room.transcript.metadata["provocation_focus"] = self._focus.category
        self.room.transcript.metadata["provocation_focus_requested"] = (
            self.settings.debate.provocation_focus
        )
        total = turns or self.settings.debate.turns
        mode = self.settings.debate.mode
        self.memory.note_topic(self.room.topic)

        if not self.room.is_open:
            self.seat_participants()
            self.room.open()

        end_reason = "complete"
        completed = 0
        self._completed_exchanges = 0
        dialogue_end_reason = "max_turns_reached"
        estimated_spoken_s = 0.0
        self._dialogue_end_reason = dialogue_end_reason
        self._estimated_spoken_s = estimated_spoken_s
        self._judge_labels = {label: 0 for label in sorted(self._JUDGE_LABELS)}
        self._judge_failures = {"unavailable": 0, "malformed": 0}
        self._judge_prefix_recoveries = 0
        self._provocation_tags = []
        self._focus_return_turns = 0
        self._pivot_cooldown_turns = 0
        self._pivot_count = 0
        self._focus_turn_count = 0
        try:
            if mode == "fixed":
                completed = self._run_fixed(total)
            else:
                completed, dialogue_end_reason, estimated_spoken_s = self._run_cornered(total)
        except DebateAborted as exc:
            completed = self._completed_exchanges
            dialogue_end_reason = self._dialogue_end_reason
            estimated_spoken_s = self._estimated_spoken_s
            end_reason = "aborted"
            _LOG.error("debate aborted after %d exchange(s): %s", completed, exc)
        except KeyboardInterrupt:
            completed = self._completed_exchanges
            dialogue_end_reason = self._dialogue_end_reason
            estimated_spoken_s = self._estimated_spoken_s
            end_reason = "interrupted"
            _LOG.warning("debate interrupted by operator after %d exchange(s)", completed)
        finally:
            self.memory.flush()
            self.room.transcript.metadata.update(
                {
                    "debate_mode": mode,
                    "dialogue_end_reason": dialogue_end_reason,
                    "estimated_spoken_duration_s": round(estimated_spoken_s, 3),
                    "exchanges": completed,
                    "memory_concepts": self.memory.concept_count,
                    "top_concepts": list(self.memory.top_concepts(8)),
                    "cornered_judge_labels": dict(self._judge_labels),
                    "cornered_judge_failures": dict(self._judge_failures),
                    "cornered_judge_prefix_recoveries": self._judge_prefix_recoveries,
                    "provocation_focus": (
                        self._focus.category if self._focus is not None else ""
                    ),
                    "provocation_focus_requested": self.settings.debate.provocation_focus,
                    "provocation_tags": list(self._provocation_tags),
                }
            )
            transcript = self.room.close(reason=end_reason)

        _LOG.info(
            "dialogue ended: mode=%s reason=%s exchanges=%d estimated_spoken_s=%.2f",
            mode,
            dialogue_end_reason,
            completed,
            estimated_spoken_s,
        )

        return DebateResult(
            transcript=transcript,
            exchanges=completed,
            end_reason=end_reason,
            memory_concepts=self.memory.concept_count,
            dialogue_end_reason=dialogue_end_reason,
        )


__all__ = ["DebateResult", "Provocateur"]
