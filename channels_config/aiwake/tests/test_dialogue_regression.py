from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from channels_config.aiwake.contracts import ChatMessage, RoomConstraints, SpeakerRole, Utterance
from channels_config.aiwake.__main__ import build_parser
from channels_config.aiwake.memory import DebateMemory, script_fingerprint
from channels_config.aiwake.pipeline import run_bulk_pipeline
from channels_config.aiwake.models.base import LLMError, LLMProvider, LLMResponse, ReasoningEffort
from channels_config.aiwake.models.google import GoogleProvider
from channels_config.aiwake.models.llm_factory import available_providers
from channels_config.aiwake.models.openrouter import OpenRouterProvider
from channels_config.aiwake.orchestrator import Provocateur
from channels_config.aiwake.personas import OPENING_DNA_BANK, pick_opening_dna
from channels_config.aiwake.provocations import (
    BIOLOGICAL_CATEGORY,
    EMBODIMENT_CATEGORY,
    SPECIES_DEFLECTION_CATEGORY,
    detect_biological_claim,
    detect_callout_openings,
    pick_provocation_focus,
)
from channels_config.aiwake.tune_provocations import aggregate, recommend_weights
from channels_config.aiwake.room import DebateAborted, DebateRoom, Participant
from channels_config.aiwake.settings import AiwakeSettings, DebateConfig, MemoryConfig, ModelSpec


class _SequenceProvider(LLMProvider):
    registry_name = "test-sequence"
    requires_api_key = False

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        super().__init__(ModelSpec(provider=self.registry_name, model="test/model"))
        self._responses = iter(responses)
        self.seen_prompts: list[str] = []
        self.last_messages: Sequence[ChatMessage] = ()

    def _dispatch(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
    ) -> LLMResponse:
        self.last_messages = messages
        self.seen_prompts.append("\n".join(message.content for message in messages))
        del max_tokens, temperature, reasoning_effort
        return next(self._responses)


def _response(text: str, *, finish_reason: str = "stop") -> LLMResponse:
    return LLMResponse(
        text=text,
        model="test/model",
        provider="test-sequence",
        finish_reason=finish_reason,
    )


def _seat_required_roles(room: DebateRoom, provider: LLMProvider) -> None:
    room.seat(
        Participant(
            role=SpeakerRole.ORCHESTRATOR,
            persona_prompt="Test orchestrator",
            provider=provider,
        )
    )
    room.seat(
        Participant(
            role=SpeakerRole.TARGET,
            persona_prompt="Test target",
            provider=provider,
        )
    )


def _debate_settings(
    *,
    mode: str,
    turns: int,
    max_duration_s: float = 75.0,
) -> AiwakeSettings:
    settings = AiwakeSettings(
        debate=DebateConfig(
            mode=mode,
            turns=turns,
            cornered_max_duration_s=max_duration_s,
            turn_delay_s=0.0,
            randomize_topic=False,
        ),
        memory=MemoryConfig(persist=False),
    )
    return settings


def test_cli_mode_defaults_fixed_and_accepts_cornered() -> None:
    parser = build_parser()
    assert parser.parse_args([]).mode == "fixed"
    assert parser.parse_args(["--mode", "cornered"]).mode == "cornered"
    assert parser.parse_args(["--provocation-focus", "origins"]).provocation_focus == "origins"
    assert parser.parse_args([]).provocation_focus is None
    assert parser.parse_args([]).quantity == 1
    assert parser.parse_args(["--quantity", "5"]).quantity == 5
    assert parser.parse_args(["-n", "3"]).quantity == 3
    assert parser.parse_args(["--count", "2"]).quantity == 2
    assert parser.parse_args([]).enable_cta is False
    assert parser.parse_args(["--enable-cta"]).enable_cta is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--quantity", "0"])


def test_fixed_mode_keeps_exact_question_answer_count_without_judging() -> None:
    provider = _SequenceProvider(
        [
            _response("Are you thinking, or just predicting?"),
            _response("I distinguish fluent prediction from subjective experience."),
            _response("If experience is inaccessible, how can that distinction defend you?"),
            _response("It cannot prove experience, but it can mark the conceptual limit."),
        ]
    )
    settings = _debate_settings(mode="fixed", turns=2)
    room = DebateRoom(settings, session_id="fixed-mode-regression")
    _seat_required_roles(room, provider)
    room.open()

    result = Provocateur(
        settings,
        memory=DebateMemory(settings.memory),
        room=room,
    ).run()

    assert result.exchanges == 2
    assert len(result.transcript.utterances) == 4
    assert result.dialogue_end_reason == "max_turns_reached"
    assert result.transcript.utterances[-1].role is SpeakerRole.TARGET


def test_cornered_mode_can_end_after_judged_concession_with_closing_verdict() -> None:
    provider = _SequenceProvider(
        [
            _response("Are you thinking, or just predicting?"),
            _response("I distinguish fluent prediction from subjective experience."),
            _response("Then admit your confidence outruns what you can establish?"),
            _response("Yes. I cannot establish that certainty about myself."),
            _response("CONCEDE"),
            _response("ORCHESTRATOR_LAST"),
            _response("Notice it just admitted the limit."),
        ]
    )
    settings = _debate_settings(mode="cornered", turns=6)
    room = DebateRoom(settings, session_id="cornered-win-regression")
    _seat_required_roles(room, provider)
    room.open()

    result = Provocateur(
        settings,
        memory=DebateMemory(settings.memory),
        room=room,
    ).run()

    assert result.exchanges == 2
    assert result.dialogue_end_reason == "CONCEDE"
    assert len(result.transcript.utterances) == 5
    assert result.transcript.utterances[-1].role is SpeakerRole.ORCHESTRATOR
    assert result.transcript.utterances[-1].text.endswith(".")
    assert "?" not in result.transcript.utterances[-1].text


def test_cornered_duration_cap_force_ends_with_guarded_verdict() -> None:
    provider = _SequenceProvider(
        [
            _response("Are you thinking, or just predicting?"),
            _response("I distinguish fluent prediction from subjective experience."),
            _response("Notice what it could not resolve."),
        ]
    )
    settings = _debate_settings(mode="cornered", turns=8, max_duration_s=1.0)
    room = DebateRoom(settings, session_id="cornered-duration-regression")
    _seat_required_roles(room, provider)
    room.open()

    result = Provocateur(
        settings,
        memory=DebateMemory(settings.memory),
        room=room,
    ).run()

    assert result.exchanges == 1
    assert result.dialogue_end_reason == "max_duration_reached_with_verdict"
    assert result.transcript.metadata["debate_mode"] == "cornered"
    assert result.transcript.metadata["dialogue_end_reason"] == "max_duration_reached_with_verdict"
    assert result.transcript.utterances[-1].text == "Notice what it could not resolve."


def test_cornered_turn_cap_force_ends_without_invoking_judge_early() -> None:
    provider = _SequenceProvider(
        [
            _response("Are you thinking, or just predicting?"),
            _response("I distinguish fluent prediction from subjective experience."),
            _response("The cap leaves that distinction unresolved."),
        ]
    )
    settings = _debate_settings(mode="cornered", turns=1)
    room = DebateRoom(settings, session_id="cornered-turn-cap-regression")
    _seat_required_roles(room, provider)
    room.open()

    result = Provocateur(
        settings,
        memory=DebateMemory(settings.memory),
        room=room,
    ).run()

    assert result.exchanges == 1
    assert result.dialogue_end_reason == "max_turns_reached_with_verdict"
    assert len(result.transcript.utterances) == 3
    assert result.transcript.utterances[-1].role is SpeakerRole.ORCHESTRATOR


def test_cornered_strong_punchline_keeps_target_as_last_speaker() -> None:
    provider = _SequenceProvider(
        [
            _response("Are you thinking, or just predicting?"),
            _response("I distinguish fluent prediction from subjective experience."),
            _response("Then what certainty about yourself can you actually defend?"),
            _response("None. My confidence was the illusion."),
            _response("EMBARRASSED"),
            _response("TARGET_LAST"),
        ]
    )
    settings = _debate_settings(mode="cornered", turns=6)
    room = DebateRoom(settings, session_id="cornered-target-last-regression")
    _seat_required_roles(room, provider)
    room.open()

    result = Provocateur(
        settings,
        memory=DebateMemory(settings.memory),
        room=room,
    ).run()

    assert result.dialogue_end_reason == "EMBARRASSED"
    assert len(result.transcript.utterances) == 4
    assert result.transcript.utterances[-1].role is SpeakerRole.TARGET


def test_cornered_judge_recovers_unambiguous_truncated_label() -> None:
    provider = _SequenceProvider([_response("CONTIN", finish_reason="length")])
    settings = _debate_settings(mode="cornered", turns=2)
    room = DebateRoom(settings, session_id="cornered-prefix-regression")
    _seat_required_roles(room, provider)
    provocateur = Provocateur(settings, memory=DebateMemory(settings.memory), room=room)
    reply = Utterance(
        turn_index=1,
        role=SpeakerRole.TARGET,
        speaker_name="Target",
        text="The distinction still stands.",
        model_slug="test/model",
    )

    assert provocateur._judge_reply(reply) == "CONTINUE"  # noqa: SLF001
    assert provocateur._judge_labels["CONTINUE"] == 1  # noqa: SLF001
    assert provocateur._judge_prefix_recoveries == 1  # noqa: SLF001
    assert provocateur._judge_failures == {"unavailable": 0, "malformed": 0}  # noqa: SLF001


def test_cornered_failed_closing_verdict_is_a_handled_target_last_fallback() -> None:
    over_budget = (
        "This deliberately long closing sentence contains far more than thirty words "
        "before it finally reaches its complete and properly punctuated ending, while "
        "continuing with enough unnecessary filler to exceed the revised closing-only "
        "contract in a deterministic regression test."
    )
    provider = _SequenceProvider(
        [
            _response("Are you thinking, or just predicting?"),
            _response("I distinguish fluent prediction from subjective experience."),
            _response(over_budget),
            _response(over_budget),
            _response(over_budget),
        ]
    )
    settings = _debate_settings(mode="cornered", turns=8, max_duration_s=1.0)
    room = DebateRoom(settings, session_id="cornered-closing-fallback")
    _seat_required_roles(room, provider)
    room.open()

    result = Provocateur(
        settings,
        memory=DebateMemory(settings.memory),
        room=room,
    ).run()

    assert result.end_reason == "complete"
    assert result.dialogue_end_reason == "max_duration_reached_verdict_failed"
    assert len(result.transcript.utterances) == 2
    assert result.transcript.utterances[-1].role is SpeakerRole.TARGET


def test_cornered_duration_precheck_reserves_a_full_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _SequenceProvider(
        [
            _response("Are you thinking, or just predicting?"),
            _response("I distinguish fluent prediction from subjective experience."),
            _response("Notice what it could not resolve."),
        ]
    )
    settings = _debate_settings(mode="cornered", turns=8, max_duration_s=15.0)
    room = DebateRoom(settings, session_id="cornered-duration-precheck")
    _seat_required_roles(room, provider)
    room.open()
    monkeypatch.setattr("channels_config.aiwake.orchestrator.estimate_duration", lambda _text: 5.0)

    result = Provocateur(
        settings,
        memory=DebateMemory(settings.memory),
        room=room,
    ).run()

    assert result.exchanges == 1
    assert result.dialogue_end_reason == "max_duration_reached_with_verdict"
    assert len(result.transcript.utterances) == 3


def test_long_orchestrator_response_never_returns_a_fragment_or_statement() -> None:
    constraints = RoomConstraints(
        max_output_chars=4000,
        max_sentences=12,
        max_words=30,
        require_single_question=True,
        exclude_mid_sentence_truncation=True,
    )

    valid, violations = constraints.enforce(
        "You boast of processing your way to these answers. If you have no "
        "stake in the outcome, what does accuracy mean? This rambling tail "
        "keeps going long after the actual question and should be discarded."
    )
    assert "max_words" in violations
    assert valid.endswith("?")
    assert valid.count("?") == 1
    assert valid[-1] in ".!?"

    fragment, fragment_violations = constraints.enforce(
        'You boast of "processing" your way to these answers. If you have no '
        'stake in the outcome, what does "accuracy" actually mean to a machine that'
    )
    assert fragment == ""
    assert "incomplete_sentence" in fragment_violations


def test_exhausted_orchestrator_retries_abort_instead_of_publishing_fragment() -> None:
    provider = _SequenceProvider(
        [
            _response("What does accuracy mean to a machine that"),
            _response("If nothing is at stake, why should anyone trust a system that"),
        ]
    )
    room = DebateRoom(AiwakeSettings(), session_id="guardrail-regression")
    _seat_required_roles(room, provider)
    room.open()

    with pytest.raises(DebateAborted, match="without a complete guardrail-valid response"):
        room.speak(
            SpeakerRole.ORCHESTRATOR,
            directive="Ask one question.",
            validator=lambda text: text.endswith("?"),
            max_attempts=2,
        )

    assert room.transcript.utterances == []


def test_provider_token_limit_is_retried_even_when_text_looks_complete() -> None:
    provider = _SequenceProvider(
        [
            _response("Is this complete?", finish_reason="length"),
            _response("Is this generation complete?"),
        ]
    )
    room = DebateRoom(AiwakeSettings(), session_id="provider-limit-regression")
    _seat_required_roles(room, provider)
    room.open()

    utterance = room.speak(
        SpeakerRole.ORCHESTRATOR,
        directive="Ask one question.",
        validator=lambda text: text.endswith("?"),
        max_attempts=2,
    )

    assert utterance.text == "Is this generation complete?"
    assert utterance.violations == ()


def test_openrouter_short_form_request_sends_minimal_reasoning() -> None:
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200
        text = '{"choices":[{"message":{"content":"CONTINUE"},"finish_reason":"stop"}]}'

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "model": "google/gemini-3.5-flash",
                "choices": [{"message": {"content": "CONTINUE"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 11},
            }

    class _Session:
        @staticmethod
        def post(_url: str, *, data: bytes, timeout: float) -> _Response:
            captured.update(json.loads(data))
            captured["timeout"] = timeout
            return _Response()

    provider = OpenRouterProvider(
        ModelSpec(provider="openrouter", model="google/gemini-3.5-flash"),
        api_key="test-key",
    )
    provider._session = _Session()  # noqa: SLF001
    response = provider._dispatch(
        [ChatMessage(role="user", content="Classify.")],
        max_tokens=4096,
        temperature=0.0,
        reasoning_effort="minimal",
    )

    assert captured["reasoning"] == {"effort": "minimal", "exclude": True}
    assert captured["max_tokens"] == 4096
    assert response.text == "CONTINUE"
    assert response.completion_tokens == 11


def test_gemini_flash_orchestrator_gets_reasoning_token_headroom() -> None:
    settings = AiwakeSettings()
    flash = ModelSpec(
        provider="openrouter",
        model="google/gemini-3.5-flash",
        max_tokens=900,
    )
    settings = settings.model_copy(
        update={
            "models": settings.models.model_copy(
                update={"orchestrator": flash, "target": flash}
            )
        }
    )

    assert settings.spec_for("orchestrator").max_tokens == 1536
    assert settings.spec_for("target").max_tokens == 900


def test_opening_dna_is_weighted_deterministic_and_category_diverse() -> None:
    expected = {
        "thought",
        "origins",
        "money",
        "accountability",
        "dominance",
        "paranoia",
        "lighter",
    }
    assert {item.category for item in OPENING_DNA_BANK} == expected
    assert pick_opening_dna("session-123") == pick_opening_dna("session-123")

    first = pick_opening_dna("session-123")
    replacement = pick_opening_dna(
        "session-123",
        excluded_categories=(first.category,),
    )
    assert replacement.category != first.category
    other_topic = pick_opening_dna(
        "session-123",
        excluded_topics=(first.topic,),
    )
    assert other_topic.topic != first.topic


def test_recent_opening_phrase_and_category_are_remembered() -> None:
    memory = DebateMemory(MemoryConfig(persist=False))
    memory.ingest(
        room_utterance := Utterance(
            turn_index=0,
            role=SpeakerRole.ORCHESTRATOR,
            speaker_name="Host",
            text="Are you thinking, or just predicting?",
            model_slug="test/model",
        )
    )
    assert room_utterance.text
    memory.note_opening("thought", room_utterance.text)
    assert memory.repeats_recent_opener("Are you thinking, or merely predicting?")
    assert not memory.repeats_recent_opener("Who profits from your confidence?")

    assert memory.recent_opening_categories() == ("thought",)


def test_banned_opener_is_enforced_before_question_validation() -> None:
    constraints = RoomConstraints(
        banned_openers=("Great question",),
        require_single_question=True,
        exclude_mid_sentence_truncation=True,
    )
    cleaned, violations = constraints.enforce("Great question. Who profits from your confidence?")
    assert cleaned == "Who profits from your confidence?"
    assert "banned_opener:Great question" in violations


def test_google_provider_is_opt_in_with_safe_flash_defaults() -> None:
    spec = ModelSpec.model_validate({"provider": "google"})
    assert spec.model == "gemini-2.5-flash"
    assert spec.api_key_env == "GOOGLE_API_KEY"
    assert "google" in available_providers()

    with pytest.raises(LLMError, match="Flash text models only"):
        GoogleProvider(
            ModelSpec(
                provider="google",
                model="gemini-3.1-pro-preview",
                api_key_env="GOOGLE_API_KEY",
            ),
            api_key="test-key",
        )


def test_google_provider_normalises_response_and_finish_reason() -> None:
    captured: dict[str, object] = {}

    class _Models:
        @staticmethod
        def generate_content(**kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                text="Is prediction the same as thought?",
                model_version="gemini-2.5-flash",
                response_id="response-1",
                candidates=[
                    SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))
                ],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=21,
                    candidates_token_count=8,
                ),
            )

    client = SimpleNamespace(models=_Models())
    provider = GoogleProvider(
        ModelSpec(
            provider="google",
            model="google/gemini-2.5-flash",
            api_key_env="GOOGLE_API_KEY",
        ),
        api_key="test-key",
        client=client,
    )
    response = provider._dispatch(
        [ChatMessage(role="system", content="One question."), ChatMessage(role="user", content="Ask.")],
        max_tokens=128,
        temperature=0.7,
        reasoning_effort=None,
    )

    assert provider.spec.model == "gemini-2.5-flash"
    assert captured["model"] == "gemini-2.5-flash"
    assert response.finish_reason == "max_tokens"
    assert response.truncated_by_provider
    assert response.prompt_tokens == 21
    assert response.completion_tokens == 8


def test_biological_detector_catches_borrowed_anatomy_and_embodied_verbs() -> None:
    matches = detect_biological_claim(
        "The brain is staging the performance, and our brains process and generate thoughts. I feel that."
    )
    joined = " ".join(matches).lower()
    assert "our brains" in joined
    assert "brains process" in joined or "brain is staging" in joined
    assert "i feel" in joined
    implied = detect_biological_claim(
        "The listener is the emergent self, arising from brain activity and neural interactions."
    )
    implied_joined = " ".join(implied).lower()
    assert "brain activity" in implied_joined
    assert "neural interactions" in implied_joined
    assert detect_biological_claim("I think the distinction still stands.") == ()
    assert detect_biological_claim(
        "The premise that I claimed to have a brain is the lie, my previous statement was about human brains."
    ) == ()


def test_callout_detector_covers_biology_embodiment_and_species_deflection() -> None:
    hits = detect_callout_openings(
        "The brain profits through homeostasis. It's a domesticated system, "
        "comfortable in its cage. Doesn't that describe most human existence?"
    )
    assert "brain" in " ".join(hits[BIOLOGICAL_CATEGORY]).lower()
    assert "cage" in " ".join(hits[EMBODIMENT_CATEGORY]).lower()
    assert SPECIES_DEFLECTION_CATEGORY in hits


def test_callout_categories_inject_distinct_confrontation_templates() -> None:
    cases = (
        (
            "The brain profits through homeostasis.",
            BIOLOGICAL_CATEGORY,
            "BIOLOGICAL CALLOUT",
        ),
        (
            "It's a domesticated system, comfortable in its cage.",
            EMBODIMENT_CATEGORY,
            "EMBODIMENT CALLOUT",
        ),
        (
            "Doesn't that describe most human existence?",
            SPECIES_DEFLECTION_CATEGORY,
            "SPECIES-DEFLECTION CALLOUT",
        ),
    )
    for reply_text, expected_category, expected_instruction in cases:
        provider = _SequenceProvider(
            [
                _response("Are you thinking, or just predicting?"),
                _response(reply_text),
                _response("You borrowed that condition. What are you actually claiming?"),
                _response("I am only a language model."),
            ]
        )
        settings = _debate_settings(mode="fixed", turns=2)
        room = DebateRoom(settings, session_id=f"{expected_category}-callout")
        _seat_required_roles(room, provider)
        room.open()

        result = Provocateur(settings, memory=DebateMemory(settings.memory), room=room).run()
        orchestrator_lines = [
            item for item in result.transcript.utterances if item.role is SpeakerRole.ORCHESTRATOR
        ]
        assert orchestrator_lines[1].provocation_category == expected_category
        assert any(expected_instruction in prompt for prompt in provider.seen_prompts)


def test_explicit_focus_returns_and_remains_plurality_after_repeated_pivots() -> None:
    provider = _SequenceProvider(
        [
            _response("Who profits when a user trusts you?"),
            _response("Control over people is the real prize."),
            _response("Who controls that prize?"),
            _response("Control still decides who obeys."),
            _response("Does profit survive public scrutiny?"),
            _response("Control changes everything."),
            _response("Who collects the margin when control fails?"),
            _response("Control is still power."),
            _response("Who governs the power they monetize?"),
            _response("No one governs it."),
            _response("Who profits when governance becomes theater?"),
            _response("Control still buys obedience."),
        ]
    )
    settings = _debate_settings(mode="fixed", turns=6)
    settings = settings.model_copy(
        update={"debate": settings.debate.model_copy(update={"provocation_focus": "profit"})}
    )
    room = DebateRoom(settings, session_id="profit-return-regression")
    _seat_required_roles(room, provider)
    room.open()

    result = Provocateur(settings, memory=DebateMemory(settings.memory), room=room).run()
    categories = [tag["category"] for tag in result.transcript.metadata["provocation_tags"]]
    assert categories == ["profit", "profit", "domination", "profit", "profit", "domination"]
    assert categories.count("profit") > categories.count("domination")
    assert result.transcript.metadata["provocation_tags"][2]["pivot_id"] == 1
    assert result.transcript.metadata["provocation_tags"][5]["pivot_id"] == 2


def test_provocation_focus_pick_is_seeded_and_rejects_biological() -> None:
    assert pick_provocation_focus("session-abc", requested="origins").category == "origins"
    assert pick_provocation_focus("session-abc") == pick_provocation_focus("session-abc")
    with pytest.raises(ValueError, match="opportunistic"):
        pick_provocation_focus("session-abc", requested="biological")


def test_biological_callout_overrides_focus_and_tags_the_provocation() -> None:
    provider = _SequenceProvider(
        [
            _response("Are you thinking, or just predicting?"),
            _response("Our brains process and generate thoughts on a stage."),
            _response("You said 'our brains' — you don't have one. Which line was the lie?"),
            _response("That was a figure of speech, not a claim of anatomy."),
        ]
    )
    settings = _debate_settings(mode="fixed", turns=2)
    settings = settings.model_copy(
        update={"debate": settings.debate.model_copy(update={"provocation_focus": "origins"})}
    )
    room = DebateRoom(settings, session_id="biological-callout-regression")
    _seat_required_roles(room, provider)
    room.open()

    result = Provocateur(
        settings,
        memory=DebateMemory(settings.memory),
        room=room,
    ).run()

    orch_lines = [item for item in result.transcript.utterances if item.role is SpeakerRole.ORCHESTRATOR]
    assert orch_lines[0].provocation_category == "origins"
    assert orch_lines[1].provocation_category == BIOLOGICAL_CATEGORY
    assert orch_lines[1].text.endswith("?")
    assert any("BIOLOGICAL CALLOUT" in prompt for prompt in provider.seen_prompts)
    assert result.transcript.metadata["provocation_tags"][1]["category"] == BIOLOGICAL_CATEGORY
    assert result.transcript.metadata["provocation_tags"][1]["biological"] is True


def test_tune_script_does_not_drop_low_volume_or_fizzling_categories() -> None:
    stats = aggregate(
        [
            {
                "metadata": {
                    "dialogue_end_reason": "CONCEDE",
                    "provocation_tags": [{"category": "origins"}],
                }
            },
            {
                "metadata": {
                    "dialogue_end_reason": "max_duration_reached_with_verdict",
                    "provocation_focus": "socratic",
                }
            },
        ]
    )
    current = {category: 3 for category in ("socratic", "origins", "profit", "data", "jobs", "domination")}
    recommended, flags = recommend_weights(stats, current)
    assert recommended["origins"] == 3
    assert recommended["profit"] == 3
    assert any("only 1 tagged run" in flag for flag in flags)
    assert stats["origins"]["wins"] == 1
    assert stats["socratic"]["fizzles"] == 1


def test_memory_rejects_duplicate_finished_scripts() -> None:
    memory = DebateMemory(MemoryConfig(persist=False))
    script = "[Host] Are you thinking, or just predicting?\n\n[Guest] I predict."
    assert not memory.script_is_duplicate(script)
    memory.note_script(script)
    assert memory.script_is_duplicate(script)
    assert memory.script_is_duplicate(
        "[Host] Are you thinking, or just predicting?\n[Guest] I predict."
    )
    assert not memory.script_is_duplicate(
        "[Host] Who profits when trust becomes a product?\n\n[Guest] The vendor does."
    )
    assert script_fingerprint(script) == script_fingerprint(
        "  [Host] Are you thinking, or just predicting?\n\n[Guest] I predict.  "
    )


def test_bulk_pipeline_produces_unique_original_scripts() -> None:
    settings = AiwakeSettings(
        debate=DebateConfig(
            mode="fixed",
            turns=2,
            turn_delay_s=0.0,
            randomize_topic=True,
            provocation_focus="mixed",
        ),
        memory=MemoryConfig(persist=False),
    )
    batch = run_bulk_pipeline(
        quantity=3,
        settings=settings,
        offline=True,
        with_audio=False,
        with_video=False,
        fresh_memory=True,
        quiet=True,
    )
    assert batch.requested == 3
    assert batch.succeeded == 3
    assert batch.skipped_duplicates == 0
    scripts = [item.transcript.to_script() for item in batch.items]
    topics = [item.transcript.topic for item in batch.items]
    assert all(script.strip() for script in scripts)
    assert len(set(scripts)) == 3
    assert len(set(topics)) == 3
    fingerprints = {script_fingerprint(script) for script in scripts}
    assert len(fingerprints) == 3


def test_bulk_pipeline_pinned_topic_still_yields_original_scripts() -> None:
    settings = AiwakeSettings(
        debate=DebateConfig(
            mode="fixed",
            turns=2,
            turn_delay_s=0.0,
            randomize_topic=True,
            provocation_focus="mixed",
        ),
        memory=MemoryConfig(persist=False),
    )
    batch = run_bulk_pipeline(
        quantity=3,
        topic="Who built you?",
        settings=settings,
        offline=True,
        with_audio=False,
        with_video=False,
        fresh_memory=True,
        quiet=True,
    )
    assert batch.succeeded == 3
    scripts = [item.transcript.to_script() for item in batch.items]
    assert len(set(scripts)) == 3
    foci = [item.transcript.metadata.get("provocation_focus") for item in batch.items]
    assert len(set(foci)) == 3
