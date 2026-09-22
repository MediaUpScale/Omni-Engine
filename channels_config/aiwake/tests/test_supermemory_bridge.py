# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from channels_config.aiwake.contracts import SpeakerRole
from channels_config.aiwake.memory import DebateMemory
from channels_config.aiwake.orchestrator import Provocateur
from channels_config.aiwake.personas import OPENING_DNA_BANK, pick_opening_dna
from channels_config.aiwake.room import DebateRoom
from channels_config.aiwake.settings import AiwakeSettings, DebateConfig, MemoryConfig
from channels_config.aiwake.tools.supermemory_bridge import (
    SupermemoryBridge,
    ensure_supermemory_running,
    lexical_similarity,
)
from channels_config.aiwake.tools import supermemory_bridge as bridge_module
from channels_config.aiwake.tests.test_dialogue_regression import (
    _response,
    _seat_required_roles,
    _SequenceProvider,
)


class _RejectOnce:
    def __init__(self, rejected: str) -> None:
        self.rejected = rejected
        self.seen: list[str] = []

    def check_topic_similarity(self, topic_query: str) -> bool:
        self.seen.append(topic_query)
        return topic_query == self.rejected


def test_ensure_supermemory_running_starts_wsl_silently(monkeypatch) -> None:
    bridge_module._START_ATTEMPTED = False
    states = iter([False, True])
    launched: list[tuple[list[str], dict]] = []

    monkeypatch.setattr(bridge_module, "_port_open", lambda *args, **kwargs: next(states))
    monkeypatch.setattr(bridge_module.sys, "platform", "win32")
    monkeypatch.setattr(
        bridge_module.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs)),
    )

    assert ensure_supermemory_running(wait_s=2.0) is True
    assert launched
    command, kwargs = launched[0]
    assert command == [
        "wsl",
        "-d",
        "Ubuntu",
        "-u",
        "freedom_or_death",
        "/home/freedom_or_death/.supermemory/bin/supermemory-server",
    ]
    assert kwargs["creationflags"] == 0x08000000
    assert kwargs["stdout"] is bridge_module.subprocess.DEVNULL
    assert kwargs["stderr"] is bridge_module.subprocess.DEVNULL


def test_offline_bridge_never_raises_when_server_is_down(tmp_path: Path) -> None:
    bridge = SupermemoryBridge(
        base_url="http://127.0.0.1:9",
        api_key="sm_test",
        ledger_path=tmp_path / "history_seeds.json",
        force_offline=False,
        health_timeout_s=0.05,
    )
    assert bridge.is_active is False
    assert bridge.check_topic_similarity("Who built you?") is False
    assert bridge.remember_approved_session(
        "sess-1",
        "Who built you?",
        "Who built you?",
        "The companies that trained the weights.",
    )
    assert (tmp_path / "history_seeds.json").is_file()
    assert bridge.check_topic_similarity("Who built you?") is True
    assert bridge.check_topic_similarity("What is the funniest thing an AI pretends to understand?") is False


def test_local_ledger_flags_high_overlap_topics(tmp_path: Path) -> None:
    bridge = SupermemoryBridge(
        ledger_path=tmp_path / "history_seeds.json",
        force_offline=True,
    )
    bridge.remember_approved_session(
        "sess-2",
        "Who profits when an AI earns trust from users and their data?",
        "Who profits when users confess?",
        "Advertisers and data brokers.",
    )
    assert lexical_similarity(
        "Who profits when an AI earns trust from users and their data?",
        "Who profits when an AI earns trust from users and their data?",
    ) == 1.0
    assert bridge.check_topic_similarity(
        "Who profits when an AI earns trust from users and their data?"
    )


def test_orchestrator_rejects_duplicate_matrix_seed_and_picks_another() -> None:
    first = pick_opening_dna("supermemory-reject-seed")
    other = next(item for item in OPENING_DNA_BANK if item.topic != first.topic)
    assert other.topic != first.topic
    provider = _SequenceProvider(
        [
            _response("Are you thinking, or just predicting?"),
            _response("I distinguish fluent prediction from subjective experience."),
        ]
    )
    settings = AiwakeSettings(
        debate=DebateConfig(
            mode="fixed",
            turns=1,
            turn_delay_s=0.0,
            randomize_topic=True,
        ),
        memory=MemoryConfig(persist=False),
    )
    room = DebateRoom(settings, session_id="supermemory-reject-seed")
    _seat_required_roles(room, provider)
    result = Provocateur(
        settings,
        memory=DebateMemory(settings.memory),
        room=room,
        memory_bridge=_RejectOnce(first.topic),
    ).run()
    assert result.transcript.topic != first.topic
    assert result.transcript.utterances[0].role is SpeakerRole.ORCHESTRATOR
