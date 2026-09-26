"""Exclusive character-to-voice reservations for the debate engine."""

from .registry import (
    BRIAN,
    GUY,
    RYAN,
    VoiceCollisionError,
    assign_debater_voices,
    voice_for,
)

__all__ = [
    "BRIAN",
    "GUY",
    "RYAN",
    "VoiceCollisionError",
    "assign_debater_voices",
    "voice_for",
]
