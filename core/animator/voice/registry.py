"""One neural voice per puppet. Brian belongs to Gemini alone."""

from __future__ import annotations

BRIAN = "en-US-BrianNeural"
RYAN = "en-GB-RyanNeural"
GUY = "en-US-GuyNeural"

# Character id -> the only voice that puppet may speak with.
RESERVED: dict[str, str] = {
    "gemini": BRIAN,
    "gemini_cyborg_v2": BRIAN,
    "claude_cyborg_v1": RYAN,
    "claude": RYAN,
    "chatgpt_cyborg_v1": GUY,
    "chatgpt": GUY,
}


class VoiceCollisionError(RuntimeError):
    """Two active debaters resolved to the same TTS voice."""


def _is_gemini(character_id: str) -> bool:
    return "gemini" in character_id.lower()


def voice_for(character_id: str) -> str:
    """Return the reserved neural voice for one puppet."""
    token = character_id.strip()
    if token in RESERVED:
        return RESERVED[token]
    lowered = token.lower()
    if "gemini" in lowered:
        return BRIAN
    if "claude" in lowered:
        return RYAN
    if "chatgpt" in lowered:
        return GUY
    raise KeyError(f"no reserved voice for {character_id}")


def _off_brian(character_id: str, voice: str) -> str:
    """Brian never leaves the Gemini seat. Other puppets fall back to their lock."""
    if voice == BRIAN and not _is_gemini(character_id):
        return voice_for(character_id)
    return voice


def assign_debater_voices(
    character_ids: list[str],
    requested: dict[str, str] | None = None,
) -> dict[str, str]:
    """Lock one voice per active debater.

    A shared id is re-routed onto that puppet's reservation. If the
    reservation is taken as well, the engine raises instead of doubling a voice.
    """
    requested = requested or {}
    assigned: dict[str, str] = {}
    owners: dict[str, str] = {}
    for character_id in character_ids:
        voice = _off_brian(character_id, requested.get(character_id) or voice_for(character_id))
        owner = owners.get(voice)
        if owner is not None and owner != character_id:
            fallback = voice_for(character_id)
            if fallback == voice or fallback in owners:
                raise VoiceCollisionError(
                    f"{character_id} and {owner} both require {voice}"
                )
            voice = fallback
        owners[voice] = character_id
        assigned[character_id] = voice
    return assigned
