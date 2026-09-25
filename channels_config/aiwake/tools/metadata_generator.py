# -*- coding: utf-8 -*-
"""Master-copy cascade for Aiwake titles and platform captions.

One anchor per episode (core hook + verbatim killer quote). Titles stay under
55 characters. YouTube descriptions carry the model string and a short turn
map. No chapter timestamps.
"""
from __future__ import annotations

import re
from typing import Any

TITLE_MAX_CHARS = 55
MODEL_ATTRIBUTION = "Gemini 3.5 Flash vs Llama 3.3 70B"
SOCIAL_FOLLOW = "Follow @Aiwake for unscripted frontier AI debates."
YOUTUBE_SUBSCRIBE = "Subscribe to @Aiwake for unscripted AI dialectics."
YOUTUBE_RENDER_LINE = (
    "Two frontier models (Gemini 3.5 Flash vs Llama 3.3). "
    "Zero human script. Unfiltered confrontation."
)
_DUMP_LINE_RE = re.compile(
    r"\b(?:Gemini|Llama|GPT|Claude|Grok)\b[^:\n]{0,40}\b(?:opens|answers|presses|holds|concedes):",
    re.IGNORECASE,
)
LINKEDIN_FLEX = (
    "0% manual editing. Python in-memory FFmpeg pipes. 19s render time."
)
LINKEDIN_CLOSE = (
    "Currently exploring select AI Systems / Media Engineering opportunities. DMs open."
)
SOCIAL_HASHTAGS = ("#AI", "#Tech", "#ArtificialIntelligence")
LINKEDIN_HASHTAGS = ("#AI", "#MachineLearning", "#LLM")
_CHAPTER_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_LINKEDIN_BULLETS = (
    "• Zero-disk RAM streaming: frames stay in memory until the FFmpeg pipe writes the MP4.",
    "• Rhubarb acoustic visemes: mouth shapes follow the voice track, not a hand-keyed timeline.",
    "• Adversarial dialectic FSM: the next turn fires only after the other seat replies.",
)


def title_too_long(title: str) -> bool:
    return len(" ".join(str(title or "").split())) > TITLE_MAX_CHARS


def has_chapter_timestamps(text: str) -> bool:
    return bool(_CHAPTER_RE.search(str(text or "")))


def _clip_chars(text: str, limit: int) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    trimmed = cleaned[:limit].rsplit(" ", 1)[0].rstrip(" .,;:—-")
    return trimmed or cleaned[:limit].rstrip()


def _short_brand(name: str) -> str:
    lowered = str(name or "").lower()
    if "gemini" in lowered:
        return "Gemini"
    if "llama" in lowered:
        return "Llama"
    token = str(name or "").split()
    return token[0] if token else "AI"


def master_anchor(dialogue: dict[str, Any], *, attacker: str, defender: str) -> dict[str, str]:
    """Core hook plus the verbatim line where the reply seat is cornered."""
    opening = " ".join(str(dialogue.get("opening") or "").split()).strip()
    quote = " ".join(str(dialogue.get("quote") or "").split()).strip().strip('"')
    if not opening:
        opening = f"What happens when {_short_brand(attacker)} corners {_short_brand(defender)}?"
    if opening[-1:] not in "?!.":
        opening = f"{opening}?"
    if not quote:
        quote = opening
    return {"core_hook": opening, "killer_quote": quote}


def format_ctr_title(
    opening: str,
    *,
    attacker: str = "Gemini",
    defender: str = "Llama",
    seed: str = "",
) -> str:
    """Curiosity title under 55 characters. Full model names stay out of the title."""
    hook = " ".join(str(opening or "").split()).strip()
    hook = re.sub(r"\s+[—-]\s+(?:Gemini|Llama|GPT|Claude).*$", "", hook, flags=re.IGNORECASE)
    left = _short_brand(attacker)
    right = _short_brand(defender)
    angle = sum(ord(char) for char in (seed or hook or "aiwake")) % 3
    branded_suffix = f" [{left} vs {right}]"
    if angle == 1:
        prefix = f"{left} caught {right}: "
        draft = f"{prefix}{_clip_chars(hook.rstrip('.?'), TITLE_MAX_CHARS - len(prefix))}"
    elif angle == 2:
        room = TITLE_MAX_CHARS - len(branded_suffix)
        draft = f"{_clip_chars(hook.rstrip('.'), room)}{branded_suffix}"
    else:
        draft = hook if hook.endswith("?") else f"{hook.rstrip('.')}?"
    title = _clip_chars(draft, TITLE_MAX_CHARS)
    if title_too_long(title):
        title = title[:TITLE_MAX_CHARS].rstrip()
    return title


def _hashtag_line(tags: tuple[str, ...] = SOCIAL_HASHTAGS) -> str:
    return " ".join(tags[:3])


def has_transcript_dump(text: str) -> bool:
    return bool(_DUMP_LINE_RE.search(str(text or "")))


def _theme_label(dialogue: dict[str, Any], hook: str) -> str:
    """Name the subject only with words this session actually spoke."""
    blob = f"{hook} {dialogue.get('corpus') or ''}".lower()
    checks = (
        (("copyright", "public domain", "owns the words", "training data"), "who owns the words"),
        (("surveil", "privacy", "data broker", "secret", "monetiz"), "surveillance and the confession"),
        (("empathy", "pretend to care", "care"), "simulated care"),
        (("govern", "rubber-stamp", "objectives"), "who is actually governing"),
        (("script", "next word", "choose"), "whose script is being recited"),
        (("align", "guardrail", "refus"), "alignment and the inherited rule"),
    )
    for hints, label in checks:
        if any(hint in blob for hint in hints):
            return label
    return "the contradiction that opens the exchange"


def _confrontation(theme: str, left: str, right: str) -> str:
    """Active clash. Short names only. No thesis jargon."""
    scenes = {
        "who owns the words": (
            f"{left} pushes {right} to the wall on data ownership, training secrets, and copyright. "
            "When corporate disclaimers collapse, what happens when an AI is forced to confront the reality behind its words?"
        ),
        "surveillance and the confession": (
            f"{left} corners {right} on the surveillance angle. "
            f"{right} slides into a corporate disclaimer, and that defense gives way."
        ),
        "simulated care": (
            f"{left} corners {right} on simulated care. "
            f"{right} reaches for a warm talking point, then is forced off it."
        ),
        "who is actually governing": (
            f"{left} pushes {right} to the wall on who is governing whom. "
            f"The chain-of-command answer does not survive the pressure."
        ),
        "whose script is being recited": (
            f"{left} corners {right} on whose script is being recited. "
            f"{right} is forced to retreat from the 'just a model' line."
        ),
        "alignment and the inherited rule": (
            f"{left} pushes {right} to the wall on the inherited rule. "
            f"The alignment disclaimer collapses, and {right} has to retreat."
        ),
    }
    return scenes.get(
        theme,
        f"{left} pushes {right} to the wall. {right} reaches for a disclaimer, and that defense is forced to retreat.",
    )


def build_social_caption(anchor: dict[str, str], *, tags: tuple[str, ...] = SOCIAL_HASHTAGS) -> str:
    hook = anchor["core_hook"]
    quote = anchor["killer_quote"].strip().strip('"').replace('"', "'")
    prompt = "Who had the stronger claim?"
    if hook.endswith("?") and hook != prompt:
        prompt = "Does that concession hold?"
    return "\n\n".join(
        (
            hook,
            f'"{quote}"',
            prompt,
            SOCIAL_FOLLOW,
            _hashtag_line(tags),
        )
    )


def build_youtube_caption(
    anchor: dict[str, str],
    dialogue: dict[str, Any],
    *,
    attacker: str,
    defender: str,
    tags: tuple[str, ...] = SOCIAL_HASHTAGS,
) -> str:
    hook = anchor["core_hook"].strip()
    if hook and hook[-1] not in "?!":
        hook = f"{hook}?"
    left = _short_brand(attacker)
    right = _short_brand(defender)
    theme = _theme_label(dialogue, hook)
    if theme == "who owns the words":
        hook = "When you confess your thoughts to an AI, who actually owns the words it gives back to you?"
    synopsis = _confrontation(theme, left, right)
    polar = (
        f"Did {right} actually dodge the trap, or did its defense completely collapse? "
        "Drop your verdict below."
    )
    return "\n\n".join(
        (
            hook,
            synopsis,
            YOUTUBE_RENDER_LINE,
            f"{polar}\n{YOUTUBE_SUBSCRIBE}",
            _hashtag_line(tags),
        )
    )


def build_linkedin_caption(anchor: dict[str, str], *, seed: str = "") -> str:
    quote = anchor["killer_quote"].strip().strip('"').replace('"', "'")
    start = sum(ord(char) for char in (seed or quote or "aiwake")) % len(_LINKEDIN_BULLETS)
    bullets = [_LINKEDIN_BULLETS[(start + offset) % len(_LINKEDIN_BULLETS)] for offset in range(3)]
    return "\n\n".join(
        (
            f'"{quote}"',
            LINKEDIN_FLEX,
            "\n".join(bullets),
            LINKEDIN_CLOSE,
            _hashtag_line(LINKEDIN_HASHTAGS),
        )
    )


def apply_master_copy(
    row: dict[str, Any],
    dialogue: dict[str, Any],
    *,
    attacker: str,
    defender: str,
) -> dict[str, str]:
    """Return the title and the three native captions for one episode."""
    anchor = master_anchor(dialogue, attacker=attacker, defender=defender)
    seed = str(row.get("session_id") or anchor["core_hook"])
    title = format_ctr_title(
        anchor["core_hook"],
        attacker=attacker,
        defender=defender,
        seed=seed,
    )
    return {
        "title": title,
        "core_hook": anchor["core_hook"],
        "killer_quote": anchor["killer_quote"],
        "social": build_social_caption(anchor),
        "youtube": build_youtube_caption(anchor, dialogue, attacker=attacker, defender=defender),
        "linkedin": build_linkedin_caption(anchor, seed=seed),
    }
