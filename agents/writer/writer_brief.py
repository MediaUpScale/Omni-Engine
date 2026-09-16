# -*- coding: utf-8 -*-
"""Input contract for the freeform script brain.

Two entry points produce the same brief, so everything downstream of the writer
(judge gate, atmosphere, image generation, assemble) is identical either way:

  * ``WriterBrief.from_theme(...)``  — theme / subtheme, as today.
  * ``WriterBrief.from_quote(...)``  — a seed quote to develop into a script.

Nothing here decides line count, cadence, or shape. Those belong to the writer.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Literal

BriefMode = Literal["emotional", "theme", "quote", "paraphrase"]

PHILOSOPHICAL_ANCHORS: tuple[str, ...] = (
    "Seneca on how we suffer more in imagination than reality",
    "Marcus Aurelius on letting go of what we cannot control",
    "Friedrich Nietzsche on the burden of masks and unspoken truths",
    "Albert Camus on quiet persistence and invincible summers",
    "Carl Jung on confronting our own shadow before blaming others",
    "Søren Kierkegaard on understanding life backwards while living it forward",
    "Arthur Schopenhauer on the pain of loneliness vs peace of solitude",
)
PSYCHOLOGICAL_REALITY_ANCHORS: tuple[str, ...] = (
    "how silence becomes an answer when someone repeatedly avoids connection",
    "why checking a phone cannot create the care that is missing",
    "how guilt steals the presence a child needs right now",
    "why control makes ordinary family moments feel unsafe",
    "how replaying yesterday quietly takes attention from today",
)
COGNITIVE_BRIDGE_STYLES: tuple[tuple[str, str], ...] = (
    (
        "direct reality check",
        "Beat 2 makes a plain observation, such as 'Real life rarely reads "
        "theory,' 'Because memory refuses logic,' or 'Theory breaks against "
        "heartbreak.'",
    ),
    (
        "reflective question",
        "Beat 2 asks a natural question, such as 'Why does the heart argue?', "
        "'Where does that calm go?', or 'Why replay an empty room?'",
    ),
    (
        "varied connective",
        "Beat 2 uses a natural connective such as 'And still...', 'Yet...', "
        "'Except we crave answers...', or 'Even when silence is clear...'.",
    ),
)


def select_single_conflict(niche: str, theme: str = "", subtheme: str = "") -> str:
    """Resolve one episode-level emotional friction from the selected theme."""
    key = f"{theme} {subtheme}".strip().lower().replace("_", " ")
    if str(niche or "").strip().lower() == "parenting":
        if any(term in key for term in ("self compassion", "good enough", "guilt", "perfect", "clean", "chore")):
            return "perfectionism guilt about keeping the home perfect"
        if any(term in key for term in ("presence", "phone", "screen", "time", "distraction")):
            return "screen distraction stealing present childhood time"
        if any(term in key for term in ("sleep", "bedtime", "routine")):
            return "bedtime control replacing calm connection"
        if any(term in key for term in ("anger", "patience", "temper", "yell")):
            return "parental impatience during one recurring moment"
    return key or (
        "one romantic disconnection"
        if str(niche or "").strip().lower() == "relationship"
        else "one parenting tension"
    )


def get_narrative_harness(
    niche: str,
    *,
    theme: str = "",
    subtheme: str = "",
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Choose one episode-level anchor before the writer model is called."""
    chooser = rng or random
    use_philosophy = chooser.random() < 0.70
    anchor = chooser.choice(
        PHILOSOPHICAL_ANCHORS
        if use_philosophy
        else PSYCHOLOGICAL_REALITY_ANCHORS
    )
    bridge_style, bridge_direction = chooser.choice(COGNITIVE_BRIDGE_STYLES)
    return {
        "narrative_mode": (
            "famous_thinker_hook" if use_philosophy else "original_freewriting"
        ),
        "use_philosophy": use_philosophy,
        "narrative_anchor": anchor,
        "narrative_niche": str(niche or "relationship").strip() or "relationship",
        "cognitive_bridge_style": bridge_style,
        "cognitive_bridge_direction": bridge_direction,
        "single_conflict": select_single_conflict(niche, theme, subtheme),
    }


@dataclass(frozen=True)
class WriterBrief:
    """What the writer is being asked to make. Not how to make it."""

    mode: BriefMode
    module: str = "relationship"
    theme: str = ""
    subtheme: str = ""
    seed_quote: str = ""
    seed_attribution: str = ""
    # Free text the caller wants in front of the writer (concrete details from
    # the theme bank, a producer note, a subject the episode must touch).
    context_notes: tuple[str, ...] = ()
    # Openings / images already used this batch, so drafts don't converge.
    avoid: tuple[str, ...] = ()
    # Judge feedback from a rejected attempt. Empty on the first pass.
    revision_note: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_theme(
        cls,
        *,
        theme: str,
        subtheme: str = "",
        module: str = "relationship",
        context_notes: list[str] | tuple[str, ...] | None = None,
        avoid: list[str] | tuple[str, ...] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> WriterBrief:
        t = str(theme or "").strip()
        if not t:
            raise ValueError("from_theme requires a theme")
        return cls(
            mode="theme",
            module=str(module or "relationship").strip() or "relationship",
            theme=t,
            subtheme=str(subtheme or "").strip(),
            context_notes=tuple(str(x).strip() for x in (context_notes or ()) if str(x).strip()),
            avoid=tuple(str(x).strip() for x in (avoid or ()) if str(x).strip()),
            meta=dict(meta or {}),
        )

    @classmethod
    def from_quote(
        cls,
        *,
        quote: str,
        attribution: str = "",
        theme: str = "",
        module: str = "relationship",
        context_notes: list[str] | tuple[str, ...] | None = None,
        avoid: list[str] | tuple[str, ...] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> WriterBrief:
        """Seed a script from one line by a writer or philosopher.

        The quote is a starting point to develop, not something to recite. The
        attribution is carried in metadata only — the script never names it.
        """
        q = str(quote or "").strip()
        if not q:
            raise ValueError("from_quote requires a quote")
        return cls(
            mode="quote",
            module=str(module or "relationship").strip() or "relationship",
            theme=str(theme or "").strip(),
            seed_quote=q,
            seed_attribution=str(attribution or "").strip(),
            context_notes=tuple(str(x).strip() for x in (context_notes or ()) if str(x).strip()),
            avoid=tuple(str(x).strip() for x in (avoid or ()) if str(x).strip()),
            meta=dict(meta or {}),
        )

    @classmethod
    def from_emotional(
        cls,
        *,
        theme: str,
        subtheme: str = "",
        module: str = "relationship",
        context_notes: list[str] | tuple[str, ...] | None = None,
        avoid: list[str] | tuple[str, ...] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> WriterBrief:
        """Request an original emotional reflection, unconstrained by literal visuals."""
        t = str(theme or "").strip() or "emotional maturity"
        return cls(
            mode="emotional",
            module=str(module or "relationship").strip() or "relationship",
            theme=t,
            subtheme=str(subtheme or "").strip(),
            context_notes=tuple(str(x).strip() for x in (context_notes or ()) if str(x).strip()),
            avoid=tuple(str(x).strip() for x in (avoid or ()) if str(x).strip()),
            meta=dict(meta or {}),
        )

    @classmethod
    def from_paraphrase(
        cls,
        *,
        aphorism: str,
        theme: str = "",
        module: str = "relationship",
        source_id: str = "",
        meta: dict[str, Any] | None = None,
    ) -> WriterBrief:
        """Request a light structural paraphrase of one complete aphorism."""
        text = str(aphorism or "").strip()
        if not text:
            raise ValueError("from_paraphrase requires an aphorism")
        details = dict(meta or {})
        if source_id:
            details["aphorism_id"] = source_id
        return cls(
            mode="paraphrase",
            module=str(module or "relationship").strip() or "relationship",
            theme=str(theme or "").strip(),
            seed_quote=text,
            meta=details,
        )

    def with_revision(self, note: str) -> WriterBrief:
        """Same assignment, carrying the judge's reason the last draft failed."""
        from dataclasses import replace

        return replace(self, revision_note=str(note or "").strip())

    @property
    def label(self) -> str:
        if self.mode == "quote":
            head = self.seed_quote[:48].rstrip()
            return f"quote:{head}…" if len(self.seed_quote) > 48 else f"quote:{head}"
        return f"{self.theme}/{self.subtheme}" if self.subtheme else self.theme

    def assignment_block(self) -> str:
        """The part of the writer prompt that changes per episode."""
        parts: list[str] = []
        from core.economic_reel_lofi import config as lofi_cfg

        if self.mode == "quote":
            parts.append(lofi_cfg.hook_line_brevity_clause())
            parts.append(
                "SEED IDEA — one line from another writer:\n"
                f"  \u201c{self.seed_quote}\u201d\n"
                "Develop what is true in it into your own piece. Do not quote it, "
                "do not paraphrase it as a line, do not name or gesture at whoever "
                "said it. It is the thought you are arguing with or extending, not "
                "material to reuse. If the seed is abstract, find the specific human "
                "situation underneath it and write that instead.\n"
                "PARABLE ARC: carry one concrete parable through four ordered "
                "movements across the beats: (1) concept/setup, (2) contact creates "
                "conflict or pain, (3) retreat creates loneliness or cost, "
                "(4) a workable equilibrium that preserves connection without "
                "repeating the harm. Each movement must change the situation; do "
                "not flatten the seed into repeated commentary. End with a direct, "
                "usable instruction the listener can act on; never end with "
                "\"that's what love/healing/lasting looks like.\""
            )
            if self.theme:
                parts.append(f"It should land somewhere near: {self.theme.replace('_', ' ')}")
        elif self.mode == "paraphrase":
            source_structure = str(
                (self.meta or {}).get("source_structure") or ""
            ).lower()
            starts = [
                word.lower()
                for word in re.findall(
                    r"(?:^|[.!?]\s+)([A-Za-z']+)",
                    str(self.seed_quote or ""),
                )
            ]
            repeated_opening = (
                max((starts.count(word) for word in set(starts)), default=0) >= 3
            )
            if "anaphora" in source_structure or repeated_opening:
                variation_rule = (
                    "This source is ANAPHORA: its repeated opening and repeated "
                    "closing refrain are protected. Keep every parallel repeat "
                    "as its own sentence; do not merge or split those sentences. "
                    "Preserve the repeated opening and refrain count. Create "
                    "structural variation inside exactly 1–2 sentences by "
                    "reordering clause elements or changing word order/tense. "
                    "Never join two repeats with 'or'."
                )
            else:
                variation_rule = (
                    "Make exactly one sentence-level structural change: either "
                    "merge one adjacent pair of short sentences into one, OR split "
                    "one sentence into two. Do not do both. Do not keep a 1:1 "
                    "sentence mirror."
                )
            parts.append(lofi_cfg.hook_line_brevity_clause(paraphrase=True))
            parts.append(
                "SOURCE APHORISM:\n"
                f"  \u201c{self.seed_quote}\u201d\n"
                "Make one light rewording only. Preserve the progression, repeated "
                "refrain, approximate total word count, rhythm, and meaning. "
                f"Substitute roughly 20–30% of the words. {variation_rule} "
                "Do not expand "
                "it into a story, add an arc, add examples, explain it, or improve "
                "its argument. Return a genuine variation, not a new composition."
            )
        elif self.mode == "emotional":
            niche = str(self.module or "relationship").strip().lower()
            narrative_mode = str(
                (self.meta or {}).get("narrative_mode") or "philosophy"
            )
            narrative_anchor = str(
                (self.meta or {}).get("narrative_anchor")
                or "a specific psychological paradox"
            )
            bridge_style = str(
                (self.meta or {}).get("cognitive_bridge_style")
                or "a naturally varied bridge"
            )
            bridge_direction = str(
                (self.meta or {}).get("cognitive_bridge_direction")
                or "Use either a direct reality check, a reflective question, "
                "or a connective that fits this story."
            )
            single_conflict = str(
                (self.meta or {}).get("single_conflict")
                or self.subtheme
                or self.theme
                or "one emotional friction"
            )
            parts.append(
                "EVERYDAY CADENCE & ANAPHORA EPISODE:\n"
                "Write a relatable, emotionally devastating micro-narrative in "
                "plain, everyday language.\n"
                f"  Niche: {niche}\n"
                f"  Theme: {self.theme.replace('_', ' ')}\n"
                + (
                    f"  Angle: {self.subtheme.replace('_', ' ')}\n"
                    if self.subtheme
                    else ""
                )
                + f"  Narrative mode: {narrative_mode}\n"
                + f"  Selected anchor: {narrative_anchor}\n"
                + f"  Selected Beat 2 bridge style: {bridge_style}\n"
                + f"  Beat 2 direction: {bridge_direction}\n"
                + f"  ONE LOCKED CONFLICT: {single_conflict}\n"
                + "Use the proven Momma Circle blueprint exactly: beat 1 profound "
                "universal hook or famous-thinker premise; beat 2 immediate reality "
                "check; beat 3 specific tangible daily behavior; beat 4 painful "
                "contrast showing what happens instead; beat 5 time or presence "
                "lost; beat 6 emotional or psychological reframe; beat 7 tangible "
                "release action; beat 8 poignant, grounded peace. Beats 1–3 must "
                "be one continuous "
                "logical bridge, never an abrupt jump from abstract philosophy to "
                "unrelated advice. Each later beat must follow because of the "
                "previous beat. "
                "Do not default mechanically to 'Yet'. Yet remains valid, but it "
                "must compete naturally across episodes with direct observations, "
                "reflective questions, and other connectives. Follow the selected "
                "Beat 2 bridge style above. If Beat 1 names only a philosopher, "
                "Beat 2 must explicitly name the concrete subject—partner, parent, "
                "mother, father, or child—inside that bridge; never postpone the "
                "subject until Beat 3. When famous-thinker mode is selected, "
                "state the philosopher and plain idea "
                "in beat 1 or 2, then connect it to everyday hurt. Never invent a "
                "quotation. Never use a grandfather, grandmother, father, mother, "
                "or loose family anecdote as the narrative authority. In "
                "original-freewriting mode, open with an original universal "
                "behavioral truth tied directly to the locked theme; do not name "
                "a thinker or imitate a quotation. POV SUBJECT INTEGRITY: choose "
                "one unified human "
                "relationship and never switch it. Relationship scripts stay with "
                "the same romantic partner dynamic for all eight beats. Parenting "
                "scripts stay with the same parent-child dynamic for all eight "
                "beats. A thinker is only the hook, not a second story subject. "
                "Never jump from father, mother, child, friend, or family to a "
                "romantic partner, or the reverse. "
                "SINGLE-CONFLICT UNITY: every beat must deepen only the locked "
                "conflict above. Never introduce a second flaw, wound, habit, or "
                "lesson. Parenting scripts must not mix guilt/perfectionism, "
                "chores, anger/patience, screen distraction, and bedtime control. "
                "Choose only the locked family tension and keep every behavior, "
                "cost, reframe, and action causally attached to it. "
                "Use intentional repetition of "
                "core anchor words for "
                "momentum, not eight disconnected sayings. Beats 6–8 must deliver "
                "one clear, useful lesson or psychological concept earned by the "
                "story—never a loose motivational slogan. SUBJECT GROUNDING: beat "
                "1 or beat 2 must name the person plainly—child, parent, partner, "
                "mother, father, friend, or another concrete subject—before using "
                "they, them, their, he, or she. Never open with an unclear pronoun. "
                "Every beat contains 4–6 simple spoken words, "
                "and the full script contains at most 48 words. CAMERA CONTRACT: "
                "use in order a wide establishing exterior, architectural interior "
                "framing, tactile prop detail, dramatic close-up profile, motion "
                "transit, rain or reflective glass, walking silhouette, and "
                "expansive panoramic closure. SPATIAL ISOLATION: bedrooms and "
                "nurseries are strictly indoor, with a cozy interior and wooden "
                "floorboards. Exterior scenes never contain beds, mattresses, "
                "dressers, or other indoor furniture. Human subjects stay upright: "
                "standing, walking, seated properly on furniture, or tucked into "
                "bed with visible pillows—never lying on floors, rugs, streets, or "
                "doorsteps, and never crawling or prone."
            )
        else:
            parts.append(lofi_cfg.hook_line_brevity_clause())
            parts.append(f"SUBJECT: {self.theme.replace('_', ' ')}")
            if self.subtheme:
                parts.append(f"NARROWER ANGLE: {self.subtheme.replace('_', ' ')}")
        if self.context_notes:
            joined = "\n".join(f"  - {n}" for n in self.context_notes)
            parts.append(
                "CONTEXT (raw material you may ignore entirely — never quote it "
                f"verbatim):\n{joined}"
            )
        if self.avoid:
            joined = "\n".join(f"  - {n}" for n in self.avoid)
            parts.append(
                "ALREADY USED in this batch — do not repeat these openings, images, "
                f"or moves:\n{joined}"
            )
        if self.revision_note:
            parts.append(
                "THE LAST DRAFT WAS REJECTED. Do not patch it — throw it out and "
                "write a different piece. Here is why it failed:\n"
                f"{self.revision_note}"
            )
        return "\n\n".join(parts)
