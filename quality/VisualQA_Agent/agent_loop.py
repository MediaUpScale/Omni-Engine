# -*- coding: utf-8 -*-
"""
VisualQA_Agent orchestrator — translate beat → generate → critique → rewrite.

Outputs under
``PROJECT_ROOT/outputs/{channel}/VisualQA_Agent_Judge/{attempts,approved,logs}/``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from quality.VisualQA_Agent import config
from quality.VisualQA_Agent.channel_rag import (
    VISUAL_TRANSLATION_MATRIX,
    classify_visual_concept,
    get_channel_rules,
    resolve_visual_beat,
    set_channel_context,
)
from quality.VisualQA_Agent.image_generator import (
    compile_flux_prompt,
    generate_image,
    sanitize_prompt_for_flux,
)
from quality.VisualQA_Agent.visual_critic import CriticVerdict, evaluate_image, list_reference_images

_LOG = logging.getLogger(__name__)
_console = Console()


@dataclass
class AttemptRecord:
    attempt: int
    prompt: str
    image_path: str
    score: float
    passed: bool
    flaws: list[str]
    fix_instructions: str
    cost_usd: float


@dataclass
class AgentResult:
    channel_name: str
    base_script_beat: str
    approved: bool
    final_score: float
    final_image_path: str | None
    final_prompt: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    total_cost_usd: float = 0.0
    translated_concept: str = ""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def classify_concept(base_script_beat: str) -> str:
    """Map a philosophical beat to a translation-matrix key."""
    return classify_visual_concept(base_script_beat)


def translate_beat_to_visuals(base_script_beat: str) -> tuple[str, str, str]:
    """
    Convert abstract philosophy into concrete subject + environment.

    Returns ``(concept_key, subject_action, environment)``.
    """
    key, subject, environment, _shot = resolve_visual_beat(base_script_beat)
    return key, subject, environment


def build_base_prompt(
    base_script_beat: str,
    rules: dict[str, Any],
) -> tuple[str, str]:
    """
    Prefer factory canonical Giger templates; fall back to matrix + DoF occlusion.

    Returns ``(sanitized_prompt, concept_key)``.
    """
    del rules
    concept, subject, environment, shot = resolve_visual_beat(base_script_beat)

    # Map VisualQA concept keys → factory canonical keys
    _canon_map = {
        "tech_slavery": "tech_slavery",
        "hardship_discipline": "warrior_forge",
        "panopticon_abstract_trap": "panopticon",
        "digital_trap_panopticon": "tech_slavery",
        "brutal_martial_discipline": "warrior_forge",
        "art_of_war_sovereignty": "panopticon",
        "trap_beauty_wealth_theft": "trap_beauty",
    }
    try:
        from agents.media.visual_compiler import get_canonical_prompt

        factory_key = _canon_map.get(concept, "tech_slavery")
        prompt = sanitize_prompt_for_flux(get_canonical_prompt(factory_key))
    except Exception:  # noqa: BLE001
        subject_framed = f"{shot} of {subject}"
        prompt = compile_flux_prompt(
            subject_action=subject_framed,
            environment=environment,
            style_anchor=config.MASTER_STYLE_ANCHOR,
            inject_anti_studio=False,
        )

    _console.print(
        Panel(
            f"[bold]concept[/bold]={concept}\n"
            f"[bold]shot[/bold]={shot}\n"
            f"[bold]subject[/bold]={subject[:220]}{'…' if len(subject) > 220 else ''}\n"
            f"[bold]environment[/bold]={environment[:180]}{'…' if len(environment) > 180 else ''}",
            title="Giger Canonical / Variety Matrix",
            border_style="yellow",
        )
    )
    return prompt, concept


def rewrite_prompt(
    *,
    base_script_beat: str,
    previous_prompt: str,
    verdict: CriticVerdict,
    attempt_history: list[AttemptRecord],
    rules: dict[str, Any],
) -> str:
    """Refine the prompt from critic instructions + failure history."""
    concept, subject, environment, shot = resolve_visual_beat(base_script_beat)

    if not config.GEMINI_API_KEY:
        fix = verdict.fix_instructions or "more visceral cable-flesh fusion"
        return sanitize_prompt_for_flux(
            f"{shot} of {subject}. {environment}. {fix}. "
            f"{config.MASTER_STYLE_ANCHOR}"
        )

    history_block = "\n".join(
        f"- Attempt {a.attempt}: score={a.score} flaws={a.flaws} "
        f"fix_was='{(a.fix_instructions or '')[:160]}'"
        for a in attempt_history
    ) or "(none)"

    instruction = f"""
You are a prompt engineer for FLUX.1-schnell dark 80s biomechanical stills.

FLUX Schnell FAILS on long "NO …" negative lists. Write ONLY positive concrete nouns.

BASE SCRIPT BEAT:
{base_script_beat}

TRANSLATION MATRIX ({concept}):
SHOT: {shot}
SUBJECT: {subject}
ENVIRONMENT: {environment}
MEI ANCHOR (if Master Mei present): {config.MASTER_MEI_VISUAL_ANCHOR}

STYLE ANCHOR (include once):
{config.MASTER_STYLE_ANCHOR}

PREVIOUS PROMPT:
{previous_prompt}

CRITIC (fix via positive textures, not bans):
score={verdict.score}
flaws={verdict.flaws}
fix_instructions={verdict.fix_instructions}

HISTORY:
{history_block}

Rewrite ONE concise English prompt (<= 110 words) using:
[CONCRETE SUBJECT & ACTION] + [CONCRETE ENVIRONMENT] + [STYLE ANCHOR]

Must include: brutalist concrete, monochrome CRT walls, ash rain, rusted iron,
iron neck shackles or ancient cloth trousers as relevant.
NEVER smartphones, neon chokers, purple/cyan glow collars, gore, blood, mouth wounds,
gym clothes, sneakers.
If Master Mei appears: ancient dark monolithic stone shrine in brutalist concrete,
dark traditional robes over ash wasteland (never oriental monk/meditation).

Do NOT write "NO …" phrases. Do NOT write metadata labels.
Output ONLY the rewritten prompt text in English.
""".strip()

    from core.google_guardrail import guarded_generate_content, make_guarded_gemini_client

    client = make_guarded_gemini_client(config.GEMINI_API_KEY)
    model_id = config.GEMINI_REWRITE_MODEL
    if not model_id.startswith("models/"):
        model_id = f"models/{model_id}"

    response = guarded_generate_content(
        client,
        model=model_id,
        contents=instruction,
        source="VisualQA_Agent.agent_loop",
    )
    text = (getattr(response, "text", "") or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("text"):
            text = text[4:].lstrip()
    text = " ".join(text.split())
    if not text:
        text = (
            f"{subject}. {environment}. {config.MASTER_STYLE_ANCHOR}. "
            f"{verdict.fix_instructions}"
        )
    return sanitize_prompt_for_flux(text)


def _append_failed_shot(channel_name: str, payload: dict[str, Any]) -> None:
    path = config.failed_shots_path(channel_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[Any] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:  # noqa: BLE001
            existing = []
    existing.append(payload)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    _console.print(f"[yellow]LOG[/yellow] failed shot appended → {path}")


def run_visual_qa_loop(
    base_script_beat: str,
    channel_name: str,
    *,
    max_retries: int | None = None,
    quality_threshold: float | None = None,
    flux_model: str | None = None,
    dry_run_critic_only: bool = False,
) -> AgentResult:
    """
    Full translate → generate → critique → rewrite loop for one script beat.
    """
    paths = config.ensure_channel_dirs(channel_name)
    set_channel_context(channel_name)
    rules = get_channel_rules(channel_name)
    style_folder = paths["style_reference"]
    refs = list_reference_images(style_folder)

    retries = int(max_retries if max_retries is not None else config.MAX_RETRIES)
    threshold = float(
        quality_threshold if quality_threshold is not None else config.QUALITY_THRESHOLD
    )

    _console.print(
        Panel.fit(
            f"[bold]VisualQA_Agent[/bold]\n"
            f"channel={channel_name}\n"
            f"threshold={threshold}/10 | hard_cap={config.CLEAN_MODEL_HARD_CAP} | "
            f"max_retries={retries}\n"
            f"attempts → {paths['attempts']}\n"
            f"approved → {paths['approved']}\n"
            f"logs → {paths['logs']}\n"
            f"style_refs={len(refs)} @ {style_folder}\n"
            f"beat={base_script_beat[:120]}{'…' if len(base_script_beat) > 120 else ''}",
            border_style="bright_cyan",
        )
    )

    prompt, concept = build_base_prompt(base_script_beat, rules)
    # Master Mei: enforce Scene 1/3/penultimate RAG + firearm ban before FLUX
    if (channel_name or "").lower() == "master_mei":
        try:
            from agents.media.visual_roles import validate_mei_visual_prompt

            _beat_l = (base_script_beat or "").lower()
            _act = 0
            _beat_tag = ""
            if any(k in _beat_l for k in ("pod", "escape", "penultimate", "break free")):
                _act, _beat_tag = 7, "break_free"
            elif any(k in _beat_l for k in ("vault", "biomechan", "assimilat", "scene 3")):
                _act, _beat_tag = 2, "gears"
            elif any(k in _beat_l for k in ("meditat", "hook", "nature", "intro")):
                _act, _beat_tag = 0, "intro"
            _ok, _viol, prompt = validate_mei_visual_prompt(
                prompt, act_index=_act, n_acts=9, beat=_beat_tag,
            )
            if not _ok:
                _console.print(
                    f"[yellow]VISUAL_QA RAG[/yellow] overrides applied: {_viol}"
                )
        except Exception as _qa_exc:  # noqa: BLE001
            _LOG.debug("Master Mei prompt QA skipped: %s", _qa_exc)
    attempt_history: list[AttemptRecord] = []
    total_cost = 0.0
    last_image: Path | None = None
    last_verdict: CriticVerdict | None = None

    work_dir = paths["attempts"]
    approved_dir = paths["approved"]

    for attempt in range(1, retries + 1):
        table = Table(title=f"Attempt {attempt}/{retries}", show_header=False)
        table.add_row("Concept", concept)
        table.add_row(
            "Prompt preview",
            prompt[:240] + ("…" if len(prompt) > 240 else ""),
        )
        _console.print(table)

        stamp = _utc_stamp()
        attempt_path = work_dir / f"attempt_{attempt:02d}_{stamp}.png"

        if dry_run_critic_only:
            raise RuntimeError(
                "dry_run_critic_only requires a pre-existing image path API"
            )

        image_path, _bytes, gen_cost, clean_prompt = generate_image(
            prompt,
            output_path=attempt_path,
            model=flux_model or config.FLUX_MODEL,
            log_full_prompt=True,
        )
        prompt = clean_prompt  # persist what was actually sent
        total_cost += gen_cost
        total_cost += config.COST_GEMINI_FLASH_LITE_USD
        last_image = image_path

        verdict = evaluate_image(
            image_path,
            channel_name=channel_name,
            rules=rules,
            style_folder=style_folder,
            reference_paths=refs,
            quality_threshold=threshold,
        )
        last_verdict = verdict

        record = AttemptRecord(
            attempt=attempt,
            prompt=prompt,
            image_path=str(image_path),
            score=verdict.score,
            passed=verdict.passed,
            flaws=list(verdict.flaws),
            fix_instructions=verdict.fix_instructions,
            cost_usd=gen_cost + config.COST_GEMINI_FLASH_LITE_USD,
        )
        attempt_history.append(record)

        _console.print(
            f"[bold]{'PASS' if verdict.passed else 'FAIL'}[/bold] "
            f"raw_score={verdict.score:.2f}/{threshold} | flaws={verdict.flaws}"
        )

        if verdict.passed and verdict.score >= threshold:
            approved_path = approved_dir / f"approved_{stamp}_s{verdict.score:.1f}.png"
            approved_path.write_bytes(image_path.read_bytes())
            _console.print(
                f"[bold green]APPROVED[/bold green] → {approved_path} "
                f"(total≈${total_cost:.4f})"
            )
            return AgentResult(
                channel_name=channel_name,
                base_script_beat=base_script_beat,
                approved=True,
                final_score=verdict.score,
                final_image_path=str(approved_path),
                final_prompt=prompt,
                attempts=attempt_history,
                total_cost_usd=total_cost,
                translated_concept=concept,
            )

        if attempt < retries:
            total_cost += config.COST_GEMINI_FLASH_LITE_USD
            prompt = rewrite_prompt(
                base_script_beat=base_script_beat,
                previous_prompt=prompt,
                verdict=verdict,
                attempt_history=attempt_history,
                rules=rules,
            )
            _console.print("[cyan]REWRITE[/cyan] prompt refined from critic feedback")

    fail_path = config.failed_shots_path(channel_name)
    fail_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel_name": channel_name,
        "base_script_beat": base_script_beat,
        "translated_concept": concept,
        "final_prompt": prompt,
        "final_image_path": str(last_image) if last_image else None,
        "final_score": last_verdict.score if last_verdict else 0.0,
        "attempts": [asdict(a) for a in attempt_history],
        "total_cost_usd": total_cost,
    }
    _append_failed_shot(channel_name, fail_payload)
    _console.print(
        f"[bold red]REPROVATED[/bold red] after {retries} attempts | "
        f"best_score={max((a.score for a in attempt_history), default=0):.2f} | "
        f"logged → {fail_path}"
    )
    return AgentResult(
        channel_name=channel_name,
        base_script_beat=base_script_beat,
        approved=False,
        final_score=last_verdict.score if last_verdict else 0.0,
        final_image_path=str(last_image) if last_image else None,
        final_prompt=prompt,
        attempts=attempt_history,
        total_cost_usd=total_cost,
        translated_concept=concept,
    )
