# -*- coding: utf-8 -*-
"""End-to-end pipeline: debate -> voice -> video.

This is the module's public façade and the only thing the parent engine needs to
import::

    from channels_config.aiwake import run_pipeline
    result = run_pipeline(topic="Is grief a slow update?", turns=4)

All wiring decisions (which observers attach, which engine speaks, whether video
is produced) are made here, so ``room.py`` and ``orchestrator.py`` stay free of
side-effect policy.
"""
from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from .contracts import DebateTranscript
    from .media.audio import build_engine
    from .memory import DebateMemory, script_fingerprint, scripts_overlap
    from .models.llm_factory import force_offline
    from .observers.core import (
        ConsoleObserver,
        MemoryObserver,
        MetricsObserver,
        TranscriptObserver,
        VoiceObserver,
    )
    from .orchestrator import Provocateur
    from .room import DebateRoom, RoomEvent
    from .settings import AiwakeSettings, load_settings, resolve_outputs_dir
    from .utils.event_bus import (
        ON_PIPELINE_FINISH,
        ON_PIPELINE_START,
        attach_optional_plugins,
        emit,
    )
except ImportError:  # pragma: no cover — standalone extraction
    from contracts import DebateTranscript  # type: ignore[no-redef]
    from media.audio import build_engine  # type: ignore[no-redef]
    from memory import DebateMemory, script_fingerprint, scripts_overlap  # type: ignore[no-redef]
    from models.llm_factory import force_offline  # type: ignore[no-redef]
    from observers.core import (  # type: ignore[no-redef]
        ConsoleObserver,
        MemoryObserver,
        MetricsObserver,
        TranscriptObserver,
        VoiceObserver,
    )
    from orchestrator import Provocateur  # type: ignore[no-redef]
    from room import DebateRoom, RoomEvent  # type: ignore[no-redef]
    from settings import AiwakeSettings, load_settings, resolve_outputs_dir  # type: ignore[no-redef]
    from utils.event_bus import (  # type: ignore[no-redef]
        ON_PIPELINE_FINISH,
        ON_PIPELINE_START,
        attach_optional_plugins,
        emit,
    )

_LOG = logging.getLogger("aiwake.pipeline")


@dataclass(slots=True)
class PipelineResult:
    """What a full run produced.

    Attributes:
        transcript: The debate record.
        video_path: Rendered MP4, or None when rendering was skipped or failed.
        exchanges: Completed provocation/rebuttal pairs.
        end_reason: ``complete``, ``aborted`` or ``interrupted``.
        audio_seconds: Total synthesised voice duration.
    """

    transcript: DebateTranscript
    video_path: Path | None
    exchanges: int
    end_reason: str
    audio_seconds: float
    dialogue_end_reason: str

    @property
    def succeeded(self) -> bool:
        return self.end_reason == "complete" and self.exchanges > 0


@dataclass(slots=True)
class BulkPipelineResult:
    """What a ``--quantity N`` run produced."""

    items: list[PipelineResult] = field(default_factory=list)
    requested: int = 0
    succeeded: int = 0
    skipped_duplicates: int = 0

    @property
    def complete(self) -> bool:
        return self.succeeded == self.requested and self.requested > 0


_BULK_SCRIPT_RETRIES = 3
_PROTOTYPE_NAMES = {
    "aiwake_full_battle_v2.mp4",
}


def archive_animation_prototypes(animation_dir: Path) -> list[Path]:
    """Move test-era animation artifacts out of the production namespace."""
    root = Path(animation_dir)
    archive = root / "test_archive"
    archive.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for item in tuple(root.iterdir()):
        if item == archive:
            continue
        name = item.name.lower()
        prototype = (
            name in _PROTOTYPE_NAMES
            or name.startswith(("test_avatar_battle", "proof_", "spoken_test_", "synthetic_test_"))
            or "_test_render_battle" in name
            or "_render." in name
            or "_render_" in name
        )
        if not prototype:
            continue
        destination = archive / item.name
        if destination.exists():
            suffix = int(time.time())
            destination = archive / f"{item.stem}_{suffix}{item.suffix}"
        shutil.move(str(item), str(destination))
        moved.append(destination)
    if moved:
        _LOG.info("archived %d prototype animation artifact(s) -> %s", len(moved), archive)
    return moved


def _transcript_script(result: PipelineResult) -> str:
    return result.transcript.to_script()


def _record_produced_script(memory: DebateMemory, script: str) -> None:
    if not script:
        return
    memory.note_script(script)
    memory.flush()


def _script_conflicts(script: str, seen_scripts: Sequence[str], seen_fps: set[str]) -> bool:
    fingerprint = script_fingerprint(script)
    if fingerprint and fingerprint in seen_fps:
        return True
    return any(scripts_overlap(script, prior) for prior in seen_scripts)


def _require_production_memory(topic: str | None) -> object:
    from .tools.supermemory_bridge import HISTORY_CONTAINER, get_bridge

    bridge = get_bridge()
    if not bridge.is_active:
        raise RuntimeError(
            "production animation requires Supermemory at localhost:6767; "
            "the self-healing startup hook could not make it ready"
        )
    _LOG.info(
        "Supermemory production gate ready at %s (container_tag=%s)",
        bridge.base_url,
        HISTORY_CONTAINER,
    )
    if topic and bridge.check_topic_similarity(topic):
        raise ValueError(f"Supermemory rejected repetitive production topic: {topic}")
    return bridge


def _persist_production_package(
    *,
    transcript: DebateTranscript,
    video_path: Path,
    media_dir: Path,
) -> None:
    from .settings import resolve_store_dir
    from .tools.backfill_metadata import (
        build_record,
        load_transcript_file,
        persist_record,
        probe_duration_s,
    )
    from .tools.post_planner import run_planner
    from .tools.supermemory_bridge import remember_approved_session
    from modules.distribution_contract import content_library_path

    transcript_dir = resolve_store_dir() / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"{transcript.session_id}.json"
    transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    doc = load_transcript_file(transcript_path)
    if doc is None:
        raise RuntimeError(f"could not reload production transcript {transcript_path}")
    record = build_record(Path(video_path), doc)
    persist_record(
        record,
        library_path=content_library_path("aiwake"),
        duration_s=probe_duration_s(Path(video_path)),
    )
    utterances = list(transcript.utterances)
    hook = utterances[0].text if utterances else ""
    quote = utterances[-1].text if utterances else ""
    remember_approved_session(transcript.session_id, transcript.topic, hook, quote)
    planner_path, entries = run_planner(outputs_dir=media_dir)
    if not any(str(item.get("session_id") or "") == transcript.session_id for item in entries):
        raise RuntimeError(
            f"Post Planner did not include production session {transcript.session_id}"
        )
    _LOG.info(
        "production package registered: library=%s planner=%s session=%s",
        content_library_path("aiwake"),
        planner_path,
        transcript.session_id,
    )


def run_pipeline(
    *,
    topic: str | None = None,
    turns: int | None = None,
    mode: Literal["fixed", "cornered"] | None = None,
    provocation_focus: str | None = None,
    settings: AiwakeSettings | None = None,
    orchestrator_model: str | None = None,
    target_model: str | None = None,
    offline: bool = False,
    with_audio: bool = True,
    with_video: bool = True,
    fresh_memory: bool = False,
    output_dir: Path | None = None,
    quiet: bool = False,
    excluded_topics: Sequence[str] = (),
    excluded_foci: Sequence[str] = (),
    record_script: bool = True,
    dynamic_animation: bool = False,
    enable_cta: bool = False,
    animation_skin: str = "v2",
    left_puppet: str | None = None,
    right_puppet: str | None = None,
    production_publish: bool = False,
) -> PipelineResult:
    """Run a debate and (optionally) produce the video.

    Args:
        topic: Debate subject. Defaults to the configured one.
        turns: Fixed exchange count, or the hard iteration cap in cornered mode.
        mode: Debate-ending strategy. None preserves the configured mode.
        provocation_focus: Attack-angle bias, orthogonal to topic. None keeps YAML.
        settings: Pre-loaded settings; loaded from YAML when omitted.
        orchestrator_model: Override the interrogator's brain. Accepts an alias
            from the model reference dictionary or a full provider slug — the
            programmatic twin of the ``--orchestrator`` CLI flag.
        target_model: Same, for the seat under interrogation.
        offline: Route both seats to the deterministic stub provider — no key,
            no network. Exercises the full media path.
        with_audio: Synthesise voices. False forces the silent engine, which
            still yields a coherent (estimated) timeline for layout checks.
        with_video: Render the MP4. False stops after the transcript.
        fresh_memory: Wipe persisted memory before starting.
        output_dir: Override the media destination.
        quiet: Suppress the live console stream.
        excluded_topics: Subjects already used in a bulk batch.
        excluded_foci: Attack angles already used in a bulk batch.
        record_script: Persist the finished script fingerprint so later runs
            cannot reprint the same video.
        dynamic_animation: False (default) keeps the classic terminal/
            typewriter render 100% intact. True routes the finished
            transcript + per-turn voice tracks to the parametric dual
            face-off avatar animation engine instead — same inputs, same
            ``video_path`` output slot, zero changes to the debate loop.
        enable_cta: Append the dynamic-animation terminal CTA. False (default)
            ends 0.4 seconds after the final spoken word for a seamless loop.
        animation_skin: Versioned dynamic-animation preset (``v1`` or ``v2``).
        left_puppet: Optional orchestrator puppet ID override.
        right_puppet: Optional target puppet ID override.

    Returns:
        A :class:`PipelineResult`. Partial runs still return their transcript
        and any video that could be built from it.
    """
    cfg = settings or load_settings()
    debate_update: dict[str, object] = {}
    if mode is not None:
        debate_update["mode"] = mode
    if provocation_focus is not None:
        debate_update["provocation_focus"] = provocation_focus
    if debate_update:
        cfg = cfg.model_copy(
            update={"debate": cfg.debate.model_copy(update=debate_update)}
        )
    pipeline_t0 = time.perf_counter()
    # Seat overrides first: force_offline() reads the resolved routing, so
    # swapping a brain and then dropping to the stub must not resurrect the
    # configured model.
    if orchestrator_model:
        cfg = cfg.with_model_override("orchestrator", orchestrator_model)
    if target_model:
        cfg = cfg.with_model_override("target", target_model)
    if offline:
        cfg = force_offline(cfg)

    if production_publish:
        if not dynamic_animation or offline:
            raise ValueError("production_publish requires a live dynamic_animation run")
        _require_production_memory(topic)

    memory = DebateMemory(cfg.memory)
    if fresh_memory:
        memory.reset()
        _LOG.info("memory reset")

    room = DebateRoom(cfg, topic=topic)
    media_dir = output_dir or resolve_outputs_dir(cfg)

    voice_observer: VoiceObserver | None = None
    if with_audio or with_video:
        # The silent engine keeps the timeline honest when audio is off.
        audio_cfg = cfg.audio if with_audio else cfg.audio.model_copy(update={"engine": "silent"})
        voice_observer = VoiceObserver(build_engine(audio_cfg), room.session_id)

    attach_optional_plugins()
    emit(
        ON_PIPELINE_START,
        {
            "session_id": room.session_id,
            "topic": room.topic,
            "debate_mode": cfg.debate.mode,
            "provocation_focus": cfg.debate.provocation_focus,
            "cornered_max_duration_s": cfg.debate.cornered_max_duration_s,
            "orchestrator_model": cfg.spec_for("orchestrator").model,
            "target_model": cfg.spec_for("target").model,
            "llm_providers": {
                "orchestrator": cfg.spec_for("orchestrator").provider,
                "target": cfg.spec_for("target").provider,
            },
            "models": {
                "orchestrator": cfg.spec_for("orchestrator").model,
                "target": cfg.spec_for("target").model,
            },
            "audio_engine": (voice_observer.engine.engine_name if voice_observer else cfg.audio.engine),
            "typewriter_gain_db": cfg.audio.typewriter.gain_db,
            "scroll_s": cfg.render.scroll_s,
            "preroll_s": cfg.render.preroll_s,
            "reply_gap_s": cfg.render.reply_gap_s,
            "send_flash_s": cfg.render.send_flash_s,
        },
    )

    room.subscribe(MemoryObserver(memory), TranscriptObserver(room.session_id), MetricsObserver())
    if voice_observer is not None:
        room.subscribe(voice_observer)
    if not quiet:
        room.subscribe(ConsoleObserver(verbose=True))

    provocateur = Provocateur(cfg, memory=memory, room=room)
    video_path: Path | None = None
    audio_seconds = 0.0
    end_reason = "interrupted"
    exchanges = 0
    try:
        result = provocateur.run(
            topic=topic,
            turns=turns,
            excluded_topics=excluded_topics,
            excluded_foci=excluded_foci,
        )
        exchanges = result.exchanges
        end_reason = result.end_reason

        audio_seconds = voice_observer.total_duration_s if voice_observer else 0.0

        if voice_observer is not None:
            room.broadcast(
                RoomEvent.AUDIO_MIXED,
                tracks=len(voice_observer.assets),
                audio_seconds=round(audio_seconds, 3),
                engine=voice_observer.engine.engine_name,
                typewriter_gain_db=cfg.audio.typewriter.gain_db,
                typewriter_core_only=True,
            )

        if with_video and result.transcript.utterances:
            if dynamic_animation:
                # New parametric avatar battle path. Imported lazily so the
                # classic pipeline never needs numpy/soundfile/ffmpeg-pipe
                # machinery when this flag is off (the default).
                try:
                    from .animator_bridge import render_debate_animation  # noqa: PLC0415
                except ImportError:  # pragma: no cover — standalone extraction
                    from animator_bridge import render_debate_animation  # type: ignore[no-redef]

                # Experimental animation battles never mix with production
                # terminal reels: dedicated subfolder under the same
                # {OUTPUT_PATH}/aiwake/ tree, not the shared media_dir.
                animation_dir = media_dir / "animation_clips"
                if production_publish:
                    archive_animation_prototypes(animation_dir)
                try:
                    video_path = render_debate_animation(
                        result.transcript,
                        audio_by_turn=voice_observer.assets if voice_observer else None,
                        output_dir=animation_dir,
                        skin=animation_skin,
                        left_puppet=left_puppet,
                        right_puppet=right_puppet,
                        audio_config=cfg.audio,
                        enable_cta=enable_cta,
                    )
                except Exception as exc:  # noqa: BLE001 — a failed render must not lose the transcript
                    _LOG.error("dynamic_animation render failed: %s", exc)
            else:
                # Legacy terminal/typewriter path — 100% unchanged, still the
                # default. Imported here so a headless run never needs
                # MoviePy/Pillow installed.
                try:
                    from .media.renderer import render_transcript  # noqa: PLC0415
                except ImportError:  # pragma: no cover — standalone extraction
                    from media.renderer import render_transcript  # type: ignore[no-redef]

                try:
                    video_path = render_transcript(
                        result.transcript,
                        cfg,
                        audio_by_turn=voice_observer.assets if voice_observer else None,
                        output_dir=media_dir,
                    )
                except Exception as exc:  # noqa: BLE001 — a failed render must not lose the transcript
                    _LOG.error("render failed: %s", exc)

        if video_path is not None:
            result.transcript.metadata["video_path"] = str(video_path)
            script_path = media_dir / f"aiwake_debate_{result.transcript.session_id}.txt"
            script_path.write_text(result.transcript.to_script(), encoding="utf-8")
            room.broadcast(
                RoomEvent.VIDEO_RENDERED,
                video_path=str(video_path),
                scroll_s=cfg.render.scroll_s,
                preroll_s=cfg.render.preroll_s,
                reply_gap_s=cfg.render.reply_gap_s,
                send_flash_s=cfg.render.send_flash_s,
            )
            if production_publish:
                _persist_production_package(
                    transcript=result.transcript,
                    video_path=video_path,
                    media_dir=media_dir,
                )

        pipeline_result = PipelineResult(
            transcript=result.transcript,
            video_path=video_path,
            exchanges=result.exchanges,
            end_reason=result.end_reason,
            audio_seconds=audio_seconds,
            dialogue_end_reason=result.dialogue_end_reason,
        )
        if record_script:
            _record_produced_script(memory, _transcript_script(pipeline_result))
        return pipeline_result
    finally:
        emit(
            ON_PIPELINE_FINISH,
            {
                "session_id": room.session_id,
                "pipeline_s": time.perf_counter() - pipeline_t0,
                "end_reason": end_reason,
                "debate_mode": cfg.debate.mode,
                "dialogue_end_reason": (
                    room.transcript.metadata.get("dialogue_end_reason")
                ),
                "exchanges": exchanges,
                "video_path": str(video_path) if video_path else None,
            },
        )


def run_bulk_pipeline(
    *,
    quantity: int,
    topic: str | None = None,
    turns: int | None = None,
    mode: Literal["fixed", "cornered"] | None = None,
    provocation_focus: str | None = None,
    settings: AiwakeSettings | None = None,
    orchestrator_model: str | None = None,
    target_model: str | None = None,
    offline: bool = False,
    with_audio: bool = True,
    with_video: bool = True,
    fresh_memory: bool = False,
    output_dir: Path | None = None,
    quiet: bool = False,
    dynamic_animation: bool = False,
    enable_cta: bool = False,
    animation_skin: str = "v2",
    left_puppet: str | None = None,
    right_puppet: str | None = None,
    production_publish: bool = False,
) -> BulkPipelineResult:
    """Produce ``quantity`` original videos. Never reprints a prior script.

    Each item draws a fresh topic (unless ``topic`` is pinned) and a distinct
    provocation focus, then fingerprints the finished transcript. A collision
    with this batch or with persisted memory is retried on a new opening axis.
    ``--fresh-memory`` wipes history once, before the first item.
    """
    qty = max(1, int(quantity))
    cfg = settings or load_settings()
    if fresh_memory:
        DebateMemory(cfg.memory).reset()
        _LOG.info("memory reset for bulk run of %d", qty)

    history = DebateMemory(cfg.memory)
    seen_scripts = [" ".join(tokens) for tokens in history.state.script_token_prints]
    seen_fps = {item for item in history.state.script_fingerprints if item}
    used_topics: list[str] = list(history.recent_topics())
    used_foci: list[str] = list(history.recent_focus_categories())

    items: list[PipelineResult] = []
    skipped = 0
    shared = {
        "turns": turns,
        "mode": mode,
        "provocation_focus": provocation_focus,
        "settings": cfg,
        "orchestrator_model": orchestrator_model,
        "target_model": target_model,
        "offline": offline,
        "with_audio": with_audio,
        "with_video": with_video,
        "output_dir": output_dir,
        "quiet": quiet,
        "dynamic_animation": dynamic_animation,
        "enable_cta": enable_cta,
        "animation_skin": animation_skin,
        "left_puppet": left_puppet,
        "right_puppet": right_puppet,
        "production_publish": production_publish,
    }

    for index in range(qty):
        accepted: PipelineResult | None = None
        item_topic = topic
        for attempt in range(1, _BULK_SCRIPT_RETRIES + 1):
            _LOG.info(
                "bulk item %d/%d attempt %d topic=%s",
                index + 1,
                qty,
                attempt,
                item_topic or "(randomised)",
            )
            result = run_pipeline(
                topic=item_topic,
                fresh_memory=False,
                excluded_topics=() if item_topic else tuple(used_topics),
                excluded_foci=tuple(used_foci),
                record_script=False,
                **shared,
            )
            if result.end_reason == "interrupted":
                _LOG.warning("bulk run interrupted at item %d/%d", index + 1, qty)
                if result.exchanges > 0:
                    items.append(result)
                return BulkPipelineResult(
                    items=items,
                    requested=qty,
                    succeeded=sum(1 for item in items if item.succeeded),
                    skipped_duplicates=skipped,
                )

            script = _transcript_script(result)
            if script and _script_conflicts(script, seen_scripts, seen_fps):
                _LOG.warning(
                    "bulk item %d/%d produced a repeated script (attempt %d); regenerating",
                    index + 1,
                    qty,
                    attempt,
                )
                # A pinned topic that collided must not be reused on retry.
                item_topic = None
                continue
            if not result.succeeded:
                _LOG.error(
                    "bulk item %d/%d failed (%s / %s)",
                    index + 1,
                    qty,
                    result.end_reason,
                    result.dialogue_end_reason,
                )
                break

            _record_produced_script(DebateMemory(cfg.memory), script)
            seen_scripts.append(script)
            fingerprint = script_fingerprint(script)
            if fingerprint:
                seen_fps.add(fingerprint)
            used_topics.append(result.transcript.topic)
            focus = str(result.transcript.metadata.get("provocation_focus") or "")
            if focus:
                used_foci.append(focus)
            accepted = result
            break

        if accepted is None:
            skipped += 1
            _LOG.error(
                "bulk item %d/%d skipped: could not produce an original script",
                index + 1,
                qty,
            )
            continue
        items.append(accepted)

    succeeded = sum(1 for item in items if item.succeeded)
    _LOG.info(
        "bulk finished: requested=%d succeeded=%d skipped_duplicates=%d",
        qty,
        succeeded,
        skipped,
    )
    return BulkPipelineResult(
        items=items,
        requested=qty,
        succeeded=succeeded,
        skipped_duplicates=skipped,
    )


__all__ = ["BulkPipelineResult", "PipelineResult", "run_bulk_pipeline", "run_pipeline"]
