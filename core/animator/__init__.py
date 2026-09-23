# -*- coding: utf-8 -*-
"""Universal, multi-niche 2D parametric avatar animation engine.

This is a foundational ``core/`` capability — completely decoupled from any
single channel's theme. Every module here accepts generic inputs: a dialogue
turn ledger, any audio track, a ``puppet.json`` skin directory, and a
:class:`~core.animator.types.SpeakerStyle` per speaker describing how that
speaker is presented (HUD label, accent colour, which way the hero looks).
Dropping in a new pair of characters re-skins the video without touching a
single line of animation logic.

Pipeline::

    audio + turns
        -> AudioAnalyzer (RMS, blinks) + Rhubarb (phonetic visemes)
        -> ShotReverseShotCompositor (HUD / single hero / subtitle zone)
        -> subtitles.build_ass (word-level karaoke)
        -> AnimationRenderer (raw frame pipe + libass burn-in + AAC mux)

Channel-specific wiring lives in that channel's own bridge module — see
``channels_config/aiwake/animator_bridge.py`` — never here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .types import AnalyzedAudio, DialogueTurn, PuppetAnchors, PuppetTheme, SpeakerStyle, WordTiming
from .puppet import PuppetRig, PuppetSkin
from .audio_analyzer import AudioAnalyzer
from .compositor import ShotReverseShotCompositor
from .renderer import AnimationRenderer, RenderStats
from . import subtitles as subtitles_module

_LOG = logging.getLogger("animator")

__all__ = [
    "AnalyzedAudio",
    "AnimationRenderer",
    "AudioAnalyzer",
    "DialogueTurn",
    "PuppetAnchors",
    "PuppetRig",
    "PuppetSkin",
    "PuppetTheme",
    "RenderStats",
    "ShotReverseShotCompositor",
    "SpeakerStyle",
    "WordTiming",
    "render_dynamic_animation",
]


def render_dynamic_animation(
    *,
    turns: Sequence[DialogueTurn],
    audio_path,
    styles: Sequence[SpeakerStyle],
    output_path,
    puppets_dir=None,
    fps: int = 30,
    width: int = 1080,
    height: int = 1920,
    duration_override: float | None = None,
    seed: int = 0,
    burn_subtitles: bool = True,
    use_rhubarb: bool = True,
    enable_cta: bool = False,
    outro_start_s: float | None = None,
    outro_frame=None,
    subtitle_fade_s: float = 0.0,
) -> RenderStats:
    """Analyze audio, direct the shot-reverse-shot, render to mp4.

    Kept free of any channel-specific type so any caller (a test harness, a
    kids'-cartoon channel, a future puppet pair) can drive the whole engine
    with plain dataclasses and paths.
    """
    from .asset_generator import DEFAULT_PUPPETS_DIR, generate_default_puppet

    puppets_root = Path(puppets_dir) if puppets_dir else DEFAULT_PUPPETS_DIR
    style_map = {style.character_id: style for style in styles}
    if not style_map:
        raise ValueError("render_dynamic_animation needs at least one SpeakerStyle")

    rigs: dict[str, PuppetRig] = {}
    for character_id in style_map:
        generate_default_puppet(character_id, puppets_dir=puppets_root)
        rigs[character_id] = PuppetRig(
            PuppetSkin.load_or_create(puppets_root / character_id)
        )

    analyzer = AudioAnalyzer(fps=fps, seed=seed)
    analyzed = analyzer.analyze(
        audio_path, list(turns), duration_override=duration_override, use_rhubarb=use_rhubarb
    )

    compositor = ShotReverseShotCompositor(
        rigs=rigs,
        styles=style_map,
        width=width,
        height=height,
        enable_cta=enable_cta,
        outro_start_s=outro_start_s,
        outro_frame=outro_frame,
    )

    output_path = Path(output_path)
    subtitles_path: Path | None = None
    if burn_subtitles:
        subtitles_path = subtitles_module.build_ass(
            list(turns),
            style_map,
            destination=output_path.with_suffix(".ass"),
            width=width,
            height=height,
            fade_out_s=subtitle_fade_s,
        )

    renderer = AnimationRenderer(width=width, height=height, fps=fps)
    return renderer.render(
        frame_iter=compositor.iter_frames(analyzed),
        audio_path=audio_path,
        output_path=output_path,
        duration_s=analyzed.duration_s,
        subtitles_path=subtitles_path,
    )
