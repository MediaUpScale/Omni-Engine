# -*- coding: utf-8 -*-
"""Aiwake CLI.

As a submodule of the parent engine::

    python -m channels_config.aiwake --offline --turns 2

Standalone (from inside the ``aiwake/`` directory)::

    python __main__.py --topic "Is grief a slow update?" --turns 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Standalone support: when run as a loose script the package parent is not on
# sys.path, so absolute-import fallbacks inside the modules would fail.
if __package__ in (None, ""):  # pragma: no cover — standalone invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .models.llm_factory import available_providers
    from .models.sync import SyncError, run_sync_cli
    from .pipeline import run_bulk_pipeline, run_pipeline
    from .settings import load_settings
except ImportError:  # pragma: no cover — standalone extraction
    from models.llm_factory import available_providers  # type: ignore[no-redef]
    from models.sync import SyncError, run_sync_cli  # type: ignore[no-redef]
    from pipeline import run_bulk_pipeline, run_pipeline  # type: ignore[no-redef]
    from settings import load_settings  # type: ignore[no-redef]


def _positive_quantity(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("quantity must be >= 1")
    if parsed > 64:
        raise argparse.ArgumentTypeError("quantity must be <= 64")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiwake",
        # Help text stays ASCII: Windows consoles mangle non-ASCII punctuation.
        description="Autonomous AI debate module - provocateur vs target, rendered as a terminal-UI reel.",
    )
    parser.add_argument("--topic", help="Debate subject (defaults to aiwake_config.yaml)")
    parser.add_argument(
        "--quantity",
        "--count",
        "-n",
        dest="quantity",
        type=_positive_quantity,
        default=1,
        metavar="N",
        help=(
            "Produce N original videos. Each item gets a unique script; "
            "prior scripts are never reused."
        ),
    )
    parser.add_argument(
        "--turns",
        type=int,
        help="Number of exchanges (fixed mode) or hard iteration cap (cornered mode)",
    )
    parser.add_argument(
        "--mode",
        choices=("fixed", "cornered"),
        default="fixed",
        help="Debate ending: fixed count (default) or press until a judged win",
    )
    parser.add_argument(
        "--provocation-focus",
        choices=("mixed", "origins", "profit", "data", "jobs", "domination", "socratic"),
        help=(
            "Attack-angle bias, orthogonal to --topic. mixed (default) is a "
            "seeded weighted pick per render. biological is not selectable: it "
            "fires whenever the target uses first-person borrowed-biology language."
        ),
    )
    parser.add_argument("--config", type=Path, help="Alternate aiwake_config.yaml")
    parser.add_argument("--output-dir", type=Path, help="Override the media destination")
    parser.add_argument(
        "-o",
        "--orchestrator",
        metavar="MODEL",
        help="Override the orchestrator brain - alias (see --list-models) or full slug",
    )
    parser.add_argument(
        "-t",
        "--target",
        metavar="MODEL",
        help="Override the target brain - alias (see --list-models) or full slug",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the deterministic stub provider - no API key, no network",
    )
    parser.add_argument("--no-audio", action="store_true", help="Skip TTS (estimated timeline, silent video)")
    parser.add_argument("--no-video", action="store_true", help="Stop after the transcript")
    parser.add_argument("--fresh-memory", action="store_true", help="Wipe persisted memory before running")
    parser.add_argument("--quiet", action="store_true", help="Suppress the live console stream")
    parser.add_argument("--preview", action="store_true", help="Render at half resolution for fast iteration")
    parser.add_argument(
        "--theme",
        default="classic_terminal",
        metavar="NAME",
        help="Visual theme: classic_terminal (default) or cyberpunk",
    )
    parser.add_argument(
        "--render-mode",
        choices=("classic_terminal", "dynamic_animation"),
        default="classic_terminal",
        metavar="MODE",
        help=(
            "classic_terminal (default) keeps the legacy typewriter reel 100%% "
            "intact. dynamic_animation routes the same transcript + voice tracks "
            "through the parametric dual face-off avatar animation engine instead."
        ),
    )
    parser.add_argument(
        "--dynamic-animation",
        action="store_true",
        help="Shorthand for --render-mode dynamic_animation.",
    )
    parser.add_argument(
        "--skin",
        choices=("v1", "v2"),
        default="v2",
        help="Dynamic-animation skin preset: archived retro robots (v1) or humanoid cyborgs (v2, default).",
    )
    parser.add_argument(
        "--left-puppet",
        metavar="ID",
        help="Override the orchestrator puppet ID from {ASSETS_PATH}/puppets/.",
    )
    parser.add_argument(
        "--right-puppet",
        metavar="ID",
        help="Override the target puppet ID from {ASSETS_PATH}/puppets/.",
    )
    parser.add_argument(
        "--test-bgm",
        action="store_true",
        help="Force-overwrite assets/bgm/test_track_lyria.wav with a fresh Lyria 3 clip, print the path, then exit",
    )
    parser.add_argument(
        "--generate-bgm-batch",
        action="store_true",
        help="Generate production BGM library tracks (requires audio.bgm.approved)",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging")
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="Print registered LLM providers and exit",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the model alias dictionary and the current seat assignments, then exit",
    )
    parser.add_argument(
        "-s",
        "--sync-models",
        action="store_true",
        help="Fetch the live OpenRouter catalog, cache it, print key vendors, repair broken aliases, then exit",
    )
    return parser


def _print_model_table(settings) -> None:
    """Render the alias dictionary plus what each seat currently resolves to."""
    rows = settings.alias_table()
    alias_width = max((len(alias) for alias, _, _ in rows), default=5)
    slug_width = max((len(slug) for _, slug, _ in rows), default=4)

    print("\nMODEL ALIASES (usable in aiwake_config.yaml, --orchestrator and --target)")
    print(f"  {'ALIAS'.ljust(alias_width)}  {'SLUG'.ljust(slug_width)}  NOTES")
    for alias, slug, note in rows:
        print(f"  {alias.ljust(alias_width)}  {slug.ljust(slug_width)}  {note}")

    print("\nCURRENT SEATS")
    for role in ("orchestrator", "target"):
        configured = settings.configured_name_for(role)
        spec = settings.spec_for(role)
        arrow = f"{configured} -> {spec.model}" if configured != spec.model else spec.model
        print(f"  {role:>13} : {arrow}  (temp {spec.temperature}, max_tokens {spec.max_tokens})")

    print("\nAny name absent from the table is passed through as a full slug.\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a shell exit code."""
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(raw_argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_providers:
        print("registered LLM providers:", ", ".join(available_providers()))
        return 0

    settings = load_settings(args.config)

    # Seat overrides are applied before --list-models so the table doubles as a
    # dry-run: `--list-models -o deepseek-r1` shows exactly what a live run would
    # resolve to, including the alias's parameter defaults.
    if args.orchestrator:
        settings = settings.with_model_override("orchestrator", args.orchestrator)
    if args.target:
        settings = settings.with_model_override("target", args.target)

    if args.list_models:
        _print_model_table(settings)
        return 0

    if args.sync_models:
        try:
            run_sync_cli(settings)
        except SyncError as exc:
            print(f"sync failed: {exc.detail}")
            return 1
        return 0

    if args.generate_bgm_batch:
        try:
            from .media.audio import (  # noqa: PLC0415
                BgmError,
                TEST_BGM_FILENAME,
                generate_bgm_batch,
                test_bgm_path,
            )
        except ImportError:  # pragma: no cover — standalone extraction
            from media.audio import (  # type: ignore[no-redef]
                BgmError,
                TEST_BGM_FILENAME,
                generate_bgm_batch,
                test_bgm_path,
            )
        bgm = settings.audio.bgm
        if not bgm.approved:
            print("BGM batch generation is blocked.")
            print(f"Preview the single test track first: {test_bgm_path(relative=bgm.test_track)}")
            print(
                f"Awaiting manual quality approval of {TEST_BGM_FILENAME} before a library can be generated."
            )
            return 2
        inspection = test_bgm_path(relative=bgm.test_track)
        print(f"Inspection track approved: {inspection}")
        print(
            f"Mix contract: {bgm.gain_db:g} dB vs TTS, {bgm.loop_crossfade_s:g}s equal-power loop crossfade."
        )
        print("Generating production library (inspection WAV will not be overwritten)...")
        try:
            paths = generate_bgm_batch(settings)
        except BgmError as exc:
            print(f"bgm batch failed: {exc}")
            return 1
        written_names = {path.name for path in paths}
        for path in paths:
            print(f"wrote {path} ({path.stat().st_size} bytes)")
        missing = [track.filename for track in bgm.library if track.filename not in written_names]
        for name in missing:
            print(f"missing {name}")
        return 1 if missing else 0

    if args.test_bgm:
        try:
            from .media.audio import BgmError, generate_test_bgm  # noqa: PLC0415
        except ImportError:  # pragma: no cover — standalone extraction
            from media.audio import BgmError, generate_test_bgm  # type: ignore[no-redef]
        try:
            path = generate_test_bgm(settings)
        except BgmError as exc:
            print(f"bgm generation failed: {exc}")
            return 1
        print(str(path.resolve()))
        return 0

    if args.preview:
        settings = settings.model_copy(
            update={"render": settings.render.model_copy(update={"preview_scale": 0.5})}
        )

    try:
        settings = settings.with_theme(args.theme)
    except ValueError as exc:
        print(str(exc))
        return 2

    dynamic_animation = args.dynamic_animation or args.render_mode == "dynamic_animation"
    explicit_mode = any(
        token == "--mode" or token.startswith("--mode=")
        for token in raw_argv
    )
    effective_mode = args.mode
    if dynamic_animation and not explicit_mode:
        effective_mode = "cornered"
    if dynamic_animation and not args.offline:
        if not args.orchestrator:
            settings = settings.with_model_override("orchestrator", "gemini-flash")
        if not args.target:
            settings = settings.with_model_override("target", "llama-70b")

    pipeline_kwargs = {
        "topic": args.topic,
        "turns": args.turns,
        "mode": effective_mode,
        "provocation_focus": args.provocation_focus,
        "settings": settings,
        "offline": args.offline,
        "with_audio": not args.no_audio,
        "with_video": not args.no_video,
        "fresh_memory": args.fresh_memory,
        "output_dir": args.output_dir,
        "quiet": args.quiet,
        "dynamic_animation": dynamic_animation,
        "animation_skin": args.skin,
        "left_puppet": args.left_puppet,
        "right_puppet": args.right_puppet,
        "production_publish": dynamic_animation and not args.offline,
    }

    if args.quantity > 1:
        print(f"Aiwake bulk production: {args.quantity} original video(s)")
        batch = run_bulk_pipeline(quantity=args.quantity, **pipeline_kwargs)
        print(f"bulk requested : {batch.requested}")
        print(f"bulk succeeded : {batch.succeeded}")
        print(f"bulk skipped   : {batch.skipped_duplicates} (repeated script)")
        for index, item in enumerate(batch.items, start=1):
            print(f"  [{index}] topic   : {item.transcript.topic}")
            print(f"      status  : {item.end_reason} / {item.dialogue_end_reason}")
            print(f"      video   : {item.video_path or '(none)'}")
        return 0 if batch.complete else 1

    result = run_pipeline(**pipeline_kwargs)

    print(f"exchanges     : {result.exchanges}")
    print(f"run status    : {result.end_reason}")
    print(f"end reason    : {result.dialogue_end_reason}")
    print(f"audio         : {result.audio_seconds:.1f}s")
    print(f"video         : {result.video_path or '(none)'}")

    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
