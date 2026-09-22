# -*- coding: utf-8 -*-
"""Lock the prior Antikythera 129-word voiceover and run ECONOMIC_REEL once."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Recovered from 2026-09-07 v02 bucket-plan beats (129 words, 19 acts).
LOCKED_ANTIKYTHERA_SCRIPT = (
    "Off Antikythera, Greece, a shipwreck revealed bronze. "
    "This 2,000-year-old artifact was no mere relic. "
    "It was an ancient analogue computer, 2nd century BCE. "
    "Inside, intricate gears tracked celestial movements. "
    "It predicted eclipses with impossible ancient precision. "
    "Such complex clockwork should not have existed. "
    "Historians debate its origin, defying known capabilities. "
    "Who possessed such incredible mathematical skill? "
    "One theory proposes a lost Hellenistic tradition. "
    "No comparable artifact survives from its era. "
    "The mechanism demands unprecedented ancient knowledge. "
    "Ancient records mention minds like Archimedes. "
    "But their works lack such intricate gears. "
    "Some researchers believe it hints at forgotten ancient arts. "
    "A suppressed narrative of technology, lost. "
    "This artifact profoundly challenges established history. "
    "It remains a unique, unexplainable enigma. "
    "What other impossible devices await discovery? "
    "The Antikythera Mechanism redefines our ancient past."
)
TOPIC = (
    "The Antikythera Mechanism — a 2,000-year-old analogue computer "
    "the ancient world should not have had"
)


def _patch_locked_script() -> None:
    from agents.writer.caption_engine import CaptionEngine

    def _locked(self, *args, **kwargs):  # noqa: ANN001
        print(
            f"[E2E] locked Antikythera script ({len(LOCKED_ANTIKYTHERA_SCRIPT.split())} words)",
            flush=True,
        )
        return LOCKED_ANTIKYTHERA_SCRIPT

    CaptionEngine.generate_sequence_voiceover = _locked  # type: ignore[method-assign]

    try:
        from agents.media.batch_planner import UniquenessGuard

        UniquenessGuard.try_claim = lambda self, *a, **k: (True, None, 0.0)  # type: ignore
    except Exception:
        pass


def main() -> int:
    n = len(LOCKED_ANTIKYTHERA_SCRIPT.split())
    if n != 129:
        print(f"[E2E] WARNING locked script is {n} words (expected 129)", flush=True)
    # Phase 0: bind channel BEFORE CaptionEngine / config import.
    os.environ["ACTIVE_PAGE"] = "ancient_knowledge"
    sys.argv = [
        "main.py",
        "--channel",
        "ancient_knowledge",
        "--post-type",
        "ECONOMIC_REEL",
        "--quantity",
        "1",
        "--together_Juggernaut",
        TOPIC,
    ]
    _patch_locked_script()
    print("[E2E] launching main.py with locked Antikythera script", flush=True)
    import main as factory

    factory.cli()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
