# -*- coding: utf-8 -*-
"""
agents/posting/facebook_scheduler/config.py
=============================
All configuration constants for the Facebook Scheduler.

Design rule: to adjust any behaviour, edit ONLY this file.
All other modules read from here — never hardcode values elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

from utils.pipeline_paths import page_outputs_dir

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[2]  # facebook_scheduler → posting → agents → factory root

# Google Sheets source — accepts either a .gsheet shortcut file or a raw ID.
# Override via env var GSHEET_ID or GSHEET_FILE.
GSHEET_FILE: Path = Path(
    r"G:\My Drive\Z sosFiles\Z_act\@ NETWORK\@MEDIAUPSCALE_FACTORY_DYNAMIC_CONTENT"
    r"\Unified Multi-Page Factory\outputs\momma_circle\quotes\quotes_to_post.gsheet"
)
GSHEET_ID: str = os.getenv("GSHEET_ID", "")   # set in .env to skip file parse

# Path to the Google service-account JSON (for gspread auth).
SERVICE_ACCOUNT_JSON: Path = Path(
    os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        str(PROJECT_ROOT / "credentials" / "gsheet_service_account.json"),
    )
)

WORKSHEET_NAME: str = "Ready_to_post"

# Screenshot + log directories (created at runtime if missing)
_FB_SCHED_OUT = page_outputs_dir("facebook_scheduler")
SCREENSHOTS_DIR: Path = _FB_SCHED_OUT / "screenshots"
LOGS_DIR: Path        = _FB_SCHED_OUT / "logs"

# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
POSTING_INTERVAL_HOURS: int = 3          # gap between posts in hours
SCHEDULE_COLUMN_FILL_START: str = "now"  # "now" = start from current time

# Reels / media scheduler (LocalMediaQueue + UniversalComposerScheduler)
# First post: now + random(FIRST_OFFSET_MIN..FIRST_OFFSET_MAX) minutes
# Subsequent: last + BASE_INTERVAL_HOURS + random(JITTER_MIN..JITTER_MAX) minutes
REELS_FIRST_OFFSET_MIN_MINUTES: int = 25  # Meta requires ≥20 min lead
REELS_FIRST_OFFSET_MAX_MINUTES: int = 60
REELS_BASE_INTERVAL_HOURS: int = 4
REELS_JITTER_MIN_MINUTES: int = 10
REELS_JITTER_MAX_MINUTES: int = 30
REELS_MIN_LEAD_MINUTES: int = 25

# Loose-folder model CTAs (--modelCTA). Same pool used for LADA Processed.
# Written into <folder>/asset_library.json only when that file is missing.
MODEL_CTA_DEFAULT: str = "link in bio for more fun 😋"
MODEL_CTA_CAPTIONS: list[str] = [
    "link in bio for more fun 😋",
    "link in bio for more fun 😉",
    "link in the bio for more fun 😋",
    "link in bio for more fun 😘",
    "more fun waiting in the bio 😋",
    "link in bio if you want more fun 😋",
    "fun’s in the bio 😋",
    "link in bio for more fun",
    "link in bio — more fun 😋",
    "link in bio for more fun 😋✨",
    "the fun’s in the bio 😋",
    "link in bio for more fun 😋",
]

# Loose-folder model hashtag captions (--modelHashtag). Brand-safe Reels
# growth pool: exactly 2 hashtags + 1–2 clean emojis. No adult terms.
# Written into <folder>/asset_library.json only when that file is missing.
MODEL_HASHTAG_DEFAULT: str = "#ViralReels #ModelStyle ✨"
MODEL_HASHTAG_CAPTIONS: list[str] = [
    "#ViralReels #ModelStyle ✨",
    "#Trending #OOTD 📸",
    "#FBReels #FashionVibes 💎",
    "#ReelsChallenge #ModelLook 🌟",
    "#ExplorePage #Aesthetic 🔥",
    "#ForYou #StyleInspo ✨👗",
    "#ModelLife #StudioDay 📸",
    "#FashionReel #GlowUp 💫",
    "#TrendAlert #LookOfTheDay 💎",
    "#ViralFashion #ReelsDaily 🔥",
    "#EditorialLook #PhotoDay 📷",
    "#StyleGoals #FBReels 💫",
    "#StreetStyle #Trending ⚡",
    "#WeekendLook #OOTD 🌸",
    "#GoldenHour #ModelLook 🌅",
    "#BehindTheScenes #OnSet 🎥",
    "#CastingReady #ModelMood ✨",
    "#NewDrop #FashionVibes ✨",
    "#BeautyInMotion #ReelsChallenge 💫",
    "#RunwayVibes #ModelStyle 🌟",
]

# ---------------------------------------------------------------------------
# Playwright / browser connection
# ---------------------------------------------------------------------------
# Connect to an already-running Chrome via remote debugging.
# Launch Chrome with: chrome.exe --remote-debugging-port=9222
CDP_ENDPOINT: str = os.getenv("CDP_ENDPOINT", "http://localhost:9222")

# Fallback: launch a new browser (useful for record_mode and dry-run)
HEADLESS: bool = False
BROWSER_CHANNEL: str = "chrome"   # "chrome" | "msedge" | "chromium"

# AdsPower / Chromium launch_args — pass these when starting a profile via
# Local API so background/occluded windows are not timer-throttled.
# Example AdsPower body: {"user_id": "...", "launch_args": CHROME_ANTI_THROTTLE_ARGS}
CHROME_ANTI_THROTTLE_ARGS: list[str] = [
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
]

# ---------------------------------------------------------------------------
# Timeouts (milliseconds)
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_MS: int      = 15_000   # default Playwright timeout
LONG_TIMEOUT_MS: int         = 30_000   # for slow dialogs / uploads
SUCCESS_WAIT_MS: int         = 8_000    # wait for post confirmation banner

# ---------------------------------------------------------------------------
# Human behaviour timing (seconds unless stated)
# ---------------------------------------------------------------------------
CLICK_PAUSE_MIN: float  = 1.2
CLICK_PAUSE_MAX: float  = 3.5
TYPE_DELAY_MIN_MS: int  = 45            # ms per character
TYPE_DELAY_MAX_MS: int  = 135
POST_COOLDOWN_MIN: float = 12.0         # seconds between scheduled posts
POST_COOLDOWN_MAX: float = 28.0
SCROLL_AMOUNT_MIN: int   = 80           # pixels
SCROLL_AMOUNT_MAX: int   = 250

# Mouse movement simulation (before every click)
MOUSE_DRIFT_STEPS: int      = 7         # intermediate waypoints towards target
MOUSE_DRIFT_JITTER_PX: int  = 10        # max random pixel offset per step
MOUSE_DRIFT_STEP_DELAY: float = 0.022   # seconds between each drift step

# Composer trigger discovery
# Max ms to test each individual candidate selector before moving to the next
COMPOSER_CANDIDATE_TIMEOUT_MS: int = 4_000
# Total budget (ms) to find any working trigger
COMPOSER_OPEN_TIMEOUT_MS: int = 20_000

# ---------------------------------------------------------------------------
# Sheet column letters (A=0 in gspread zero-indexed, but 1-indexed for update)
# ---------------------------------------------------------------------------
COL_TEXT     = "A"
COL_DATETIME = "B"
COL_STATUS   = "C"
COL_TEXT_IDX     = 0   # 0-based for list access
COL_DATETIME_IDX = 1
COL_STATUS_IDX   = 2

STATUS_DONE    = "DONE"
STATUS_PENDING = "PENDING"
STATUS_FAILED  = "FAILED"

# ---------------------------------------------------------------------------
# Facebook page target (displayed in logs / screenshots only; no auth)
# ---------------------------------------------------------------------------
FB_PAGE_NAME: str = "Momma Circle"

# ---------------------------------------------------------------------------
# Dry-run flag (can be overridden by CLI --dry-run)
# ---------------------------------------------------------------------------
DRY_RUN: bool = bool(os.getenv("DRY_RUN", ""))
