# -*- coding: utf-8 -*-
"""
test_momma_circle_post.py
=========================
Quick connection & publish test for the Momma Circle Facebook Page.

Usage
-----
    python channels_config/momma_circle/tests/test_momma_circle_post.py

Options
-------
    --message "Custom text"     Override the default test message
    --dry-run                   Verify token / page info WITHOUT posting
    --verify-only               Same as --dry-run

The script loads credentials from .env automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass  # dotenv optional — fall through to raw os.getenv

# ---------------------------------------------------------------------------
# Credentials (can also be passed as CLI args for CI use)
# ---------------------------------------------------------------------------
PAGE_ID      = os.getenv("FB_MOMMA_CIRCLE_PAGE_ID", "415144745345466").strip()
ACCESS_TOKEN = os.getenv(
    "FB_MOMMA_CIRCLE_ACCESS_TOKEN",
    "EAAPbmJ8XohkBSGmLeFXx5tRzM6ndvgSFGtWpU7O4FRN3BYUagpZB9XF4WxSYanAdzdNDhqg7unbx5QVm5ULsl4Pbq5fBklyrdX0EAfCn1H8ahS8uI9QfeUZCO9iqDo63GXhXWFVZCIFAZCAopnUee1fZAKuqiKpAq59Nnxnxpa3lPRIDSgT0EerIzFN5ZAcX8moaxn7w2ZA1g7pu2DIocW8t9P01ikAdqUb04ryWXQdFlHcYUD18aYrEhZB9ceAZD",
).strip()
API_VERSION  = os.getenv("FB_GRAPH_API_VERSION", "v25.0").strip()

_DEFAULT_MESSAGE = (
    "Testing automated queue engine on Momma Circle! "
    "This post was published by the Unified Multi-Page Factory. [test]"
)


# ---------------------------------------------------------------------------
# Core test functions
# ---------------------------------------------------------------------------

def verify_page(page_id: str = PAGE_ID, access_token: str = ACCESS_TOKEN) -> dict:
    """
    Call GET /{page_id}?fields=id,name,fan_count,category to confirm the
    token is valid and the page ID is correct, WITHOUT making any post.
    """
    import requests

    version = API_VERSION.lstrip("v")
    url = f"https://graph.facebook.com/v{version}/{page_id}"
    params = {
        "fields": "id,name,fan_count,category",
        "access_token": access_token,
    }
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()

    if "error" in data:
        err = data["error"]
        print(f"[FAIL] Token verification failed:")
        print(f"       Code   : {err.get('code')}")
        print(f"       Message: {err.get('message')}")
        return {}

    print("[OK] Page token verified successfully:")
    print(f"     Page ID   : {data.get('id')}")
    print(f"     Page Name : {data.get('name')}")
    print(f"     Fans      : {data.get('fan_count', 'N/A')}")
    print(f"     Category  : {data.get('category', 'N/A')}")
    return data


def test_publish(
    message_text: str = _DEFAULT_MESSAGE,
    page_id: str = PAGE_ID,
    access_token: str = ACCESS_TOKEN,
) -> "str | None":
    """
    Publish a text post to the Momma Circle Facebook Page.

    Returns the Facebook post ID on success, None on failure.
    """
    import requests

    version = API_VERSION.lstrip("v")
    url = f"https://graph.facebook.com/v{version}/{page_id}/feed"
    payload = {
        "message": message_text,
        "access_token": access_token,
    }

    print(f"\n[Facebook] Posting to page {page_id}...")
    print(f"[Facebook] Message preview: {message_text[:80]}...")

    response = requests.post(url, data=payload, timeout=30)
    res_data = response.json()

    if "id" in res_data:
        post_id = res_data["id"]
        print(f"[OK] Successfully posted to Momma Circle!")
        print(f"     Post ID: {post_id}")
        print(f"     View at: https://www.facebook.com/{post_id.replace('_', '/posts/')}")
        return post_id
    else:
        err = res_data.get("error", res_data)
        print(f"[FAIL] Post failed:")
        print(f"       Code   : {err.get('code', '?')}")
        print(f"       Message: {err.get('message', str(err))}")
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test Facebook Graph API connection for Momma Circle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--message", "-m",
        default=_DEFAULT_MESSAGE,
        help="Text to post (default: canned test message).",
    )
    ap.add_argument(
        "--dry-run", "--verify-only",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Only verify the token/page — do NOT publish.",
    )
    ns = ap.parse_args()

    print("=" * 60)
    print("  Momma Circle -- Facebook Connection Test")
    print(f"  Page ID    : {PAGE_ID}")
    print(f"  API Version: {API_VERSION}")
    print("=" * 60)

    # Step 1: always verify the page first
    info = verify_page()
    if not info:
        print("\n[ABORT] Cannot proceed — token or page ID invalid.")
        sys.exit(1)

    if ns.dry_run:
        print("\n[dry-run] Token verified. Skipping post (--dry-run flag set).")
        sys.exit(0)

    # Step 2: publish the test post
    print()
    post_id = test_publish(ns.message)
    if post_id is None:
        sys.exit(1)

    print("\n[DONE] Connection test passed.")


if __name__ == "__main__":
    main()
