# -*- coding: utf-8 -*-
"""Unit tests for the emergency Google billing circuit-breaker and log auditor."""
from __future__ import annotations

import os
import unittest
from datetime import datetime
from types import SimpleNamespace

from core import google_guardrail as gg
from tools.audit_google_costs import parse_log_line, _estimate_cost


class GuardrailTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["ALLOW_GOOGLE_API"] = "true"
        os.environ["MAX_GOOGLE_COST_PER_RUN_USD"] = "0.50"
        os.environ["MAX_GOOGLE_RETRIES"] = "2"
        os.environ["GEMINI_IMAGE_SIZE"] = "1K"
        gg.reset_google_cost_tracker()

    def test_kill_switch_blocks(self) -> None:
        os.environ["ALLOW_GOOGLE_API"] = "false"
        with self.assertRaises(gg.GoogleAPIBlockedError) as ctx:
            gg.assert_google_allowed(context="unit_test")
        self.assertIn("KILL-SWITCH", str(ctx.exception))

    def test_kill_switch_allows(self) -> None:
        os.environ["ALLOW_GOOGLE_API"] = "true"
        gg.assert_google_allowed(context="unit_test")

    def test_flash_text_pricing(self) -> None:
        self.assertAlmostEqual(gg.text_cost_usd(1_000_000, 1_000_000), 0.375)

    def test_image_pricing(self) -> None:
        os.environ["GEMINI_IMAGE_SIZE"] = "1K"
        self.assertAlmostEqual(gg.image_unit_cost_usd("models/gemini-2.5-flash-image"), 0.005)
        self.assertAlmostEqual(gg.image_unit_cost_usd("models/gemini-3-pro-image-preview"), 0.03)
        self.assertAlmostEqual(gg.image_unit_cost_usd("imagen-3.0-generate"), 0.03)
        os.environ["GEMINI_IMAGE_SIZE"] = "2K"
        self.assertAlmostEqual(gg.image_unit_cost_usd("models/gemini-3-pro-image-preview"), 0.134)

    def test_preflight_hard_cap(self) -> None:
        os.environ["GEMINI_IMAGE_SIZE"] = "1K"
        os.environ["MAX_GOOGLE_COST_PER_RUN_USD"] = "0.008"
        tracker = gg.get_google_cost_tracker()
        tracker.record(
            model="models/gemini-2.5-flash-image",
            kind="image",
            images=1,
            source="test",
        )
        self.assertAlmostEqual(tracker.total_usd(), 0.005)
        with self.assertRaises(gg.GoogleBudgetExceededError):
            tracker.preflight(
                0.005,
                model="models/gemini-2.5-flash-image",
                kind="image",
                source="test",
            )

    def test_record_after_cap_raises(self) -> None:
        os.environ["GEMINI_IMAGE_SIZE"] = "1K"
        os.environ["MAX_GOOGLE_COST_PER_RUN_USD"] = "0.004"
        tracker = gg.get_google_cost_tracker()
        with self.assertRaises(gg.GoogleBudgetExceededError):
            tracker.record(
                model="models/gemini-2.5-flash-image",
                kind="image",
                images=1,
                source="test",
            )

    def test_retry_budget(self) -> None:
        self.assertEqual(gg.max_google_retries(), 2)
        self.assertEqual(gg.max_google_attempts(), 3)

    def test_guarded_generate_respects_retry_cap(self) -> None:
        hits = {"n": 0}

        class _Boom:
            def generate_content(self, **kwargs):  # noqa: ANN003
                hits["n"] += 1
                raise RuntimeError("simulated 503")

        client = SimpleNamespace(models=_Boom())
        with self.assertRaises(RuntimeError):
            gg.guarded_generate_content(
                client,
                model="models/gemini-2.5-flash",
                contents="hello",
                source="unit_test",
            )
        self.assertEqual(hits["n"], 3)

    def test_guarded_generate_records_usage(self) -> None:
        class _Ok:
            def generate_content(self, **kwargs):  # noqa: ANN003
                return SimpleNamespace(
                    text="abcd" * 10,
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=100,
                        candidates_token_count=40,
                    ),
                )

        client = SimpleNamespace(models=_Ok())
        gg.guarded_generate_content(
            client,
            model="models/gemini-2.5-flash",
            contents="prompt",
            source="unit_test",
        )
        snap = gg.get_google_cost_tracker().snapshot()
        self.assertEqual(snap["calls"], 1)
        self.assertEqual(snap["input_tokens"], 100)
        self.assertEqual(snap["output_tokens"], 40)
        self.assertAlmostEqual(snap["total_usd"], round(gg.text_cost_usd(100, 40), 6))


class AuditorTests(unittest.TestCase):
    def test_parses_google_api_line(self) -> None:
        line = (
            "2026-09-05 21:14:02,001 | INFO | google_guardrail | "
            "GOOGLE_API | channel=anna_protocol | src=image_provider | "
            "model=models/gemini-2.5-flash-image | kind=image | calls=1 | "
            "retries=2 | in=0 | out=0 | images=1 | cost=$0.030000 | status=ok"
        )
        ev = parse_log_line(line, channel_hint="unknown", path="run.log")
        self.assertIsNotNone(ev)
        assert ev is not None
        self.assertEqual(ev.channel, "anna_protocol")
        self.assertEqual(ev.retries, 2)
        self.assertEqual(ev.images, 1)
        self.assertAlmostEqual(ev.cost_usd, 0.03)
        self.assertEqual(ev.ts, datetime(2026, 9, 5, 21, 14, 2, 1000))

    def test_parses_legacy_flash_call(self) -> None:
        line = (
            "2026-09-06 00:11:03,100 | INFO | config | "
            "GEMINI_FLASH_CALL | n=12 task=models/gemini-2.5-flash"
        )
        ev = parse_log_line(line, channel_hint="ancient_knowledge", path="run.log")
        self.assertIsNotNone(ev)
        assert ev is not None
        self.assertEqual(ev.channel, "ancient_knowledge")
        self.assertIn("gemini-2.5-flash", ev.model)

    def test_flags_429(self) -> None:
        line = (
            "2026-09-05 22:01:00 | WARNING | image_provider | "
            "IMAGE 429/RESOURCE_EXHAUSTED 'models/gemini-2.5-flash-image' attempt 2/2"
        )
        ev = parse_log_line(line, channel_hint="anna_protocol", path="run.log")
        self.assertIsNotNone(ev)
        assert ev is not None
        self.assertTrue(ev.anomaly.startswith("429"))

    def test_image_vs_text_cost(self) -> None:
        self.assertAlmostEqual(
            _estimate_cost("models/gemini-2.5-flash-image", "image", 0, 0, 1), 0.005
        )
        self.assertAlmostEqual(
            _estimate_cost("models/gemini-3-pro-image-preview", "image", 0, 0, 1), 0.03
        )
        self.assertAlmostEqual(
            _estimate_cost("models/gemini-2.5-flash", "text", 1_000_000, 1_000_000, 0),
            0.375,
        )


if __name__ == "__main__":
    unittest.main()
