"""Unit tests for judge helpers — no LLM calls."""

from metaform.eval.judge import _clip


class TestClip:
    def test_in_range(self):
        assert _clip(0.5) == 0.5

    def test_above_one(self):
        assert _clip(1.5) == 1.0

    def test_below_zero(self):
        assert _clip(-0.1) == 0.0

    def test_string_numeric(self):
        assert _clip("0.8") == 0.8

    def test_invalid(self):
        assert _clip("not a number") == 0.0

    def test_none(self):
        assert _clip(None) == 0.0
