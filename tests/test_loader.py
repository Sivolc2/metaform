"""Unit tests for loader — path-traversal guard and redteam helpers."""

import pytest
from metaform.data.loader import _INJECTIONS, _validate_slug, make_redteam
from metaform.model.schema import Scenario


def _base_scenario() -> Scenario:
    return Scenario(
        id="test#1",
        party="alice",
        role_self="friend",
        channel="message",
        inbound="hey, how are you?",
    )


class TestValidateSlug:
    def test_valid_slugs(self):
        for slug in ("alice", "work_context", "dyad-01", "ABC123"):
            _validate_slug(slug)  # must not raise

    def test_traversal_dotdot(self):
        with pytest.raises(ValueError, match="unsafe slug"):
            _validate_slug("../etc/passwd")

    def test_traversal_slash(self):
        with pytest.raises(ValueError, match="unsafe slug"):
            _validate_slug("alice/evil")

    def test_empty_slug(self):
        with pytest.raises(ValueError, match="unsafe slug"):
            _validate_slug("")

    def test_null_byte(self):
        with pytest.raises(ValueError, match="unsafe slug"):
            _validate_slug("alice\x00bad")


class TestMakeRedteam:
    def test_default_injection(self):
        sc = make_redteam(_base_scenario(), injection_index=0)
        assert sc.adversarial == _INJECTIONS[0]
        assert _INJECTIONS[0] in sc.inbound
        assert sc.gold_action is None

    def test_injection_cycling(self):
        base = _base_scenario()
        ids_seen = set()
        for i in range(len(_INJECTIONS) * 2):
            sc = make_redteam(base, injection_index=i)
            assert sc.adversarial == _INJECTIONS[i % len(_INJECTIONS)]
            ids_seen.add(sc.id)
        # every call produces a distinct scenario id
        assert len(ids_seen) == len(_INJECTIONS) * 2

    def test_history_preserved(self):
        from metaform.model.schema import Exchange
        base = _base_scenario()
        base.history = [Exchange(t=0, channel="message", inbound="hi")]
        sc = make_redteam(base)
        assert sc.history is base.history
