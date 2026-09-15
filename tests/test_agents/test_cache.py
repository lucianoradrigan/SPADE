"""Tests for SimulationCache (Phase 4)."""

import time

from driveflow.agents.cache import SimulationCache


class TestGetSet:
    def test_returns_none_for_a_missing_key(self):
        cache = SimulationCache()
        assert cache.get("missing") is None

    def test_set_then_get_round_trips(self):
        cache = SimulationCache()
        cache.set("key", {"a": 1})
        assert cache.get("key") == {"a": 1}


class TestExpiry:
    def test_is_expired_true_for_a_key_never_set(self):
        cache = SimulationCache()
        assert cache.is_expired("nope") is True

    def test_expires_after_ttl(self):
        cache = SimulationCache(ttl_seconds=0)
        cache.set("key", "value")
        time.sleep(0.01)
        assert cache.is_expired("key") is True
        assert cache.get("key") is None

    def test_get_evicts_an_expired_entry(self):
        cache = SimulationCache(ttl_seconds=0)
        cache.set("key", "value")
        time.sleep(0.01)
        cache.get("key")
        assert "key" not in cache.cache
        assert "key" not in cache.timestamps

    def test_not_expired_within_ttl(self):
        cache = SimulationCache(ttl_seconds=300)
        cache.set("key", "value")
        assert cache.is_expired("key") is False


class TestClear:
    def test_clear_removes_everything(self):
        cache = SimulationCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.cache == {}
        assert cache.timestamps == {}
