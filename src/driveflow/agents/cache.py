"""Simulation cache: avoids re-running the same hypothesis simulation on every telemetry window.

Phase 4: a real TTL-based cache, used as SimulationBasedAgent's own self.cache (base.py) --
keyed by whatever hashable key the caller uses (SimulationBasedAgent uses
(hypothesis_name, rounded_duration_s)), each entry expiring `ttl_seconds` after it was set.
max_size_mb is recorded but not enforced -- no eviction-by-size here, since doing that properly
needs knowing each stored value's real memory footprint, and what this caches (a handful of
short float arrays per hypothesis) is small relative to typical process memory. Add real
size-based eviction if a caller ever stores something large enough for that to matter.
"""

from __future__ import annotations

import time


class SimulationCache:
    """Caché con TTL para resultados de simulate_scenario()."""

    def __init__(self, ttl_seconds: int = 300, max_size_mb: int = 100):
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        self.cache: dict = {}
        self.timestamps: dict = {}

    def get(self, key):
        """Valor cacheado, o None si no existe o ya expiró (y en ese caso lo descarta)."""
        if key not in self.cache:
            return None
        if self.is_expired(key):
            self._evict(key)
            return None
        return self.cache[key]

    def set(self, key, value) -> None:
        self.cache[key] = value
        self.timestamps[key] = time.monotonic()

    def is_expired(self, key) -> bool:
        if key not in self.timestamps:
            return True
        return (time.monotonic() - self.timestamps[key]) > self.ttl_seconds

    def _evict(self, key) -> None:
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

    def clear(self) -> None:
        """Limpiar caché completamente."""
        self.cache.clear()
        self.timestamps.clear()
