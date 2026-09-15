"""Simulation cache: avoids re-running the same hypothesis simulation on every telemetry window.

Phase 1 scaffolding only.
"""

from __future__ import annotations


class SimulationCache:
    """Caché con TTL para resultados de simulate_scenario() -- generar una simulación es
    costoso (corre la física real), así que escenarios ya vistos se reusan por ttl_seconds."""

    def __init__(self, ttl_seconds: int = 300, max_size_mb: int = 100):
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        self.cache: dict = {}
        self.timestamps: dict = {}

    def get(self, key: str) -> dict | None:
        raise NotImplementedError("Phase 4")

    def set(self, key: str, value: dict) -> None:
        raise NotImplementedError("Phase 4")

    def clear(self) -> None:
        self.cache.clear()
        self.timestamps.clear()
