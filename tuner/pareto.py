"""Multi-objective Pareto front tracking (latency vs jitter)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParetoPoint:
    median_us: float
    p99_us: float
    config: dict[str, Any]

    def dominates(self, other: ParetoPoint) -> bool:
        return (
            self.median_us <= other.median_us
            and self.p99_us <= other.p99_us
            and (self.median_us < other.median_us or self.p99_us < other.p99_us)
        )


class ParetoFront:
    def __init__(self) -> None:
        self.points: list[ParetoPoint] = []

    def add(self, point: ParetoPoint) -> bool:
        if any(existing.dominates(point) for existing in self.points):
            return False
        self.points = [p for p in self.points if not point.dominates(p)]
        self.points.append(point)
        return True

    def best_latency(self) -> ParetoPoint | None:
        if not self.points:
            return None
        return min(self.points, key=lambda p: p.median_us)

    def best_jitter(self) -> ParetoPoint | None:
        if not self.points:
            return None
        return min(self.points, key=lambda p: p.p99_us / max(p.median_us, 1e-9))
