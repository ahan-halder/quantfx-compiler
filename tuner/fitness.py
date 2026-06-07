"""Fitness functions for the genetic auto-tuner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    median_us: float
    p99_us: float
    mean_us: float
    warmup: int
    iters: int


@dataclass
class FitnessScore:
    primary: float
    jitter_penalty: float
    size_penalty: float
    total: float


def compute_fitness(
    result: BenchmarkResult,
    *,
    jitter_weight: float = 0.25,
    size_weight: float = 0.05,
    binary_bytes: int = 0,
) -> FitnessScore:
    """Lower total fitness is better (minimization objective for pygad)."""
    primary = result.median_us
    jitter_ratio = result.p99_us / max(result.median_us, 1e-9)
    jitter_penalty = max(0.0, jitter_ratio - 2.0) * result.median_us * jitter_weight
    size_penalty = (binary_bytes / 1_000_000.0) * size_weight
    total = primary + jitter_penalty + size_penalty
    return FitnessScore(
        primary=primary,
        jitter_penalty=jitter_penalty,
        size_penalty=size_penalty,
        total=total,
    )
