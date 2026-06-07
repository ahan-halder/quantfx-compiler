#!/usr/bin/env python3
"""Genetic auto-tuner over quantfx-compiler configuration space."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pygad

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuner.cache_db import DEFAULT_DB, connect, lookup_best, report, store_pareto_point, store_result
from tuner.chromosome import GENE_COUNT, TunerConfig, decode, encode, random_gene_space
from tuner.fitness import compute_fitness
from tuner.harness import run_benchmark
from tuner.pareto import ParetoFront, ParetoPoint

DEFAULT_KERNELS = [
    ROOT / "examples" / "corr_matrix.qfx",
    ROOT / "examples" / "vwap_signal.qfx",
    ROOT / "examples" / "volatility_surface.qfx",
]


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def hardware_id() -> str:
    return f"{platform.machine()}-{platform.processor() or platform.node()}"


def baseline_config() -> TunerConfig:
    return TunerConfig(opt_level=3, window_fusion=True, loop_tiling=True, prefetch=True)


def run_ga(
    kernel: Path,
    *,
    generations: int,
    population: int,
    warmup: int,
    iters: int,
    target: str,
    db_path: Path,
) -> dict:
    compiler = ROOT / "build" / "tools" / "quantfx-compiler" / "quantfx-compiler"
    if not compiler.exists():
        raise SystemExit(f"Compiler not built: {compiler}\nRun cmake && make first.")

    src_hash = source_hash(kernel)
    hw = hardware_id()
    conn = connect(db_path)
    pareto = ParetoFront()

    cached = lookup_best(conn, src_hash, hw)
    if cached:
        print(f"Cache hit: median={cached['median_us']:.2f} us")

    baseline = run_benchmark(kernel, baseline_config(), warmup=warmup, iters=iters, target=target)
    if baseline is None:
        raise SystemExit("Baseline benchmark failed")
    baseline_median = baseline.median_us
    print(f"Baseline -O3 median: {baseline_median:.2f} us")

    best_config: TunerConfig | None = None
    best_median = float("inf")

    def fitness_func(ga_instance, solution, solution_idx):
        nonlocal best_config, best_median
        genes = [int(g) for g in solution]
        config = decode(genes)
        result = run_benchmark(kernel, config, warmup=warmup, iters=iters, target=target)
        if result is None:
            return 1e9

        score = compute_fitness(result)
        pareto.add(ParetoPoint(result.median_us, result.p99_us, config.to_dict()))
        store_pareto_point(
            conn,
            source_hash=src_hash,
            hardware_id=hw,
            config=config.to_dict(),
            median_us=result.median_us,
            p99_us=result.p99_us,
        )
        store_result(
            conn,
            source_hash=src_hash,
            hardware_id=hw,
            config=config.to_dict(),
            median_us=result.median_us,
            p99_us=result.p99_us,
            fitness=score.total,
        )

        if result.median_us < best_median:
            best_median = result.median_us
            best_config = config

        return score.total

    gene_space = random_gene_space()
    ga = pygad.GA(
        num_generations=generations,
        num_parents_mating=max(8, population // 4),
        sol_per_pop=population,
        num_genes=GENE_COUNT,
        gene_type=int,
        gene_space=gene_space,
        parent_selection_type="tournament",
        K_tournament=4,
        crossover_type="uniform",
        mutation_type="random",
        mutation_percent_genes=15,
        keep_elitism=8,
        fitness_func=fitness_func,
        save_best_solutions=True,
    )

    print(f"Tuning {kernel} | pop={population} gen={generations}")
    ga.run()

    if best_config is None:
        raise SystemExit("Tuning produced no valid configuration")

    improvement = (baseline_median - best_median) / baseline_median * 100.0
    summary = {
        "kernel": str(kernel),
        "source_hash": src_hash,
        "hardware_id": hw,
        "baseline_median_us": baseline_median,
        "best_median_us": best_median,
        "improvement_pct": improvement,
        "best_config": best_config.to_dict(),
        "best_genes": encode(best_config),
    }

    print("\n=== Tuning complete ===")
    print(json.dumps(summary, indent=2))
    if improvement >= 8.0:
        print(f"Milestone met: {improvement:.1f}% improvement over -O3 baseline")
    else:
        print(f"Improvement: {improvement:.1f}% (target >= 8%)")

    pf = pareto.best_latency()
    if pf:
        print(f"Pareto best latency: {pf.median_us:.2f} us (p99={pf.p99_us:.2f} us)")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="quantfx genetic auto-tuner")
    parser.add_argument("--kernel", type=Path, help="Path to .qfx kernel")
    parser.add_argument("--target", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--cache", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", action="store_true", help="Print cache report")
    parser.add_argument("--all-kernels", action="store_true", help="Tune all example kernels")
    args = parser.parse_args()

    if args.report:
        conn = connect(args.cache)
        rows = report(conn)
        if not rows:
            print("Cache is empty.")
            return 0
        for row in rows:
            print(
                f"{row['source_hash']} | {row['hardware_id']} | "
                f"median={row['median_us']:.2f}us p99={row['p99_us']:.2f}us | "
                f"{json.dumps(row['config'])}"
            )
        return 0

    kernels = DEFAULT_KERNELS if args.all_kernels else [args.kernel]
    if not kernels or kernels[0] is None:
        parser.error("--kernel is required unless --report or --all-kernels")

    for kernel in kernels:
        if not kernel.exists():
            print(f"Skipping missing kernel: {kernel}")
            continue
        run_ga(
            kernel,
            generations=args.generations,
            population=args.population,
            warmup=args.warmup,
            iters=args.iters,
            target=args.target,
            db_path=args.cache,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
