"""Python wrapper around quantfx-compiler --benchmark."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tuner.chromosome import TunerConfig
from tuner.fitness import BenchmarkResult

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPILER = ROOT / "build" / "tools" / "quantfx-compiler" / "quantfx-compiler"


def config_to_args(config: TunerConfig) -> list[str]:
    args = [
        f"-O{config.opt_level}",
        f"--tile-size={config.tile_size}",
        f"--vector-width={config.vector_width}",
        f"--prefetch-distance={config.prefetch_distance}",
        f"--llvm-codegen-opt={config.llvm_codegen_opt}",
    ]
    if config.streaming:
        args.append("--streaming")
    if not config.window_fusion:
        args.append("--no-fusion")
    if not config.loop_tiling:
        args.append("--no-tiling")
    if not config.prefetch:
        args.append("--no-prefetch")
    if config.vectorize:
        args.append("--vectorize")
    return args


def run_benchmark(
    kernel_path: Path | str,
    config: TunerConfig,
    *,
    compiler: Path | None = None,
    warmup: int = 100,
    iters: int = 1000,
    target: str = "cpu",
) -> BenchmarkResult | None:
    compiler_path = compiler or DEFAULT_COMPILER
    if not compiler_path.exists():
        return None

    cmd = [
        str(compiler_path),
        str(kernel_path),
        "--target",
        target,
        "--benchmark",
        f"--bench-warmup={warmup}",
        f"--bench-iters={iters}",
        *config_to_args(config),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        data = json.loads(line)
        return BenchmarkResult(
            median_us=float(data["median_us"]),
            p99_us=float(data["p99_us"]),
            mean_us=float(data["mean_us"]),
            warmup=int(data.get("warmup", warmup)),
            iters=int(data.get("iters", iters)),
        )
    return None
