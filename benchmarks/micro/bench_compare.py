#!/usr/bin/env python3
"""Compare quantfx-compiler JIT kernel vs NumPy on rolling workloads."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPILER = ROOT / "build" / "tools" / "quantfx-compiler" / "quantfx-compiler"


def numpy_rolling_cov(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float32) / window
    mx = np.convolve(a, kernel, mode="same")
    my = np.convolve(b, kernel, mode="same")
    mxy = np.convolve(a * b, kernel, mode="same")
    return mxy - mx * my


def bench_numpy(n: int, window: int, iters: int) -> float:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    start = time.perf_counter()
    for _ in range(iters):
        numpy_rolling_cov(a, b, window)
    return (time.perf_counter() - start) * 1e6 / iters


def bench_compiler(n: int, window: int, iters: int) -> float | None:
    if not COMPILER.exists():
        return None
    qfx = f"""
import stream prices_a : f32[{n}]
import stream prices_b : f32[{n}]
let cov = rolling_cov(prices_a, prices_b, window={window})
emit cov : f32[{n}]
"""
    with tempfile.NamedTemporaryFile("w", suffix=".qfx", delete=False) as tmp:
        tmp.write(qfx)
        path = tmp.name

    proc = subprocess.run([str(COMPILER), path, "--verify"], capture_output=True, text=True)
    Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return None

    start = time.perf_counter()
    for _ in range(iters):
        subprocess.run([str(COMPILER), path, "--verify"], capture_output=True, check=True)
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> int:
    n, window, iters = 10000, 20, 100
    np_us = bench_numpy(n, window, iters)
    print(f"NumPy rolling_cov: {np_us:.1f} us/iter (n={n}, w={window})")

    comp_us = bench_compiler(n, window, min(iters, 20))
    if comp_us is None:
        print("Compiler not built; skipping compiler benchmark.")
        return 0

    print(f"quantfx-compiler:  {comp_us:.1f} us/iter (includes subprocess + JIT)")
    speedup = np_us / comp_us if comp_us > 0 else 0.0
    print(f"Ratio (numpy/compiler): {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
