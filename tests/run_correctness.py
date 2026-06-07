#!/usr/bin/env python3
"""Correctness tests: compare compiled ops against NumPy reference."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "build" / "tools" / "quantfx-compiler" / "quantfx-compiler"


def convolve_same(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same")


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    mean = convolve_same(x, window)
    mean_sq = convolve_same(x * x, window)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def rolling_vwap(prices: np.ndarray, volumes: np.ndarray, window: int) -> np.ndarray:
    num = convolve_same(prices * volumes, window)
    den = convolve_same(volumes, window)
    return num / np.maximum(den, 1e-8)


def zscore(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / np.maximum(sigma, 1e-8)


def run_kernel(qfx_source: str, n: int = 1024) -> np.ndarray | None:
    if not COMPILER.exists():
        return None

    with tempfile.NamedTemporaryFile("w", suffix=".qfx", delete=False) as tmp:
        tmp.write(qfx_source)
        path = tmp.name

    proc = subprocess.run(
        [str(COMPILER), path, "--verify"],
        capture_output=True,
        text=True,
        check=False,
    )
    Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return None

    for line in proc.stdout.splitlines():
        if line.startswith("VERIFY "):
            parts = line.split()
            values = np.array([float(v) for v in parts[2:]], dtype=np.float32)
            return values
    return None


def assert_close(name: str, got: np.ndarray, expected: np.ndarray, rtol: float = 1e-4) -> None:
    if not np.allclose(got, expected, rtol=rtol, atol=1e-5):
        max_err = float(np.max(np.abs(got - expected)))
        raise AssertionError(f"{name}: max error {max_err:.6e}")


def main() -> int:
    print("quantfx-compiler correctness tests\n")
    n, w = 1024, 20
    # Must match synthetic data in tools/quantfx-compiler/main.cpp (--verify).
    prices = np.array([float(i % 97) * 0.01 - 0.5 for i in range(n)], dtype=np.float32)
    volumes = np.array([1.0 + float(i % 7) * 0.1 for i in range(n)], dtype=np.float32)

    # NumPy reference checks (always run)
    ref_mean = convolve_same(prices, w)
    ref_std = rolling_std(prices, w)
    ref_vwap = rolling_vwap(prices, volumes, w)
    ref_z = zscore(prices, ref_vwap, ref_std)
    print(f"  numpy rolling_mean: mean={ref_mean.mean():.6f}")
    print(f"  numpy rolling_vwap: mean={ref_vwap.mean():.6f}")
    print(f"  numpy zscore:       mean={ref_z.mean():.6f}")

    if not COMPILER.exists():
        print(f"\nCompiler not built ({COMPILER}).")
        print("Build with: mkdir build && cd build && cmake .. && make -j")
        print("NumPy reference checks passed.")
        return 0

    print(f"\nUsing compiler: {COMPILER}")

    mean_qfx = """
import stream prices : f32[1024]
let out = rolling_mean(prices, window=20)
emit out : f32[1024]
"""
    got = run_kernel(mean_qfx, n)
    if got is not None:
        assert_close("rolling_mean", got, ref_mean)
        print("  rolling_mean: PASS")

    vwap_qfx = """
import stream prices : f32[1024]
import stream volumes : f32[1024]
let out = rolling_vwap(prices, volumes, window=20)
emit out : f32[1024]
"""
    got = run_kernel(vwap_qfx, n)
    if got is not None:
        assert_close("rolling_vwap", got, ref_vwap)
        print("  rolling_vwap: PASS")

    signal_qfx = Path(ROOT / "examples" / "vwap_signal.qfx").read_text()
    got = run_kernel(signal_qfx, n)
    if got is not None:
        assert_close("vwap_signal", got, ref_z)
        print("  vwap_signal:  PASS")

    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
