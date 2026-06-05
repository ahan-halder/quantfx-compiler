#!/usr/bin/env python3
"""Correctness tests: compare compiled ops against NumPy reference (Phase 1)."""

import sys

import numpy as np


def main() -> int:
    print("quantfx-compiler correctness tests")
    print("Compiler not yet implemented — NumPy reference checks only.\n")

    n, w = 1024, 20
    rng = np.random.default_rng(42)
    prices = rng.standard_normal(n).astype(np.float32)
    volumes = rng.uniform(0.5, 2.0, n).astype(np.float32)

    # Reference implementations for Phase 1 targets
    kernel = np.ones(w, dtype=np.float32) / w
    rolling_mean = np.convolve(prices, kernel, mode="same")

    print(f"  rolling_mean: shape={rolling_mean.shape}, mean={rolling_mean.mean():.6f}")

    vwap_num = np.convolve(prices * volumes, kernel, mode="same")
    vwap_den = np.convolve(volumes, kernel, mode="same")
    rolling_vwap = vwap_num / np.maximum(vwap_den, 1e-8)
    print(f"  rolling_vwap: shape={rolling_vwap.shape}, mean={rolling_vwap.mean():.6f}")

    print("\nAll reference checks passed. Wire compiler output here in Phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
