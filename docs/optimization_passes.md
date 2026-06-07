# Phase 2 Optimization Passes

## Pipeline order (CPU, `-O2` default)

1. **qfx-window-fusion** — Merges `rolling_mean` + `rolling_std` on the same input/window into `fused_rolling_stats`. Replaces `rolling_cov` with `fused_rolling_cov`.
2. **qfx-sliding-window** (`--streaming`) — Marks rolling ops for trailing-window O(1) incremental updates.
3. **qfx-to-affine** — Lowers qfx ops to scf/arith/memref loops.
4. **qfx-loop-tiling** (`-O2+`) — Annotates large loops with tile-size hints (64).
5. **qfx-prefetch** (`-O2+`) — Inserts `memref.prefetch` ahead of loads (distance 8).
6. **qfx-vectorize** (`-O3`) — Vectorizes simple elementwise loops (width 16 ≈ AVX-512).

## GPU backend (`--target cuda`)

Generates CUDA C with a persistent-style launcher (`qfx_launch`). Use `-o kernel.ptx` to invoke `nvcc -ptx` when available.

## Benchmarks

```bash
./build/benchmarks/bench_rolling --n 10000 --window 20 --iters 10000
.venv/bin/python benchmarks/micro/bench_compare.py
```
