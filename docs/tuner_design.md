# Auto-Tuner Design (Phase 3)

## Overview

The genetic auto-tuner (`tuner/search.py`) searches compiler knob settings to minimize kernel latency. It treats `quantfx-compiler` as a black box: each chromosome is compiled and benchmarked via `--benchmark`.

## Search space

| Gene | Values | Maps to |
|------|--------|---------|
| opt_level | 0–3 | `-O0` … `-O3` |
| streaming | 0/1 | `--streaming` |
| window_fusion | 0/1 | fusion pass on/off |
| loop_tiling | 0/1 | tiling pass on/off |
| prefetch | 0/1 | prefetch pass on/off |
| vectorize | 0/1 | `--vectorize` |
| tile_size | index | `--tile-size` ∈ {16,32,64,128,256} |
| vector_width | index | `--vector-width` ∈ {4,8,16} |
| prefetch_distance | index | `--prefetch-distance` ∈ {4,8,16,32} |
| llvm_codegen_opt | 0–3 | `--llvm-codegen-opt` |

## Fitness

- **Primary:** median wall-clock latency (μs)
- **Secondary:** p99 jitter penalty when p99/median > 2×
- **Tertiary:** soft binary-size penalty (reserved)

## Algorithm

μ+λ evolution strategy via `pygad`:

- Population 32 (default), elite 8
- Tournament selection (k=4)
- Uniform crossover, 15% mutation rate

## Cache

SQLite database at `tuner/cache/results.db`:

- Key: `(source_hash, hardware_id, config_json)`
- Stores median, p99, fitness, timestamp
- Pareto front table tracks (latency, jitter) pairs

## Usage

```bash
python tuner/search.py --kernel examples/corr_matrix.qfx --target cpu --generations 50 --population 32
python tuner/search.py --all-kernels --generations 20 --population 16
python tuner/search.py --report --cache tuner/cache/results.db
```

## Milestone

≥8% median latency improvement over default `-O3` baseline on at least one kernel.
