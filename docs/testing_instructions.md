# Testing Instructions

Commands used to set up, build, verify, benchmark, and tune `quantfx-compiler`. Run from the repository root unless noted.

## 1. Prerequisites

```bash
# System packages (Ubuntu/Debian, requires sudo)
./scripts/setup-deps.sh

# Or manually:
sudo apt install llvm-18 llvm-18-dev llvm-18-tools mlir-18-tools libmlir-18-dev build-essential

# Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Build

```bash
mkdir -p build && cd build
cmake .. -DMLIR_DIR=$(llvm-config-18 --cmakedir)/../mlir -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
```

Expected binaries:

- `build/tools/quantfx-compiler/quantfx-compiler`
- `build/benchmarks/bench_rolling`
- `build/tuner/qfx-harness` (after adding tuner to root CMake)

## 3. Frontend-only compile check (no MLIR)

```bash
g++ -std=c++20 -Icompiler/frontend -c compiler/frontend/Token.cpp -o /tmp/Token.o
g++ -std=c++20 -Icompiler/frontend -c compiler/frontend/Lexer.cpp -o /tmp/Lexer.o
g++ -std=c++20 -Icompiler/frontend -c compiler/frontend/Parser.cpp -o /tmp/Parser.o
g++ -std=c++20 -Icompiler/frontend -c compiler/frontend/HIR.cpp -o /tmp/HIR.o
echo "frontend compiles OK"
```

## 4. Correctness tests (NumPy reference)

```bash
source .venv/bin/activate
python tests/run_correctness.py
```

With compiler built, this also runs E2E checks for `rolling_mean`, `rolling_vwap`, and `vwap_signal.qfx`.

## 5. Compiler smoke tests

```bash
COMPILER=build/tools/quantfx-compiler/quantfx-compiler

# Print MLIR
$COMPILER examples/vwap_signal.qfx --emit-mlir

# JIT verify (centered-window synthetic data)
$COMPILER examples/vwap_signal.qfx --verify

# Emit CPU object file
$COMPILER examples/vwap_signal.qfx --target cpu -o /tmp/vwap.o

# Emit LLVM IR
$COMPILER examples/vwap_signal.qfx --emit-llvm -o /tmp/vwap.ll

# CUDA source / PTX (PTX requires nvcc)
$COMPILER examples/corr_matrix.qfx --target cuda -o /tmp/corr.cu
$COMPILER examples/corr_matrix.qfx --target cuda -o /tmp/corr.ptx
```

## 6. Phase 2 optimization flags

```bash
# Default pipeline (-O2)
$COMPILER examples/corr_matrix.qfx --benchmark --bench-warmup=10 --bench-iters=100

# Full optimizations
$COMPILER examples/corr_matrix.qfx -O3 --vectorize --benchmark --bench-warmup=10 --bench-iters=100

# Streaming / trailing-window incremental
$COMPILER examples/vwap_signal.qfx --streaming -O2 --benchmark

# Custom tile / prefetch / vector width
$COMPILER examples/corr_matrix.qfx -O2 --tile-size=128 --prefetch-distance=16 --benchmark
```

Benchmark output is JSON on stdout, e.g.:

```json
{"median_us":12.5,"p99_us":15.2,"mean_us":12.8,"warmup":100,"iters":1000}
```

## 7. Microbenchmarks

```bash
# Standalone C++ rolling-mean loop
./build/benchmarks/bench_rolling --n 10000 --window 20 --iters 10000

# NumPy vs compiler comparison (Python)
.venv/bin/python benchmarks/micro/bench_compare.py
```

## 8. Phase 3 auto-tuner

```bash
source .venv/bin/activate

# Tune a single kernel
python tuner/search.py --kernel examples/corr_matrix.qfx \
  --target cpu --generations 50 --population 32

# Tune all three example kernels
python tuner/search.py --all-kernels --generations 20 --population 16

# View cached results
python tuner/search.py --report --cache tuner/cache/results.db

# C++ harness wrapper (subprocess to compiler)
./build/tuner/qfx-harness --kernel examples/corr_matrix.qfx \
  --compiler build/tools/quantfx-compiler/quantfx-compiler \
  --warmup 100 --iters 1000
```

## 9. Quick validation checklist

| Step | Command | Pass criteria |
|------|---------|---------------|
| Build | `make -j` in `build/` | No compile errors |
| Correctness | `python tests/run_correctness.py` | All tests PASS |
| Verify | `$COMPILER examples/vwap_signal.qfx --verify` | `VERIFY 1024 ...` on stdout |
| Benchmark | `$COMPILER examples/corr_matrix.qfx --benchmark` | JSON with `median_us` |
| Tuner | `python tuner/search.py --kernel examples/corr_matrix.qfx --generations 5 --population 8` | Completes, prints summary |

## 10. Troubleshooting

```bash
# Confirm LLVM/MLIR
llvm-config-18 --version
mlir-opt --version

# Confirm compiler exists
ls -la build/tools/quantfx-compiler/quantfx-compiler

# Reconfigure from scratch
rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)
```
