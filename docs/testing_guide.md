# QuantFX Compiler - Beginner Testing & Setup Guide

Welcome to the `quantfx-compiler`! This guide is designed to help beginners set up the project on their local machine, compile the compiler, and run our benchmark and correctness tests.

## 1. Prerequisites (Ubuntu / Debian)

This project relies on LLVM and MLIR to parse our DSL, optimize it, and generate highly optimized machine code. Since MLIR APIs change frequently, we recommend installing the packages via `apt`. The compiler supports LLVM/MLIR 15 and 16.

```bash
# Update your package list
sudo apt-get update

# Install LLVM and MLIR development packages (Version 15 or 16)
sudo apt-get install -y llvm-15 llvm-15-dev mlir-15-tools libmlir-15-dev

# Ensure you have build tools installed
sudo apt-get install -y cmake build-essential ccache python3 python3-pip
```

> [!TIP]
> **LLVM Config:** You may need to create a symlink so `cmake` finds `llvm-config` automatically:
> `sudo ln -s /usr/bin/llvm-config-15 /usr/bin/llvm-config`

Install the Python dependencies for the auto-tuner and tests:
```bash
pip3 install pygad numpy pandas
```

## 2. Building the Compiler

Once your dependencies are installed, you can build the compiler using CMake. 

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Run CMake configuration. We point it to the MLIR CMake files provided by your LLVM installation:
```bash
# Point CMAKE to the correct MLIR directory based on your LLVM version
cmake .. -DMLIR_DIR=$(llvm-config --cmakedir)/../mlir -DCMAKE_BUILD_TYPE=Release
```

3. Compile the project:
```bash
make -j$(nproc)
```
After a successful build, you will see the `quantfx-compiler` executable inside `build/tools/quantfx-compiler/` (or directly in the `build/` root depending on the binary output directory).

## 3. Trying Out the Compiler

You can try compiling one of the example `.qfx` DSL files. We provide multiple examples in the `examples/` directory.

To see the generated MLIR (Mid-Level Intermediate Representation):
```bash
cd .. # Go back to project root
./build/quantfx-compiler examples/vwap_signal.qfx --emit-mlir
```

To see the generated LLVM IR (Low-Level Virtual Machine IR):
```bash
./build/quantfx-compiler examples/vwap_signal.qfx --emit-llvm
```

## 4. Running the Tests

We ensure that the `quantfx-compiler` produces numerically correct results by comparing its outputs against standard Python/NumPy implementations.

To run the full correctness test suite:
```bash
cd tests
python3 run_correctness.py
```
*You should see all operations (rolling mean, VWAP, z-score, etc.) report `PASS`.*

## 5. Running the STAC-A2 Benchmark

The real power of `quantfx-compiler` is its low-latency execution of financial workloads. You can test the end-to-end streaming latency.

1. **Generate Synthetic Tick Data**:
```bash
# Generates 100,000 mock LOBSTER messages for the test
python3 scripts/generate_lobster.py benchmarks/tick/lobster_sample.csv 100000
```

2. **Run the Streaming Benchmark**:
```bash
# Execute the full pipeline
./build/run_stac_a2 --data benchmarks/tick/lobster_sample.csv
```
*You will see the per-tick median and p99 latencies printed to your terminal. On a modern CPU, the median latency per tick should be under 100ns!*

## 6. Auto-Tuning a Kernel

If you want to try the genetic algorithm auto-tuner, you can run it on a specific kernel to let it find the best optimization passes:

```bash
python3 tuner/search.py --kernel examples/corr_matrix.qfx --target cpu --generations 10 --population 16
```
This will compile the kernel with different flags, benchmark it, and evolve the best configuration.

---

> [!NOTE]
> If you run into any issues during setup, please make sure your `LLVM_DIR` and `MLIR_DIR` are correctly detected by CMake. You can manually pass `-DLLVM_DIR=/usr/lib/llvm-15/cmake` if CMake complains about missing LLVM.
